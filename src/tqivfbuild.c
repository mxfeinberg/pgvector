#include "postgres.h"

#include <float.h>
#include <math.h>

#include "access/generic_xlog.h"
#include "access/tableam.h"
#include "catalog/index.h"
#include "catalog/pg_operator_d.h"
#include "catalog/pg_type_d.h"
#include "executor/executor.h"
#include "miscadmin.h"
#include "storage/bufmgr.h"
#include "storage/lmgr.h"
#include "tqivf.h"
#include "utils/memutils.h"
#include "utils/rel.h"
#include "utils/tuplesort.h"
#include "vector.h"

/*
 * Per-list streaming cursor for emitting one list's TurboQuant v4 block stream.
 *
 * tqflat keeps a single global code/side cursor in its TqBuildState; tqivf
 * needs one cursor per list (lists are written contiguously, one at a time, so
 * a single reused cursor structure that is reset per list suffices).  The
 * on-disk byte layout produced here is byte-for-byte identical to tqflat's
 * (same MAXALIGN chunking via TqIvfCodeAppend, same per-block side records via
 * PageAddItem), so the scan path can read it back with TqReadBytes +
 * TqScoreBlockRange exactly as for a tqflat index.
 */
typedef struct TqivfListCursor
{
	/* Code-plane chain */
	BlockNumber codeStart;
	Buffer		codeBuf;
	Page		codePage;
	GenericXLogState *codeState;

	/* Side-record chain */
	BlockNumber sideStart;
	Buffer		sideBuf;
	Page		sidePage;
	GenericXLogState *sideState;

	/* Block staging */
	uint8	   *codeStage;		/* TQ_BLOCK_CODE_BYTES(dimCodes) */
	TqBlockSideRec sideStage;
	int			slot;			/* next free lane 0..TQ_BLOCK_WIDTH-1 */
	uint32		blockCount;
	uint32		nvectors;
}			TqivfListCursor;

typedef struct TqivfBuildState
{
	Relation	heap;
	Relation	index;
	IndexInfo  *indexInfo;
	ForkNumber	forkNum;

	/* Model parameters */
	int			dim;
	int			bits;
	int			nLevels;
	TqMetric	metric;
	bool		fastRotation;
	int			dimPadded;
	int			dimCodes;		/* dim, or next_pow2(dim) in fast mode */
	TqModel		model;

	/* Clustering (full-precision, un-rotated centers) */
	int			lists;

	/* Assignment / sorting */
	FmgrInfo   *procinfo;		/* exact distance (proc 1) for assignment */
	FmgrInfo   *normprocinfo;	/* proc 2 (ip/cosine); NULL for l2 */
	Oid			collation;
	Tuplesortstate *sortstate;
	TupleDesc	sortdesc;
	TupleDesc	tupdesc;
	TupleTableSlot *slot;
	VectorArray centers;

	/* List directory locations for back-patching */
	ListInfo   *listInfo;

	/* Counters */
	double		reltuples;
	double		indtuples;

	MemoryContext tmpCtx;
}			TqivfBuildState;

/*
 * Append raw bytes to a list's streaming code-plane chain (replicated from
 * tqbuild.c's TqCodeAppend, operating on the per-list cursor).  Keep the
 * chunking logic byte-identical so the reassembled stream matches tqflat's.
 */
static void
TqIvfCodeAppend(Relation index, ForkNumber forkNum, TqivfListCursor * cur,
				const char *bytes, Size nbytes)
{
	Size		offset = 0;

	/* Lazily open the code chain on the first block flush. */
	if (cur->codePage == NULL)
	{
		cur->codeBuf = TqNewBuffer(index, forkNum);
		TqInitRegisterPage(index, &cur->codeBuf, &cur->codePage, &cur->codeState,
						   TQIVF_PAGE_ID);
		cur->codeStart = BufferGetBlockNumber(cur->codeBuf);
	}

	while (offset < nbytes)
	{
		Size		avail = PageGetFreeSpace(cur->codePage);
		Size		chunk;
		OffsetNumber offno;

		if (avail <= sizeof(ItemIdData))
		{
			TqAppendPage(index, &cur->codeBuf, &cur->codePage, &cur->codeState, forkNum,
						 TQIVF_PAGE_ID);
			continue;
		}

		chunk = avail - sizeof(ItemIdData);
		chunk = chunk - (chunk % MAXIMUM_ALIGNOF);
		if (chunk == 0)
		{
			TqAppendPage(index, &cur->codeBuf, &cur->codePage, &cur->codeState, forkNum,
						 TQIVF_PAGE_ID);
			continue;
		}
		if (chunk > nbytes - offset)
			chunk = nbytes - offset;

		offno = PageAddItem(cur->codePage, (Item) (bytes + offset), chunk,
							InvalidOffsetNumber, false, false);
		if (offno == InvalidOffsetNumber)
			elog(ERROR, "failed to add code-plane item to \"%s\"", RelationGetRelationName(index));

		offset += chunk;
	}
}

/*
 * Append one TqBlockSideRec to a list's side chain (replicated from tqbuild.c's
 * TqAppendSideRec, operating on the per-list cursor).
 */
static void
TqIvfAppendSideRec(Relation index, ForkNumber forkNum, TqivfListCursor * cur,
				   const TqBlockSideRec *rec)
{
	OffsetNumber offno;

	if (cur->sidePage == NULL)
	{
		cur->sideBuf = TqNewBuffer(index, forkNum);
		TqInitRegisterPage(index, &cur->sideBuf, &cur->sidePage, &cur->sideState,
						   TQIVF_PAGE_ID);
		cur->sideStart = BufferGetBlockNumber(cur->sideBuf);
	}

	if (PageGetFreeSpace(cur->sidePage) < sizeof(TqBlockSideRec))
		TqAppendPage(index, &cur->sideBuf, &cur->sidePage, &cur->sideState, forkNum,
					 TQIVF_PAGE_ID);

	offno = PageAddItem(cur->sidePage, (Item) rec, sizeof(TqBlockSideRec),
						InvalidOffsetNumber, false, false);
	if (offno == InvalidOffsetNumber)
		elog(ERROR, "failed to add tqivf side record to \"%s\"",
			 RelationGetRelationName(index));
}

/*
 * Flush the currently-staged block of a list cursor (replicated from
 * tqbuild.c's TqFlushBlock).
 */
static void
TqIvfFlushBlock(TqivfBuildState * buildstate, TqivfListCursor * cur)
{
	int			dc = buildstate->dimCodes;

	cur->sideStage.nvecs = (uint16) cur->slot;
	cur->sideStage.deletedMask = 0;
	cur->sideStage.pad = 0;

	TqIvfCodeAppend(buildstate->index, buildstate->forkNum, cur,
					(char *) cur->codeStage, TQ_BLOCK_CODE_BYTES(dc));
	TqIvfAppendSideRec(buildstate->index, buildstate->forkNum, cur, &cur->sideStage);

	memset(cur->codeStage, 0, TQ_BLOCK_CODE_BYTES(dc));
	cur->slot = 0;
	cur->blockCount++;
	MemSet(&cur->sideStage, 0, sizeof(cur->sideStage));
}

/*
 * Build the global model (rotation + codebook) and write the codebook (and, in
 * dense mode, the rotation) to side pages.  This is TurboQuant's
 * data-OBLIVIOUS, list-independent model: ONE rotation + ONE codebook shared by
 * every list.  Replicated from tqbuild.c's TqBuildModelAndSidePages (the model
 * fields it writes into TqBuildState live in TqivfBuildState here, and it uses
 * the now-exported TqWriteBytes).  tqProd / QJL are unsupported in the v4
 * blocked layout, so the QJL branch is omitted.
 */
static void
TqivfBuildModelAndSidePages(TqivfBuildState * buildstate, BlockNumber *rotStart,
							BlockNumber *cbStart)
{
	int			dim = buildstate->dim;
	int			bits = buildstate->bits;
	int			nLevels = buildstate->nLevels;
	int			nBnd = nLevels - 1;
	bool		fastRotation = buildstate->fastRotation;
	int			dimPadded = fastRotation ? TqNextPow2(dim) : dim;
	int			dimCodes = fastRotation ? dimPadded : dim;
	Size		rotBytes = (Size) sizeof(float) * dim * dim;
	Size		cbBytes = (Size) sizeof(float) * (nBnd + nLevels);
	char	   *cbBuf;

	buildstate->model.dim = dim;
	buildstate->model.bits = bits;
	buildstate->model.nLevels = nLevels;
	buildstate->model.metric = buildstate->metric;
	buildstate->model.tqProd = false;
	buildstate->model.qjl = NULL;
	buildstate->model.qjlScale = 0.0f;
	buildstate->model.fastRotation = fastRotation;
	buildstate->model.dimPadded = dimPadded;
	buildstate->model.dimCodes = dimCodes;
	buildstate->model.rotation = NULL;
	buildstate->model.rotSeed = TQ_ROTATION_SEED;
	buildstate->model.qjlSeed = TQ_QJL_SEED;
	buildstate->model.boundaries = palloc(sizeof(float) * nBnd);
	buildstate->model.centroids = palloc(sizeof(float) * nLevels);

	buildstate->dimPadded = dimPadded;
	buildstate->dimCodes = dimCodes;

	TqBuildCodebook(dimCodes, bits, buildstate->model.boundaries, buildstate->model.centroids);

	if (fastRotation)
	{
		*rotStart = InvalidBlockNumber;
	}
	else
	{
		buildstate->model.rotation = palloc(rotBytes);
		TqBuildRotation(dim, TQ_ROTATION_SEED, buildstate->model.rotation);
		*rotStart = TqWriteBytes(buildstate->index, buildstate->forkNum,
								 (const char *) buildstate->model.rotation, rotBytes,
								 TQIVF_PAGE_ID);
	}

	cbBuf = palloc(cbBytes);
	memcpy(cbBuf, buildstate->model.boundaries, sizeof(float) * nBnd);
	memcpy(cbBuf + sizeof(float) * nBnd, buildstate->model.centroids, sizeof(float) * nLevels);
	*cbStart = TqWriteBytes(buildstate->index, buildstate->forkNum, cbBuf, cbBytes,
							TQIVF_PAGE_ID);
	pfree(cbBuf);
}

/*
 * Initialize the build state from index relation / options.  Mirrors
 * TqInitBuildState (model params) + InitBuildState (clustering params).
 */
static void
TqivfInitBuildState(TqivfBuildState * buildstate, Relation heap, Relation index,
					IndexInfo *indexInfo, ForkNumber forkNum)
{
	TqivfOptions *opts = (TqivfOptions *) index->rd_options;
	const		IvfflatTypeInfo *typeInfo;

	buildstate->heap = heap;
	buildstate->index = index;
	buildstate->indexInfo = indexInfo;
	buildstate->forkNum = forkNum;

	/* Dimensions from the index column typmod (mirror tqflat/ivfflat) */
	buildstate->dim = TupleDescAttr(index->rd_att, 0)->atttypmod;
	if (buildstate->dim < 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("column does not have dimensions")));
	if (buildstate->dim > TQ_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("column cannot have more than %d dimensions for tqivf index", TQ_MAX_DIM)));
	if (buildstate->dim < 3)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("tqivf index requires at least 3 dimensions")));

	/* Options.  bits is FIXED at 4 (the v4 blocked layout). */
	buildstate->bits = 4;
	buildstate->fastRotation = opts ? opts->fastRotation : TQ_DEFAULT_FAST_ROTATION;
	buildstate->lists = opts ? opts->lists : TQIVF_DEFAULT_LISTS;
	buildstate->nLevels = 1 << buildstate->bits;

	/* Metric from the opclass type-info support function (default L2) */
	buildstate->metric = TQ_METRIC_L2;
	if (OidIsValid(index_getprocid(index, 1, TQIVF_TYPE_INFO_PROC)))
	{
		FmgrInfo   *typeProc = index_getprocinfo(index, 1, TQIVF_TYPE_INFO_PROC);
		TqTypeInfo *ti = (TqTypeInfo *) DatumGetPointer(FunctionCall0Coll(typeProc, InvalidOid));

		buildstate->metric = ti->metric;
	}

	/* Assignment / probe support functions (FUNCTION 1 = exact distance) */
	buildstate->procinfo = index_getprocinfo(index, 1, TQIVF_DISTANCE_PROC);
	buildstate->normprocinfo = IvfflatOptionalProcInfo(index, TQIVF_NORM_PROC);
	buildstate->collation = index->rd_indcollation[0];

	/*
	 * Clustering context: tqivf deliberately leaves opclass FUNCTION slot 5
	 * unregistered, so IvfflatGetTypeInfo() returns the default Vector
	 * typeInfo.  Centers are sized to `lists`; IvfflatKmeans reads k from
	 * centers->maxlen / centers->length and never touches reloptions.
	 */
	typeInfo = IvfflatGetTypeInfo(index);
	buildstate->centers = VectorArrayInit(buildstate->lists, buildstate->dim,
										  typeInfo->itemSize(buildstate->dim));
	buildstate->listInfo = palloc(sizeof(ListInfo) * buildstate->lists);

	/* Sort tuple descriptor: (list int4, tid tid, vector) keyed by list */
	buildstate->tupdesc = RelationGetDescr(index);
	buildstate->sortdesc = CreateTemplateTupleDesc(3);
	TupleDescInitEntry(buildstate->sortdesc, (AttrNumber) 1, "list", INT4OID, -1, 0);
	TupleDescInitEntry(buildstate->sortdesc, (AttrNumber) 2, "tid", TIDOID, -1, 0);
	TupleDescInitEntry(buildstate->sortdesc, (AttrNumber) 3, "vector",
					   TupleDescAttr(buildstate->tupdesc, 0)->atttypid, -1, 0);
#if PG_VERSION_NUM >= 190000
	TupleDescFinalize(buildstate->sortdesc);
#endif
	buildstate->slot = MakeSingleTupleTableSlot(buildstate->sortdesc, &TTSOpsVirtual);

	buildstate->reltuples = 0;
	buildstate->indtuples = 0;

	buildstate->tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
											   "Tqivf build temporary context",
											   ALLOCSET_DEFAULT_SIZES);
}

static void
TqivfFreeBuildState(TqivfBuildState * buildstate)
{
	VectorArrayFree(buildstate->centers);
	pfree(buildstate->listInfo);
	MemoryContextDelete(buildstate->tmpCtx);
}

/*
 * Run sample + k-means to produce full-precision, un-rotated centers.  Reuses
 * ivfflat's SampleRows + IvfflatKmeans via a locally-populated
 * IvfflatBuildState (only the fields those routines read are set).
 */
static void
TqivfComputeCenters(TqivfBuildState * buildstate)
{
	IvfflatBuildState ivfstate;
	const		IvfflatTypeInfo *typeInfo = IvfflatGetTypeInfo(buildstate->index);
	int			numSamples;

	MemSet(&ivfstate, 0, sizeof(ivfstate));
	ivfstate.heap = buildstate->heap;
	ivfstate.index = buildstate->index;
	ivfstate.indexInfo = buildstate->indexInfo;
	ivfstate.typeInfo = typeInfo;
	ivfstate.dimensions = buildstate->dim;
	ivfstate.lists = buildstate->lists;
	ivfstate.centers = buildstate->centers;
	ivfstate.collation = buildstate->collation;

	/* k-means uses proc 3; spherical norm proc 4 (ip/cosine only) */
	ivfstate.procinfo = index_getprocinfo(buildstate->index, 1, TQIVF_KMEANS_DISTANCE_PROC);
	ivfstate.normprocinfo = IvfflatOptionalProcInfo(buildstate->index, TQIVF_NORM_PROC);
	ivfstate.kmeansnormprocinfo = IvfflatOptionalProcInfo(buildstate->index, TQIVF_KMEANS_NORM_PROC);

	ivfstate.tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
											"Tqivf kmeans sample context",
											ALLOCSET_DEFAULT_SIZES);

	/* Target 50 samples per list, with at least 10000 samples (mirror ivfflat) */
	numSamples = buildstate->lists * 50;
	if (numSamples < 10000)
		numSamples = 10000;
	if (buildstate->heap == NULL)
		numSamples = 1;

	ivfstate.samples = VectorArrayInit(numSamples, buildstate->dim, buildstate->centers->itemsize);
	if (buildstate->heap != NULL)
	{
		SampleRows(&ivfstate);

		if (ivfstate.samples->length < buildstate->lists)
			ereport(NOTICE,
					(errmsg("tqivf index created with little data"),
					 errdetail("This will cause low recall."),
					 errhint("Drop the index until the table has more data.")));
	}

	IvfflatKmeans(buildstate->index, ivfstate.samples, buildstate->centers, typeInfo);

	VectorArrayFree(ivfstate.samples);
	MemoryContextDelete(ivfstate.tmpCtx);
}

/*
 * Write the `lists` list-directory records up front (Invalid chain heads, zero
 * counts), recording each record's (blkno, offno) for later back-patching.
 * Mirrors ivfbuild.c's CreateListPages, using TQIVF_LIST_SIZE for item size.
 */
static BlockNumber
TqivfCreateListPages(TqivfBuildState * buildstate)
{
	Relation	index = buildstate->index;
	ForkNumber	forkNum = buildstate->forkNum;
	Buffer		buf;
	Page		page;
	GenericXLogState *state;
	BlockNumber listStart;
	Size		listSize = MAXALIGN(TQIVF_LIST_SIZE(buildstate->dim));
	TqivfList	list = palloc0(listSize);

	buf = TqNewBuffer(index, forkNum);
	TqInitRegisterPage(index, &buf, &page, &state, TQIVF_PAGE_ID);
	listStart = BufferGetBlockNumber(buf);

	for (int i = 0; i < buildstate->lists; i++)
	{
		OffsetNumber offno;
		Pointer		center = VectorArrayGet(buildstate->centers, i);

		MemSet(list, 0, listSize);
		list->codeStart = InvalidBlockNumber;
		list->sideStart = InvalidBlockNumber;
		list->tailStart = InvalidBlockNumber;
		list->tailInsertPage = InvalidBlockNumber;
		list->blockCount = 0;
		list->nvectors = 0;
		memcpy(&list->center, center, VARSIZE_ANY(center));

		if (PageGetFreeSpace(page) < listSize)
			TqAppendPage(index, &buf, &page, &state, forkNum, TQIVF_PAGE_ID);

		offno = PageAddItem(page, (Item) list, listSize, InvalidOffsetNumber, false, false);
		if (offno == InvalidOffsetNumber)
			elog(ERROR, "failed to add tqivf list item to \"%s\"", RelationGetRelationName(index));

		buildstate->listInfo[i].blkno = BufferGetBlockNumber(buf);
		buildstate->listInfo[i].offno = offno;
	}

	TqCommitBuffer(buf, state);
	pfree(list);

	return listStart;
}

/*
 * Assignment callback: find the nearest center (FUNCTION 1 distance over
 * full-precision, un-rotated centers) and push (list, tid, vector) into the
 * tuplesort keyed by list.  Mirrors ivfbuild.c's AddTupleToSort/BuildCallback.
 */
static void
TqivfBuildCallback(Relation index, ItemPointer tid, Datum *values,
				   bool *isnull, bool tupleIsAlive, void *state)
{
	TqivfBuildState *buildstate = (TqivfBuildState *) state;
	MemoryContext oldCtx;
	Datum		value;
	double		minDistance = DBL_MAX;
	int			closestCenter = 0;
	VectorArray centers = buildstate->centers;
	TupleTableSlot *slot = buildstate->slot;

	if (isnull[0])
		return;

	oldCtx = MemoryContextSwitchTo(buildstate->tmpCtx);

	value = PointerGetDatum(PG_DETOAST_DATUM(values[0]));

	/* Normalize for spherical assignment (ip/cosine), mirror ivfflat */
	if (buildstate->normprocinfo != NULL)
	{
		if (!IvfflatCheckNorm(buildstate->normprocinfo, buildstate->collation, value))
		{
			MemoryContextSwitchTo(oldCtx);
			MemoryContextReset(buildstate->tmpCtx);
			return;
		}

		value = IvfflatNormValue(IvfflatGetTypeInfo(index), buildstate->collation, value);
	}

	for (int i = 0; i < centers->length; i++)
	{
		double		distance = DatumGetFloat8(FunctionCall2Coll(buildstate->procinfo,
																buildstate->collation, value,
																PointerGetDatum(VectorArrayGet(centers, i))));

		if (distance < minDistance)
		{
			minDistance = distance;
			closestCenter = i;
		}
	}

	ExecClearTuple(slot);
	slot->tts_values[0] = Int32GetDatum(closestCenter);
	slot->tts_isnull[0] = false;
	slot->tts_values[1] = PointerGetDatum(tid);
	slot->tts_isnull[1] = false;
	slot->tts_values[2] = value;
	slot->tts_isnull[2] = false;
	ExecStoreVirtualTuple(slot);

	tuplesort_puttupleslot(buildstate->sortstate, slot);

	buildstate->indtuples++;

	MemoryContextSwitchTo(oldCtx);
	MemoryContextReset(buildstate->tmpCtx);
}

/*
 * Emit the sorted tuples into per-list block streams, back-patching each list
 * directory record.  Replicates tqflat's per-block flush over a per-list cursor.
 */
static void
TqivfEmitLists(TqivfBuildState * buildstate)
{
	Relation	index = buildstate->index;
	ForkNumber	forkNum = buildstate->forkNum;
	int			dc = buildstate->dimCodes;
	TupleTableSlot *sortSlot = MakeSingleTupleTableSlot(buildstate->sortdesc, &TTSOpsMinimalTuple);
	TqivfListCursor cur;
	TqEntry    *entry;
	Size		entrySize;
	int			list;
	bool		haveTuple;
	bool		isnull;

	/* Scratch entry sized over dimCodes (mirror tqflat) */
	entrySize = TqEntrySize(buildstate->dimCodes, buildstate->bits, false);
	entry = palloc0(entrySize);

	cur.codeStage = palloc0(TQ_BLOCK_CODE_BYTES(dc));

	/* Prime the first sorted tuple */
	haveTuple = tuplesort_gettupleslot(buildstate->sortstate, true, false, sortSlot, NULL);
	if (haveTuple)
		list = DatumGetInt32(slot_getattr(sortSlot, 1, &isnull));
	else
		list = -1;

	for (int i = 0; i < buildstate->lists; i++)
	{
		CHECK_FOR_INTERRUPTS();

		/* Reset the per-list cursor + block staging */
		cur.codeStart = InvalidBlockNumber;
		cur.codeBuf = InvalidBuffer;
		cur.codePage = NULL;
		cur.codeState = NULL;
		cur.sideStart = InvalidBlockNumber;
		cur.sideBuf = InvalidBuffer;
		cur.sidePage = NULL;
		cur.sideState = NULL;
		cur.slot = 0;
		cur.blockCount = 0;
		cur.nvectors = 0;
		MemSet(&cur.sideStage, 0, sizeof(cur.sideStage));
		memset(cur.codeStage, 0, TQ_BLOCK_CODE_BYTES(dc));

		while (haveTuple && list == i)
		{
			Datum		value = slot_getattr(sortSlot, 3, &isnull);
			Vector	   *vec = DatumGetVector(value);
			ItemPointer tidp = (ItemPointer) DatumGetPointer(slot_getattr(sortSlot, 2, &isnull));
			int			lane;

			memset(entry, 0, entrySize);
			TqEncode(&buildstate->model, vec->x, entry);

			lane = cur.slot;
			TqScatterCodes(&buildstate->model, entry->data, lane, cur.codeStage);
			cur.sideStage.side[lane].heaptid = *tidp;
			cur.sideStage.side[lane].norm = entry->norm;
			cur.sideStage.side[lane].scale = entry->scale;
			cur.slot++;
			cur.nvectors++;

			if (cur.slot == TQ_BLOCK_WIDTH)
				TqIvfFlushBlock(buildstate, &cur);

			haveTuple = tuplesort_gettupleslot(buildstate->sortstate, true, false, sortSlot, NULL);
			if (haveTuple)
				list = DatumGetInt32(slot_getattr(sortSlot, 1, &isnull));
			else
				list = -1;
		}

		/* Flush trailing partial block for this list */
		if (cur.slot > 0)
			TqIvfFlushBlock(buildstate, &cur);

		/* Close the per-list page cursors */
		if (cur.sidePage != NULL)
			TqCommitBuffer(cur.sideBuf, cur.sideState);
		if (cur.codePage != NULL)
			TqCommitBuffer(cur.codeBuf, cur.codeState);

		/* Back-patch the list directory record in place */
		{
			Buffer		buf;
			Page		page;
			GenericXLogState *state;
			TqivfList	dirlist;

			buf = ReadBufferExtended(index, forkNum, buildstate->listInfo[i].blkno, RBM_NORMAL, NULL);
			LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
			state = GenericXLogStart(index);
			page = GenericXLogRegisterBuffer(state, buf, 0);
			dirlist = (TqivfList) PageGetItem(page, PageGetItemId(page, buildstate->listInfo[i].offno));

			dirlist->codeStart = cur.codeStart;
			dirlist->sideStart = cur.sideStart;
			dirlist->tailStart = InvalidBlockNumber;
			dirlist->tailInsertPage = InvalidBlockNumber;
			dirlist->blockCount = cur.blockCount;
			dirlist->nvectors = cur.nvectors;

			TqCommitBuffer(buf, state);
		}
	}

	pfree(cur.codeStage);
	pfree(entry);
	ExecDropSingleTupleTableSlot(sortSlot);
}

/*
 * Write the meta page at block TQIVF_METAPAGE_BLKNO.  Variable block numbers /
 * nVectors are back-patched after the side pages and lists are written.
 */
static void
TqivfCreateMetaPage(TqivfBuildState * buildstate)
{
	Relation	index = buildstate->index;
	Buffer		buf;
	Page		page;
	GenericXLogState *state;
	TqivfMetaPage metap;

	buf = TqNewBuffer(index, buildstate->forkNum);
	Assert(BufferGetBlockNumber(buf) == TQIVF_METAPAGE_BLKNO);
	TqInitRegisterPage(index, &buf, &page, &state, TQIVF_PAGE_ID);

	metap = TqivfPageGetMeta(page);
	metap->magicNumber = TQIVF_MAGIC_NUMBER;
	metap->version = TQIVF_VERSION;
	metap->dimensions = (uint16) buildstate->dim;
	metap->bits = (uint16) buildstate->bits;
	metap->metric = (uint16) buildstate->metric;
	metap->fastRotation = (uint16) (buildstate->fastRotation ? 1 : 0);
	metap->dimPadded = (uint16) buildstate->dimPadded;
	metap->lists = (uint16) buildstate->lists;
	metap->nLevels = (uint32) buildstate->nLevels;
	metap->nVectors = 0;
	metap->listStart = InvalidBlockNumber;
	metap->codebookStart = InvalidBlockNumber;
	metap->rotationStart = InvalidBlockNumber;
	metap->rotSeed = TQ_ROTATION_SEED;
	metap->qjlSeed = TQ_QJL_SEED;
	metap->qjlScale = 0.0f;

	((PageHeader) page)->pd_lower =
		((char *) metap + sizeof(TqivfMetaPageData)) - (char *) page;

	TqCommitBuffer(buf, state);
}

/*
 * Back-patch the variable fields of the meta page after the directory + side
 * pages are written.
 */
static void
TqivfUpdateMeta(TqivfBuildState * buildstate, BlockNumber listStart,
				BlockNumber codebookStart, BlockNumber rotationStart,
				uint32 nVectors)
{
	Relation	index = buildstate->index;
	Buffer		buf;
	Page		page;
	GenericXLogState *state;
	TqivfMetaPage metap;

	buf = ReadBufferExtended(index, buildstate->forkNum, TQIVF_METAPAGE_BLKNO, RBM_NORMAL, NULL);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	state = GenericXLogStart(index);
	page = GenericXLogRegisterBuffer(state, buf, 0);
	metap = TqivfPageGetMeta(page);

	metap->listStart = listStart;
	metap->codebookStart = codebookStart;
	metap->rotationStart = rotationStart;
	metap->nVectors = nVectors;

	/*
	 * dimPadded is only known after TqivfBuildModelAndSidePages runs, which is
	 * after the meta page is first written; back-patch it here.
	 */
	metap->dimPadded = (uint16) buildstate->dimPadded;

	TqCommitBuffer(buf, state);
}

/*
 * Build the index (shared by tqivfbuild and tqivfbuildempty).
 */
static void
TqivfBuildIndex(Relation heap, Relation index, IndexInfo *indexInfo,
				TqivfBuildState * buildstate, ForkNumber forkNum)
{
	BlockNumber rotStart;
	BlockNumber cbStart;
	BlockNumber listStart;
	AttrNumber	attNums[] = {1};
	Oid			sortOperators[] = {Int4LessOperator};
	Oid			sortCollations[] = {InvalidOid};
	bool		nullsFirstFlags[] = {false};

	TqivfInitBuildState(buildstate, heap, index, indexInfo, forkNum);

	/* Meta page first (block 0), then the global model side pages */
	TqivfCreateMetaPage(buildstate);
	TqivfBuildModelAndSidePages(buildstate, &rotStart, &cbStart);

	/* Sample + k-means → full-precision, un-rotated centers */
	TqivfComputeCenters(buildstate);

	/* List directory up front; record its head for the meta page */
	listStart = TqivfCreateListPages(buildstate);

	/* Assign + sort by list id */
	buildstate->sortstate = tuplesort_begin_heap(buildstate->sortdesc, 1, attNums,
												 sortOperators, sortCollations,
												 nullsFirstFlags, maintenance_work_mem,
												 NULL, false);

	if (heap != NULL)
		buildstate->reltuples = table_index_build_scan(heap, index, indexInfo,
													   true, true, TqivfBuildCallback,
													   (void *) buildstate, NULL);

	tuplesort_performsort(buildstate->sortstate);

	/* Per-list block emit + directory back-patch */
	TqivfEmitLists(buildstate);

	tuplesort_end(buildstate->sortstate);

	/* Finalize the meta page */
	TqivfUpdateMeta(buildstate, listStart, cbStart, rotStart,
					(uint32) buildstate->indtuples);

	/* Write WAL for the init fork since GenericXLog does not */
	if (forkNum == INIT_FORKNUM)
		log_newpage_range(index, forkNum, 0, RelationGetNumberOfBlocksInFork(index, forkNum), true);

	TqivfFreeBuildState(buildstate);
}

IndexBuildResult *
tqivfbuild(Relation heap, Relation index, struct IndexInfo *indexInfo)
{
	IndexBuildResult *result;
	TqivfBuildState buildstate;

	MemSet(&buildstate, 0, sizeof(buildstate));
	TqivfBuildIndex(heap, index, indexInfo, &buildstate, MAIN_FORKNUM);

	result = (IndexBuildResult *) palloc(sizeof(IndexBuildResult));
	result->heap_tuples = buildstate.reltuples;
	result->index_tuples = buildstate.indtuples;

	return result;
}

void
tqivfbuildempty(Relation index)
{
	IndexInfo  *indexInfo = BuildIndexInfo(index);
	TqivfBuildState buildstate;

	MemSet(&buildstate, 0, sizeof(buildstate));
	TqivfBuildIndex(NULL, index, indexInfo, &buildstate, INIT_FORKNUM);
}

/*
 * tqivfinsert -- insert one tuple into an existing tqivf index.
 *
 * Steps:
 *   1. Skip nulls; create a short-lived memory context (mirrors tqinsert).
 *   2. Detoast vector, load cached model, get metric/lists/listStart.
 *   3. Optionally normalize (FUNCTION 2 / TQIVF_NORM_PROC), same as build.
 *   4. Walk the list directory to find the nearest centroid (FUNCTION 1
 *      exact distance).  Record its blkno/offno plus tailStart/tailInsertPage.
 *   5. Encode the vector into a row-major TqEntry (same format the tail-scan
 *      path reads via TqScoreEntry).
 *   6. Append the entry to the list's tail chain at tailInsertPage (O(1)).
 *      If no tail chain yet: allocate first page, set both tailStart and
 *      tailInsertPage.  Otherwise walk/extend as needed (mirrors tqinsert's
 *      append_entry loop with ivfinsert's concurrency pattern).
 *   7. If tailStart/tailInsertPage changed, update the directory record in
 *      place under GenericXLog (mirrors IvfflatUpdateList).
 *   8. Clean up and return false (same as tqinsert/ivfflatinsert).
 *
 * Concurrency: mirrors tqinsert / ivfinsert.  We never hold two data-page
 * buffer locks simultaneously.  Directory-record update uses a separate
 * GenericXLog after all data-page locks are released.
 */
bool
tqivfinsert(Relation index, Datum *values, bool *isnull, ItemPointer heap_tid,
			Relation heap, IndexUniqueCheck checkUnique, bool indexUnchanged,
			struct IndexInfo *indexInfo)
{
	TqModel    *model;
	Vector	   *vec;
	TqEntry    *entry;
	Size		entrySize;
	MemoryContext insertCtx;
	MemoryContext oldCtx;
	FmgrInfo   *procinfo;
	FmgrInfo   *normprocinfo;
	Oid			collation;
	TqMetric	metric;
	int			lists;
	BlockNumber listStart;

	/* The list directory location of the best list. */
	ListInfo	bestListInfo;
	BlockNumber bestTailStart;
	BlockNumber bestTailInsertPage;

	/* Skip nulls (mirror ivfflat) */
	if (isnull[0])
		return false;

	/*
	 * Short-lived memory context for detoast / encode scratch (mirrors
	 * tqinsert / ivfflatinsert).
	 */
	insertCtx = AllocSetContextCreate(CurrentMemoryContext,
									  "Tqivf insert temporary context",
									  ALLOCSET_DEFAULT_SIZES);
	oldCtx = MemoryContextSwitchTo(insertCtx);

	/* Load (or reuse cached) model */
	model = TqivfGetCachedModel(index);

	/* Get meta info: metric, list count, directory head. */
	TqivfGetMetaInfo(index, NULL, &metric, &lists, &listStart);

	/* Detoast once for all calls (DatumGetVector already calls PG_DETOAST_DATUM) */
	vec = DatumGetVector(values[0]);

	/*
	 * Normalize if needed (FUNCTION 2 / TQIVF_NORM_PROC).  For inner product
	 * without a norm proc, skip — same as build.
	 */
	normprocinfo = IvfflatOptionalProcInfo(index, TQIVF_NORM_PROC);
	if (normprocinfo != NULL)
	{
		const		IvfflatTypeInfo *typeInfo = IvfflatGetTypeInfo(index);

		collation = index->rd_indcollation[0];

		if (!IvfflatCheckNorm(normprocinfo, collation, PointerGetDatum(vec)))
		{
			MemoryContextSwitchTo(oldCtx);
			MemoryContextDelete(insertCtx);
			return false;
		}

		vec = DatumGetVector(IvfflatNormValue(typeInfo, collation, PointerGetDatum(vec)));
	}

	/*
	 * Find the nearest list: walk the list-directory chain and compute the
	 * exact FUNCTION 1 distance from the vector to each centroid.  Record the
	 * directory location (blkno/offno) and the tail-chain pointers of the
	 * best list.  Mirrors ivfinsert's FindInsertPage but reading TqivfListData
	 * instead of IvfflatListData.
	 */
	{
		double		minDistance = DBL_MAX;
		BlockNumber nextblkno = listStart;
		bool		found = false;

		procinfo = index_getprocinfo(index, 1, TQIVF_DISTANCE_PROC);
		collation = index->rd_indcollation[0];

		/* Initialise to suppress compiler warnings */
		bestListInfo.blkno = listStart;
		bestListInfo.offno = FirstOffsetNumber;
		bestTailStart = InvalidBlockNumber;
		bestTailInsertPage = InvalidBlockNumber;

		while (BlockNumberIsValid(nextblkno))
		{
			Buffer		cbuf;
			Page		cpage;
			OffsetNumber maxoffno;
			OffsetNumber offno;
			BlockNumber nxt;

			cbuf = ReadBuffer(index, nextblkno);
			LockBuffer(cbuf, BUFFER_LOCK_SHARE);
			cpage = BufferGetPage(cbuf);
			maxoffno = PageGetMaxOffsetNumber(cpage);

			for (offno = FirstOffsetNumber; offno <= maxoffno; offno = OffsetNumberNext(offno))
			{
				TqivfList	list = (TqivfList) PageGetItem(cpage, PageGetItemId(cpage, offno));
				double		distance;

				distance = DatumGetFloat8(FunctionCall2Coll(procinfo, collation,
															PointerGetDatum(vec),
															PointerGetDatum(&list->center)));

				if (!found || distance < minDistance)
				{
					minDistance = distance;
					bestListInfo.blkno = nextblkno;
					bestListInfo.offno = offno;
					bestTailStart = list->tailStart;
					bestTailInsertPage = list->tailInsertPage;
					found = true;
				}
			}

			nxt = TqPageGetOpaque(cpage)->nextblkno;
			UnlockReleaseBuffer(cbuf);
			nextblkno = nxt;
		}

		if (!found)
		{
			MemoryContextSwitchTo(oldCtx);
			MemoryContextDelete(insertCtx);
			return false;
		}
	}

	/*
	 * Encode: allocate a row-major TqEntry (same format TqScoreEntry reads in
	 * the tail scan path).  bits=4, tqProd=false (tqivf v4 layout).
	 */
	entrySize = MAXALIGN(TqEntrySize(model->dimCodes, 4, false));
	entry = palloc0(entrySize);
	entry->heaptid = *heap_tid;
	entry->deleted = 0;
	TqEncode(model, vec->x, entry);

	MemoryContextSwitchTo(oldCtx);

	/*
	 * ---- Append to the list's tail chain ----
	 *
	 * Two cases:
	 *   A. No tail chain yet (tailStart == Invalid): allocate the first page,
	 *      add the entry, then under the directory lock set tailStart (if it is
	 *      still Invalid — another session may have won the race) and update
	 *      tailInsertPage.  If we lose the race, the just-allocated page becomes
	 *      a stranded page (reclaimed at REINDEX), and we fall through to Case B
	 *      to insert our entry into the winner's chain so no row is lost.
	 *   B. Tail chain exists: go to tailInsertPage; if there is room, add and
	 *      we are done; if full, walk/extend the chain (mirrors tqinsert's
	 *      append_entry loop), then update tailInsertPage in the directory.
	 *
	 * Chain invariant: every tail page that holds an entry MUST be reachable
	 * from the directory's tailStart by following nextblkno.  tailInsertPage is
	 * only a hint (the scan never uses it directly) and may be stale — it just
	 * saves walking the whole chain on the next insert.
	 *
	 * We track whether tailInsertPage actually changed so we only write the
	 * directory record when needed.
	 */
	{
		BlockNumber newTailInsertPage = bestTailInsertPage;
		bool		dirtyDir = false;

		if (!BlockNumberIsValid(bestTailStart))
		{
			/*
			 * Case A: first insert into this list's tail chain.
			 *
			 * Allocate and populate a new page before taking the directory lock
			 * (mirrors tqinsert's first-insert path).
			 */
			Buffer		newbuf;
			Page		newpage;
			GenericXLogState *newstate;
			BlockNumber newblk;
			OffsetNumber offno;

			LockRelationForExtension(index, ExclusiveLock);
			newbuf = TqNewBuffer(index, MAIN_FORKNUM);
			UnlockRelationForExtension(index, ExclusiveLock);

			newblk = BufferGetBlockNumber(newbuf);
			newstate = GenericXLogStart(index);
			newpage = GenericXLogRegisterBuffer(newstate, newbuf, GENERIC_XLOG_FULL_IMAGE);
			TqInitPage(newbuf, newpage, TQIVF_PAGE_ID);

			offno = PageAddItem(newpage, (Item) entry, entrySize,
								InvalidOffsetNumber, false, false);
			if (offno == InvalidOffsetNumber)
			{
				GenericXLogAbort(newstate);
				UnlockReleaseBuffer(newbuf);
				elog(ERROR, "failed to add tqivf tail entry to \"%s\"",
					 RelationGetRelationName(index));
			}

			GenericXLogFinish(newstate);
			UnlockReleaseBuffer(newbuf);

			/*
			 * Now take the directory lock and try to claim tailStart.  Another
			 * session may have won the race and already set tailStart while we
			 * were allocating and writing newblk.
			 */
			{
				Buffer		dbuf;
				Page		dpage;
				GenericXLogState *dstate;
				TqivfList	dirlist;

				dbuf = ReadBufferExtended(index, MAIN_FORKNUM,
										  bestListInfo.blkno, RBM_NORMAL, NULL);
				LockBuffer(dbuf, BUFFER_LOCK_EXCLUSIVE);
				dstate = GenericXLogStart(index);
				dpage = GenericXLogRegisterBuffer(dstate, dbuf, 0);
				dirlist = (TqivfList) PageGetItem(dpage,
												  PageGetItemId(dpage, bestListInfo.offno));

				if (BlockNumberIsValid(dirlist->tailStart))
				{
					/*
					 * Another session won the first-insert race: it set
					 * tailStart before we acquired the directory lock.  Our
					 * newblk page is committed but not linked into the chain —
					 * it becomes a stranded page (reclaimed at REINDEX), the
					 * same trade-off as tqinsert / ivfinsert.  Abort our
					 * directory xlog, capture the winner's tailInsertPage, and
					 * fall through to Case B so our entry is inserted into the
					 * winner's chain and is not lost.
					 */
					newTailInsertPage = dirlist->tailInsertPage;
					GenericXLogAbort(dstate);
					UnlockReleaseBuffer(dbuf);

					goto append_to_existing_chain;
				}

				/* We won: claim tailStart and tailInsertPage. */
				dirlist->tailStart = newblk;
				dirlist->tailInsertPage = newblk;

				GenericXLogFinish(dstate);
				UnlockReleaseBuffer(dbuf);
			}

			/* Winner path: entry is already on newblk, chain is set. Done. */
			goto insert_done;
		}

		/*
		 * Case B: tail chain already exists (normal path or loser fallback).
		 * Append to tailInsertPage, extending when full.  Mirrors tqinsert's
		 * append_entry loop.
		 */
append_to_existing_chain:
		{
			BlockNumber insertPage = newTailInsertPage;
			Buffer		buf;
			Page		page;
			GenericXLogState *state;

			for (;;)
			{
				OffsetNumber offno;
				BlockNumber nextblkno;

				buf = ReadBuffer(index, insertPage);
				LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);

				state = GenericXLogStart(index);
				page = GenericXLogRegisterBuffer(state, buf, 0);

				if (PageGetFreeSpace(page) >= entrySize)
				{
					/* Room on this page. */
					offno = PageAddItem(page, (Item) entry, entrySize,
										InvalidOffsetNumber, false, false);
					if (offno == InvalidOffsetNumber)
					{
						GenericXLogAbort(state);
						UnlockReleaseBuffer(buf);
						elog(ERROR, "failed to add tqivf tail entry to \"%s\"",
							 RelationGetRelationName(index));
					}

					GenericXLogFinish(state);
					UnlockReleaseBuffer(buf);
					break;
				}

				nextblkno = TqPageGetOpaque(page)->nextblkno;

				if (BlockNumberIsValid(nextblkno))
				{
					/*
					 * Concurrent inserter already extended; follow the link
					 * rather than overwriting it (mirrors tqinsert).  Both
					 * pages remain linked via nextblkno and are reachable from
					 * tailStart, preserving the chain invariant.
					 */
					GenericXLogAbort(state);
					UnlockReleaseBuffer(buf);
					insertPage = nextblkno;
					continue;
				}

				/* No room and no concurrent extension: allocate a new page. */
				{
					Buffer		newbuf;
					Page		newpage;
					BlockNumber newblkno;

					LockRelationForExtension(index, ExclusiveLock);
					newbuf = TqNewBuffer(index, MAIN_FORKNUM);
					UnlockRelationForExtension(index, ExclusiveLock);

					/*
					 * Register the new page in the same xlog as the old page
					 * so that setting nextblkno on the old page and
					 * initializing the new page are atomic.  This ensures the
					 * new page is linked into the chain (reachable from
					 * tailStart) before any reader can see it, preserving the
					 * chain invariant.
					 */
					newpage = GenericXLogRegisterBuffer(state, newbuf,
														GENERIC_XLOG_FULL_IMAGE);
					TqInitPage(newbuf, newpage, TQIVF_PAGE_ID);

					newblkno = BufferGetBlockNumber(newbuf);
					TqPageGetOpaque(page)->nextblkno = newblkno;

					/* Commit link on old page + init of new page. */
					GenericXLogFinish(state);
					UnlockReleaseBuffer(buf);

					/* Now insert into the new page under a fresh xlog. */
					state = GenericXLogStart(index);
					buf = newbuf;
					page = GenericXLogRegisterBuffer(state, buf, 0);

					offno = PageAddItem(page, (Item) entry, entrySize,
										InvalidOffsetNumber, false, false);
					if (offno == InvalidOffsetNumber)
					{
						GenericXLogAbort(state);
						UnlockReleaseBuffer(buf);
						elog(ERROR, "failed to add tqivf tail entry to \"%s\"",
							 RelationGetRelationName(index));
					}

					GenericXLogFinish(state);
					UnlockReleaseBuffer(buf);

					insertPage = newblkno;
					newTailInsertPage = newblkno;
					dirtyDir = true;
					break;
				}
			}
		}

		/*
		 * Update the list directory record if tailInsertPage changed (i.e. we
		 * extended the chain).  tailStart is already valid at this point (Case B
		 * only runs when a chain exists), so we never overwrite it.  This is a
		 * separate GenericXLog after all data-page locks have been released
		 * (mirrors IvfflatUpdateList's lock ordering).
		 */
		if (dirtyDir)
		{
			Buffer		dbuf;
			Page		dpage;
			GenericXLogState *dstate;
			TqivfList	dirlist;

			dbuf = ReadBufferExtended(index, MAIN_FORKNUM,
									  bestListInfo.blkno, RBM_NORMAL, NULL);
			LockBuffer(dbuf, BUFFER_LOCK_EXCLUSIVE);
			dstate = GenericXLogStart(index);
			dpage = GenericXLogRegisterBuffer(dstate, dbuf, 0);
			dirlist = (TqivfList) PageGetItem(dpage,
											  PageGetItemId(dpage, bestListInfo.offno));

			/*
			 * tailStart is guaranteed valid here (Case B path).  Only update
			 * tailInsertPage — it is a hint and may be stale, but must point
			 * to a page that is reachable from tailStart via nextblkno.
			 */
			Assert(BlockNumberIsValid(dirlist->tailStart));
			dirlist->tailInsertPage = newTailInsertPage;

			GenericXLogFinish(dstate);
			UnlockReleaseBuffer(dbuf);
		}
	}

insert_done:;

	MemoryContextDelete(insertCtx);

	return false;
}

/*
 * TqivfGetCachedModel -- load the TqModel from rd_amcache, or (re)load it.
 * Mirrors TqGetCachedModel.
 */
TqModel *
TqivfGetCachedModel(Relation index)
{
	if (index->rd_amcache != NULL)
		return (TqModel *) index->rd_amcache;

	index->rd_amcache = TqivfLoadModel(index, index->rd_indexcxt);
	return (TqModel *) index->rd_amcache;
}
