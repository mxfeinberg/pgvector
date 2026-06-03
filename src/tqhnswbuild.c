#include "postgres.h"

#include <math.h>

#include "access/generic_xlog.h"
#include "access/tableam.h"
#include "catalog/index.h"
#include "common/pg_prng.h"
#include "miscadmin.h"
#include "storage/bufmgr.h"
#include "tqhnsw.h"
#include "utils/memutils.h"
#include "utils/rel.h"

/*
 * tqhnswbuild reuses TqInitRegisterPage / TqInitPage, which size the page
 * special area with sizeof(TqPageOpaqueData), while tqhnsw readers cast the
 * special area to TqhnswPageOpaqueData.  They are currently byte-identical; this
 * assertion catches a future divergence at compile time.
 */
StaticAssertDecl(sizeof(TqhnswPageOpaqueData) == sizeof(TqPageOpaqueData),
				 "tqhnsw page opaque must match tq page opaque");

PGDLLEXPORT Datum l2_normalize(PG_FUNCTION_ARGS);

/*
 * Serial in-memory build state.  Holds the model, the in-memory graph (a
 * singly-linked build list rooted at head), the entry point, and the build
 * parameters / scratch encode buffer.
 */
typedef struct TqhnswBuildState
{
	Relation	index;
	TqModel    *model;
	int			dim;
	TqMetric	metric;
	int			m;
	int			efConstruction;
	bool		fastRotation;
	int			dimCodes;
	int			codesBytes;
	double		ml;
	int			maxLevel;

	TqEntry    *scratch;		/* TqEntrySize(dimCodes, bits, false) bytes */
	Size		scratchSize;

	TqhnswElement *head;		/* in-memory build list */
	TqhnswElement *entryPoint;
	int64		nVectors;
	BlockNumber firstElementPage;	/* first element page written during flush */

	MemoryContext graphCtx;		/* owns all in-memory graph nodes */
	MemoryContext tmpCtx;		/* per-tuple scratch */
}			TqhnswBuildState;

/*
 * TqhnswWriteModelAndMeta -- build the codebook (+ dense rotation), write the
 * meta page at block 0, and write the codebook/rotation side pages.
 *
 * Replicates TqBuildModelAndSidePages (static in tqbuild.c) + TqBuildIndex's
 * meta-first page ordering: the meta page placeholder is created at block 0
 * FIRST (so it is always the first block in the fork), THEN the side pages
 * append at blocks >= 1, THEN the meta page is back-patched with the side-chain
 * heads and header fields.  The entry point is written "empty" (entryLevel =
 * -1); the real build (Task 4) back-patches it.
 *
 * modelOut->boundaries / modelOut->centroids must be preallocated by the caller.
 */
static void
TqhnswWriteModelAndMeta(Relation index, ForkNumber forkNum, int dim, TqMetric metric,
						int m, int efConstruction, bool fastRotation,
						BlockNumber *codebookStart, BlockNumber *rotationStart,
						TqModel *modelOut)
{
	int			bits = TQ_DEFAULT_BITS;
	int			nLevels = 1 << bits;
	int			nBnd = nLevels - 1;
	int			dimPadded = fastRotation ? TqNextPow2(dim) : dim;
	int			dimCodes = fastRotation ? dimPadded : dim;
	Size		cbBytes = (Size) sizeof(float) * (nBnd + nLevels);
	char	   *cbBuf;
	Buffer		metabuf;
	Page		metapage;
	GenericXLogState *state;
	TqhnswMetaPage metap;

	/*
	 * Meta page is block 0; it must be the first block created in the fork.
	 * Create the placeholder (with Invalid side-chain heads) and commit it so
	 * the subsequent TqWriteBytes side pages land at blocks >= 1.  We
	 * back-patch the meta page below (mirror tqbuild.c's
	 * TqCreateMetaPage + TqUpdateMeta ordering).
	 */
	metabuf = TqNewBuffer(index, forkNum);
	Assert(BufferGetBlockNumber(metabuf) == TQHNSW_METAPAGE_BLKNO);
	TqInitRegisterPage(index, &metabuf, &metapage, &state, TQHNSW_PAGE_ID);
	metap = TqhnswPageGetMeta(metapage);
	memset(metap, 0, sizeof(TqhnswMetaPageData));
	metap->magicNumber = TQHNSW_MAGIC_NUMBER;
	metap->version = TQHNSW_VERSION;
	((PageHeader) metapage)->pd_lower =
		((char *) metap + sizeof(TqhnswMetaPageData)) - (char *) metapage;
	TqCommitBuffer(metabuf, state);

	/* Build the model (codebook + optional dense rotation) in caller memory. */
	TqBuildCodebook(dimCodes, bits, modelOut->boundaries, modelOut->centroids);
	modelOut->dim = dim;
	modelOut->bits = bits;
	modelOut->nLevels = nLevels;
	modelOut->metric = metric;
	modelOut->tqProd = false;
	modelOut->fastRotation = fastRotation;
	modelOut->dimPadded = dimPadded;
	modelOut->dimCodes = dimCodes;
	modelOut->rotation = NULL;
	modelOut->qjl = NULL;
	modelOut->rotSeed = TQ_ROTATION_SEED;
	modelOut->qjlSeed = TQ_QJL_SEED;
	modelOut->qjlScale = 0.0f;

	*rotationStart = InvalidBlockNumber;
	if (!fastRotation)
	{
		Size		rotBytes = (Size) sizeof(float) * dim * dim;

		modelOut->rotation = palloc(rotBytes);
		TqBuildRotation(dim, TQ_ROTATION_SEED, modelOut->rotation);
		*rotationStart = TqWriteBytes(index, forkNum, (const char *) modelOut->rotation,
									  rotBytes, TQHNSW_PAGE_ID);
	}

	cbBuf = palloc(cbBytes);
	memcpy(cbBuf, modelOut->boundaries, sizeof(float) * nBnd);
	memcpy(cbBuf + sizeof(float) * nBnd, modelOut->centroids, sizeof(float) * nLevels);
	*codebookStart = TqWriteBytes(index, forkNum, cbBuf, cbBytes, TQHNSW_PAGE_ID);
	pfree(cbBuf);

	/* Back-patch the meta page (block 0) with the header + side-chain heads. */
	metabuf = ReadBufferExtended(index, forkNum, TQHNSW_METAPAGE_BLKNO, RBM_NORMAL, NULL);
	LockBuffer(metabuf, BUFFER_LOCK_EXCLUSIVE);
	state = GenericXLogStart(index);
	metapage = GenericXLogRegisterBuffer(state, metabuf, 0);
	metap = TqhnswPageGetMeta(metapage);

	metap->magicNumber = TQHNSW_MAGIC_NUMBER;
	metap->version = TQHNSW_VERSION;
	metap->dimensions = (uint16) dim;
	metap->bits = (uint16) bits;
	metap->metric = (uint16) metric;
	metap->fastRotation = (uint16) (fastRotation ? 1 : 0);
	metap->dimPadded = (uint16) dimPadded;
	metap->m = (uint16) m;
	metap->efConstruction = (uint16) efConstruction;
	metap->nLevels = (uint32) nLevels;
	metap->nVectors = 0;
	metap->rotSeed = TQ_ROTATION_SEED;
	metap->codebookStart = *codebookStart;
	metap->rotationStart = *rotationStart;
	metap->entryBlkno = InvalidBlockNumber;
	metap->entryOffno = InvalidOffsetNumber;
	metap->entryLevel = -1;
	metap->insertPage = InvalidBlockNumber;
	metap->firstElementPage = InvalidBlockNumber;

	TqCommitBuffer(metabuf, state);
}

/*
 * Resolve build parameters (dim, metric, options) from the index relation.
 * Mirrors TqInitBuildState's dim/metric resolution.
 */
static void
TqhnswResolveBuildParams(Relation index, int *dim, TqMetric *metric,
						 int *m, int *efc, bool *fast)
{
	TqhnswOptions *opts = (TqhnswOptions *) index->rd_options;

	*m = opts ? opts->m : TQHNSW_DEFAULT_M;
	*efc = opts ? opts->efConstruction : TQHNSW_DEFAULT_EF_CONSTRUCTION;
	*fast = opts ? opts->fastRotation : TQ_DEFAULT_FAST_ROTATION;

	*dim = TupleDescAttr(index->rd_att, 0)->atttypmod;	/* vector dim via typmod */
	if (*dim < 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("column does not have dimensions")));

	/* Resolve metric from FUNCTION 3 support proc (mirror TqInitBuildState). */
	*metric = TQ_METRIC_L2;
	if (OidIsValid(index_getprocid(index, 1, TQHNSW_TYPE_INFO_PROC)))
	{
		FmgrInfo   *typeProc = index_getprocinfo(index, 1, TQHNSW_TYPE_INFO_PROC);
		TqTypeInfo *ti = (TqTypeInfo *) DatumGetPointer(FunctionCall0Coll(typeProc, InvalidOid));

		*metric = ti->metric;
	}
}

/*
 * Write a valid empty index (model + codebook side pages + meta with no graph)
 * to the given fork.  Shared by tqhnswbuildempty (INIT fork) and the empty-heap
 * fast path in tqhnswbuild (MAIN fork).
 */
static void
TqhnswWriteEmptyIndex(Relation index, ForkNumber forkNum)
{
	int			dim;
	TqMetric	metric;
	int			m;
	int			efc;
	bool		fast;
	TqModel		model;
	BlockNumber cbStart;
	BlockNumber rotStart;

	TqhnswResolveBuildParams(index, &dim, &metric, &m, &efc, &fast);

	model.boundaries = palloc(sizeof(float) * ((1 << TQ_DEFAULT_BITS) - 1));
	model.centroids = palloc(sizeof(float) * (1 << TQ_DEFAULT_BITS));
	TqhnswWriteModelAndMeta(index, forkNum, dim, metric, m, efc, fast,
							&cbStart, &rotStart, &model);

	/* Write WAL for the init fork (GenericXLog does not on INIT_FORKNUM). */
	if (forkNum == INIT_FORKNUM)
		log_newpage_range(index, forkNum, 0,
						  RelationGetNumberOfBlocksInFork(index, forkNum), true);
}

/*
 * Assign a random level (HnswInitElement: level = -log(U) * ml, capped at
 * maxLevel).  RandomDouble() is a macro in hnsw.h; replicate its body so levels
 * are drawn the same way (unseeded global PRNG -> nondeterministic across runs).
 */
int
TqhnswRandomLevel(double ml, int maxLevel)
{
	int			level = (int) (-log(pg_prng_double(&pg_global_prng_state)) * ml);

	if (level > maxLevel)
		level = maxLevel;
	return level;
}

/*
 * Allocate an in-memory element (level pre-assigned).  Neighbor arrays are sized
 * per layer (level 0 doubled) and zero-initialized.  Allocated in graphCtx.
 */
static TqhnswElement *
TqhnswAllocElement(TqhnswBuildState * buildstate, ItemPointer tid, int level)
{
	TqhnswElement *element = palloc0(sizeof(TqhnswElement));
	int			lc;

	element->heaptid = *tid;
	element->level = (uint8) level;
	element->rhat = palloc(sizeof(float) * buildstate->dimCodes);
	element->codes = palloc0(buildstate->codesBytes);
	element->visitedGeneration = 0;
	element->neighbors = palloc(sizeof(TqhnswNeighborArray *) * (level + 1));
	for (lc = 0; lc <= level; lc++)
	{
		int			lm = TqhnswGetLayerM(buildstate->m, lc);

		element->neighbors[lc] = palloc(TQHNSW_NEIGHBOR_ARRAY_SIZE(lm));
		element->neighbors[lc]->count = 0;
	}
	element->blkno = InvalidBlockNumber;
	element->offno = InvalidOffsetNumber;
	element->neighborPage = InvalidBlockNumber;
	element->neighborOffno = InvalidOffsetNumber;
	element->next = NULL;
	return element;
}

/*
 * Build callback.  Detoast -> (cosine) normalize -> encode -> reconstruct rhat
 * -> in-memory insert.  Mirrors hnswbuild.c's serial BuildCallback ->
 * InsertTupleInMemory path.
 */
static void
TqhnswBuildCallback(Relation index, ItemPointer tid, Datum *values,
					bool *isnull, bool tupleIsAlive, void *state)
{
	TqhnswBuildState *buildstate = (TqhnswBuildState *) state;
	TqModel    *model = buildstate->model;
	MemoryContext oldCtx;
	Datum		value;
	Vector	   *vec;
	int			level;
	TqhnswElement *element;

	if (isnull[0])
		return;

	oldCtx = MemoryContextSwitchTo(buildstate->tmpCtx);

	value = PointerGetDatum(PG_DETOAST_DATUM(values[0]));

	/* Cosine: normalize before encode so the stripped norm is unit. */
	if (buildstate->metric == TQ_METRIC_COSINE)
		value = DirectFunctionCall1Coll(l2_normalize,
										index->rd_indcollation[0], value);

	vec = DatumGetVector(value);

	/* Encode into the scratch entry. */
	memset(buildstate->scratch, 0, buildstate->scratchSize);
	TqEncode(model, vec->x, buildstate->scratch);

	MemoryContextSwitchTo(oldCtx);

	/* Allocate the persistent element in the graph context. */
	oldCtx = MemoryContextSwitchTo(buildstate->graphCtx);

	level = TqhnswRandomLevel(buildstate->ml, buildstate->maxLevel);
	element = TqhnswAllocElement(buildstate, tid, level);
	element->norm = buildstate->scratch->norm;
	element->scale = buildstate->scratch->scale;
	memcpy(element->codes, buildstate->scratch->data, buildstate->codesBytes);

	TqhnswReconstruct(model, element->codes, element->norm, element->scale,
					  element->rhat);

	/* Cosine: unit-normalize rhat so -IP orders by cosine (spike caveat). */
	if (buildstate->metric == TQ_METRIC_COSINE)
	{
		double		n = 0.0;
		int			i;

		for (i = 0; i < buildstate->dimCodes; i++)
			n += (double) element->rhat[i] * (double) element->rhat[i];
		n = sqrt(n);
		if (n > 1e-20)
		{
			float		inv = (float) (1.0 / n);

			for (i = 0; i < buildstate->dimCodes; i++)
				element->rhat[i] *= inv;
		}
	}

	/*
	 * The durable element + its rhat/codes/neighbor arrays now live in
	 * graphCtx.  The per-insert search/select scratch (pairing heaps,
	 * search-candidate nodes, candidate lists) is transient: run the insert in
	 * the per-tuple tmpCtx so it is reclaimed by the reset below instead of
	 * accumulating in graphCtx for the whole build.  This is safe because the
	 * only durable outputs of the insert are pointer writes into the already-
	 * durable neighbor arrays (element->neighbors / reciprocal targets), and
	 * TqhnswUpdateConnection only overwrites pre-allocated neighbor slots --
	 * it never grows an array in the current (scratch) context.
	 */
	MemoryContextSwitchTo(buildstate->tmpCtx);

	/* Insert into the in-memory graph. */
	if (buildstate->entryPoint == NULL)
	{
		buildstate->entryPoint = element;
	}
	else
	{
		TqhnswInsertElement(NULL /* base */, NULL /* index */,
							NULL /* model */, NULL /* cache */,
							CurrentMemoryContext /* ctx (unused on build path) */,
							element, buildstate->entryPoint, buildstate->m,
							buildstate->efConstruction, buildstate->dimCodes,
							buildstate->metric, false /* existing */);

		if (element->level > buildstate->entryPoint->level)
			buildstate->entryPoint = element;
	}

	/* Append to the build list (graphCtx-owned element, durable next pointer). */
	element->next = buildstate->head;
	buildstate->head = element;
	buildstate->nVectors++;

	MemoryContextSwitchTo(oldCtx);

	MemoryContextReset(buildstate->tmpCtx);
}

/*
 * Flush the in-memory graph to disk: element tuples + neighbor-tuple placeholders
 * (pass 1), then fill the neighbor tuples (pass 2), then back-patch the meta page
 * with the entry point + nVectors.  Mirrors hnswbuild.c CreateGraphPages +
 * WriteNeighborTuples + HnswUpdateMetaPage.
 */
static void
TqhnswFlushGraph(TqhnswBuildState * buildstate)
{
	Relation	index = buildstate->index;
	int			m = buildstate->m;
	Size		maxSize = TqPageCapacity();
	Buffer		buf;
	Page		page;
	GenericXLogState *state;
	TqhnswElement *iter;
	char	   *etupBuf;
	char	   *ntupBuf;
	Size		etupAlloc;
	Size		ntupAlloc;

	/* Worst-case tuple sizes (level <= maxLevel). */
	etupAlloc = TQHNSW_ELEMENT_TUPLE_SIZE(buildstate->codesBytes);
	ntupAlloc = TQHNSW_NEIGHBOR_TUPLE_SIZE(buildstate->maxLevel, m);
	etupBuf = palloc0(etupAlloc);
	ntupBuf = palloc0(ntupAlloc);

	/* Pass 1: write element tuples + neighbor-tuple placeholders. */
	buf = TqNewBuffer(index, MAIN_FORKNUM);
	TqInitRegisterPage(index, &buf, &page, &state, TQHNSW_PAGE_ID);
	buildstate->firstElementPage = BufferGetBlockNumber(buf);

	for (iter = buildstate->head; iter != NULL; iter = iter->next)
	{
		TqhnswElementTuple etup = (TqhnswElementTuple) etupBuf;
		Size		etupSize = TQHNSW_ELEMENT_TUPLE_SIZE(buildstate->codesBytes);
		Size		ntupSize = TQHNSW_NEIGHBOR_TUPLE_SIZE(iter->level, m);
		Size		combinedSize = etupSize + ntupSize + sizeof(ItemIdData);
		OffsetNumber offno;

		MemSet(etupBuf, 0, etupAlloc);

		etup->type = TQHNSW_ELEMENT_TUPLE_TYPE;
		etup->level = iter->level;
		etup->deleted = 0;
		etup->version = 0;
		etup->heaptid = iter->heaptid;
		etup->norm = iter->norm;
		etup->scale = iter->scale;
		memcpy(etup->codes, iter->codes, buildstate->codesBytes);

		/* Keep element + its neighbor tuple on the same page when possible. */
		if (PageGetFreeSpace(page) < etupSize ||
			(combinedSize <= maxSize && PageGetFreeSpace(page) < combinedSize))
			TqAppendPage(index, &buf, &page, &state, MAIN_FORKNUM, TQHNSW_PAGE_ID);

		iter->blkno = BufferGetBlockNumber(buf);
		iter->offno = OffsetNumberNext(PageGetMaxOffsetNumber(page));
		if (combinedSize <= maxSize)
		{
			iter->neighborPage = iter->blkno;
			iter->neighborOffno = OffsetNumberNext(iter->offno);
		}
		else
		{
			iter->neighborPage = iter->blkno + 1;
			iter->neighborOffno = FirstOffsetNumber;
		}

		ItemPointerSet(&etup->neighbortid, iter->neighborPage, iter->neighborOffno);

		offno = PageAddItem(page, (Item) etup, etupSize, InvalidOffsetNumber,
							false, false);
		if (offno != iter->offno)
			elog(ERROR, "failed to add index item to \"%s\"",
				 RelationGetRelationName(index));

		/* Neighbor-tuple placeholder. */
		if (PageGetFreeSpace(page) < ntupSize)
			TqAppendPage(index, &buf, &page, &state, MAIN_FORKNUM, TQHNSW_PAGE_ID);

		offno = PageAddItem(page, (Item) ntupBuf, ntupSize, InvalidOffsetNumber,
							false, false);
		if (offno != iter->neighborOffno)
			elog(ERROR, "failed to add index item to \"%s\"",
				 RelationGetRelationName(index));
	}

	TqCommitBuffer(buf, state);

	/* Pass 2: fill neighbor tuples. */
	for (iter = buildstate->head; iter != NULL; iter = iter->next)
	{
		TqhnswNeighborTuple ntup = (TqhnswNeighborTuple) ntupBuf;
		Size		ntupSize = TQHNSW_NEIGHBOR_TUPLE_SIZE(iter->level, m);
		int			idx = 0;
		int			lc;

		CHECK_FOR_INTERRUPTS();

		MemSet(ntupBuf, 0, ntupAlloc);
		ntup->type = TQHNSW_NEIGHBOR_TUPLE_TYPE;
		ntup->version = 0;

		for (lc = iter->level; lc >= 0; lc--)
		{
			int			lm = TqhnswGetLayerM(m, lc);
			int			i;

			for (i = 0; i < lm; i++)
			{
				ItemPointer indextid = &ntup->indextids[idx++];

				{
					TqhnswNeighborArray *na = iter->neighbors[lc];

					if (i < na->count)
					{
						TqhnswElement *ne = TqhnswPtrAccess(NULL, na->items[i].element);

						ItemPointerSet(indextid, ne->blkno, ne->offno);
					}
					else
						ItemPointerSetInvalid(indextid);
				}
			}
		}
		ntup->count = (uint16) idx;

		buf = ReadBufferExtended(index, MAIN_FORKNUM, iter->neighborPage,
								 RBM_NORMAL, NULL);
		LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
		state = GenericXLogStart(index);
		page = GenericXLogRegisterBuffer(state, buf, 0);

		if (!PageIndexTupleOverwrite(page, iter->neighborOffno, (Item) ntup, ntupSize))
			elog(ERROR, "failed to add index item to \"%s\"",
				 RelationGetRelationName(index));

		TqCommitBuffer(buf, state);
	}

	/* Back-patch the meta page entry point + nVectors. */
	buf = ReadBufferExtended(index, MAIN_FORKNUM, TQHNSW_METAPAGE_BLKNO,
							 RBM_NORMAL, NULL);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	state = GenericXLogStart(index);
	page = GenericXLogRegisterBuffer(state, buf, 0);
	{
		TqhnswMetaPage metap = TqhnswPageGetMeta(page);

		metap->nVectors = (uint32) buildstate->nVectors;
		metap->firstElementPage = buildstate->firstElementPage;
		metap->insertPage = buildstate->firstElementPage;
		if (buildstate->entryPoint != NULL)
		{
			metap->entryBlkno = buildstate->entryPoint->blkno;
			metap->entryOffno = buildstate->entryPoint->offno;
			metap->entryLevel = buildstate->entryPoint->level;
		}
	}
	TqCommitBuffer(buf, state);

	pfree(etupBuf);
	pfree(ntupBuf);
}

IndexBuildResult *
tqhnswbuild(Relation heap, Relation index, struct IndexInfo *indexInfo)
{
	IndexBuildResult *result;
	double		reltuples;
	TqhnswBuildState buildstate;
	TqModel		model;
	BlockNumber cbStart;
	BlockNumber rotStart;
	int			bits = TQ_DEFAULT_BITS;

	memset(&buildstate, 0, sizeof(buildstate));
	buildstate.index = index;

	TqhnswResolveBuildParams(index, &buildstate.dim, &buildstate.metric,
							 &buildstate.m, &buildstate.efConstruction,
							 &buildstate.fastRotation);

	/* Build the model + write meta + codebook side pages (MAIN fork). */
	model.boundaries = palloc(sizeof(float) * ((1 << bits) - 1));
	model.centroids = palloc(sizeof(float) * (1 << bits));
	TqhnswWriteModelAndMeta(index, MAIN_FORKNUM, buildstate.dim, buildstate.metric,
							buildstate.m, buildstate.efConstruction,
							buildstate.fastRotation, &cbStart, &rotStart, &model);

	buildstate.model = &model;
	buildstate.dimCodes = model.dimCodes;
	buildstate.codesBytes = TQ_CODES_BYTES(model.dimCodes, bits);
	buildstate.ml = TqhnswGetMl(buildstate.m);
	buildstate.maxLevel = TqhnswGetMaxLevel(buildstate.m);
	buildstate.head = NULL;
	buildstate.entryPoint = NULL;
	buildstate.nVectors = 0;

	/* Scratch encode entry. */
	buildstate.scratchSize = TqEntrySize(model.dimCodes, bits, false);
	buildstate.scratch = (TqEntry *) palloc(buildstate.scratchSize);

	buildstate.graphCtx = AllocSetContextCreate(CurrentMemoryContext,
												"tqhnsw build graph",
												ALLOCSET_DEFAULT_SIZES);
	buildstate.tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
											  "tqhnsw build temporary",
											  ALLOCSET_DEFAULT_SIZES);

	/*
	 * The callback manages contexts itself: durable graph nodes are allocated
	 * into graphCtx, while the per-insert search/select scratch goes into tmpCtx
	 * and is reset after every tuple.  Run the scan in the build context.
	 */
	reltuples = table_index_build_scan(heap, index, indexInfo, true, true,
									   TqhnswBuildCallback, &buildstate, NULL);

	if (buildstate.nVectors > 0)
		TqhnswFlushGraph(&buildstate);
	/* else: empty heap -> the meta written above already describes an empty index */

	MemoryContextDelete(buildstate.tmpCtx);
	MemoryContextDelete(buildstate.graphCtx);

	result = (IndexBuildResult *) palloc(sizeof(IndexBuildResult));
	result->heap_tuples = reltuples;
	result->index_tuples = buildstate.nVectors;
	return result;
}

void
tqhnswbuildempty(Relation index)
{
	TqhnswWriteEmptyIndex(index, INIT_FORKNUM);
}
