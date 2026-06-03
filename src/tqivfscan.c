#include "postgres.h"

#include <float.h>
#include <math.h>

#include "access/relscan.h"
#include "access/table.h"
#include "access/tableam.h"
#include "catalog/index.h"
#include "executor/executor.h"
#include "executor/tuptable.h"
#include "fmgr.h"
#include "lib/pairingheap.h"
#include "miscadmin.h"
#include "pgstat.h"
#include "storage/bufmgr.h"
#include "tqivf.h"
#include "utils/float.h"
#include "utils/memutils.h"
#include "utils/rel.h"
#include "vector.h"

#if PG_VERSION_NUM >= 160000
#include "varatt.h"
#endif

PGDLLEXPORT Datum l2_normalize(PG_FUNCTION_ARGS);

#define GetTqivfScanList(ptr) pairingheap_container(TqivfScanList, ph_node, ptr)
#define GetTqivfScanListConst(ptr) pairingheap_const_container(TqivfScanList, ph_node, ptr)

/*
 * One scored candidate.  dist is the value we MINIMIZE (smaller == closer);
 * for the reranked subset it is the exact FUNCTION 1 distance, otherwise the
 * quantized estimate.  Mirrors tqscan's TqScanResult.
 */
typedef struct TqivfScanResult
{
	double		dist;
	ItemPointerData tid;
}			TqivfScanResult;

/*
 * Scan state.  Mirrors tqscan's TqScanOpaqueData (model, per-query LUT, exact
 * distance procinfo + collation for rerank, ||q||^2, ordered results + cursor,
 * lazy heap-fetch handle) plus the IVF probe-selection fields (probes /
 * maxProbes / lists, list-directory head, the ordered probed-list array, and a
 * pairing heap used during probe selection).
 */
typedef struct TqivfScanOpaqueData
{
	TqModel    *model;
	TqMetric	metric;
	bool		first;

	/* Exact distance (FUNCTION 1) for rerank + probe selection. */
	FmgrInfo   *procinfo;
	Oid			collation;

	/* Per-query state (rebuilt on each rescan). */
	float	   *lut;			/* dimCodes * nLevels */
	uint8	   *lut8;			/* 8-bit query LUT for the block kernel */
	float		lutBias;		/* affine recovery: mse = lutScale*sum + dc*lutBias */
	float		lutScale;
	Vector	   *queryVec;		/* normalized query (for rerank / cosine) */
	double		qNormSq;		/* ||q||^2 (1.0 for cosine after normalize) */

	/* Results. */
	TqivfScanResult *results;
	int			nresults;
	int			cursor;

	/* Heap access for rerank. */
	Relation	heapRel;		/* opened here iff heapOpened */
	bool		heapOpened;
	IndexFetchTableData *fetch;
	TupleTableSlot *slot;
	AttrNumber	heapAttno;		/* heap attribute backing the index column */

	/* IVF probe selection. */
	int			probes;			/* lists scored per batch */
	int			maxProbes;		/* total lists scored across all batches */
	int			lists;			/* total lists in the index */
	BlockNumber listStart;		/* first list-directory page */
	pairingheap *listQueue;		/* max-heap during probe selection */
	TqivfScanList *probeLists;	/* materialized, ascending distance */
	int			nProbeLists;	/* number of entries in probeLists */
	int			listIndex;		/* next probed list to score (Task 9 seam) */

	MemoryContext tmpCtx;
}			TqivfScanOpaqueData;

typedef TqivfScanOpaqueData * TqivfScanOpaque;

/*
 * qsort comparator: ascending distance.
 */
static int
CompareResults(const void *a, const void *b)
{
	double		da = ((const TqivfScanResult *) a)->dist;
	double		db = ((const TqivfScanResult *) b)->dist;

	if (da < db)
		return -1;
	if (da > db)
		return 1;
	return 0;
}

/*
 * Pairing-heap comparator for probe selection: a max-heap keyed on centroid
 * distance, so the farthest kept list sits at the root and is the first
 * evicted.  Mirrors ivfscan's CompareLists.
 */
static int
CompareLists(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
	if (GetTqivfScanListConst(a)->distance > GetTqivfScanListConst(b)->distance)
		return 1;
	if (GetTqivfScanListConst(a)->distance < GetTqivfScanListConst(b)->distance)
		return -1;
	return 0;
}

/*
 * Convert a raw inner-product estimate to a distance to MINIMIZE, taking the
 * stored norm directly.  MUST mirror tqscan's TqEstToDist / TqEstToDistSide
 * formulas EXACTLY: IP -> -est; L2 -> qNormSq + norm^2 - 2*est; cosine ->
 * 1 - est/norm.
 */
static inline double
TqivfEstToDist(TqMetric metric, double qNormSq, float norm, float est)
{
	switch (metric)
	{
		case TQ_METRIC_IP:
			return -(double) est;
		case TQ_METRIC_COSINE:
			{
				double		n = (double) norm;

				if (n < 1e-12)
					return 1.0;
				return 1.0 - (double) est / n;
			}
		case TQ_METRIC_L2:
		default:
			return qNormSq + (double) norm * (double) norm - 2.0 * (double) est;
	}
}

/*
 * Fetch the original vector for tid from the heap.  Returns a palloc'd copy in
 * the current memory context, or NULL if the tuple is no longer visible.
 * Replicates tqscan's TqHeapFetchVector scoped to TqivfScanOpaque.
 */
static Vector *
TqivfHeapFetchVector(IndexScanDesc scan, TqivfScanOpaque so, ItemPointer tid)
{
	bool		call_again = false;
	bool		all_dead = false;
	Datum		datum;
	bool		isnull;
	Vector	   *vec;

	if (so->fetch == NULL)
	{
		Relation	heap = scan->heapRelation;

		if (heap == NULL)
		{
			Oid			heapoid = IndexGetRelation(RelationGetRelid(scan->indexRelation), false);

			so->heapRel = table_open(heapoid, AccessShareLock);
			so->heapOpened = true;
			heap = so->heapRel;
		}
		else
			so->heapRel = heap;

		so->fetch = table_index_fetch_begin(heap);
		so->slot = table_slot_create(heap, NULL);
	}

	ExecClearTuple(so->slot);
	if (!table_index_fetch_tuple(so->fetch, tid, scan->xs_snapshot, so->slot,
								 &call_again, &all_dead))
		return NULL;

	datum = slot_getattr(so->slot, so->heapAttno, &isnull);
	if (isnull)
		return NULL;

	/*
	 * Detoast first, then size from the DETOASTED pointer (see tqscan's
	 * TqHeapFetchVector for the short-header truncation hazard).  The slot's
	 * storage may be reused on the next fetch, so always return a private copy.
	 */
	vec = DatumGetVector(datum);
	{
		Size		sz = VARSIZE_ANY(vec);
		Vector	   *copy = palloc(sz);

		memcpy(copy, vec, sz);
		return copy;
	}
}

/*
 * TqivfGetScanLists -- probe selection.  Walk the list-directory chain, compute
 * the exact query->centroid distance for each list, and keep the nearest
 * maxProbes in a max-heap (evicting the farthest).  Materialize the kept lists
 * into probeLists ordered by ascending distance.  Mirrors ivfscan's
 * GetScanLists.
 */
static void
TqivfGetScanLists(IndexScanDesc scan)
{
	TqivfScanOpaque so = (TqivfScanOpaque) scan->opaque;
	Relation	index = scan->indexRelation;
	BlockNumber blkno = so->listStart;
	int			listCount = 0;
	double		maxDistance = DBL_MAX;
	MemoryContext oldCtx;
	int			i;

	oldCtx = MemoryContextSwitchTo(so->tmpCtx);

	so->probeLists = palloc(so->maxProbes * sizeof(TqivfScanList));
	pairingheap_reset(so->listQueue);

	while (BlockNumberIsValid(blkno))
	{
		Buffer		buf;
		Page		page;
		OffsetNumber maxoff;
		OffsetNumber offno;
		BlockNumber nextblk;

		buf = ReadBuffer(index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		maxoff = PageGetMaxOffsetNumber(page);

		for (offno = FirstOffsetNumber; offno <= maxoff; offno = OffsetNumberNext(offno))
		{
			TqivfList	list = (TqivfList) PageGetItem(page, PageGetItemId(page, offno));
			double		distance;

			/* Exact query->centroid distance via FUNCTION 1. */
			distance = DatumGetFloat8(FunctionCall2Coll(so->procinfo, so->collation,
														PointerGetDatum(&list->center),
														PointerGetDatum(so->queryVec)));

			if (listCount < so->maxProbes)
			{
				TqivfScanList *sl = &so->probeLists[listCount];

				sl->codeStart = list->codeStart;
				sl->sideStart = list->sideStart;
				sl->tailStart = list->tailStart;
				sl->blockCount = list->blockCount;
				sl->nvectors = list->nvectors;
				sl->distance = distance;
				listCount++;

				pairingheap_add(so->listQueue, &sl->ph_node);

				if (listCount == so->maxProbes)
					maxDistance = GetTqivfScanList(pairingheap_first(so->listQueue))->distance;
			}
			else if (distance < maxDistance)
			{
				TqivfScanList *sl;

				/* Evict the farthest, reuse its slot. */
				sl = GetTqivfScanList(pairingheap_remove_first(so->listQueue));
				sl->codeStart = list->codeStart;
				sl->sideStart = list->sideStart;
				sl->tailStart = list->tailStart;
				sl->blockCount = list->blockCount;
				sl->nvectors = list->nvectors;
				sl->distance = distance;
				pairingheap_add(so->listQueue, &sl->ph_node);

				maxDistance = GetTqivfScanList(pairingheap_first(so->listQueue))->distance;
			}
		}

		nextblk = TqPageGetOpaque(page)->nextblkno;
		UnlockReleaseBuffer(buf);
		blkno = nextblk;
	}

	/*
	 * Drain the max-heap into a temporary array (the heap nodes alias the
	 * probeLists slots), then copy back in ascending distance order.  Removing
	 * from a max-heap yields descending distance, so fill from the back.
	 */
	{
		TqivfScanList *ordered = palloc(listCount * sizeof(TqivfScanList));

		for (i = listCount - 1; i >= 0; i--)
			ordered[i] = *GetTqivfScanList(pairingheap_remove_first(so->listQueue));

		Assert(pairingheap_is_empty(so->listQueue));

		for (i = 0; i < listCount; i++)
			so->probeLists[i] = ordered[i];

		pfree(ordered);
	}

	so->nProbeLists = listCount;

	MemoryContextSwitchTo(oldCtx);
}

/*
 * TqivfScoreList -- score one probed list.  Replicates tqscan's TqDoScan block
 * + tail loops, scoped to this list's chains.  Appends candidates to *results
 * (growing via repalloc), updating *pn / *pcapacity.  Must be called with the
 * scan's tmpCtx current (caller switches).
 */
static void
TqivfScoreList(IndexScanDesc scan, TqivfScanList *L,
			   TqivfScanResult **presults, int *pn, int *pcapacity)
{
	TqivfScanOpaque so = (TqivfScanOpaque) scan->opaque;
	TqModel    *model = so->model;
	Relation	index = scan->indexRelation;
	int			dc = model->dimCodes;
	TqivfScanResult *results = *presults;
	int			n = *pn;
	int			capacity = *pcapacity;

	/* Block scan: read this list's code-plane chain, walk its side chain. */
	if (L->blockCount > 0 && BlockNumberIsValid(L->codeStart))
	{
		Size		blockCodeBytes = TQ_BLOCK_CODE_BYTES(dc);
		Size		codeLen = (Size) L->blockCount * blockCodeBytes;
		char	   *codeBuf = palloc(codeLen);
		BlockNumber sblk = L->sideStart;
		uint32		b = 0;

		TqReadBytes(index, L->codeStart, codeBuf, codeLen);

		while (BlockNumberIsValid(sblk))
		{
			Buffer		sbuf = ReadBuffer(index, sblk);
			Page		spage;
			OffsetNumber soff,
						smax;
			BlockNumber nextblk;

			LockBuffer(sbuf, BUFFER_LOCK_SHARE);
			spage = BufferGetPage(sbuf);
			smax = PageGetMaxOffsetNumber(spage);

			for (soff = FirstOffsetNumber; soff <= smax; soff = OffsetNumberNext(soff))
			{
				TqBlockSideRec *srec = (TqBlockSideRec *) PageGetItem(spage, PageGetItemId(spage, soff));
				const uint8 *plane = (const uint8 *) codeBuf + (Size) b * blockCodeBytes;

				if (tqivf_force_scalar)
				{
					int			nLevels = model->nLevels;
					int			j;

					for (j = 0; j < srec->nvecs; j++)
					{
						int			lane = j & 15;
						bool		high = j >= 16;
						double		mse = 0.0;
						float		est;
						TqBlockSide *sd;
						int			ii;

						if (srec->deletedMask & (1u << j))
							continue;

						for (ii = 0; ii < dc; ii++)
						{
							uint8		cellb = plane[(Size) ii * 16 + lane];
							uint8		code = high ? (uint8) (cellb >> 4) : (uint8) (cellb & 0x0F);

							mse += (double) so->lut[(Size) ii * nLevels + code];
						}
						sd = &srec->side[j];
						est = (float) ((double) sd->scale * mse);

						if (n >= capacity)
						{
							capacity *= 2;
							results = repalloc(results, sizeof(TqivfScanResult) * capacity);
						}
						results[n].dist = TqivfEstToDist(so->metric, so->qNormSq, sd->norm, est);
						results[n].tid = sd->heaptid;
						n++;
					}
				}
				else
				{
					TqBlockAccum acc;
					int			j;

					TqBlockAccumInit(&acc);
					TqScoreBlockRange(so->lut8, plane, 0, dc, &acc);
					TqBlockAccumFinish(&acc);

					for (j = 0; j < srec->nvecs; j++)
					{
						double		mse;
						float		est;
						TqBlockSide *sd;

						if (srec->deletedMask & (1u << j))
							continue;

						sd = &srec->side[j];
						mse = (double) so->lutScale * acc.acc32[j] + (double) dc * so->lutBias;
						est = (float) ((double) sd->scale * mse);

						if (n >= capacity)
						{
							capacity *= 2;
							results = repalloc(results, sizeof(TqivfScanResult) * capacity);
						}
						results[n].dist = TqivfEstToDist(so->metric, so->qNormSq, sd->norm, est);
						results[n].tid = sd->heaptid;
						n++;
					}
				}
				b++;
			}

			nextblk = TqPageGetOpaque(spage)->nextblkno;
			UnlockReleaseBuffer(sbuf);
			sblk = nextblk;
		}
		Assert(b == L->blockCount);

		pfree(codeBuf);
	}

	/* Tail scan: row-major insert tail (Invalid until first insert). */
	if (BlockNumberIsValid(L->tailStart))
	{
		BlockNumber tblk = L->tailStart;

		while (BlockNumberIsValid(tblk))
		{
			Buffer		tbuf = ReadBuffer(index, tblk);
			Page		tpage;
			OffsetNumber toff,
						tmax;
			BlockNumber nextblk;

			LockBuffer(tbuf, BUFFER_LOCK_SHARE);
			tpage = BufferGetPage(tbuf);
			tmax = PageGetMaxOffsetNumber(tpage);

			for (toff = FirstOffsetNumber; toff <= tmax; toff = OffsetNumberNext(toff))
			{
				TqEntry    *entry = (TqEntry *) PageGetItem(tpage, PageGetItemId(tpage, toff));
				float		est;

				if (entry->deleted)
					continue;

				est = TqScoreEntry(model, so->lut, NULL, entry, entry->data);

				if (n >= capacity)
				{
					capacity *= 2;
					results = repalloc(results, sizeof(TqivfScanResult) * capacity);
				}
				results[n].dist = TqivfEstToDist(so->metric, so->qNormSq, entry->norm, est);
				results[n].tid = entry->heaptid;
				n++;
			}

			nextblk = TqPageGetOpaque(tpage)->nextblkno;
			UnlockReleaseBuffer(tbuf);
			tblk = nextblk;
		}
	}

	*presults = results;
	*pn = n;
	*pcapacity = capacity;
}

/*
 * TqivfLoadBatch -- score the next batch of probed lists (up to `probes` lists,
 * advancing listIndex), rerank the top-K, and sort ascending.  Populates
 * so->results / so->nresults / so->cursor.
 *
 * Iterative scan / relaxed_order semantics: when tqivf.iterative_scan =
 * relaxed_order, maxProbes > probes and tqivfgettuple calls this function
 * repeatedly as each batch is consumed.  Results are sorted within each
 * batch (exact distances after rerank), but rows returned from later batches
 * may rank ahead of rows already returned from earlier batches -- the output
 * is approximately ordered across batches, not globally ordered.  This
 * mirrors ivfflat's relaxed_order behaviour and is acceptable for index-scan
 * callers that apply a top-level Sort node (e.g. ORDER BY ... LIMIT).
 *
 * With iterative scan OFF, maxProbes == probes so this runs exactly once and
 * the single batch covers every probed list -- non-iterative behaviour is
 * unchanged.
 */
static void
TqivfLoadBatch(IndexScanDesc scan)
{
	TqivfScanOpaque so = (TqivfScanOpaque) scan->opaque;
	int			capacity;
	uint64		capacity64;
	int			n = 0;
	int			batchEnd;
	int			i;
	TqivfScanResult *results;
	MemoryContext oldCtx;

	oldCtx = MemoryContextSwitchTo(so->tmpCtx);

	/*
	 * Estimate the initial capacity from the lists in this batch (nvectors is a
	 * hint; the array grows by doubling in TqivfScoreList as needed).  Each
	 * nvectors is a uint32 and the per-index cap is ~4.29B, so accumulate into a
	 * uint64 to avoid signed-int overflow when many large lists are probed, then
	 * clamp to INT_MAX so the (int) cast is well-defined.  The doubling repalloc
	 * recovers any under-estimate.
	 */
	batchEnd = Min(so->listIndex + so->probes, so->nProbeLists);
	capacity64 = 0;
	for (i = so->listIndex; i < batchEnd; i++)
		capacity64 += so->probeLists[i].nvectors;
	if (capacity64 < 1)
		capacity64 = 1;
	if (capacity64 > INT_MAX)
		capacity64 = INT_MAX;
	capacity = (int) capacity64;
	results = palloc(sizeof(TqivfScanResult) * capacity);

	for (i = so->listIndex; i < batchEnd; i++)
		TqivfScoreList(scan, &so->probeLists[i], &results, &n, &capacity);

	so->listIndex = batchEnd;

	/*
	 * Rerank: for the top-K candidates by estimate, fetch the original vector
	 * and replace the estimate with the exact FUNCTION 1 distance.  Mirrors
	 * tqscan's rerank block.
	 *
	 * Documented prototype tradeoff (inherited from tqflat): the reranked subset
	 * carries an exact FUNCTION 1 distance while the rest carry the quantized
	 * estimate, and the two are not on a perfectly comparable scale -- most
	 * visibly for cosine, where the exact value is -cos(q, x) (range [-1, 1])
	 * while the estimate-based distance is 1 - cos(q, x) (range [0, 2]).  The two
	 * are monotone in cos(q, x) and the reranked (exact) values dominate the
	 * front of the array, which is what matters for the returned neighbors when
	 * rerank >= LIMIT.  Distances are never surfaced to the caller (the AM does
	 * not set xs_orderbyvals), so the scale difference affects only internal
	 * ordering.
	 */
	if (tqivf_rerank > 0 && n > 0 && so->queryVec != NULL)
	{
		int			k = Min(tqivf_rerank, n);

		qsort(results, n, sizeof(TqivfScanResult), CompareResults);

		for (i = 0; i < k; i++)
		{
			Vector	   *heapVec = TqivfHeapFetchVector(scan, so, &results[i].tid);

			if (heapVec != NULL)
			{
				Datum		heapDatum = PointerGetDatum(heapVec);
				double		d;

				/*
				 * For cosine, normalize the heap vector so FUNCTION 1
				 * (vector_negative_inner_product) computes -cos(q, x), ranking
				 * identically to the estimate-based distance 1 - cos(q, x).
				 */
				if (so->metric == TQ_METRIC_COSINE)
					heapDatum = DirectFunctionCall1Coll(l2_normalize,
														so->collation,
														heapDatum);

				d = DatumGetFloat8(FunctionCall2Coll(so->procinfo,
													 so->collation,
													 PointerGetDatum(so->queryVec),
													 heapDatum));

				results[i].dist = d;

				if (so->metric == TQ_METRIC_COSINE)
					pfree(DatumGetPointer(heapDatum));
				pfree(heapVec);
			}
			else
			{
				/* No longer visible: push to the end so it is not returned. */
				results[i].dist = get_float8_infinity();
			}
		}
	}

	/* Final ordering. */
	qsort(results, n, sizeof(TqivfScanResult), CompareResults);

	so->results = results;
	so->nresults = n;
	so->cursor = 0;

	MemoryContextSwitchTo(oldCtx);
}

/*
 * tqivfbeginscan -- initialize a scan descriptor.  Mirrors tqbeginscan +
 * ivfflatbeginscan: load the model, resolve metric / lists / listStart, set up
 * probes / maxProbes, and look up the exact-distance procinfo for probe
 * selection and rerank.
 */
IndexScanDesc
tqivfbeginscan(Relation index, int nkeys, int norderbys)
{
	IndexScanDesc scan;
	TqivfScanOpaque so;
	int			lists;
	TqMetric	metric;
	BlockNumber listStart;
	int			probes = tqivf_probes;
	int			maxProbes;
	MemoryContext oldCtx;

	scan = RelationGetIndexScan(index, nkeys, norderbys);

	TqivfGetMetaInfo(index, NULL, &metric, &lists, &listStart);

	if (tqivf_iterative_scan != TQIVF_ITERATIVE_SCAN_OFF)
		maxProbes = Max(tqivf_max_probes, probes);
	else
		maxProbes = probes;

	if (probes > lists)
		probes = lists;
	if (maxProbes > lists)
		maxProbes = lists;

	so = (TqivfScanOpaque) palloc0(sizeof(TqivfScanOpaqueData));
	so->first = true;

	so->tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
									   "Tqivf scan temporary context",
									   ALLOCSET_DEFAULT_SIZES);

	/* Cached model (owned by rd_indexcxt; must NOT be freed by the scan). */
	so->model = TqivfGetCachedModel(index);
	so->metric = metric;

	/* Exact distance (FUNCTION 1) + collation for probe selection / rerank. */
	so->procinfo = index_getprocinfo(index, 1, TQIVF_DISTANCE_PROC);
	so->collation = index->rd_indcollation[0];

	/* Map the index column to its backing heap attribute. */
	so->heapAttno = index->rd_index->indkey.values[0];

	so->probes = probes;
	so->maxProbes = maxProbes;
	so->lists = lists;
	so->listStart = listStart;
	so->listIndex = 0;
	so->nProbeLists = 0;
	so->probeLists = NULL;

	so->lut = NULL;
	so->lut8 = NULL;
	so->queryVec = NULL;
	so->qNormSq = 0.0;
	so->results = NULL;
	so->nresults = 0;
	so->cursor = 0;
	so->heapRel = NULL;
	so->heapOpened = false;
	so->fetch = NULL;
	so->slot = NULL;

	oldCtx = MemoryContextSwitchTo(so->tmpCtx);
	so->listQueue = pairingheap_allocate(CompareLists, scan);
	MemoryContextSwitchTo(oldCtx);

	scan->opaque = so;

	return scan;
}

/*
 * tqivfrescan -- start or restart the scan: extract and (for cosine) normalize
 * the order-by query, compute ||q||^2, and build the per-query LUT (once,
 * reused across all probed lists).
 */
void
tqivfrescan(IndexScanDesc scan, ScanKey keys, int nkeys,
			ScanKey orderbys, int norderbys)
{
	TqivfScanOpaque so = (TqivfScanOpaque) scan->opaque;
	TqModel    *model;
	MemoryContext oldCtx;

	so->first = true;
	so->cursor = 0;
	so->nresults = 0;
	so->listIndex = 0;
	so->nProbeLists = 0;

	if (keys && scan->numberOfKeys > 0)
		memmove(scan->keyData, keys, scan->numberOfKeys * sizeof(ScanKeyData));
	if (orderbys && scan->numberOfOrderBys > 0)
		memmove(scan->orderByData, orderbys, scan->numberOfOrderBys * sizeof(ScanKeyData));

	/* Tear down any heap-fetch state from a prior scan (lives in tmpCtx). */
	if (so->fetch != NULL)
	{
		table_index_fetch_end(so->fetch);
		so->fetch = NULL;
	}
	if (so->slot != NULL)
	{
		ExecDropSingleTupleTableSlot(so->slot);
		so->slot = NULL;
	}
	if (so->heapOpened && so->heapRel != NULL)
	{
		table_close(so->heapRel, AccessShareLock);
		so->heapOpened = false;
	}
	so->heapRel = NULL;

	/*
	 * Reset per-query allocations (LUT, queryVec, results, probeLists, listQueue).
	 * The model lives in rd_indexcxt and is unaffected.  listQueue lives in tmpCtx
	 * and is freed by the reset, so re-allocate it afterward.
	 */
	pairingheap_reset(so->listQueue);
	MemoryContextReset(so->tmpCtx);
	so->lut = NULL;
	so->lut8 = NULL;
	so->queryVec = NULL;
	so->results = NULL;
	so->probeLists = NULL;

	oldCtx = MemoryContextSwitchTo(so->tmpCtx);
	so->listQueue = pairingheap_allocate(CompareLists, scan);
	MemoryContextSwitchTo(oldCtx);

	model = so->model;

	oldCtx = MemoryContextSwitchTo(so->tmpCtx);
	if (scan->orderByData != NULL &&
		!(scan->orderByData->sk_flags & SK_ISNULL))
	{
		Datum		value = scan->orderByData->sk_argument;
		Vector	   *q;
		int			i;
		double		s = 0.0;

		Assert(!VARATT_IS_COMPRESSED(DatumGetPointer(value)));
		Assert(!VARATT_IS_EXTENDED(DatumGetPointer(value)));

		/* Normalize the query for cosine (||q|| = 1). */
		if (so->metric == TQ_METRIC_COSINE)
			value = DirectFunctionCall1Coll(l2_normalize, so->collation, value);

		q = DatumGetVector(value);
		so->queryVec = q;

		for (i = 0; i < q->dim; i++)
			s += (double) q->x[i] * q->x[i];
		so->qNormSq = s;

		/* Build the LUT (sized over dimCodes), then the 8-bit block LUT. */
		so->lut = palloc(sizeof(float) * model->dimCodes * model->nLevels);
		TqBuildQueryLut(model, q->x, so->lut, NULL);

		so->lut8 = palloc(model->dimCodes * model->nLevels);
		TqBuildLut8(model, so->lut, so->lut8, &so->lutBias, &so->lutScale);
	}

	MemoryContextSwitchTo(oldCtx);
}

/*
 * tqivfgettuple -- on the first call, select probes + score the first batch +
 * rerank; then return one heap tid per call in ascending distance order.
 */
bool
tqivfgettuple(IndexScanDesc scan, ScanDirection dir)
{
	TqivfScanOpaque so = (TqivfScanOpaque) scan->opaque;

	Assert(ScanDirectionIsForward(dir));

	if (so->first)
	{
		pgstat_count_index_scan(scan->indexRelation);
#if PG_VERSION_NUM >= 180000
		if (scan->instrument)
			scan->instrument->nsearches++;
#endif

		if (scan->orderByData == NULL)
			elog(ERROR, "cannot scan tqivf index without order");

		if (!IsMVCCSnapshot(scan->xs_snapshot))
			elog(ERROR, "non-MVCC snapshots are not supported with tqivf");

		if (so->lut == NULL)
		{
			so->first = false;
			return false;		/* NULL order-by query: no results */
		}

		TqivfGetScanLists(scan);
		TqivfLoadBatch(scan);
		so->first = false;
	}

	/*
	 * Task-9 seam: when iterative scan is enabled, score the next batch of
	 * probed lists once the current batch is exhausted.  With iterative scan
	 * OFF, maxProbes == probes so the single batch covers every probed list and
	 * this loop never re-enters.
	 */
	while (so->cursor >= so->nresults)
	{
		if (so->listIndex >= so->nProbeLists)
			return false;

		TqivfLoadBatch(scan);
	}

	scan->xs_heaptid = so->results[so->cursor].tid;
	scan->xs_recheck = false;
	scan->xs_recheckorderby = false;
	so->cursor++;

	return true;
}

/*
 * tqivfendscan -- release scan resources (and the heap relation if we opened
 * it).  Mirrors tqendscan.
 */
void
tqivfendscan(IndexScanDesc scan)
{
	TqivfScanOpaque so = (TqivfScanOpaque) scan->opaque;

	if (so->fetch != NULL)
		table_index_fetch_end(so->fetch);
	if (so->slot != NULL)
		ExecDropSingleTupleTableSlot(so->slot);
	if (so->heapOpened && so->heapRel != NULL)
		table_close(so->heapRel, AccessShareLock);

	MemoryContextDelete(so->tmpCtx);

	pfree(so);
	scan->opaque = NULL;
}
