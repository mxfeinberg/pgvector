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
#include "miscadmin.h"
#include "pgstat.h"
#include "storage/bufmgr.h"
#include "tq.h"
#include "utils/float.h"
#include "utils/memutils.h"
#include "utils/rel.h"
#include "vector.h"

#if PG_VERSION_NUM >= 160000
#include "varatt.h"
#endif

PGDLLEXPORT Datum l2_normalize(PG_FUNCTION_ARGS);

/*
 * One scored candidate.  dist is the value we MINIMIZE (smaller == closer);
 * for the reranked subset it is the exact FUNCTION 1 distance, otherwise the
 * quantized estimate.
 */
typedef struct TqScanResult
{
	double		dist;
	ItemPointerData tid;
}			TqScanResult;

/*
 * Scan state.  Mirrors IvfflatScanOpaqueData: the loaded model, the per-query
 * LUT (+ qjlQuery when tqProd), the metric, the exact-distance procinfo +
 * collation used for rerank, ||q||^2, an ordered results array, and a cursor.
 */
typedef struct TqScanOpaqueData
{
	TqModel    *model;
	TqMetric	metric;
	bool		first;

	/* Exact distance (FUNCTION 1) for rerank. */
	FmgrInfo   *procinfo;
	Oid			collation;

	/* Per-query state (rebuilt on each rescan). */
	float	   *lut;			/* dim * nLevels */
	uint8	   *lut8;			/* 8-bit query LUT for the block kernel */
	float		lutBias;		/* affine recovery: mse = lutScale*sum + dc*lutBias */
	float		lutScale;
	float	   *qjlQuery;		/* dim, or NULL when !tqProd */
	Vector	   *queryVec;		/* normalized query (for rerank / cosine) */
	double		qNormSq;		/* ||q||^2 (1.0 for cosine after normalize) */

	/* Results. */
	TqScanResult *results;
	int			nresults;
	int			cursor;

	/* Heap access for rerank. */
	Relation	heapRel;		/* opened here iff heapOpened */
	bool		heapOpened;
	IndexFetchTableData *fetch;
	TupleTableSlot *slot;
	AttrNumber	heapAttno;		/* heap attribute backing the index column */

	MemoryContext tmpCtx;
}			TqScanOpaqueData;

typedef TqScanOpaqueData * TqScanOpaque;

/*
 * qsort comparator: ascending distance.
 */
static int
CompareResults(const void *a, const void *b)
{
	double		da = ((const TqScanResult *) a)->dist;
	double		db = ((const TqScanResult *) b)->dist;

	if (da < db)
		return -1;
	if (da > db)
		return 1;
	return 0;
}

/*
 * tqbeginscan -- initialize a scan descriptor.
 *
 * Loads the model (via the same rd_amcache path inserts use, falling back to
 * TqLoadModel), resolves the metric from the meta page, and looks up the exact
 * distance procinfo (FUNCTION 1) + collation for rerank.  Mirrors
 * ivfflatbeginscan.
 */
IndexScanDesc
tqbeginscan(Relation index, int nkeys, int norderbys)
{
	IndexScanDesc scan;
	TqScanOpaque so;

	scan = RelationGetIndexScan(index, nkeys, norderbys);

	so = (TqScanOpaque) palloc0(sizeof(TqScanOpaqueData));
	so->first = true;

	so->tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
									   "Tqflat scan temporary context",
									   ALLOCSET_DEFAULT_SIZES);

	/*
	 * Load the model from the relcache (rd_amcache / rd_indexcxt).  The
	 * cached model is owned by rd_indexcxt and must NOT be freed by the scan
	 * or live in tmpCtx.
	 */
	so->model = TqGetCachedModel(index);

	so->metric = so->model->metric;

	/* Exact distance (FUNCTION 1) + collation for rerank. */
	so->procinfo = index_getprocinfo(index, 1, TQ_DISTANCE_PROC);
	so->collation = index->rd_indcollation[0];

	/* Map the index column to its backing heap attribute. */
	so->heapAttno = index->rd_index->indkey.values[0];

	so->lut = NULL;
	so->lut8 = NULL;
	so->qjlQuery = NULL;
	so->queryVec = NULL;
	so->results = NULL;
	so->nresults = 0;
	so->cursor = 0;
	so->heapRel = NULL;
	so->heapOpened = false;
	so->fetch = NULL;
	so->slot = NULL;

	scan->opaque = so;

	return scan;
}

/*
 * tqrescan -- start or restart the scan: extract and (for cosine) normalize the
 * order-by query, compute ||q||^2, and build the per-query LUT.
 */
void
tqrescan(IndexScanDesc scan, ScanKey keys, int nkeys,
		 ScanKey orderbys, int norderbys)
{
	TqScanOpaque so = (TqScanOpaque) scan->opaque;
	TqModel    *model;
	MemoryContext oldCtx;

	so->first = true;
	so->cursor = 0;
	so->nresults = 0;

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
	 * Reset per-query allocations (LUT, qjlQuery, queryVec, results array).
	 * The model lives in rd_indexcxt (via TqGetCachedModel) and must NOT be
	 * freed here — assign model from so->model after the reset.
	 */
	MemoryContextReset(so->tmpCtx);
	so->lut = NULL;
	so->lut8 = NULL;
	so->qjlQuery = NULL;
	so->queryVec = NULL;
	so->results = NULL;

	/* so->model still points at the cached model (unaffected by the reset). */
	model = so->model;

	/*
	 * Build the LUT once we have the query (deferred to first gettuple call
	 * if there is no order-by, but a tqflat scan always has one).
	 */
	oldCtx = MemoryContextSwitchTo(so->tmpCtx);
	if (scan->orderByData != NULL &&
		!(scan->orderByData->sk_flags & SK_ISNULL))
	{
		Datum		value = scan->orderByData->sk_argument;
		Vector	   *q;

		Assert(!VARATT_IS_COMPRESSED(DatumGetPointer(value)));
		Assert(!VARATT_IS_EXTENDED(DatumGetPointer(value)));

		/* Normalize the query for cosine (||q|| = 1). */
		if (so->metric == TQ_METRIC_COSINE)
		{
			value = DirectFunctionCall1Coll(l2_normalize, so->collation, value);
		}

		q = DatumGetVector(value);
		so->queryVec = q;

		/* ||q||^2 (1 for cosine after normalize, but compute generally). */
		{
			double		s = 0.0;
			int			i;

			for (i = 0; i < q->dim; i++)
				s += (double) q->x[i] * q->x[i];
			so->qNormSq = s;
		}

		/* Build the LUT (+ qjlQuery when tqProd), sized over dimCodes. */
		so->lut = palloc(sizeof(float) * model->dimCodes * model->nLevels);
		if (model->tqProd)
			so->qjlQuery = palloc(sizeof(float) * model->dimCodes);
		TqBuildQueryLut(model, q->x, so->lut, so->qjlQuery);

		/* Build the 8-bit LUT + affine recovery constants for the block kernel. */
		so->lut8 = palloc(model->dimCodes * model->nLevels);
		TqBuildLut8(model, so->lut, so->lut8, &so->lutBias, &so->lutScale);
	}

	MemoryContextSwitchTo(oldCtx);
}

/*
 * Convert a raw inner-product estimate to a distance to MINIMIZE.
 */
static inline double
TqEstToDist(TqScanOpaque so, const TqEntry *entry, float est)
{
	switch (so->metric)
	{
		case TQ_METRIC_IP:
			return -(double) est;
		case TQ_METRIC_COSINE:
			{
				double		n = (double) entry->norm;

				if (n < 1e-12)
					return 1.0;
				return 1.0 - (double) est / n;
			}
		case TQ_METRIC_L2:
		default:
			return so->qNormSq + (double) entry->norm * (double) entry->norm
				- 2.0 * (double) est;
	}
}

/*
 * Like TqEstToDist but takes the stored norm directly (the block path has no
 * TqEntry, only the side record's norm).  MUST mirror TqEstToDist's formulas.
 */
static inline double
TqEstToDistSide(TqScanOpaque so, float norm, float est)
{
	switch (so->metric)
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
			return so->qNormSq + (double) norm * (double) norm
				- 2.0 * (double) est;
	}
}

/*
 * Fetch the original vector for tid from the heap.  Returns a palloc'd copy in
 * the current memory context, or NULL if the tuple is no longer visible.
 */
static Vector *
TqHeapFetchVector(IndexScanDesc scan, TqScanOpaque so, ItemPointer tid)
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
	 * Detoast first, then size from the DETOASTED pointer.  A small inline
	 * value may carry a 1-byte (short) varlena header in the slot but expand to
	 * a 4-byte header after detoast; sizing from the original short-header datum
	 * would under-copy and truncate the tail of the vector.  The slot's storage
	 * may be reused on the next fetch, so we always return a private copy.
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
 * Perform the full scan: score every live entry, optionally rerank the top
 * candidates against full-precision heap vectors, and sort the results array
 * in ascending distance order.
 */
static void
TqDoScan(IndexScanDesc scan)
{
	TqScanOpaque so = (TqScanOpaque) scan->opaque;
	TqModel    *model = so->model;
	Relation	index = scan->indexRelation;
	int			dc = model->dimCodes;
	BlockNumber codeStart;
	BlockNumber sideStart;
	BlockNumber tailStart;
	uint32		blockCount;
	int			capacity;
	int			n = 0;
	TqScanResult *results;
	MemoryContext oldCtx;

	/* Block-layout chain heads + counts from the meta page. */
	{
		Buffer		metabuf;
		Page		metapage;
		TqMetaPage	metap;

		metabuf = ReadBuffer(index, TQ_METAPAGE_BLKNO);
		LockBuffer(metabuf, BUFFER_LOCK_SHARE);
		metapage = BufferGetPage(metabuf);
		metap = TqPageGetMeta(metapage);
		codeStart = metap->codeStart;
		sideStart = metap->sideStart;
		tailStart = metap->tailStart;
		blockCount = metap->blockCount;
		capacity = (int) metap->nVectors;
		UnlockReleaseBuffer(metabuf);
	}

	oldCtx = MemoryContextSwitchTo(so->tmpCtx);

	if (capacity < 1)
		capacity = 1;
	results = palloc(sizeof(TqScanResult) * capacity);

	/*
	 * Block scan: read the whole code-plane chain into one contiguous buffer,
	 * then walk the side chain in block order.  For each block, fold its
	 * code-plane into the 32-lane accumulator with the SIMD kernel and recover
	 * each live lane's estimate from the side record's per-lane scale/norm.
	 */
	if (blockCount > 0 && BlockNumberIsValid(codeStart))
	{
		Size		blockCodeBytes = TQ_BLOCK_CODE_BYTES(dc);
		Size		codeLen = (Size) blockCount * blockCodeBytes;
		char	   *codeBuf = palloc(codeLen);
		BlockNumber sblk = sideStart;
		uint32		b = 0;

		TqReadBytes(index, codeStart, codeBuf, codeLen);

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

				if (tqflat_force_scalar)
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
							results = repalloc(results, sizeof(TqScanResult) * capacity);
						}
						results[n].dist = TqEstToDistSide(so, sd->norm, est);
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
							results = repalloc(results, sizeof(TqScanResult) * capacity);
						}
						results[n].dist = TqEstToDistSide(so, sd->norm, est);
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
		Assert(b == blockCount);
	}

	/*
	 * Tail scan: rows inserted post-build live in the row-major tail chain
	 * (Invalid until the first insert).  Score them with the scalar kernel over
	 * the float LUT, exactly as the pre-block layout did.
	 */
	if (BlockNumberIsValid(tailStart))
	{
		BlockNumber tblk = tailStart;

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

				est = TqScoreEntry(model, so->lut, so->qjlQuery, entry, entry->data);

				if (n >= capacity)
				{
					capacity *= 2;
					results = repalloc(results, sizeof(TqScanResult) * capacity);
				}
				results[n].dist = TqEstToDist(so, entry, est);
				results[n].tid = entry->heaptid;
				n++;
			}

			nextblk = TqPageGetOpaque(tpage)->nextblkno;
			UnlockReleaseBuffer(tbuf);
			tblk = nextblk;
		}
	}

	/*
	 * Rerank: for the top-K candidates by estimate, fetch the original vector
	 * and replace the estimate with the exact FUNCTION 1 distance.
	 *
	 * Approach (documented prototype tradeoff): rerank the top-K exactly, then
	 * sort the whole array by distance.  Exact and estimated distances are not
	 * perfectly comparable, but in the top-K region the exact values dominate,
	 * which is what matters for the returned neighbors.
	 */
	if (tqflat_rerank > 0 && n > 0 && so->queryVec != NULL)
	{
		int			k = Min(tqflat_rerank, n);
		int			i;

		/* Full sort; the top-K smallest estimates then sit at the front.  K is
		 * typically small (<= a few hundred), so a full sort is simplest and
		 * cheap relative to the scan itself. */
		qsort(results, n, sizeof(TqScanResult), CompareResults);

		for (i = 0; i < k; i++)
		{
			Vector	   *heapVec = TqHeapFetchVector(scan, so, &results[i].tid);

			if (heapVec != NULL)
			{
				Datum		heapDatum = PointerGetDatum(heapVec);
				double		d;

				/*
				 * For cosine, normalize the heap vector so that
				 * FUNCTION 1 (vector_negative_inner_product) computes
				 * -dot(q_norm, x_norm) = -cos(q, x), which ranks identically
				 * to the estimate-based distance 1 - cos(q, x).  Without this,
				 * it computes -||x|| * cos(q, x), giving wrong ranking when
				 * heap vectors differ in magnitude.
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

				/*
				 * For cosine, heapDatum points to a freshly palloc'd normalized
				 * copy returned by l2_normalize — distinct from heapVec — so free
				 * it separately.  For IP/L2, heapDatum == PointerGetDatum(heapVec)
				 * (no new allocation), so only heapVec is freed.
				 */
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
	qsort(results, n, sizeof(TqScanResult), CompareResults);

	so->results = results;
	so->nresults = n;
	so->cursor = 0;

	MemoryContextSwitchTo(oldCtx);
}

/*
 * tqgettuple -- on the first call, run the full scan + rerank; then return one
 * heap tid per call in ascending distance order.
 */
bool
tqgettuple(IndexScanDesc scan, ScanDirection dir)
{
	TqScanOpaque so = (TqScanOpaque) scan->opaque;

	Assert(ScanDirectionIsForward(dir));

	if (so->first)
	{
		pgstat_count_index_scan(scan->indexRelation);
#if PG_VERSION_NUM >= 180000
		if (scan->instrument)
			scan->instrument->nsearches++;
#endif

		if (scan->orderByData == NULL)
			elog(ERROR, "cannot scan tqflat index without order");

		if (!IsMVCCSnapshot(scan->xs_snapshot))
			elog(ERROR, "non-MVCC snapshots are not supported with tqflat");

		if (so->lut == NULL)
		{
			so->first = false;
			return false;		/* NULL order-by query: no results */
		}

		TqDoScan(scan);
		so->first = false;
	}

	if (so->cursor >= so->nresults)
		return false;

	scan->xs_heaptid = so->results[so->cursor].tid;
	scan->xs_recheck = false;
	scan->xs_recheckorderby = false;
	so->cursor++;

	return true;
}

/*
 * tqendscan -- release scan resources (and the heap relation if we opened it).
 */
void
tqendscan(IndexScanDesc scan)
{
	TqScanOpaque so = (TqScanOpaque) scan->opaque;

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
