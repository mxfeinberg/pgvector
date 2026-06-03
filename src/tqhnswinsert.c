#include "postgres.h"

#include <math.h>

#include "access/generic_xlog.h"
#include "storage/bufmgr.h"
#include "storage/lmgr.h"
#include "tqhnsw.h"
#include "utils/memutils.h"
#include "utils/rel.h"

#if PG_VERSION_NUM >= 160000
#include "varatt.h"
#endif

PGDLLEXPORT Datum l2_normalize(PG_FUNCTION_ARGS);

/*
 * Get the insert page (mirrors hnswinsert.c GetInsertPage).  Read under a SHARE
 * lock on the meta page; the field is advisory (a hint for where a new element
 * tuple is likely to fit), so a stale value only costs an extra page scan.
 */
static BlockNumber
GetInsertPage(Relation index)
{
	Buffer		buf;
	Page		page;
	TqhnswMetaPage metap;
	BlockNumber insertPage;

	buf = ReadBuffer(index, TQHNSW_METAPAGE_BLKNO);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	metap = TqhnswPageGetMeta(page);

	insertPage = metap->insertPage;

	UnlockReleaseBuffer(buf);

	return insertPage;
}

/*
 * Quantize a heap value into a fresh in-memory element (codes + rhat), drawing a
 * random level.  Mirrors TqhnswBuildCallback's encode block exactly (detoast ->
 * cosine l2_normalize -> TqEncode -> reconstruct rhat -> cosine unit-normalize
 * rhat) so on-disk inserts are numerically identical to build-time nodes.
 *
 * Allocated in the current (per-insert) memory context.  Neighbor arrays are
 * sized per layer (level 0 doubled) and zero-initialized; forward links are
 * filled by TqhnswInsertElement.
 */
static TqhnswElement *
TqhnswQuantizeElement(Relation index, const TqModel *model, TqMetric metric,
					  int m, int dimCodes, int codesBytes, ItemPointer heaptid,
					  Datum value)
{
	TqhnswElement *element;
	Vector	   *vec;
	int			level;
	int			lc;
	Size		scratchSize = TqEntrySize(dimCodes, model->bits, false);
	TqEntry    *scratch = (TqEntry *) palloc(scratchSize);

	value = PointerGetDatum(PG_DETOAST_DATUM(value));

	/* Cosine: normalize before encode so the stripped norm is unit. */
	if (metric == TQ_METRIC_COSINE)
		value = DirectFunctionCall1Coll(l2_normalize,
										index->rd_indcollation[0], value);

	vec = DatumGetVector(value);

	memset(scratch, 0, scratchSize);
	TqEncode(model, vec->x, scratch);

	level = TqhnswRandomLevel(TqhnswGetMl(m), TqhnswGetMaxLevel(m));

	element = palloc0(sizeof(TqhnswElement));
	element->heaptid = *heaptid;
	element->level = (uint8) level;
	element->visitedGeneration = 0;
	element->norm = scratch->norm;
	element->scale = scratch->scale;
	element->codes = palloc0(codesBytes);
	memcpy(element->codes, scratch->data, codesBytes);
	element->rhat = palloc(sizeof(float) * dimCodes);

	element->neighbors = palloc(sizeof(TqhnswNeighborArray *) * (level + 1));
	for (lc = 0; lc <= level; lc++)
	{
		int			lm = TqhnswGetLayerM(m, lc);

		element->neighbors[lc] = palloc(TQHNSW_NEIGHBOR_ARRAY_SIZE(lm));
		element->neighbors[lc]->count = 0;
	}

	element->blkno = InvalidBlockNumber;
	element->offno = InvalidOffsetNumber;
	element->neighborPage = InvalidBlockNumber;
	element->neighborOffno = InvalidOffsetNumber;
	element->next = NULL;

	TqhnswReconstruct(model, element->codes, element->norm, element->scale,
					  element->rhat);

	/* Cosine: unit-normalize rhat so -IP orders by cosine (mirrors build). */
	if (metric == TQ_METRIC_COSINE)
	{
		double		n = 0.0;
		int			i;

		for (i = 0; i < dimCodes; i++)
			n += (double) element->rhat[i] * (double) element->rhat[i];
		n = sqrt(n);
		if (n > 1e-20)
		{
			float		inv = (float) (1.0 / n);

			for (i = 0; i < dimCodes; i++)
				element->rhat[i] *= inv;
		}
	}

	pfree(scratch);
	return element;
}

/*
 * Fill an element tuple from an in-memory element (mirrors the etup field writes
 * in TqhnswFlushGraph).  neighbortid is set by the caller after page placement.
 */
static void
TqhnswSetElementTuple(TqhnswElementTuple etup, TqhnswElement *e, int codesBytes)
{
	etup->type = TQHNSW_ELEMENT_TUPLE_TYPE;
	etup->level = e->level;
	etup->deleted = 0;
	etup->version = 0;
	etup->heaptid = e->heaptid;
	etup->norm = e->norm;
	etup->scale = e->scale;
	memcpy(etup->codes, e->codes, codesBytes);
}

/*
 * Fill a neighbor tuple from an in-memory element's forward links (mirrors the
 * pass-2 fill in TqhnswFlushGraph).  Slots beyond the selected count are set
 * invalid; reciprocal edges are written separately by UpdateNeighborsOnDisk.
 */
static void
TqhnswSetNeighborTuple(TqhnswNeighborTuple ntup, TqhnswElement *e, int m)
{
	int			idx = 0;
	int			lc;

	ntup->type = TQHNSW_NEIGHBOR_TUPLE_TYPE;
	ntup->version = 0;

	for (lc = e->level; lc >= 0; lc--)
	{
		int			lm = TqhnswGetLayerM(m, lc);
		TqhnswNeighborArray *na = e->neighbors[lc];
		int			i;

		for (i = 0; i < lm; i++)
		{
			ItemPointer indextid = &ntup->indextids[idx++];

			if (i < na->count)
			{
				TqhnswElement *ne = TqhnswPtrAccess(NULL, na->items[i].element);

				ItemPointerSet(indextid, ne->blkno, ne->offno);
			}
			else
				ItemPointerSetInvalid(indextid);
		}
	}
	ntup->count = (uint16) idx;
}

/*
 * Append a new page after the current one within a GenericXLog transaction
 * (mirrors hnswinsert.c HnswInsertAppendPage; on-disk path only).
 */
static void
TqhnswAppendInsertPage(Relation index, Buffer *nbuf, Page *npage,
					   GenericXLogState *state, Page page)
{
	LockRelationForExtension(index, ExclusiveLock);
	*nbuf = TqNewBuffer(index, MAIN_FORKNUM);
	UnlockRelationForExtension(index, ExclusiveLock);

	*npage = GenericXLogRegisterBuffer(state, *nbuf, GENERIC_XLOG_FULL_IMAGE);
	TqInitPage(*nbuf, *npage, TQHNSW_PAGE_ID);

	/* Link the previous page to the new one. */
	TqhnswPageGetOpaque(page)->nextblkno = BufferGetBlockNumber(*nbuf);
}

/*
 * Add an element + its neighbor tuple to disk (mirrors hnswinsert.c
 * AddElementOnDisk, simplified: tqhnsw never sets the deleted flag, so there is
 * no deleted-tuple reuse path).  Walks the element-page chain from insertPage for
 * a page with room, appending pages as needed, and assigns e->blkno/offno/
 * neighborPage/neighborOffno.  Returns the (possibly updated) insert-page hint in
 * *updatedInsertPage.
 */
static void
TqhnswAddElementOnDisk(Relation index, TqhnswElement *e, int m, int codesBytes,
					   BlockNumber insertPage, BlockNumber *updatedInsertPage)
{
	Buffer		buf;
	Page		page;
	GenericXLogState *state;
	Size		etupSize = TQHNSW_ELEMENT_TUPLE_SIZE(codesBytes);
	Size		ntupSize = TQHNSW_NEIGHBOR_TUPLE_SIZE(e->level, m);
	Size		combinedSize = etupSize + ntupSize + sizeof(ItemIdData);
	Size		maxSize = TqPageCapacity();
	Size		minCombinedSize = etupSize + TQHNSW_NEIGHBOR_TUPLE_SIZE(0, m) + sizeof(ItemIdData);
	TqhnswElementTuple etup;
	TqhnswNeighborTuple ntup;
	Buffer		nbuf;
	Page		npage;
	BlockNumber currentPage = insertPage;
	BlockNumber newInsertPage = InvalidBlockNumber;

	/* Prepare tuples. */
	etup = palloc0(etupSize);
	TqhnswSetElementTuple(etup, e, codesBytes);
	ntup = palloc0(ntupSize);
	TqhnswSetNeighborTuple(ntup, e, m);

	if (!BlockNumberIsValid(currentPage))
		currentPage = TQHNSW_METAPAGE_BLKNO + 1;	/* first non-meta page */

	/* Find a page (or two if needed) to insert the tuples. */
	for (;;)
	{
		buf = ReadBuffer(index, currentPage);
		LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
		state = GenericXLogStart(index);
		page = GenericXLogRegisterBuffer(state, buf, 0);

		/* First page where a level-0 element can fit -> next insert hint. */
		if (!BlockNumberIsValid(newInsertPage) && PageGetFreeSpace(page) >= minCombinedSize)
			newInsertPage = currentPage;

		/* Fast path: both tuples fit on the current page. */
		if (PageGetFreeSpace(page) >= combinedSize)
		{
			nbuf = buf;
			npage = page;
			break;
		}

		/* Element only, if both can't share a page and this is the last page. */
		if (combinedSize > maxSize && PageGetFreeSpace(page) >= etupSize &&
			!BlockNumberIsValid(TqhnswPageGetOpaque(page)->nextblkno))
		{
			TqhnswAppendInsertPage(index, &nbuf, &npage, state, page);
			break;
		}

		currentPage = TqhnswPageGetOpaque(page)->nextblkno;

		if (BlockNumberIsValid(currentPage))
		{
			/* Move to the next page. */
			GenericXLogAbort(state);
			UnlockReleaseBuffer(buf);
		}
		else
		{
			Buffer		newbuf;
			Page		newpage;

			/* Append a fresh page after the last and commit the link. */
			TqhnswAppendInsertPage(index, &newbuf, &newpage, state, page);
			GenericXLogFinish(state);
			UnlockReleaseBuffer(buf);

			/* Continue on the new page. */
			buf = newbuf;
			state = GenericXLogStart(index);
			page = GenericXLogRegisterBuffer(state, buf, 0);

			/* Add a second page for the neighbor tuple if needed. */
			if (PageGetFreeSpace(page) < combinedSize)
				TqhnswAppendInsertPage(index, &nbuf, &npage, state, page);
			else
			{
				nbuf = buf;
				npage = page;
			}

			break;
		}
	}

	e->blkno = BufferGetBlockNumber(buf);
	e->neighborPage = BufferGetBlockNumber(nbuf);

	/* If we only allocated new pages, hint at the neighbor page next time. */
	if (!BlockNumberIsValid(newInsertPage))
		newInsertPage = e->neighborPage;

	e->offno = OffsetNumberNext(PageGetMaxOffsetNumber(page));
	if (nbuf == buf)
		e->neighborOffno = OffsetNumberNext(e->offno);
	else
		e->neighborOffno = FirstOffsetNumber;

	ItemPointerSet(&etup->neighbortid, e->neighborPage, e->neighborOffno);

	if (PageAddItem(page, (Item) etup, etupSize, InvalidOffsetNumber, false, false) != e->offno)
		elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));

	if (PageAddItem(npage, (Item) ntup, ntupSize, InvalidOffsetNumber, false, false) != e->neighborOffno)
		elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));

	GenericXLogFinish(state);
	UnlockReleaseBuffer(buf);
	if (nbuf != buf)
		UnlockReleaseBuffer(nbuf);

	pfree(etup);
	pfree(ntup);

	if (BlockNumberIsValid(newInsertPage) && newInsertPage != insertPage)
		*updatedInsertPage = newInsertPage;
}

/*
 * Check whether an edge to newElement already exists in the layer-lc slice
 * [startIdx, startIdx+lm) of ntup (mirrors hnswinsert.c ConnectionExists).
 */
static bool
ConnectionExists(TqhnswElement *newElement, TqhnswNeighborTuple ntup, int startIdx, int lm)
{
	for (int i = 0; i < lm; i++)
	{
		ItemPointer indextid = &ntup->indextids[startIdx + i];

		if (!ItemPointerIsValid(indextid))
			break;

		if (ItemPointerGetBlockNumber(indextid) == newElement->blkno &&
			ItemPointerGetOffsetNumber(indextid) == newElement->offno)
			return true;
	}
	return false;
}

/*
 * Compute the single slice slot (relative to the layer-lc slice) that
 * newElement should occupy in neighborElement's neighbor tuple (mirrors
 * hnswinsert.c GetUpdateIndex).  Done UNLOCKED (SHARE via TqhnswLoadNeighbors)
 * since selecting neighbors can take time; the chosen slot is re-validated
 * under the EXCLUSIVE lock in TqhnswUpdateNeighborOnDisk.
 *
 * Returns:
 *   -1  newElement should NOT be added (it loses the prune against the
 *       neighbor's current edges).
 *   -2  the slice has a free/invalid slot; the writer fills the first one.
 *   >=0 the (slice-relative) index whose current occupant is pruned OUT in
 *       favor of newElement.
 *
 * Crucially this never mutates neighborElement's cached neighbor array: the
 * prune is computed into a LOCAL scratch element (palloc'd in updateCtx) so the
 * shared cached element is left untouched (FIX 2).
 */
static int
GetUpdateIndex(Relation index, const TqModel *model, TqMetric metric, HTAB *cache,
			   MemoryContext cacheCtx, MemoryContext updateCtx,
			   TqhnswElement *neighborElement,
			   TqhnswElement *newElement, double distance, int m, int dc, int lc)
{
	int			lm = TqhnswGetLayerM(m, lc);
	int			idx = -1;
	TqhnswNeighborArray *na;
	TqhnswNeighborArray *savedNeighbors;
	MemoryContext oldCtx = MemoryContextSwitchTo(updateCtx);

	/*
	 * Load the neighbor's current layer-lc edges (SHARE).  TqhnswLoadNeighbors
	 * assigns the freshly loaded array (in updateCtx) to
	 * neighborElement->neighbors[lc]; since this is a cache-shared element and
	 * updateCtx is reset between neighbors, save and restore the original
	 * pointer so we never leave a dangling array on, or otherwise mutate, the
	 * cached element's authoritative neighbor state (FIX 2).
	 */
	savedNeighbors = neighborElement->neighbors[lc];
	na = TqhnswLoadNeighbors(index, model, metric, neighborElement, lc, m, updateCtx);

	if (na->count < lm)
	{
		/* Free slot exists; writer will pick the first invalid one. */
		idx = -2;
	}
	else
	{
		/*
		 * Full: run the prune in a LOCAL scratch element so neighborElement's
		 * cached neighbor array is never mutated (FIX 2).  The scratch element
		 * shares neighborElement's rhat (used only for distance) and carries a
		 * private neighbor slice seeded from the loaded edges.
		 */
		TqhnswElement scratch;
		TqhnswNeighborArray *sna;
		int			i;

		scratch.rhat = neighborElement->rhat;
		scratch.level = (uint8) lc;		/* neighbors[] sized [0..lc] */
		scratch.neighbors = (TqhnswNeighborArray **)
			palloc0(sizeof(TqhnswNeighborArray *) * (lc + 1));
		sna = palloc(TQHNSW_NEIGHBOR_ARRAY_SIZE(lm));
		sna->count = na->count;
		scratch.neighbors[lc] = sna;

		/* Resolve each loaded TID to an element so rhat is available. */
		for (i = 0; i < na->count; i++)
		{
			TqhnswElement *ne = TqhnswLoadElement(index, model, metric,
												  &na->items[i].tid, cacheCtx, cache);

			TqhnswPtrStore(NULL, sna->items[i].element, ne);
			sna->items[i].distance = TqhnswBuildDistance(neighborElement->rhat,
														 ne->rhat, dc, metric);
		}

		/* Re-select into the scratch slice (newElement competes). */
		TqhnswUpdateConnection(NULL, &scratch, newElement, distance, lm, lc,
							   dc, metric);

		/*
		 * Find newElement's slot in the pruned result.  If it is absent,
		 * newElement lost the prune -> do not add (idx stays -1), mirroring
		 * HNSW returning -1 when newElement is not selected.
		 */
		for (i = 0; i < sna->count; i++)
		{
			if (TqhnswPtrAccess(NULL, sna->items[i].element) == newElement)
			{
				idx = i;
				break;
			}
		}
	}

	/* Restore the cached element's neighbor pointer (FIX 2). */
	neighborElement->neighbors[lc] = savedNeighbors;

	MemoryContextSwitchTo(oldCtx);
	return idx;
}

/*
 * Persist the single reciprocal edge neighborElement -> newElement at layer lc
 * (mirrors hnswinsert.c UpdateNeighborOnDisk).  idx is the slice-relative slot
 * computed (unlocked) by GetUpdateIndex.
 *
 * Under the EXCLUSIVE lock we RE-READ the live neighbor tuple, re-check
 * ConnectionExists, resolve a -2 ("any free slot") request against the live
 * slice, and write a SINGLE ItemPointer at slot startIdx + idx -- leaving every
 * other slot (and count) untouched (FIX 1).  This preserves any edge a
 * concurrent insert added to a different slot between the two phases; the
 * residual race of two inserts choosing the same idx is the same one HNSW
 * tolerates.
 */
static void
TqhnswUpdateNeighborOnDisk(Relation index, TqhnswElement *neighborElement,
						   TqhnswElement *newElement, int idx, int m, int lc)
{
	Buffer		buf;
	Page		page;
	GenericXLogState *state;
	TqhnswNeighborTuple ntup;
	int			lm = TqhnswGetLayerM(m, lc);
	int			startIdx = (neighborElement->level - lc) * m;

	buf = ReadBuffer(index, neighborElement->neighborPage);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	state = GenericXLogStart(index);
	page = GenericXLogRegisterBuffer(state, buf, 0);

	ntup = (TqhnswNeighborTuple) PageGetItem(page,
											 PageGetItemId(page, neighborElement->neighborOffno));

	/* Already connected (e.g. by a concurrent backend). */
	if (ConnectionExists(newElement, ntup, startIdx, lm))
		idx = -1;
	else if (idx == -2)
	{
		/* Re-resolve the free slot against the live tuple. */
		idx = -1;
		for (int j = 0; j < lm; j++)
		{
			if (!ItemPointerIsValid(&ntup->indextids[startIdx + j]))
			{
				idx = startIdx + j;
				break;
			}
		}
	}
	else
		idx += startIdx;

	/* Write exactly one slot; leave all others (and count) untouched. */
	if (idx >= 0 && idx < ntup->count)
	{
		ItemPointer indextid = &ntup->indextids[idx];

		ItemPointerSet(indextid, newElement->blkno, newElement->offno);
		GenericXLogFinish(state);
	}
	else
		GenericXLogAbort(state);

	UnlockReleaseBuffer(buf);
}

/*
 * For each layer, persist the reciprocal edges from newElement's selected
 * neighbors back to newElement (mirrors hnswinsert.c HnswUpdateNeighborsOnDisk).
 * newElement->neighbors[lc] holds the forward links chosen by
 * TqhnswInsertElement; each target's authoritative on-disk neighbor tuple is
 * re-read under EXCLUSIVE lock here.
 */
static void
TqhnswUpdateNeighborsOnDisk(Relation index, const TqModel *model, TqMetric metric,
							HTAB *cache, MemoryContext ctx, TqhnswElement *newElement,
							int m, int dc)
{
	/* Throwaway context for the per-neighbor unlocked prune (mirrors HNSW). */
	MemoryContext updateCtx = AllocSetContextCreate(ctx,
													"tqhnsw insert update context",
													ALLOCSET_DEFAULT_SIZES);

	for (int lc = newElement->level; lc >= 0; lc--)
	{
		TqhnswNeighborArray *na = newElement->neighbors[lc];

		for (int i = 0; i < na->count; i++)
		{
			TqhnswElement *neighborElement = TqhnswPtrAccess(NULL, na->items[i].element);
			int			idx;

			/* Unlocked: compute the single slot newElement should occupy. */
			idx = GetUpdateIndex(index, model, metric, cache, ctx, updateCtx,
								 neighborElement, newElement,
								 na->items[i].distance, m, dc, lc);

			/* newElement was not selected as a neighbor. */
			if (idx == -1)
			{
				MemoryContextReset(updateCtx);
				continue;
			}

			/* Locked: re-read + single-slot write. */
			TqhnswUpdateNeighborOnDisk(index, neighborElement, newElement, idx, m, lc);

			MemoryContextReset(updateCtx);
		}
	}

	MemoryContextDelete(updateCtx);
}

/*
 * Set the meta-page entry point + insertPage hint + bump nVectors, under an
 * EXCLUSIVE lock on the meta page.  When updateEntry is true the entry point is
 * overwritten only if the (re-read-under-lock) entry level is still below
 * newElement's level.  newInsertPage (if valid) advances the insert-page hint.
 */
static void
TqhnswUpdateMetaPage(Relation index, TqhnswElement *newElement, bool updateEntry,
					 BlockNumber newInsertPage)
{
	Buffer		buf;
	Page		page;
	GenericXLogState *state;
	TqhnswMetaPage metap;

	buf = ReadBuffer(index, TQHNSW_METAPAGE_BLKNO);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	state = GenericXLogStart(index);
	page = GenericXLogRegisterBuffer(state, buf, 0);
	metap = TqhnswPageGetMeta(page);

	if (updateEntry && (metap->entryLevel < 0 || newElement->level > metap->entryLevel))
	{
		metap->entryBlkno = newElement->blkno;
		metap->entryOffno = newElement->offno;
		metap->entryLevel = (int16) newElement->level;
	}

	if (BlockNumberIsValid(newInsertPage))
		metap->insertPage = newInsertPage;

	metap->nVectors += 1;

	GenericXLogFinish(state);
	UnlockReleaseBuffer(buf);
}

/*
 * Read efConstruction and the current entry point from the meta page (SHARE).
 */
static void
TqhnswGetInsertMeta(Relation index, int *m, int *efConstruction,
					BlockNumber *entryBlkno, OffsetNumber *entryOffno, int *entryLevel)
{
	Buffer		buf;
	Page		page;
	TqhnswMetaPage metap;

	buf = ReadBuffer(index, TQHNSW_METAPAGE_BLKNO);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	metap = TqhnswPageGetMeta(page);

	if (unlikely(metap->magicNumber != TQHNSW_MAGIC_NUMBER))
	{
		UnlockReleaseBuffer(buf);
		elog(ERROR, "tqhnsw index is not valid");
	}

	*m = metap->m;
	*efConstruction = metap->efConstruction;
	*entryBlkno = metap->entryBlkno;
	*entryOffno = metap->entryOffno;
	*entryLevel = metap->entryLevel;

	UnlockReleaseBuffer(buf);
}

/*
 * Insert a single quantized tuple into the on-disk graph.
 */
static void
TqhnswInsertTupleOnDisk(Relation index, const TqModel *model, TqMetric metric,
						Datum value, ItemPointer heaptid, MemoryContext insertCtx)
{
	int			m;
	int			efConstruction;
	BlockNumber entryBlkno;
	OffsetNumber entryOffno;
	int			entryLevel;
	int			dimCodes = model->dimCodes;
	int			codesBytes = TQ_CODES_BYTES(dimCodes, model->bits);
	HTAB	   *cache;
	TqhnswElement *newElement;
	BlockNumber updatedInsertPage = InvalidBlockNumber;
	LOCKMODE	lockmode = ShareLock;

	/*
	 * Take a shared page lock for the whole insert.  This lets a future
	 * graph-mutating vacuum (#2) drain in-flight inserts before repairing the
	 * graph.  A page lock is used so it does not interfere with buffer locks (or
	 * reads while vacuuming).  Upgraded to ExclusiveLock below when this insert
	 * may change the entry point.  Mirrors hnswinsert.c HnswInsertTupleOnDisk.
	 */
	LockPage(index, TQHNSW_UPDATE_LOCK, lockmode);

	TqhnswGetInsertMeta(index, &m, &efConstruction, &entryBlkno, &entryOffno, &entryLevel);

	cache = TqhnswCreateElementCache(insertCtx);
	newElement = TqhnswQuantizeElement(index, model, metric, m, dimCodes,
									   codesBytes, heaptid, value);

	/*
	 * Prevent concurrent inserts when this element may change the entry point
	 * (empty index, or its level exceeds the current entry level) -- the same
	 * condition HNSW uses.  Re-read the entry point after the upgrade since it
	 * may have advanced while we held only the shared lock.
	 */
	if (!BlockNumberIsValid(entryBlkno) || entryLevel < 0 ||
		newElement->level > entryLevel)
	{
		UnlockPage(index, TQHNSW_UPDATE_LOCK, lockmode);
		lockmode = ExclusiveLock;
		LockPage(index, TQHNSW_UPDATE_LOCK, lockmode);

		TqhnswGetInsertMeta(index, &m, &efConstruction, &entryBlkno, &entryOffno, &entryLevel);
	}

	if (!BlockNumberIsValid(entryBlkno) || entryLevel < 0)
	{
		/*
		 * Empty index: the new element becomes the sole node and the entry point.
		 * Its neighbor tuple is all-invalid (no forward links to write).
		 */
		TqhnswAddElementOnDisk(index, newElement, m, codesBytes,
							   GetInsertPage(index), &updatedInsertPage);
		TqhnswUpdateMetaPage(index, newElement, true, updatedInsertPage);

		UnlockPage(index, TQHNSW_UPDATE_LOCK, lockmode);
		return;
	}

	/* Non-empty: materialize the entry point and run Alg 1 over the disk graph. */
	{
		ItemPointerData entryTid;
		TqhnswElement *entryPoint;

		ItemPointerSet(&entryTid, entryBlkno, entryOffno);
		entryPoint = TqhnswLoadElement(index, model, metric, &entryTid,
									   insertCtx, cache);

		TqhnswInsertElement(NULL /* base */, index, model, cache, insertCtx,
							newElement, entryPoint, m, efConstruction, dimCodes,
							metric);
	}

	/* Write the new element (forward links from newElement->neighbors). */
	TqhnswAddElementOnDisk(index, newElement, m, codesBytes,
						   GetInsertPage(index), &updatedInsertPage);

	/* Persist reciprocal edges under per-neighbor EXCLUSIVE locks. */
	TqhnswUpdateNeighborsOnDisk(index, model, metric, cache, insertCtx, newElement, m, dimCodes);

	/* Entry-point update (if higher) + insert-page hint + nVectors bump. */
	TqhnswUpdateMetaPage(index, newElement, newElement->level > entryLevel,
						 updatedInsertPage);

	/* Release the page-level update lock. */
	UnlockPage(index, TQHNSW_UPDATE_LOCK, lockmode);
}

bool
tqhnswinsert(Relation index, Datum *values, bool *isnull, ItemPointer heap_tid,
			 Relation heap, IndexUniqueCheck checkUnique, bool indexUnchanged,
			 struct IndexInfo *indexInfo)
{
	MemoryContext oldCtx;
	MemoryContext insertCtx;
	TqModel    *model;
	TqMetric	metric;

	/* Skip nulls. */
	if (isnull[0])
		return false;

	insertCtx = AllocSetContextCreate(CurrentMemoryContext,
									  "tqhnsw insert temporary context",
									  ALLOCSET_DEFAULT_SIZES);
	oldCtx = MemoryContextSwitchTo(insertCtx);

	model = TqhnswGetCachedModel(index);
	metric = model->metric;

	TqhnswInsertTupleOnDisk(index, model, metric, values[0], heap_tid, insertCtx);

	MemoryContextSwitchTo(oldCtx);
	MemoryContextDelete(insertCtx);

	/* tqhnsw is never a unique index. */
	return false;
}
