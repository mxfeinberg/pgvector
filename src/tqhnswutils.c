#include "postgres.h"

#include <math.h>

#include "common/pg_prng.h"
#include "lib/pairingheap.h"
#include "nodes/pg_list.h"
#include "storage/bufmgr.h"
#include "utils/memutils.h"
#include "tqhnsw.h"

/*
 * Reconstruct the rotated, full-magnitude vector rhat[i] = norm*scale*centroids[code_i]
 * from an entry's packed codes.  Because the RHT rotation is orthonormal, L2/IP
 * distances between two rhat's equal the distances between the original vectors, so
 * the graph can be built entirely in rotated space without an inverse rotation.
 */
void
TqhnswReconstruct(const TqModel *model, const char *codes,
				  float norm, float scale, float *rhat /* dimCodes */)
{
	int			dc = model->dimCodes;
	int			codesBytes = TQ_CODES_BYTES(dc, model->bits);
	float		s = norm * scale;
	int			i;

	for (i = 0; i < dc; i++)
	{
		uint8		code = TqUnpackCode(codes, codesBytes, i, model->bits);

		rhat[i] = s * model->centroids[code];
	}
}

/*
 * Build-time distance on the reconstructed rotated vectors (variant A).  Smaller
 * is nearer, matching the HNSW convention.
 *
 *   L2          -> squared Euclidean distance
 *   IP / cosine -> negative inner product
 *
 * For cosine the rhat vectors are pre-normalized to unit length at node creation,
 * so -IP orders by cosine distance.
 */
double
TqhnswBuildDistance(const float *a, const float *b, int dc, TqMetric metric)
{
	double		acc = 0.0;
	int			i;

	if (metric == TQ_METRIC_L2)
	{
		for (i = 0; i < dc; i++)
		{
			double		d = (double) a[i] - (double) b[i];

			acc += d * d;
		}
		return acc;
	}

	for (i = 0; i < dc; i++)
		acc += (double) a[i] * (double) b[i];
	return -acc;
}

/* ------------------------------------------------------------------------- *
 * In-memory serial graph build (absolute pointers, no relptr, no LWLocks).  *
 * Ported from hnswutils.c's in-memory branch: HnswSearchLayer (Alg 2),      *
 * SelectNeighbors (Alg 4), HnswFindElementNeighbors (Alg 1) + reciprocal    *
 * pruning (HnswUpdateConnection).  The only seam vs HNSW is the distance     *
 * call, which becomes TqhnswBuildDistance on the nodes' rhat vectors.       *
 * ------------------------------------------------------------------------- */

/*
 * A search candidate: an element plus its distance to the query, with the two
 * pairing-heap link nodes (c_node for the nearest/min heap C, w_node for the
 * furthest/max heap W).  Mirrors HnswSearchCandidate.
 */
typedef struct TqhnswSearchCandidate
{
	pairingheap_node c_node;
	pairingheap_node w_node;
	TqhnswElement *element;
	double		distance;
}			TqhnswSearchCandidate;

#define TqhnswGetSearchCandidate(membername, ptr) \
	pairingheap_container(TqhnswSearchCandidate, membername, ptr)
#define TqhnswGetSearchCandidateConst(membername, ptr) \
	pairingheap_const_container(TqhnswSearchCandidate, membername, ptr)

/* Per-search visited stamp, bumped once per TqhnswSearchLayer call. */
static uint32 tqhnsw_visited_generation = 0;

/* C heap: nearest first (min by distance). */
static int
CompareNearestCandidates(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
	double		da = TqhnswGetSearchCandidateConst(c_node, a)->distance;
	double		db = TqhnswGetSearchCandidateConst(c_node, b)->distance;

	if (da < db)
		return 1;
	if (da > db)
		return -1;
	return 0;
}

/* W heap: furthest first (max by distance). */
static int
CompareFurthestCandidates(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
	double		da = TqhnswGetSearchCandidateConst(w_node, a)->distance;
	double		db = TqhnswGetSearchCandidateConst(w_node, b)->distance;

	if (da < db)
		return -1;
	if (da > db)
		return 1;
	return 0;
}

static TqhnswSearchCandidate *
TqhnswInitSearchCandidate(TqhnswElement *element, double distance)
{
	TqhnswSearchCandidate *sc = palloc(sizeof(TqhnswSearchCandidate));

	sc->element = element;
	sc->distance = distance;
	return sc;
}

/*
 * Whether element e counts toward ef during search.  Mirrors hnswutils.c
 * CountElement: when skipElement is non-NULL (vacuum repair), elements being
 * deleted (invalid heaptid) do NOT count toward ef, so the beam is sized over
 * survivors.  For build/insert (skipElement == NULL) every element counts.
 *
 * Two intentional divergences from hnswutils.c CountElement:
 *   1. Live-ness check uses ItemPointerIsValid(&e->heaptid) (single heaptid)
 *      rather than heaptidsLength > 0, because tqhnsw stores exactly one heap
 *      TID per element (no multi-TID deduplication).
 *   2. The pg_memory_barrier() that hnswutils.c places here is omitted: tqhnsw
 *      has no concurrent shared-memory in-memory build phase (single writer),
 *      so no barrier is needed to order visibility of heaptid writes.
 */
static inline bool
TqhnswCountElement(TqhnswElement *skipElement, TqhnswElement *e)
{
	if (skipElement == NULL)
		return true;
	return ItemPointerIsValid(&e->heaptid);
}

/*
 * Algorithm 2 from the paper (in-memory branch of HnswSearchLayer).
 *
 * ep is a List of TqhnswSearchCandidate * entry points (already distance-scored).
 * Returns a List of TqhnswSearchCandidate * (the W set), ordered furthest-first
 * as the source returns it (callers do not rely on the order, they re-sort).
 */
static List *
TqhnswSearchLayer(char *base, Relation index, const TqModel *model, HTAB *cache,
				  MemoryContext ctx, TqhnswElement *query, List *ep, int ef, int lc,
				  int m, int dc, TqMetric metric, TqhnswElement *skipElement)
{
	List	   *w = NIL;
	pairingheap *C = pairingheap_allocate(CompareNearestCandidates, NULL);
	pairingheap *W = pairingheap_allocate(CompareFurthestCandidates, NULL);
	int			wlen = 0;
	ListCell   *lc2;
	uint32		gen = ++tqhnsw_visited_generation;

	/* Add entry points to C and W, marking them visited. */
	foreach(lc2, ep)
	{
		TqhnswSearchCandidate *sc = (TqhnswSearchCandidate *) lfirst(lc2);

		sc->element->visitedGeneration = gen;
		pairingheap_add(C, &sc->c_node);
		pairingheap_add(W, &sc->w_node);
		if (TqhnswCountElement(skipElement, sc->element))
			wlen++;
	}

	while (!pairingheap_is_empty(C))
	{
		TqhnswSearchCandidate *c = TqhnswGetSearchCandidate(c_node, pairingheap_remove_first(C));
		TqhnswSearchCandidate *f = TqhnswGetSearchCandidate(w_node, pairingheap_first(W));
		TqhnswElement *cElement = c->element;
		int			i;

		if (c->distance > f->distance)
			break;

		/* Iterate the candidate's neighbors at layer lc (skip levels it lacks). */
		if (lc > cElement->level)
			continue;

		/* On-disk path: load neighbors for this layer lazily. */
		if (index != NULL && cElement->neighbors[lc] == NULL)
			TqhnswLoadNeighbors(index, model, metric, cElement, lc, m, ctx);

		{
			TqhnswNeighborArray *na = TqhnswGetNeighbors(base, cElement, lc);

			for (i = 0; i < na->count; i++)
			{
				TqhnswElement *eElement = TqhnswPtrAccess(base, na->items[i].element);
				TqhnswSearchCandidate *e;
				double		eDistance;
				bool		alwaysAdd;

				/* On-disk path: resolve TID to element lazily. */
				if (index != NULL && eElement == NULL)
				{
					eElement = TqhnswLoadElement(index, model, metric,
												 &na->items[i].tid,
												 ctx, cache);
					TqhnswPtrStore(base, na->items[i].element, eElement);
				}

				if (eElement->visitedGeneration == gen)
					continue;
				eElement->visitedGeneration = gen;

				f = TqhnswGetSearchCandidate(w_node, pairingheap_first(W));
				alwaysAdd = wlen < ef;

				eDistance = TqhnswBuildDistance(query->rhat, eElement->rhat, dc, metric);

				if (!(eDistance < f->distance || alwaysAdd))
					continue;

				e = TqhnswInitSearchCandidate(eElement, eDistance);
				pairingheap_add(C, &e->c_node);
				pairingheap_add(W, &e->w_node);
				if (TqhnswCountElement(skipElement, eElement))
				{
					wlen++;

					/*
					 * tqhnsw keeps the textbook Alg-2 invariant wlen == |W|
					 * (decrement on eviction).  This intentionally differs from
					 * hnswutils.c, which never decrements wlen because it tracks
					 * "admitted-live-elements" for its discarded-heap /
					 * iterative-scan path, which tqhnsw lacks.  For build/insert
					 * (skipElement==NULL) the two are equivalent; on the vacuum
					 * repair path (skipElement!=NULL) repaired-node recall is
					 * validated separately by the vacuum recall TAP test.
					 */
					if (wlen > ef)
					{
						pairingheap_remove_first(W);
						wlen--;
					}
				}
			}
		}
	}

	/* Drain W into a List. */
	while (!pairingheap_is_empty(W))
	{
		TqhnswSearchCandidate *sc = TqhnswGetSearchCandidate(w_node, pairingheap_remove_first(W));

		w = lappend(w, sc);
	}

	return w;
}

/* Compare two TqhnswCandidate by distance desc (furthest first), ptr tie-break. */
static int
CompareCandidateDistances(const ListCell *a, const ListCell *b)
{
	TqhnswCandidate *ca = lfirst(a);
	TqhnswCandidate *cb = lfirst(b);

	if (ca->distance < cb->distance)
		return 1;
	if (ca->distance > cb->distance)
		return -1;
	/* Tie-break on raw pointer (build) or relptr offset (disk) — both use .ptr for
	 * consistent ordering; the exact value is immaterial, only stability matters. */
	if (ca->element.ptr < cb->element.ptr)
		return 1;
	if (ca->element.ptr > cb->element.ptr)
		return -1;
	return 0;
}

/*
 * Check if candidate e is closer to the query than to any element already in r.
 * Mirrors CheckElementCloser; distance(e, ri) via TqhnswBuildDistance.
 */
static bool
CheckElementCloser(char *base, TqhnswCandidate *e, List *r, int dc, TqMetric metric)
{
	ListCell   *lc2;

	foreach(lc2, r)
	{
		TqhnswCandidate *ri = lfirst(lc2);
		double		distance = TqhnswBuildDistance(TqhnswPtrAccess(base, e->element)->rhat,
												   TqhnswPtrAccess(base, ri->element)->rhat,
												   dc, metric);

		if (distance <= e->distance)
			return false;
	}

	return true;
}

/*
 * Algorithm 4 from the paper (SelectNeighbors heuristic).  Returns up to lm
 * selected neighbors as a List of TqhnswCandidate *.  c is a List of
 * TqhnswCandidate *.
 */
static List *
TqhnswSelectNeighbors(char *base, List *c, int lm, int dc, TqMetric metric)
{
	List	   *r = NIL;
	List	   *w = list_copy(c);
	TqhnswCandidate **wd;
	int			wdlen = 0;
	int			wdoff = 0;

	if (list_length(w) <= lm)
		return w;

	wd = palloc(sizeof(TqhnswCandidate *) * list_length(w));

	/* Order descending by distance so llast() is the nearest. */
	list_sort(w, CompareCandidateDistances);

	while (list_length(w) > 0 && list_length(r) < lm)
	{
		/* w is ordered desc; llast is the current nearest. */
		TqhnswCandidate *e = llast(w);
		bool		closer;

		w = list_delete_last(w);

		closer = CheckElementCloser(base, e, r, dc, metric);

		if (closer)
			r = lappend(r, e);
		else
			wd[wdlen++] = e;
	}

	/* Keep pruned connections to fill up to lm. */
	while (wdoff < wdlen && list_length(r) < lm)
		r = lappend(r, wd[wdoff++]);

	return r;
}

/*
 * Add element as a neighbor of target at layer lc, pruning target's neighbor
 * list back to lm via SelectNeighbors if it overflows.  Mirrors
 * HnswUpdateConnection.
 */
void
TqhnswUpdateConnection(char *base, TqhnswElement *target, TqhnswElement *element,
					   double distance, int lm, int lc, int dc, TqMetric metric)
{
	TqhnswNeighborArray *na = TqhnswGetNeighbors(base, target, lc);

	if (na->count < lm)
	{
		TqhnswPtrStore(base, na->items[na->count].element, element);
		na->items[na->count].distance = distance;
		na->count++;
		return;
	}

	/* Full: rebuild the candidate set (existing + new) and re-select. */
	{
		List	   *c = NIL;
		List	   *selected;
		ListCell   *lc2;
		int			i;
		TqhnswCandidate *newCand;

		for (i = 0; i < na->count; i++)
		{
			TqhnswCandidate *hc = palloc(sizeof(TqhnswCandidate));
			TqhnswElement *ne = TqhnswPtrAccess(base, na->items[i].element);

			TqhnswPtrStore(base, hc->element, ne);
			hc->distance = TqhnswBuildDistance(target->rhat, ne->rhat,
											   dc, metric);
			c = lappend(c, hc);
		}
		newCand = palloc(sizeof(TqhnswCandidate));
		TqhnswPtrStore(base, newCand->element, element);
		newCand->distance = distance;
		c = lappend(c, newCand);

		selected = TqhnswSelectNeighbors(base, c, lm, dc, metric);

		na->count = 0;
		foreach(lc2, selected)
		{
			TqhnswCandidate *hc = lfirst(lc2);

			TqhnswPtrStore(base, na->items[na->count].element,
						   TqhnswPtrAccess(base, hc->element));
			na->items[na->count].distance = hc->distance;
			na->count++;
		}
	}
}

/* ------------------------------------------------------------------------- *
 * Disk-load helpers (on-disk insert / scan path; build path never calls     *
 * these because index==NULL on that path).                                  *
 * ------------------------------------------------------------------------- */

/*
 * Create the TID->element cache.  The entry layout is TqhnswElementCacheEntry
 * (key first, as required by dynahash).
 */
HTAB *
TqhnswCreateElementCache(MemoryContext ctx)
{
	HASHCTL		hashctl;

	memset(&hashctl, 0, sizeof(hashctl));
	hashctl.keysize = sizeof(ItemPointerData);
	hashctl.entrysize = sizeof(TqhnswElementCacheEntry);
	hashctl.hcxt = ctx;
	return hash_create("tqhnsw element cache", 256, &hashctl,
					   HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);
}

/*
 * Materialize a disk element into ctx, caching it by TID in cache.
 * Loads the element tuple, reconstructs rhat, and lazily allocates the
 * neighbors pointer array (each layer populated by TqhnswLoadNeighbors).
 */
TqhnswElement *
TqhnswLoadElement(Relation index, const TqModel *model, TqMetric metric,
				  ItemPointer tid, MemoryContext ctx, HTAB *cache)
{
	bool		found;
	TqhnswElementCacheEntry *entry;
	TqhnswElement *e;
	Buffer		buf;
	Page		page;
	TqhnswElementTuple etup;
	int			codesBytes = TQ_CODES_BYTES(model->dimCodes, model->bits);
	MemoryContext old;

	/* Cache keyed by ItemPointerData; dynahash copies tid into entry->tid. */
	entry = (TqhnswElementCacheEntry *) hash_search(cache, tid, HASH_ENTER, &found);
	if (found)
		return entry->element;

	old = MemoryContextSwitchTo(ctx);
	e = palloc0(sizeof(TqhnswElement));

	buf = ReadBuffer(index, ItemPointerGetBlockNumber(tid));
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	etup = (TqhnswElementTuple) PageGetItem(page,
											PageGetItemId(page,
														  ItemPointerGetOffsetNumber(tid)));

	e->level = etup->level;
	e->norm = etup->norm;
	e->scale = etup->scale;
	e->heaptid = etup->heaptid;
	e->blkno = ItemPointerGetBlockNumber(tid);
	e->offno = ItemPointerGetOffsetNumber(tid);
	e->neighborPage = ItemPointerGetBlockNumber(&etup->neighbortid);
	e->neighborOffno = ItemPointerGetOffsetNumber(&etup->neighbortid);
	e->codes = palloc(codesBytes);
	memcpy(e->codes, etup->codes, codesBytes);

	UnlockReleaseBuffer(buf);

	e->rhat = palloc(sizeof(float) * model->dimCodes);
	TqhnswReconstruct(model, e->codes, e->norm, e->scale, e->rhat);

	/* Cosine: unit-normalize rhat so -IP orders by cosine (mirrors build). */
	if (metric == TQ_METRIC_COSINE)
	{
		double		n = 0.0;
		int			i;

		for (i = 0; i < model->dimCodes; i++)
			n += (double) e->rhat[i] * (double) e->rhat[i];
		n = sqrt(n);
		if (n > 1e-20)
		{
			float		inv = (float) (1.0 / n);

			for (i = 0; i < model->dimCodes; i++)
				e->rhat[i] *= inv;
		}
	}

	/* Neighbor arrays populated lazily per layer by TqhnswLoadNeighbors. */
	e->neighbors = palloc0(sizeof(TqhnswNeighborArray *) * (e->level + 1));

	MemoryContextSwitchTo(old);
	entry->element = e;
	return e;
}

/*
 * Load element's layer-lc neighbor TIDs into a TqhnswNeighborArray.  The
 * items[].element pointers start NULL and are resolved lazily in
 * TqhnswSearchLayer; items[].tid carry the on-disk TIDs.
 * Mirrors TqhnswLoadNeighborTids (tqhnswscan.c) but produces a NeighborArray
 * rather than a raw TID array.
 */
TqhnswNeighborArray *
TqhnswLoadNeighbors(Relation index, const TqModel *model, TqMetric metric,
					TqhnswElement *element, int lc, int m, MemoryContext ctx)
{
	Buffer		buf;
	Page		page;
	TqhnswNeighborTuple ntup;
	int			lm = TqhnswGetLayerM(m, lc);
	int			level;
	int			start;
	TqhnswNeighborArray *na;
	int			i;
	MemoryContext old = MemoryContextSwitchTo(ctx);

	na = palloc(TQHNSW_NEIGHBOR_ARRAY_SIZE(lm));
	na->count = 0;

	buf = ReadBuffer(index, element->neighborPage);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	ntup = (TqhnswNeighborTuple) PageGetItem(page,
											 PageGetItemId(page,
														   element->neighborOffno));

	/* count == (level + 2) * m -> recover element's level from tuple. */
	level = (int) (ntup->count / m) - 2;
	Assert(level == element->level);
	start = (level - lc) * m;

	for (i = 0; i < lm; i++)
	{
		ItemPointer t = &ntup->indextids[start + i];

		if (!ItemPointerIsValid(t))
			continue;
		na->items[na->count].tid = *t;
		na->items[na->count].element.ptr = NULL; /* lazy */
		na->items[na->count].distance = 0;
		na->count++;
	}
	UnlockReleaseBuffer(buf);
	MemoryContextSwitchTo(old);

	element->neighbors[lc] = na;
	return na;
}

/*
 * Remove self (skipElement, matched by block/offset) and any elements being
 * deleted (invalid heaptid) from a candidate list before neighbor selection.
 * Mirrors hnswutils.c RemoveElements.  Disk path only.
 *
 * As with TqhnswCountElement, live-ness is checked via a single heaptid
 * (ItemPointerIsValid) rather than heaptidsLength, and no pg_memory_barrier()
 * is needed (single-writer build; no concurrent in-memory graph).
 */
static List *
TqhnswRemoveElements(List *w, TqhnswElement *skipElement)
{
	ListCell   *lc2;
	List	   *w2 = NIL;

	foreach(lc2, w)
	{
		TqhnswCandidate *hc = (TqhnswCandidate *) lfirst(lc2);
		TqhnswElement *hce = TqhnswPtrAccess(NULL, hc->element);

		if (skipElement != NULL &&
			hce->blkno == skipElement->blkno && hce->offno == skipElement->offno)
			continue;

		if (ItemPointerIsValid(&hce->heaptid))
			w2 = lappend(w2, hc);
	}
	return w2;
}

/*
 * Algorithm 1 from the paper (HnswFindElementNeighbors): greedy descent then
 * per-layer search + select + reciprocal connect.  element already has its level,
 * rhat, codes, and zero-initialized neighbor arrays.
 *
 * build path: base=NULL, index=NULL, model=NULL, cache=NULL.
 * existing=true is used by vacuum repair: element is already in the graph and
 * needs its neighbors re-selected (skip self, widen beam, no inline reciprocal).
 */
void
TqhnswInsertElement(char *base, Relation index, const TqModel *model, HTAB *cache,
					MemoryContext ctx, TqhnswElement *element,
					TqhnswElement *entryPoint, int m,
					int efConstruction, int dc, TqMetric metric, bool existing)
{
	List	   *ep;
	List	   *w;
	int			level = element->level;
	int			entryLevel;
	int			lc;
	TqhnswElement *skipElement = existing ? element : NULL;

	/* No neighbors to select if there is no entry point (mirrors
	 * HnswFindElementNeighbors).  Repair may pass NULL when the graph's entry
	 * point was deleted with no surviving replacement; the element then gets an
	 * empty neighbor list, consistent with HNSW. */
	if (entryPoint == NULL)
		return;

	entryLevel = entryPoint->level;

	/* Entry point candidate. */
	ep = list_make1(TqhnswInitSearchCandidate(entryPoint,
											  TqhnswBuildDistance(element->rhat,
																  entryPoint->rhat,
																  dc, metric)));

	/* 1st phase: greedy descent (ef=1) down to the element's level + 1. */
	for (lc = entryLevel; lc >= level + 1; lc--)
	{
		w = TqhnswSearchLayer(base, index, model, cache, ctx, element, ep, 1, lc, m, dc, metric,
							  skipElement);
		ep = w;
	}

	if (level > entryLevel)
		level = entryLevel;

	/* Add one for existing element (it will be filtered from its own candidate set). */
	if (existing)
		efConstruction++;

	/* 2nd phase: per-layer search + select + connect. */
	for (lc = level; lc >= 0; lc--)
	{
		int			lm = TqhnswGetLayerM(m, lc);
		List	   *lw = NIL;
		List	   *selected;
		ListCell   *lc2;

		w = TqhnswSearchLayer(base, index, model, cache, ctx, element, ep, efConstruction,
							  lc, m, dc, metric, skipElement);

		/* Convert search candidates to plain candidates. */
		foreach(lc2, w)
		{
			TqhnswSearchCandidate *sc = lfirst(lc2);
			TqhnswCandidate *hc = palloc(sizeof(TqhnswCandidate));

			TqhnswPtrStore(base, hc->element, sc->element);
			hc->distance = sc->distance;
			lw = lappend(lw, hc);
		}

		/* Disk path: drop self + deleted elements before selecting neighbors. */
		if (index != NULL)
			lw = TqhnswRemoveElements(lw, skipElement);

		selected = TqhnswSelectNeighbors(base, lw, lm, dc, metric);

		/* Connect element -> selected, and reciprocally selected -> element. */
		foreach(lc2, selected)
		{
			TqhnswCandidate *hc = lfirst(lc2);
			TqhnswNeighborArray *na = TqhnswGetNeighbors(base, element, lc);
			TqhnswElement *neighbor = TqhnswPtrAccess(base, hc->element);

			TqhnswPtrStore(base, na->items[na->count].element, neighbor);
			na->items[na->count].distance = hc->distance;
			na->count++;

			/* Build + insert keep the in-memory reciprocal; repair (existing) does
			 * not -- its reciprocity goes to disk via TqhnswUpdateNeighborsOnDisk. */
			if (!existing)
				TqhnswUpdateConnection(base, neighbor, element, hc->distance, lm, lc,
									   dc, metric);
		}

		ep = w;
	}
}
