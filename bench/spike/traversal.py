"""Real-HNSW traversal variants for the TQ+HNSW spike (faiss-cpu)."""
import heapq

import faiss
import numpy as np
import scoring as S

M = 32
EFC = 128
EFS = (50, 100, 200, 400, 800)


def _index(vectors, metric):
    d = vectors.shape[1]
    space = faiss.METRIC_INNER_PRODUCT if metric == "cosine" else faiss.METRIC_L2
    ix = faiss.IndexHNSWFlat(d, M, space)
    ix.hnsw.efConstruction = EFC
    ix.add(np.ascontiguousarray(vectors))
    return ix


def _search_recall(ix, queries, base, gt, metric, efs, rerank, k=10):
    """For each efSearch: search ix, optionally rerank candidates by true base
    distance, return {ef: recall@k}.  rerank=0 disables rerank (raw faiss order)."""
    out = {}
    for ef in efs:
        ix.hnsw.efSearch = ef
        topn = max(k, rerank) if rerank else k
        _, I = ix.search(np.ascontiguousarray(queries), topn)
        if rerank:
            approx = []
            for qi, q in enumerate(queries):
                cand = I[qi][I[qi] >= 0]
                dt = S._dist(q, base[cand], metric)
                approx.append(cand[np.argsort(dt)][:k])
            out[ef] = S.recall_at_k(np.array(approx), gt, k)
        else:
            out[ef] = S.recall_at_k(I, gt, k)
    return out


def variant_full(base, queries, gt, metric, efs=EFS, k=10):
    """Baseline: full-precision build + search."""
    ix = _index(base, metric)
    return _search_recall(ix, queries, base, gt, metric, efs, rerank=0, k=k)


def variant_A_all_quantized(base, rec, queries, gt, metric, efs=EFS, k=10, rerank=200):
    """Build + traverse on dequantized vectors; rerank candidates by true distance."""
    ix = _index(rec, metric)
    return _search_recall(ix, queries, base, gt, metric, efs, rerank=rerank, k=k)


def variant_C_faiss_sq(base, queries, gt, metric, efs=EFS, k=10):
    """faiss-native HNSWSQ (8-bit scalar quant) reference."""
    d = base.shape[1]
    space = faiss.METRIC_INNER_PRODUCT if metric == "cosine" else faiss.METRIC_L2
    ix = faiss.IndexHNSWSQ(d, faiss.ScalarQuantizer.QT_8bit, M, space)
    ix.hnsw.efConstruction = EFC
    ix.train(np.ascontiguousarray(base))
    ix.add(np.ascontiguousarray(base))
    return _search_recall(ix, queries, base, gt, metric, efs, rerank=0, k=k)


def _extract_graph(ix):
    """Pull HNSW adjacency out of a faiss IndexHNSWFlat.
    Returns (levels, offsets, neighbors, entry, max_level) as numpy arrays/ints.
    faiss flattens all levels per node; offsets[i]..offsets[i+1] spans node i's
    neighbor slots across levels, with cum_nneighbor() giving per-level boundaries."""
    h = ix.hnsw
    neighbors = faiss.vector_to_array(h.neighbors).astype(np.int64)
    offsets = faiss.vector_to_array(h.offsets).astype(np.int64)
    levels = faiss.vector_to_array(h.levels).astype(np.int64)
    cum = faiss.vector_to_array(h.cum_nneighbor_per_level).astype(np.int64)
    return dict(neighbors=neighbors, offsets=offsets, levels=levels,
                cum=cum, entry=int(h.entry_point), max_level=int(h.max_level))


def _level_neighbors(g, node, level):
    """Neighbor node ids of `node` at `level` (>=0), filtering the -1 padding."""
    base = g["offsets"][node]
    start = base + g["cum"][level]
    end = base + g["cum"][level + 1]
    nb = g["neighbors"][start:end]
    return nb[nb >= 0]


def variant_B_build_full_traverse_quant(base, rec, queries, gt, metric,
                                        efs=EFS, k=10, rerank=200):
    """Build the graph on TRUE vectors, then greedy-search scoring nodes by QUANTIZED
    (rec) distance; rerank the ef-sized candidate pool by true distance."""
    ix = _index(base, metric)
    g = _extract_graph(ix)

    def qdist(q, ids):
        return S._dist(q, rec[ids], metric)

    out = {}
    for ef in efs:
        approx = []
        for q in queries:
            # descend upper levels greedily (single best), then ef-beam at level 0
            cur = g["entry"]
            curd = qdist(q, np.array([cur]))[0]
            for lvl in range(g["max_level"], 0, -1):
                improved = True
                while improved:
                    improved = False
                    nb = _level_neighbors(g, cur, lvl)
                    if len(nb) == 0:
                        break
                    dd = qdist(q, nb)
                    j = int(np.argmin(dd))
                    if dd[j] < curd:
                        cur, curd = int(nb[j]), float(dd[j]); improved = True
            # level-0 beam search (ef)
            visited = {cur}
            cand = [(curd, cur)]            # min-heap frontier
            res = [(-curd, cur)]            # max-heap of best ef (store neg)
            while cand:
                d0, n0 = heapq.heappop(cand)
                if -res[0][0] < d0 and len(res) >= ef:
                    break
                nb = _level_neighbors(g, n0, 0)
                # O(|nb|) set-membership filter (nb is small, <=2*M); avoids the
                # O(|visited|) np.isin scan per expansion (quadratic at large ef).
                nb = [int(x) for x in nb.tolist() if x not in visited] if len(nb) else []
                if nb:
                    dd = qdist(q, np.asarray(nb))
                    for nid, ddi in zip(nb, dd.tolist()):
                        visited.add(nid)
                        if len(res) < ef:
                            heapq.heappush(res, (-ddi, nid))
                            heapq.heappush(cand, (ddi, nid))
                        elif ddi < -res[0][0]:
                            heapq.heapreplace(res, (-ddi, nid))
                            heapq.heappush(cand, (ddi, nid))
            pool = np.array([nid for _, nid in res])
            dt = S._dist(q, base[pool], metric)
            approx.append(pool[np.argsort(dt)][:k])
        out[ef] = S.recall_at_k(np.array(approx), gt, k)
    return out
