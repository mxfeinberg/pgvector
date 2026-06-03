"""Brute-force (graph-free) fidelity metrics for the TQ+HNSW spike."""
import numpy as np
from scipy.stats import kendalltau, spearmanr


def _dist(q, X, metric):
    if metric == "cosine":
        return 1.0 - X @ q            # X, q unit-normalized
    return np.linalg.norm(X - q, axis=1)


def quantized_search_vectors(rec, metric):
    """Vectors to use as the *quantized* search representation.  For cosine the
    reconstruction must be unit-normalized so 1 - rec@q is a true cosine distance
    (and matches faiss inner-product = cosine downstream); otherwise vectors with a
    larger reconstruction norm get an artificially smaller distance.  L2 uses rec
    as-is.  Normalize once here, not per query."""
    if metric == "cosine":
        n = np.linalg.norm(rec, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return rec / n
    return rec


def recall_at_k(approx_idx, gt_idx, k=10):
    return np.mean([len(set(a[:k]) & set(g[:k])) / k
                    for a, g in zip(approx_idx, gt_idx)])


def bruteforce_rerank_recall(base, rec, queries, gt, metric, rerank, k=10):
    """Rank all by quantized (rec) distance, take top-`rerank`, rerank by true base
    distance, recall@k.  rerank=0 means pure-quantized (no rerank)."""
    approx = []
    for i, q in enumerate(queries):
        dq = _dist(q, rec, metric)
        if rerank and rerank < len(base):
            cand = np.argpartition(dq, rerank)[:rerank]
            dt = _dist(q, base[cand], metric)
            order = cand[np.argsort(dt)]
        else:
            order = np.argsort(dq)
        approx.append(order[:k])
    return recall_at_k(np.array(approx), gt, k)


def overfetch_curve(base, rec, queries, gt, metric, Cs=(0, 10, 20, 50, 100, 200, 500), k=10):
    """recall@k vs rerank budget C (C=0 is pure-quantized)."""
    return {C: bruteforce_rerank_recall(base, rec, queries, gt, metric, rerank=C, k=k)
            for C in Cs}


def near_neighbor_rank_fidelity(base, rec, queries, metric, topn=200):
    """Mean Kendall-tau / Spearman between quantized and true distances over each
    query's true top-`topn` neighborhood (the routing-relevant regime)."""
    taus, rhos = [], []
    for q in queries:
        dt = _dist(q, base, metric)
        near = np.argpartition(dt, topn)[:topn]
        dq = _dist(q, rec[near], metric)
        taus.append(kendalltau(dt[near], dq).statistic)
        rhos.append(spearmanr(dt[near], dq).statistic)
    return dict(kendall_tau=float(np.nanmean(taus)),
                spearman=float(np.nanmean(rhos)))
