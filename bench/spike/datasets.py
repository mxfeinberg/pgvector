"""Dataset loader for the TQ+HNSW spike: 200k subset, normalized, exact GT."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import tqflat_bench as H  # noqa: E402  (file loaders: load_fvecs/load_hdf5/load_from_files)

SUBSET = 200_000
SEED = 1234

# key -> (metric, loader-thunk returning (base, queries_or_None))
def _sources(home):
    td = os.path.join(home, "tqdata")
    return {
        "sift1m": ("l2", lambda: (
            H.load_fvecs(os.path.join(td, "sift/sift/sift_base.fvecs")),
            H.load_fvecs(os.path.join(td, "sift/sift/sift_query.fvecs")))),
        "glove200": ("cosine", lambda: (
            H.load_hdf5(os.path.join(td, "glove/glove-200-angular.hdf5"))[0],
            H.load_hdf5(os.path.join(td, "glove/glove-200-angular.hdf5"))[1])),
        "openai1m": ("cosine", lambda: (
            H.load_from_files(os.path.join(td, "openai1m/base.npy"),
                              os.path.join(td, "openai1m/queries.npy"))[0],
            H.load_from_files(os.path.join(td, "openai1m/base.npy"),
                              os.path.join(td, "openai1m/queries.npy"))[1])),
    }


def _normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def exact_gt(base, queries, metric, k=10):
    """Exact top-k row indices per query, computed in chunks to bound memory."""
    if metric == "cosine":
        b = _normalize(base.astype(np.float32))
        q = _normalize(queries.astype(np.float32))
        # cosine distance rank == descending inner product
        sims = q @ b.T
        return np.argsort(-sims, axis=1)[:, :k]
    else:  # l2
        b = base.astype(np.float32)
        q = queries.astype(np.float32)
        bn = (b * b).sum(1)
        out = np.empty((len(q), k), dtype=np.int64)
        for i in range(len(q)):
            d = bn - 2.0 * (b @ q[i])
            out[i] = np.argpartition(d, k)[:k][np.argsort(d[np.argpartition(d, k)[:k]])]
        return out


def load(key, n_queries):
    """Return dict(base, queries, gt, metric) for a 200k subset."""
    metric, thunk = _sources(os.path.expanduser("~"))[key]
    base, queries = thunk()
    base = np.ascontiguousarray(base[:SUBSET].astype(np.float32))
    rng = np.random.default_rng(SEED)
    if queries is None:
        idx = rng.choice(len(base), size=n_queries, replace=False)
        queries = base[idx]
    else:
        queries = np.ascontiguousarray(queries[:n_queries].astype(np.float32))
    if metric == "cosine":
        base = _normalize(base)
        queries = _normalize(queries)
    gt = exact_gt(base, queries, metric, k=10)
    return dict(base=base, queries=queries, gt=gt, metric=metric, key=key)
