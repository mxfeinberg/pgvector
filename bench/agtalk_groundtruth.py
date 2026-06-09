#!/usr/bin/env python3
"""
agtalk_groundtruth.py — exact kNN ground truth for the agtalk benchmark.

Computes, for each query, the indices of its exact top-k nearest base vectors under a
given metric, and writes an int32 (n_queries, k) array of 0-based base-row indices —
the format tqflat_bench.py expects for --dataset-gt.

Uses numpy batched matmul (no faiss). Base is mmap'd; queries are processed against the
full base in chunks to bound RAM. For cosine, both sides are L2-normalized so the dot
product equals cosine similarity (argmax similarity == nearest).

Example (on the-fire):
    python3 bench/agtalk_groundtruth.py \
        --base ~/tqdata/agtalk/agtalk_base.npy \
        --queries ~/tqdata/agtalk/agtalk_queries.npy \
        --out ~/tqdata/agtalk/agtalk_gt.npy \
        --metric cosine --k 100 --chunk 200000
"""
import argparse
import time

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True)
    p.add_argument("--queries", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--metric", choices=["cosine", "l2", "ip"], default="cosine")
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--chunk", type=int, default=200000,
                   help="base rows per chunk (RAM/throughput tradeoff)")
    args = p.parse_args()

    t0 = time.time()
    base = np.load(args.base, mmap_mode="r")        # (N, dim) float32, mmap
    queries = np.load(args.queries).astype(np.float32)  # (Q, dim) in RAM
    N, dim = base.shape
    Q = queries.shape[0]
    k = args.k
    print(f"base={base.shape} queries={queries.shape} metric={args.metric} k={k}",
          flush=True)

    if args.metric in ("cosine",):
        qn = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-30)
    else:
        qn = queries

    # Running top-k per query. For cosine/ip we maximize similarity; for l2 we
    # minimize distance. Track the best-k as (score, idx) merged across chunks.
    maximize = args.metric in ("cosine", "ip")
    best_score = np.full((Q, k), -np.inf if maximize else np.inf, dtype=np.float32)
    best_idx = np.full((Q, k), -1, dtype=np.int64)

    q_sq = None
    if args.metric == "l2":
        q_sq = (qn * qn).sum(axis=1, keepdims=True)  # (Q,1)

    for start in range(0, N, args.chunk):
        end = min(start + args.chunk, N)
        blk = np.asarray(base[start:end], dtype=np.float32)   # (b, dim)
        if args.metric == "cosine":
            blk = blk / (np.linalg.norm(blk, axis=1, keepdims=True) + 1e-30)
            sim = qn @ blk.T                                   # (Q, b) similarity
            score = sim
        elif args.metric == "ip":
            score = qn @ blk.T                                # (Q, b)
        else:  # l2: -(||q||^2 - 2 q.b + ||b||^2); we store NEGATIVE sq-dist so we
               # can use the same "maximize" merge, then negate idx selection.
            b_sq = (blk * blk).sum(axis=1)[None, :]           # (1, b)
            d2 = q_sq - 2.0 * (qn @ blk.T) + b_sq             # (Q, b) squared dist
            score = -d2                                        # maximize -> nearest

        # Merge this chunk's candidates with the running best-k.
        b = end - start
        cand_idx = np.broadcast_to(np.arange(start, end), (Q, b))
        all_score = np.concatenate([best_score, score], axis=1)        # (Q, k+b)
        all_idx = np.concatenate([best_idx, cand_idx], axis=1)         # (Q, k+b)
        # top-k by score (largest), per row
        part = np.argpartition(-all_score, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(Q)[:, None]
        best_score = all_score[rows, part]
        best_idx = all_idx[rows, part]
        if (start // args.chunk) % 5 == 0:
            print(f"  base {end}/{N}  ({time.time()-t0:.0f}s)", flush=True)

    # Final sort within the k so column 0 is the true nearest.
    order = np.argsort(-best_score, axis=1)            # descending similarity
    rows = np.arange(Q)[:, None]
    gt = best_idx[rows, order].astype(np.int32)

    np.save(args.out, gt)
    print(f"Done in {time.time()-t0:.0f}s. gt={gt.shape} -> {args.out}", flush=True)
    print(f"sample gt[0,:5]={gt[0,:5].tolist()}", flush=True)


if __name__ == "__main__":
    main()
