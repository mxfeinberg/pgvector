#!/usr/bin/env python3
"""
parquet_to_npy.py — extract an embedding column from one or more parquet files
into base.npy + queries.npy for tqflat_bench.py.

The query set is a disjoint holdout (the rows immediately after the base slice),
so queries are not members of the base table — recall is then measured against a
true held-out query distribution rather than self-matches.

Example (Qdrant dbpedia OpenAI text-embedding-3-large-1536):
    python3 bench/parquet_to_npy.py \
        --parquet /tmp/tqdata/openai/train-0000*.parquet \
        --column text-embedding-3-large-1536-embedding \
        --n-base 100000 --n-queries 1000 \
        --base-out /tmp/tqdata/openai/base.npy \
        --queries-out /tmp/tqdata/openai/queries.npy
"""
import argparse
import glob
import sys

import numpy as np
import pyarrow.parquet as pq


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parquet", nargs="+", required=True,
                   help="parquet file(s) or glob(s), read in sorted order")
    p.add_argument("--column", required=True, help="embedding column name (list<float>)")
    p.add_argument("--n-base", type=int, required=True, help="rows for the base set")
    p.add_argument("--n-queries", type=int, default=1000, help="held-out query rows")
    p.add_argument("--base-out", required=True)
    p.add_argument("--queries-out", required=True)
    args = p.parse_args()

    files = []
    for pat in args.parquet:
        files.extend(sorted(glob.glob(pat)))
    if not files:
        print("No parquet files matched.", file=sys.stderr)
        sys.exit(1)

    need = args.n_base + args.n_queries
    chunks = []
    have = 0
    for f in files:
        if have >= need:
            break
        col = pq.read_table(f, columns=[args.column]).column(args.column)
        arr = np.asarray(col.to_pylist(), dtype=np.float32)
        chunks.append(arr)
        have += len(arr)
        print(f"  read {f}: {len(arr)} rows (total {have})")

    data = np.concatenate(chunks, axis=0)
    if len(data) < need:
        print(f"WARNING: only {len(data)} rows available, need {need}; "
              f"reducing query holdout.", file=sys.stderr)
    base = data[:args.n_base]
    queries = data[args.n_base:args.n_base + args.n_queries]

    np.save(args.base_out, base)
    np.save(args.queries_out, queries)
    print(f"base    -> {args.base_out}  shape={base.shape} dtype={base.dtype}")
    print(f"queries -> {args.queries_out}  shape={queries.shape}")
    print(f"mean ||v|| base = {np.linalg.norm(base, axis=1).mean():.4f} "
          f"(≈1.0 => already unit-normalized; L2 and cosine ranking coincide)")


if __name__ == "__main__":
    main()
