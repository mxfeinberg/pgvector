#!/usr/bin/env python3
"""
agtalk_extract.py — extract EmbeddingGemma-300M embeddings from the agtalk_posts.sql
dump into base.npy + queries.npy for tqflat_bench.py.

The dump is a CSV-ish row per post. The embedding is the LAST bracketed array on the
line; commas inside the array are escaped as ``\\,`` (backslash-comma), same as the body
text. Verified over 200k sample lines: every row yields exactly 768 clean floats.

    line.rsplit('[', 1)[1].split(']', 1)[0].split('\\,')   ->  768 floats

A disjoint random holdout of ``--n-queries`` rows (deterministic seed) becomes the query
set and is EXCLUDED from base.npy, so recall is measured against true held-out queries
rather than self-matches.

Memory is bounded: base.npy is written through an np.memmap, so peak RAM stays small even
for ~4.6M x 768 (~14 GB on disk).

Example (on the-fire):
    python3 bench/agtalk_extract.py \
        --sql ~/tqdata/agtalk/agtalk_posts.sql \
        --base-out ~/tqdata/agtalk/agtalk_base.npy \
        --queries-out ~/tqdata/agtalk/agtalk_queries.npy \
        --dim 768 --n-queries 1000 --seed 42
"""
import argparse
import sys
import time

import numpy as np


def count_lines(path):
    n = 0
    with open(path, "rb") as f:
        buf = f.read(1 << 24)
        while buf:
            n += buf.count(b"\n")
            buf = f.read(1 << 24)
    return n


def parse_embedding(line, dim):
    """Return a list of `dim` floats, or None if the line is malformed."""
    try:
        arr = line.rsplit("[", 1)[1].split("]", 1)[0]
    except IndexError:
        return None
    parts = arr.split("\\,")
    if len(parts) != dim:
        return None
    try:
        return [float(x) for x in parts]
    except ValueError:
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sql", required=True, help="path to agtalk_posts.sql")
    p.add_argument("--base-out", required=True)
    p.add_argument("--queries-out", required=True)
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--n-queries", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--total-rows", type=int, default=None,
                   help="known row count (skips the initial line count pass)")
    args = p.parse_args()

    dim = args.dim
    t0 = time.time()

    total = args.total_rows
    if total is None:
        print("Counting rows ...", flush=True)
        total = count_lines(args.sql)
    print(f"Total rows: {total}", flush=True)

    # Deterministic disjoint query holdout: pick n_queries distinct line indices.
    rng = np.random.default_rng(args.seed)
    q_idx = np.sort(rng.choice(total, size=args.n_queries, replace=False))
    q_set = set(int(i) for i in q_idx)
    n_base = total - args.n_queries
    print(f"n_base={n_base}  n_queries={args.n_queries}", flush=True)

    base = np.lib.format.open_memmap(
        args.base_out, mode="w+", dtype=np.float32, shape=(n_base, dim))
    queries = np.empty((args.n_queries, dim), dtype=np.float32)

    bi = 0          # next base row
    qi = 0          # next query row (q_idx is sorted, so fill in order)
    malformed = 0
    with open(args.sql, "r", encoding="utf-8", errors="replace") as f:
        for li, line in enumerate(f):
            vals = parse_embedding(line, dim)
            if vals is None:
                malformed += 1
                # A malformed query-row would desync; abort loudly.
                if li in q_set:
                    sys.exit(f"FATAL: query row {li} is malformed")
                continue
            if li in q_set:
                queries[qi] = vals
                qi += 1
            else:
                if bi >= n_base:
                    # extra non-query rows beyond expectation (count drift)
                    malformed += 1
                    continue
                base[bi] = vals
                bi += 1
            if (li + 1) % 500000 == 0:
                el = time.time() - t0
                print(f"  {li+1}/{total} rows  ({el:.0f}s, "
                      f"{(li+1)/el:.0f} rows/s)  base={bi} q={qi}", flush=True)

    base.flush()
    del base
    if qi != args.n_queries:
        sys.exit(f"FATAL: expected {args.n_queries} queries, got {qi}")
    if bi != n_base:
        print(f"WARNING: base rows {bi} != expected {n_base} "
              f"(truncating npy view); malformed={malformed}", file=sys.stderr)
        # Re-open and truncate to actual count.
        full = np.load(args.base_out, mmap_mode="r")
        np.save(args.base_out, np.asarray(full[:bi]))
    np.save(args.queries_out, queries)

    print(f"Done in {time.time()-t0:.0f}s. base={bi}x{dim} -> {args.base_out}; "
          f"queries={qi}x{dim} -> {args.queries_out}; malformed={malformed}", flush=True)


if __name__ == "__main__":
    main()
