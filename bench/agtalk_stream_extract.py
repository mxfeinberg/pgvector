#!/usr/bin/env python3
"""
agtalk_stream_extract.py — STREAMING variant of agtalk_extract.py.

Reads agtalk_posts.sql, writes a valid float32 .npy of the base embeddings to **stdout**
(so it can be piped through a compressor straight to a remote host, avoiding any large
local temp file), and saves the held-out query set to a small local .npy.

It emits EXACTLY n_base = total - n_queries rows so the streamed .npy header is correct:
malformed/short rows are zero-filled (and counted); extra rows past `total` are ignored.

Embedding parse (commas inside the array escaped as ``\\,``; verified 0 errors / 200k):
    line.rsplit('[', 1)[1].split(']', 1)[0].split('\\,')   ->  768 floats

Usage (pipe to the-fire, ~12 GB on the wire instead of 48 GB):
    python3 bench/agtalk_stream_extract.py \
        --sql agtalk_posts.sql --dim 768 --total 4595214 \
        --n-queries 1000 --seed 42 --queries-out /tmp/agtalk_queries.npy \
      | zstd -3 -c \
      | ssh max@the-fire 'zstd -dc > ~/tqdata/agtalk/agtalk_base.npy.tmp \
            && mv ~/tqdata/agtalk/agtalk_base.npy.tmp ~/tqdata/agtalk/agtalk_base.npy'
"""
import argparse
import sys
import time

import numpy as np
import numpy.lib.format as npf

BATCH = 4096


def parse_embedding(line, dim):
    try:
        arr = line.rsplit("[", 1)[1].split("]", 1)[0]
    except IndexError:
        return None
    parts = arr.split("\\,")
    if len(parts) != dim:
        return None
    try:
        return np.asarray(parts, dtype=np.float32)
    except ValueError:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sql", required=True)
    p.add_argument("--queries-out", required=True)
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--total", type=int, required=True, help="exact total row count")
    p.add_argument("--n-queries", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    dim, total = args.dim, args.total
    n_base = total - args.n_queries
    rng = np.random.default_rng(args.seed)
    q_idx = np.sort(rng.choice(total, size=args.n_queries, replace=False))
    q_set = set(int(i) for i in q_idx)
    queries = np.zeros((args.n_queries, dim), dtype=np.float32)

    out = sys.stdout.buffer
    npf.write_array_header_1_0(
        out, {"descr": npf.dtype_to_descr(np.dtype("<f4")),
              "fortran_order": False, "shape": (n_base, dim)})

    buf = np.zeros((BATCH, dim), dtype=np.float32)
    bb = 0          # rows buffered
    bi = 0          # base rows emitted
    qi = 0          # queries filled
    malformed = 0
    t0 = time.time()

    def flush():
        nonlocal bb
        if bb:
            out.write(buf[:bb].tobytes())
            bb = 0

    with open(args.sql, "r", encoding="utf-8", errors="replace") as f:
        for li, line in enumerate(f):
            if li >= total:
                break
            vals = parse_embedding(line, dim)
            if li in q_set:
                if vals is None:
                    malformed += 1
                else:
                    queries[qi] = vals
                qi += 1
                continue
            # base row
            if bi >= n_base:
                continue
            if vals is None:
                malformed += 1
                buf[bb] = 0.0
            else:
                buf[bb] = vals
            bb += 1
            bi += 1
            if bb == BATCH:
                flush()
            if (li + 1) % 500000 == 0:
                el = time.time() - t0
                print(f"  {li+1}/{total}  ({el:.0f}s, {(li+1)/max(el,1):.0f} r/s) "
                      f"base={bi} q={qi} malformed={malformed}",
                      file=sys.stderr, flush=True)
    flush()
    # Pad if the file was short, so the stream matches the declared header.
    while bi < n_base:
        out.write(np.zeros((1, dim), dtype=np.float32).tobytes())
        bi += 1
        malformed += 1
    out.flush()

    np.save(args.queries_out, queries)
    print(f"STREAM DONE in {time.time()-t0:.0f}s: base={bi}x{dim} (streamed), "
          f"queries={qi}x{dim} -> {args.queries_out}, malformed={malformed}",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
