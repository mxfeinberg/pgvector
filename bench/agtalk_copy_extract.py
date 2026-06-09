#!/usr/bin/env python3
"""
agtalk_copy_extract.py — extract the 768-d EmbeddingGemma embeddings from a pg_dump of the
``public.search_chunks`` table (the full agtalk + combineforum corpus) into base.npy +
queries.npy for tqflat_bench.py.

This is the COPY-format sibling of ``agtalk_extract.py``. The dump is a standard ``pg_dump``
plain-text dump whose data section is a ``COPY ... FROM stdin;`` block: one tab-separated row
per line, with literal tabs/newlines inside text fields escaped as ``\\t`` / ``\\n`` (so every
row is exactly one physical line). The 15 columns are::

    chunk_id, source, source_post_id, thread_id, forum_id, chunk_index, thread_title, author,
    posted_at, source_url, content_plain, content_tsv, embedding, embed_model, indexed_at

so the ``vector(768)`` embedding is **field 13** (index 12), emitted as ``[f1,f2,...,f768]`` with
plain commas (NOT the ``\\,``-escaped format of the old agtalk_posts.sql). Parsing is therefore::

    e = line.split('\\t')[12]            # the [...] vector literal
    np.fromstring(e[1:-1], sep=',', ...) # 768 floats, C-speed

The data section is found dynamically (the ``COPY public.search_chunks`` line) and consumed until
the ``\\.`` terminator, so header/footer lines are skipped without hard-coding line numbers.

A deterministic disjoint random holdout of ``--n-queries`` rows (seed) becomes the query set and is
EXCLUDED from base.npy. Memory is bounded: base.npy is written through an np.memmap.

Example (on the-fire):
    python3 bench/agtalk_copy_extract.py \\
        --sql ~/tqdata/search_chunks.sql \\
        --base-out ~/tqdata/agtalk/agtalk_base.npy \\
        --queries-out ~/tqdata/agtalk/agtalk_queries.npy \\
        --dim 768 --total 7736111 --n-queries 1000 --seed 42
"""
import argparse
import os
import sys
import time

import numpy as np

NCOLS = 15          # columns in public.search_chunks
EMB_COL = 12        # 0-based index of the embedding column


def parse_embedding(line, dim):
    """Return a (dim,) float32 array, or None if the line is malformed."""
    fields = line.split("\t")
    if len(fields) != NCOLS:
        return None
    e = fields[EMB_COL].strip()
    if len(e) < 2 or e[0] != "[" or e[-1] != "]":
        return None
    try:
        v = np.fromstring(e[1:-1], sep=",", dtype=np.float32)
    except ValueError:
        return None
    if v.shape[0] != dim:
        return None
    return v


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sql", required=True, help="path to the search_chunks pg_dump .sql")
    p.add_argument("--base-out", required=True)
    p.add_argument("--queries-out", required=True)
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--total", type=int, required=True,
                   help="exact number of data rows in the COPY block (for the holdout RNG)")
    p.add_argument("--n-queries", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    dim, total = args.dim, args.total
    t0 = time.time()

    # Deterministic disjoint query holdout: pick n_queries distinct data-row indices.
    rng = np.random.default_rng(args.seed)
    q_idx = np.sort(rng.choice(total, size=args.n_queries, replace=False))
    q_set = set(int(i) for i in q_idx)
    n_base = total - args.n_queries
    print(f"total={total}  n_base={n_base}  n_queries={args.n_queries}", flush=True)

    base = np.lib.format.open_memmap(
        args.base_out, mode="w+", dtype=np.float32, shape=(n_base, dim))
    queries = np.zeros((args.n_queries, dim), dtype=np.float32)

    li = -1         # data-row index (0-based, only data rows count)
    bi = 0          # next base row
    qi = 0          # next query row
    malformed = 0
    malformed_q = 0
    in_data = False

    with open(args.sql, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            if not in_data:
                if raw.startswith("COPY public.search_chunks "):
                    in_data = True
                continue
            if raw.startswith("\\."):           # COPY terminator
                break
            li += 1
            if li >= total:
                # extra rows past the declared total (count drift) — ignore
                continue
            line = raw[:-1] if raw.endswith("\n") else raw
            vals = parse_embedding(line, dim)
            if li in q_set:
                if vals is None:
                    malformed_q += 1            # leave this query slot unused
                else:
                    queries[qi] = vals
                    qi += 1
                continue
            # base row
            if vals is None:
                malformed += 1
                continue
            if bi >= n_base:
                malformed += 1
                continue
            base[bi] = vals
            bi += 1
            if (li + 1) % 500000 == 0:
                el = time.time() - t0
                print(f"  {li+1}/{total} rows  ({el:.0f}s, {(li+1)/max(el,1):.0f} rows/s)  "
                      f"base={bi} q={qi} malformed={malformed}", flush=True)

    base.flush()
    del base

    # Truncate base to the actual valid count. Do NOT materialize the whole array in
    # RAM or overwrite the file in place (that fails at ~23 GB): copy the first `bi`
    # rows to a temp npy in bounded chunks via memmap, then atomically replace.
    if bi != n_base:
        print(f"NOTE: valid base rows {bi} != declared {n_base} "
              f"(malformed/skipped base={malformed}); truncating npy via chunked copy.",
              file=sys.stderr)
        tmp = args.base_out + ".tmp"
        src = np.load(args.base_out, mmap_mode="r")
        dst = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(bi, dim))
        CH = 500000
        for s in range(0, bi, CH):
            e = min(s + CH, bi)
            dst[s:e] = src[s:e]
        dst.flush()
        del dst, src
        os.replace(tmp, args.base_out)
    # Truncate queries if any chosen query row was malformed (≈never).
    if qi != args.n_queries:
        print(f"NOTE: valid queries {qi} != {args.n_queries} "
              f"(malformed query rows={malformed_q}); truncating.", file=sys.stderr)
        queries = queries[:qi]
    np.save(args.queries_out, queries)

    print(f"Done in {time.time()-t0:.0f}s. base={bi}x{dim} -> {args.base_out}; "
          f"queries={qi}x{dim} -> {args.queries_out}; "
          f"data_rows_seen={li+1} malformed_base={malformed} malformed_q={malformed_q}",
          flush=True)


if __name__ == "__main__":
    main()
