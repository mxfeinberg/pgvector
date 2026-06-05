#!/usr/bin/env python3
"""TQ-HNSW parallel-vs-serial BUILD-TIME benchmark (sub-project #3 payoff).

For each dataset, loads the vectors once, then builds a tqhnsw index twice --
serial (max_parallel_maintenance_workers=0) and parallel (forced workers) --
timing each CREATE INDEX and recording index size + recall@10.  The invariant:
parallel only moves the build-time column; size and recall match serial within
noise.  maintenance_work_mem is set high (default 12GB) so the in-memory build
path is used for all datasets (the flush-to-disk fallback is exercised separately).

Reuses the loaders/DB helpers from tqflat_bench.py.  Run on the-fire against tqtest.
"""
import argparse
import sys
import time

import numpy as np
import psycopg2

import tqflat_bench as tb


def connect(args):
    return psycopg2.connect(host=args.host, dbname=args.dbname, user=args.user)


def set_guc(conn, k, v):
    tb._execute(conn, f"SET {k} = {v}")


def count_workers_hint(conn):
    """Best-effort: read the granted parallel-maintenance-worker cap."""
    try:
        row = tb._fetchone(conn, "SHOW max_parallel_maintenance_workers")
        return row[0]
    except Exception:
        return "?"


def timed_build(conn, opclass, m, efc):
    ddl = (f"CREATE INDEX {tb.INDEX_NAME} ON {tb.TABLE} USING tqhnsw (v {opclass}) "
           f"WITH (m={m}, ef_construction={efc}, fast_rotation=true)")
    return tb.build_index_and_time(conn, ddl)


def recall_of(conn, queries, gt, k, op, ef_search, rerank):
    set_guc(conn, "tqhnsw.ef_search", ef_search)
    set_guc(conn, "tqhnsw.rerank", rerank)
    set_guc(conn, "enable_seqscan", "off")
    results, _qps, _ms = tb.run_queries(conn, queries, k, op=op)
    return tb.recall_at_k(results, gt, k)


def load_dataset(name):
    """Return (vectors, queries, gt, dim, opclass, op)."""
    if name == "sift":
        base = "/home/max/tqdata/sift/sift/sift_base.fvecs"
        qry = "/home/max/tqdata/sift/sift/sift_query.fvecs"
        gtp = "/home/max/tqdata/sift/sift/sift_groundtruth.ivecs"
        vectors = tb.load_fvecs(base)
        queries = tb.load_fvecs(qry)
        gt = tb.load_ivecs(gtp)
        return vectors, queries, gt, vectors.shape[1], "vector_l2_ops", "<->"
    if name == "glove":
        vectors, queries, gt = tb.load_hdf5("/home/max/tqdata/glove/glove-200-angular.hdf5")
        return vectors, queries, gt, vectors.shape[1], "vector_cosine_ops", "<=>"
    if name == "openai":
        vectors = np.load("/home/max/tqdata/openai1m/base.npy").astype(np.float32)
        queries = np.load("/home/max/tqdata/openai1m/queries.npy").astype(np.float32)
        gt = None  # no ground truth file; recall measured vs exact below
        return vectors, queries, gt, vectors.shape[1], "vector_cosine_ops", "<=>"
    raise SystemExit(f"unknown dataset {name}")


def exact_gt(conn, queries, k, op):
    """Compute ground truth via seqscan (for datasets without a gt file)."""
    set_guc(conn, "enable_indexscan", "off")
    set_guc(conn, "enable_seqscan", "on")
    res, _q, _m = tb.run_queries(conn, queries, k, op=op)
    set_guc(conn, "enable_indexscan", "on")
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="sift,glove,openai")
    p.add_argument("--host", default="/var/run/postgresql")
    p.add_argument("--dbname", default="tqtest")
    p.add_argument("--user", default="max")
    p.add_argument("--mwm", default="12GB")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--m", type=int, default=16)
    p.add_argument("--efc", type=int, default=64)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--ef-search", type=int, default=100)
    p.add_argument("--rerank", type=int, default=100)
    p.add_argument("--max-queries", type=int, default=500)
    p.add_argument("--csv", default="bench/results/tqhnsw_parallel_build_x86.csv")
    args = p.parse_args()

    out = open(args.csv, "w")
    out.write("dataset,n,dim,serial_build_s,parallel_build_s,speedup,"
              "serial_size,parallel_size,serial_recall,parallel_recall,workers_cap\n")
    out.flush()

    for name in args.datasets.split(","):
        name = name.strip()
        print(f"\n================ {name} {time.strftime('%H:%M:%S')} ================", flush=True)
        vectors, queries, gt, dim, opclass, op = load_dataset(name)
        if args.max_queries and len(queries) > args.max_queries:
            queries = queries[:args.max_queries]
            if gt is not None:
                gt = gt[:args.max_queries]
        n = vectors.shape[0]
        print(f"  n={n} dim={dim} metric={opclass}", flush=True)

        conn = connect(args)
        conn.autocommit = True
        tb._execute(conn, f"CREATE SCHEMA IF NOT EXISTS {tb.SCHEMA}")
        tb.drop_index(conn)
        tb._execute(conn, f"DROP TABLE IF EXISTS {tb.TABLE}")
        tb.create_table(conn, dim)
        print(f"  loading {n} vectors …", flush=True)
        tb.load_vectors(conn, vectors)

        set_guc(conn, "maintenance_work_mem", f"'{args.mwm}'")

        # Ground truth (if no file): exact seqscan top-k before building.
        if gt is None:
            print("  computing exact ground truth (seqscan) …", flush=True)
            gt = exact_gt(conn, queries, args.k, op)

        # --- SERIAL build ---
        set_guc(conn, "max_parallel_maintenance_workers", 0)
        print("  SERIAL build …", flush=True)
        s_build = timed_build(conn, opclass, args.m, args.efc)
        s_size, _ = tb.index_size(conn)
        s_recall = recall_of(conn, queries, gt, args.k, op, args.ef_search, args.rerank)
        print(f"  serial: {s_build:.1f}s size={tb.format_bytes(s_size)} recall@{args.k}={s_recall:.4f}",
              flush=True)

        # --- PARALLEL build ---
        set_guc(conn, "max_parallel_maintenance_workers", args.workers)
        set_guc(conn, "min_parallel_table_scan_size", 0)
        print("  PARALLEL build …", flush=True)
        p_build = timed_build(conn, opclass, args.m, args.efc)
        p_size, _ = tb.index_size(conn)
        p_recall = recall_of(conn, queries, gt, args.k, op, args.ef_search, args.rerank)
        wcap = count_workers_hint(conn)
        print(f"  parallel: {p_build:.1f}s size={tb.format_bytes(p_size)} recall@{args.k}={p_recall:.4f} "
              f"(cap={wcap})", flush=True)

        speedup = s_build / p_build if p_build > 0 else 0.0
        print(f"  >>> {name}: speedup {speedup:.2f}x  "
              f"(serial {s_build:.0f}s -> parallel {p_build:.0f}s), "
              f"size {tb.format_bytes(s_size)}/{tb.format_bytes(p_size)}, "
              f"recall {s_recall:.4f}/{p_recall:.4f}", flush=True)
        out.write(f"{name},{n},{dim},{s_build:.2f},{p_build:.2f},{speedup:.3f},"
                  f"{s_size},{p_size},{s_recall:.4f},{p_recall:.4f},{wcap}\n")
        out.flush()

        tb.drop_index(conn)
        tb._execute(conn, f"DROP TABLE IF EXISTS {tb.TABLE}")
        conn.close()

    out.close()
    print(f"\nDONE -> {args.csv}", flush=True)


if __name__ == "__main__":
    main()
