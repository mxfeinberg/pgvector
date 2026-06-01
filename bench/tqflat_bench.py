#!/usr/bin/env python3
"""
tqflat_bench.py — benchmark harness comparing tqflat against hnsw, ivfflat, and exact seqscan.

Usage (synthetic default — runs immediately, no download needed):
    python3 bench/tqflat_bench.py

From the repo root with a custom size:
    python3 bench/tqflat_bench.py --n 10000 --dim 128 --queries 500

Point at a real dataset (numpy .npy files):
    python3 bench/tqflat_bench.py --dataset-vectors /path/to/vectors.npy \
        --dataset-queries /path/to/queries.npy \
        [--dataset-gt /path/to/ground_truth.npy]   # int32 indices, shape (nq, k)

DB connection:
    python3 bench/tqflat_bench.py --host localhost --port 5432 --dbname tqtest --user max

DB driver: psycopg2 (pip install psycopg2-binary).
If psycopg2 is unavailable, install it or rewrite _connect_psycopg2/_execute
to shell out to psql via subprocess (documented approach in bench/README.md).
"""

import argparse
import csv
import io
import os
import sys
import time
import textwrap

import numpy as np

# ---------------------------------------------------------------------------
# DB connection helpers
# ---------------------------------------------------------------------------

def _connect_psycopg2(args):
    import psycopg2
    conn = psycopg2.connect(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=args.password or None,
    )
    conn.autocommit = True
    return conn


def _execute(conn, sql, params=None, *, fetch=False, many=False):
    """Run SQL, optionally returning rows."""
    cur = conn.cursor()
    try:
        if many and params:
            cur.executemany(sql, params)
        elif params:
            cur.execute(sql, params)
        else:
            cur.execute(sql)
        if fetch:
            return cur.fetchall()
    finally:
        cur.close()


def _fetchall(conn, sql, params=None):
    return _execute(conn, sql, params, fetch=True)


def _fetchone(conn, sql, params=None):
    rows = _fetchall(conn, sql, params)
    return rows[0] if rows else None


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_synthetic(n, dim, nq, seed):
    """Generate synthetic float32 vectors with a fixed RNG seed."""
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    queries = rng.standard_normal((nq, dim)).astype(np.float32)
    return vectors, queries


def load_from_files(vec_path, q_path, gt_path=None):
    """Load vectors from .npy files.  Returns (vectors, queries, gt_or_None)."""
    vectors = np.load(vec_path).astype(np.float32)
    queries = np.load(q_path).astype(np.float32)
    gt = None
    if gt_path and os.path.exists(gt_path):
        gt = np.load(gt_path)   # shape (nq, k), int32 0-based row indices
    return vectors, queries, gt


def load_fvecs(path):
    """Load a .fvecs file (SIFT/GIST format) into an np.float32 array."""
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.int32)
    dim = data[0]
    n = len(data) // (dim + 1)
    vecs = data.reshape(n, dim + 1)[:, 1:].view(np.float32).copy()
    return vecs


def load_ivecs(path):
    """Load a .ivecs file (SIFT/GIST ground-truth format) into an np.int32 array."""
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.int32)
    k = data[0]
    n = len(data) // (k + 1)
    return data.reshape(n, k + 1)[:, 1:].copy()


def load_hdf5(path):
    """Load an ann-benchmarks .hdf5 dataset (e.g. glove-200-angular).
    Returns (base, queries, gt_or_None).  'neighbors' is 0-based row indices
    into 'train' under the dataset's native metric (angular => cosine)."""
    import h5py
    with h5py.File(path, "r") as f:
        base = f["train"][:].astype(np.float32)
        queries = f["test"][:].astype(np.float32)
        gt = f["neighbors"][:].astype(np.int32) if "neighbors" in f else None
    return base, queries, gt


# ---------------------------------------------------------------------------
# Table / index helpers
# ---------------------------------------------------------------------------

SCHEMA = "bench_tqflat"
TABLE = f"{SCHEMA}.items"
INDEX_NAME = "bench_tqflat_items_idx"   # flat name; PG places it in the same schema as the table

# Metric -> (order-by operator, opclass).  The opclass name is shared across
# hnsw, ivfflat, and tqflat (opclasses are per-AM), so one entry serves all.
METRICS = {
    "l2":     ("<->", "vector_l2_ops"),
    "cosine": ("<=>", "vector_cosine_ops"),
    "ip":     ("<#>", "vector_ip_ops"),
}

def setup_schema(conn):
    _execute(conn, f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
    _execute(conn, f"DROP TABLE IF EXISTS {TABLE} CASCADE")


def create_table(conn, dim):
    _execute(conn, f"CREATE TABLE {TABLE} (id int PRIMARY KEY, v vector({dim}))")


def load_vectors(conn, vectors, batch=20000):
    """Bulk-insert vectors as PostgreSQL vector literals via batched COPY.

    The COPY payload is streamed in batches of `batch` rows rather than
    materialized as one StringIO: at high dim and large N the full buffer is
    enormous (e.g. 1M x 1536 is ~18 GB of text), which thrashes swap / fills
    disk.  Batching bounds the in-memory buffer to ~one batch.
    """
    n, dim = vectors.shape
    print(f"  Loading {n:,} vectors (dim={dim}) …", end="", flush=True)
    t0 = time.perf_counter()

    cur = conn.cursor()
    for start in range(0, n, batch):
        buf = io.StringIO()
        for i in range(start, min(start + batch, n)):
            vec = vectors[i]
            buf.write(f"{i}\t[{','.join(str(float(x)) for x in vec)}]\n")
        buf.seek(0)
        cur.copy_expert(f"COPY {TABLE} (id, v) FROM STDIN", buf)
        buf.close()
    cur.close()

    elapsed = time.perf_counter() - t0
    print(f" done ({elapsed:.1f}s)")


# ---------------------------------------------------------------------------
# Ground-truth computation (exact seqscan from DB)
# ---------------------------------------------------------------------------

def compute_exact_ground_truth(conn, queries, op="<->", k=10, verbose=False):
    """Compute exact k-NN for every query via seqscan (no index), under op."""
    _execute(conn, "SET enable_indexscan = off")
    _execute(conn, "SET enable_bitmapscan = off")
    _execute(conn, "SET enable_seqscan = on")

    if verbose:
        plan = _fetchall(conn, f"EXPLAIN SELECT id FROM {TABLE} ORDER BY v {op} %s::vector LIMIT {k}",
                         (f"[{','.join(['0']*queries.shape[1])}]",))
        print("  [exact EXPLAIN]:", plan[0][0])

    gt = []
    n = len(queries)
    print(f"  Computing exact ground truth ({op}) for {n} queries …", end="", flush=True)
    t0 = time.perf_counter()
    for q in queries:
        q_lit = f"[{','.join(str(float(x)) for x in q)}]"
        rows = _fetchall(conn, f"SELECT id FROM {TABLE} ORDER BY v {op} %s::vector LIMIT {k}", (q_lit,))
        gt.append([r[0] for r in rows])
    elapsed = time.perf_counter() - t0
    print(f" done ({elapsed:.1f}s, {n/elapsed:.0f} QPS)")

    _execute(conn, "SET enable_indexscan = on")
    _execute(conn, "SET enable_bitmapscan = on")
    return gt


# ---------------------------------------------------------------------------
# Recall computation
# ---------------------------------------------------------------------------

def recall_at_k(results, ground_truth, k=10):
    """Mean recall@k over all queries."""
    hits = 0
    total = 0
    for res, gt in zip(results, ground_truth):
        res_set = set(res[:k])
        gt_set = set(gt[:k])
        hits += len(res_set & gt_set)
        total += k
    return hits / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Index benchmark helpers
# ---------------------------------------------------------------------------

def drop_index(conn):
    _execute(conn, f"DROP INDEX IF EXISTS {SCHEMA}.{INDEX_NAME}")


def build_index_and_time(conn, ddl):
    """Build an index, return wall-clock seconds."""
    drop_index(conn)
    t0 = time.perf_counter()
    _execute(conn, ddl)
    return time.perf_counter() - t0


def index_size(conn):
    row = _fetchone(conn, f"SELECT pg_relation_size('{SCHEMA}.{INDEX_NAME}'), pg_total_relation_size('{SCHEMA}.{INDEX_NAME}')")
    return row[0], row[1]   # (heap_size, total_size) in bytes


def format_bytes(n):
    if n >= 1 << 20:
        return f"{n / (1<<20):.1f} MB"
    if n >= 1 << 10:
        return f"{n / (1<<10):.1f} KB"
    return f"{n} B"


def run_queries(conn, queries, k, op="<->", verbose=False, explain_first=False):
    """Run the query set against the current index; return (result_ids_list, qps, avg_ms)."""
    if explain_first or verbose:
        q = queries[0]
        q_lit = f"[{','.join(str(float(x)) for x in q)}]"
        plan = _fetchall(conn, f"EXPLAIN SELECT id FROM {TABLE} ORDER BY v {op} %s::vector LIMIT {k}", (q_lit,))
        print("  [EXPLAIN]:", plan[0][0])

    results = []
    t0 = time.perf_counter()
    for q in queries:
        q_lit = f"[{','.join(str(float(x)) for x in q)}]"
        rows = _fetchall(conn, f"SELECT id FROM {TABLE} ORDER BY v {op} %s::vector LIMIT {k}", (q_lit,))
        results.append([r[0] for r in rows])
    elapsed = time.perf_counter() - t0

    nq = len(queries)
    qps = nq / elapsed
    avg_ms = elapsed / nq * 1000
    return results, qps, avg_ms


# ---------------------------------------------------------------------------
# Individual method benchmarks
# ---------------------------------------------------------------------------

def bench_exact(conn, queries, k, gt, op="<->", verbose=False):
    """Exact seqscan — both the ground truth source and a latency baseline."""
    _execute(conn, "SET enable_indexscan = off")
    _execute(conn, "SET enable_bitmapscan = off")
    _execute(conn, "SET enable_seqscan = on")
    results, qps, avg_ms = run_queries(conn, queries, k, op=op, verbose=verbose, explain_first=verbose)
    _execute(conn, "SET enable_indexscan = on")
    _execute(conn, "SET enable_bitmapscan = on")

    rec = recall_at_k(results, gt, k)
    return {
        "method": "exact (seqscan)",
        "build_s": 0.0,
        "idx_size": "—",
        "recall": rec,
        "qps": qps,
        "avg_ms": avg_ms,
        "_build_s_raw": 0.0,
        "_idx_bytes": 0,
    }


def bench_hnsw(conn, queries, k, gt, dim, op="<->", opclass="vector_l2_ops", ef_search=100,
               m=16, ef_construction=64, verbose=False):
    ddl = (f"CREATE INDEX {INDEX_NAME} ON {TABLE} USING hnsw (v {opclass}) "
           f"WITH (m={m}, ef_construction={ef_construction})")
    print(f"  Building hnsw index ({opclass}, m={m}, ef_construction={ef_construction}) …",
          end="", flush=True)
    build_s = build_index_and_time(conn, ddl)
    print(f" {build_s:.1f}s")

    sz, sz_total = index_size(conn)
    _execute(conn, f"SET hnsw.ef_search = {ef_search}")
    _execute(conn, "SET enable_seqscan = off")
    results, qps, avg_ms = run_queries(conn, queries, k, op=op, verbose=verbose, explain_first=verbose)
    _execute(conn, "SET enable_seqscan = on")
    rec = recall_at_k(results, gt, k)

    return {
        "method": f"hnsw (m={m},ef_construction={ef_construction},ef_search={ef_search})",
        "build_s": build_s,
        "idx_size": format_bytes(sz),
        "recall": rec,
        "qps": qps,
        "avg_ms": avg_ms,
        "_build_s_raw": build_s,
        "_idx_bytes": sz,
    }


def bench_ivfflat(conn, queries, k, gt, n, op="<->", opclass="vector_l2_ops", verbose=False):
    lists = max(1, int(n ** 0.5))
    ddl = f"CREATE INDEX {INDEX_NAME} ON {TABLE} USING ivfflat (v {opclass}) WITH (lists={lists})"
    print(f"  Building ivfflat index ({opclass}, lists={lists}) …", end="", flush=True)
    build_s = build_index_and_time(conn, ddl)
    print(f" {build_s:.1f}s")

    sz, sz_total = index_size(conn)
    probes = max(1, lists // 10)
    _execute(conn, f"SET ivfflat.probes = {probes}")
    _execute(conn, "SET enable_seqscan = off")
    results, qps, avg_ms = run_queries(conn, queries, k, op=op, verbose=verbose, explain_first=verbose)
    _execute(conn, "SET enable_seqscan = on")
    rec = recall_at_k(results, gt, k)

    return {
        "method": f"ivfflat (lists={lists},probes={probes})",
        "build_s": build_s,
        "idx_size": format_bytes(sz),
        "recall": rec,
        "qps": qps,
        "avg_ms": avg_ms,
        "_build_s_raw": build_s,
        "_idx_bytes": sz,
    }


def bench_tqflat_build(conn, bits, opclass="vector_l2_ops", fast_rotation=True, tq_prod=True):
    """Build tqflat index for a given bits value; return (build_s, idx_bytes)."""
    fr = "true" if fast_rotation else "false"
    tp = "true" if tq_prod else "false"
    ddl = (f"CREATE INDEX {INDEX_NAME} ON {TABLE} USING tqflat (v {opclass}) "
           f"WITH (bits={bits}, fast_rotation={fr}, tq_prod={tp})")
    print(f"  Building tqflat {opclass} bits={bits} fast_rotation={fr} tq_prod={tp} index …",
          end="", flush=True)
    build_s = build_index_and_time(conn, ddl)
    print(f" {build_s:.2f}s")
    sz, _ = index_size(conn)
    return build_s, sz


def bench_tqflat_query(conn, queries, k, gt, bits, rerank, build_s, idx_bytes, op="<->", verbose=False,
                       force_scalar="off"):
    """Query an already-built tqflat index at a given rerank level."""
    kernel = "[float]" if force_scalar == "on" else "[8bit]"
    label = f"tqflat (bits={bits},rerank={rerank}) {kernel}"
    _execute(conn, f"SET tqflat.rerank = {rerank}")
    _execute(conn, f"SET tqflat.force_scalar = {force_scalar}")
    _execute(conn, "SET enable_seqscan = off")
    results, qps, avg_ms = run_queries(conn, queries, k, op=op, verbose=verbose, explain_first=verbose)
    _execute(conn, "SET enable_seqscan = on")
    rec = recall_at_k(results, gt, k)

    return {
        "method": label,
        "build_s": build_s,
        "idx_size": format_bytes(idx_bytes),
        "recall": rec,
        "qps": qps,
        "avg_ms": avg_ms,
        "_build_s_raw": build_s,
        "_idx_bytes": idx_bytes,
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

MD_SEP = "|"

def print_markdown_table(rows):
    headers = ["Metric", "Method", "Build (s)", "Index size", "Recall@10", "QPS", "Avg latency (ms)"]

    def fmt(row):
        return [
            row.get("metric", ""),
            row["method"],
            f"{row['build_s']:.2f}",
            row["idx_size"],
            f"{row['recall']:.4f}",
            f"{row['qps']:.1f}",
            f"{row['avg_ms']:.2f}",
        ]

    formatted = [fmt(r) for r in rows]
    col_widths = [max(len(h), max((len(f[i]) for f in formatted), default=0))
                  for i, h in enumerate(headers)]

    def row_str(cells):
        return MD_SEP + MD_SEP.join(f" {c:<{w}} " for c, w in zip(cells, col_widths)) + MD_SEP

    sep_row = MD_SEP + MD_SEP.join("-" * (w + 2) for w in col_widths) + MD_SEP

    print(row_str(headers))
    print(sep_row)
    for f in formatted:
        print(row_str(f))


def write_csv(rows, path):
    fieldnames = ["metric", "method", "build_s", "idx_size", "recall", "qps", "avg_ms"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"\nResults written to: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="tqflat vs hnsw/ivfflat/exact benchmark harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          # synthetic default (fast):
          python3 bench/tqflat_bench.py

          # larger synthetic:
          python3 bench/tqflat_bench.py --n 50000 --dim 128 --queries 1000

          # real dataset (numpy .npy):
          python3 bench/tqflat_bench.py --dataset-vectors vecs.npy --dataset-queries queries.npy

          # real dataset with pre-computed ground truth:
          python3 bench/tqflat_bench.py --dataset-vectors vecs.npy \\
              --dataset-queries queries.npy --dataset-gt gt.npy

          # SIFT/GIST .fvecs/.ivecs format:
          python3 bench/tqflat_bench.py --fvecs-vectors sift_base.fvecs \\
              --fvecs-queries sift_query.fvecs --ivecs-gt sift_groundtruth.ivecs
        """),
    )
    # Synthetic
    p.add_argument("--n", type=int, default=5000, help="Number of vectors (synthetic)")
    p.add_argument("--dim", type=int, default=64, help="Vector dimension (synthetic)")
    p.add_argument("--queries", type=int, default=200, help="Number of query vectors (synthetic)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed (synthetic)")

    # Real dataset hooks (numpy)
    p.add_argument("--dataset-vectors", metavar="PATH", help=".npy file: float32 array (n, dim)")
    p.add_argument("--dataset-queries", metavar="PATH", help=".npy file: float32 array (nq, dim)")
    p.add_argument("--dataset-gt", metavar="PATH",
                   help=".npy file: int32 array (nq, k) 0-based row indices (optional)")

    # Real dataset hooks (fvecs/ivecs)
    p.add_argument("--fvecs-vectors", metavar="PATH", help=".fvecs base vectors file")
    p.add_argument("--fvecs-queries", metavar="PATH", help=".fvecs query vectors file")
    p.add_argument("--ivecs-gt", metavar="PATH", help=".ivecs ground-truth file (optional)")

    # Real dataset hook (ann-benchmarks hdf5, e.g. glove-200-angular)
    p.add_argument("--hdf5", metavar="PATH",
                   help="ann-benchmarks .hdf5 dataset (train/test/neighbors); GT is cosine")

    # Large-dataset controls
    p.add_argument("--max-queries", type=int, default=None,
                   help="Cap the query set to the first N (and align file GT); for big datasets")
    p.add_argument("--no-exact", action="store_true",
                   help="Skip the exact-seqscan baseline (prohibitively slow at large N; "
                        "use when a precomputed ground truth is supplied)")

    # DB
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=5432)
    p.add_argument("--dbname", default="tqtest")
    p.add_argument("--user", default=os.environ.get("USER", "postgres"))
    p.add_argument("--password", default=None)

    # Bench options
    p.add_argument("--k", type=int, default=10, help="k for recall@k and kNN queries")
    p.add_argument("--tqflat-bits", type=int, nargs="+", default=[2, 4],
                   help="bits values to test for tqflat")
    p.add_argument("--tqflat-reranks", type=int, nargs="+", default=[200, 0],
                   help="rerank candidate counts to test for tqflat")
    p.add_argument("--metric", nargs="+", default=["l2"], choices=list(METRICS.keys()),
                   help="Distance metric(s) to benchmark: l2, cosine, ip (space-separated for several)")
    p.add_argument("--maintenance-work-mem", default="2GB",
                   help="maintenance_work_mem for index builds (default 2GB). The default "
                        "server value (often 64-128MB) forces pgvector's slow on-disk HNSW "
                        "build path at large N, making build-time comparisons unrepresentative.")
    p.add_argument("--max-parallel-maintenance-workers", type=int, default=4,
                   help="max_parallel_maintenance_workers for index builds (default 4)")
    p.add_argument("--hnsw-ef-search", type=int, default=100,
                   help="hnsw.ef_search at query time (default 100; pgvector's default is 40, "
                        "which under-recalls on hard datasets like GloVe-angular)")
    p.add_argument("--hnsw-m", type=int, default=16,
                   help="hnsw m (max connections per layer; build reloption, default 16)")
    p.add_argument("--hnsw-ef-construction", type=int, default=64,
                   help="hnsw ef_construction (build candidate list size, default 64; "
                        "raise with m for higher recall at high dimension)")
    p.add_argument("--tqflat-fast-rotation", choices=["on", "off"], default="on",
                   help="tqflat fast_rotation reloption (default on; off = dense O(d^2) baseline for A/B)")
    p.add_argument("--tqflat-tq-prod", choices=["on", "off"], default="on",
                   help="tqflat tq_prod (QJL residual stage) reloption (default on)")
    p.add_argument("--tqflat-force-scalar", choices=["on", "off"], default="off",
                   help="A/B: score blocked tqflat with the float LUT (on) vs the 8-bit SIMD kernel (off, default)")
    p.add_argument("--no-hnsw", action="store_true", help="Skip hnsw benchmark")
    p.add_argument("--no-ivfflat", action="store_true", help="Skip ivfflat benchmark")
    p.add_argument("--csv", default="bench/results.csv", help="Output CSV path")
    p.add_argument("--verbose", action="store_true", help="Print EXPLAIN plans")

    return p.parse_args()


def main():
    args = parse_args()

    # ---- Load dataset ----
    file_gt = None
    if args.fvecs_vectors:
        print("Loading .fvecs dataset …")
        vectors = load_fvecs(args.fvecs_vectors)
        queries = load_fvecs(args.fvecs_queries) if args.fvecs_queries else vectors[:args.queries]
        file_gt = load_ivecs(args.ivecs_gt) if args.ivecs_gt else None
    elif args.hdf5:
        print("Loading .hdf5 dataset …")
        vectors, queries, file_gt = load_hdf5(args.hdf5)
    elif args.dataset_vectors:
        print("Loading .npy dataset …")
        vectors, queries, file_gt = load_from_files(
            args.dataset_vectors, args.dataset_queries, args.dataset_gt)
    else:
        print(f"Generating synthetic data: N={args.n:,}, dim={args.dim}, queries={args.queries}, seed={args.seed}")
        vectors, queries = load_synthetic(args.n, args.dim, args.queries, args.seed)

    # Cap queries (and align file GT) for large datasets with many queries.
    if args.max_queries is not None and len(queries) > args.max_queries:
        queries = queries[:args.max_queries]
        if file_gt is not None:
            file_gt = file_gt[:args.max_queries]
        print(f"Capped query set to first {args.max_queries}.")

    n, dim = vectors.shape
    nq = len(queries)
    k = args.k
    print(f"Dataset: {n:,} vectors, dim={dim}, {nq} queries, k={k}")

    # ---- Connect ----
    print(f"\nConnecting to {args.user}@{args.host}:{args.port}/{args.dbname} …")
    try:
        conn = _connect_psycopg2(args)
    except Exception as e:
        print(f"ERROR: could not connect: {e}", file=sys.stderr)
        sys.exit(1)
    print("Connected.")

    # Resource index builds adequately.  The server default maintenance_work_mem
    # (often 64-128MB) is far below the ~1GB a 1M-element HNSW graph needs, which
    # forces pgvector onto its slow on-disk build path and makes build-time
    # comparisons meaningless.  Set it (and parallel workers) per session.
    _execute(conn, f"SET maintenance_work_mem = '{args.maintenance_work_mem}'")
    _execute(conn, f"SET max_parallel_maintenance_workers = {args.max_parallel_maintenance_workers}")
    mwm = _fetchone(conn, "SHOW maintenance_work_mem")[0]
    print(f"  maintenance_work_mem={mwm}, "
          f"max_parallel_maintenance_workers={args.max_parallel_maintenance_workers}")

    # ---- Set up schema and load data ----
    print("\n--- Setup ---")
    setup_schema(conn)
    create_table(conn, dim)
    load_vectors(conn, vectors)

    # A pre-computed ground truth from file is metric-specific (fvecs/ivecs GT
    # is L2; ann-benchmarks angular GT is cosine).  Apply it only when a single
    # matching metric is requested; otherwise compute GT per metric via seqscan.
    results_rows = []
    multi = len(args.metric) > 1

    for metric in args.metric:
        op, opclass = METRICS[metric]
        print(f"\n========== metric: {metric}  (op {op}, opclass {opclass}) ==========")

        # ---- Ground truth ----
        print(f"\n--- Ground truth ({metric}) ---")
        if file_gt is not None and not multi:
            print("  Using pre-computed ground truth (from file).")
            gt = [list(file_gt[i]) for i in range(nq)]
        else:
            if file_gt is not None and multi:
                print("  (ignoring file ground truth: multiple metrics requested; computing exact per metric)")
            gt = compute_exact_ground_truth(conn, queries, op=op, k=k, verbose=args.verbose)

        # ---- Exact baseline (skippable; very slow at large N) ----
        if not args.no_exact:
            print(f"\n--- Exact seqscan ({metric}) ---")
            row = bench_exact(conn, queries, k, gt, op=op, verbose=args.verbose)
            row["metric"] = metric
            results_rows.append(row)

        # ---- tqflat: build each (bits) index once, query at each rerank ----
        for bits in args.tqflat_bits:
            print(f"\n--- tqflat {metric} bits={bits} ---")
            fast = args.tqflat_fast_rotation == "on"
            tqprod = args.tqflat_tq_prod == "on"
            build_s, idx_bytes = bench_tqflat_build(conn, bits, opclass=opclass,
                                                    fast_rotation=fast, tq_prod=tqprod)
            tag = f" [{'fast' if fast else 'dense'}{'' if tqprod else ',noqjl'}]"
            for rerank in args.tqflat_reranks:
                print(f"  Querying rerank={rerank} …", end="", flush=True)
                row = bench_tqflat_query(conn, queries, k, gt, bits, rerank,
                                         build_s, idx_bytes, op=op, verbose=args.verbose,
                                         force_scalar=args.tqflat_force_scalar)
                row["method"] += tag
                row["metric"] = metric
                print(f" {row['qps']:.0f} QPS, recall={row['recall']:.4f}")
                results_rows.append(row)

        # ---- hnsw ----
        if not args.no_hnsw:
            print(f"\n--- HNSW ({metric}) ---")
            row = bench_hnsw(conn, queries, k, gt, dim, op=op, opclass=opclass,
                             ef_search=args.hnsw_ef_search, m=args.hnsw_m,
                             ef_construction=args.hnsw_ef_construction, verbose=args.verbose)
            row["metric"] = metric
            results_rows.append(row)

        # ---- ivfflat ----
        if not args.no_ivfflat:
            print(f"\n--- IVFFlat ({metric}) ---")
            row = bench_ivfflat(conn, queries, k, gt, n, op=op, opclass=opclass, verbose=args.verbose)
            row["metric"] = metric
            results_rows.append(row)

        drop_index(conn)

    # ---- Results ----
    print("\n=== Results ===\n")
    print_markdown_table(results_rows)
    write_csv(results_rows, args.csv)

    # ---- Cleanup ----
    print(f"\nDropping scratch schema {SCHEMA} …")
    _execute(conn, f"DROP SCHEMA {SCHEMA} CASCADE")
    conn.close()
    print("Done. Connection closed.")


if __name__ == "__main__":
    main()
