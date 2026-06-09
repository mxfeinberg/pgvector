#!/usr/bin/env python3
"""
ip_accuracy.py — IP-estimation accuracy benchmark for tqflat configurations.

Measures inner-product estimation accuracy (bias, RMSE, relative RMSE, Pearson r)
for four tqflat configurations over a sample of query x base vector pairs:

  1. turboquant_mse  dense  (tq_prod=false, fast=false)
  2. turboquant_prod dense  (tq_prod=true,  fast=false)
  3. turboquant_mse  fast   (tq_prod=false, fast=true)
  4. turboquant_prod fast   (tq_prod=true,  fast=true)

The benchmark calls tqflat_test_ip_accuracy() in PostgreSQL, which scores all
n_queries * n_base pairs in a single function call and returns:
  {bias, rmse, mean_abs_true, pearson, n_pairs}

Usage:
    # Synthetic data (no files needed):
    python3 bench/ip_accuracy.py

    # Real dataset (.npy files):
    python3 bench/ip_accuracy.py --dataset-vectors base.npy --dataset-queries queries.npy

    # SIFT/GIST .fvecs format:
    python3 bench/ip_accuracy.py --fvecs-vectors sift_base.fvecs --fvecs-queries sift_query.fvecs

    # ann-benchmarks .hdf5 format:
    python3 bench/ip_accuracy.py --hdf5 glove-200-angular.hdf5

DB connection:
    python3 bench/ip_accuracy.py --dbname tqtest --user $USER --host localhost
"""

import argparse
import os
import sys
import textwrap

import numpy as np


# ---------------------------------------------------------------------------
# DB connection helpers (mirrors tqflat_bench.py)
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


def _fetchone(conn, sql, params=None):
    cur = conn.cursor()
    try:
        if params:
            cur.execute(sql, params)
        else:
            cur.execute(sql)
        return cur.fetchone()
    finally:
        cur.close()


# ---------------------------------------------------------------------------
# Dataset loaders (mirrors tqflat_bench.py)
# ---------------------------------------------------------------------------

def load_synthetic(n, dim, nq, seed=42):
    """Generate synthetic float32 vectors."""
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    queries = rng.standard_normal((nq, dim)).astype(np.float32)
    return vectors, queries


def load_from_npy(vec_path, q_path):
    """Load base and query vectors from .npy files."""
    vectors = np.load(vec_path).astype(np.float32)
    queries = np.load(q_path).astype(np.float32)
    return vectors, queries


def load_fvecs(path):
    """Load a .fvecs file (SIFT/GIST format) into float32 array."""
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.int32)
    dim = data[0]
    n = len(data) // (dim + 1)
    return data.reshape(n, dim + 1)[:, 1:].view(np.float32).copy()


def load_hdf5(path):
    """Load an ann-benchmarks .hdf5 dataset. Returns (base, queries)."""
    import h5py
    with h5py.File(path, "r") as f:
        base = f["train"][:].astype(np.float32)
        queries = f["test"][:].astype(np.float32)
    return base, queries


# ---------------------------------------------------------------------------
# Vector array -> PostgreSQL literal
# ---------------------------------------------------------------------------

def vectors_to_pg_array(vecs):
    """
    Convert a 2D float32 numpy array to a PostgreSQL vector[] array literal.
    Returns a string like: ARRAY['[1.0,2.0]'::vector,'[3.0,4.0]'::vector]
    We build a text string and cast it as vector[] via a separate SQL fragment.
    """
    # Build the array as a PostgreSQL array constructor.
    # We pass it as a single SQL text parameter to avoid psycopg2 quoting
    # issues with the vector literal format.
    parts = []
    for row in vecs:
        inner = ",".join(repr(float(x)) for x in row)
        parts.append(f"'[{inner}]'::vector")
    return "ARRAY[" + ",".join(parts) + "]"


# ---------------------------------------------------------------------------
# Call tqflat_test_ip_accuracy
# ---------------------------------------------------------------------------

def ensure_accuracy_function(conn):
    """
    tqflat_test_ip_accuracy is a test-only C wrapper that is no longer shipped
    in the extension SQL; define it from vector.so (requires superuser).
    """
    cur = conn.cursor()
    try:
        cur.execute("""
            CREATE OR REPLACE FUNCTION tqflat_test_ip_accuracy(
                vector[], vector[], int, bool, bool) RETURNS float8[]
            AS 'vector' LANGUAGE C IMMUTABLE STRICT
        """)
    finally:
        cur.close()


def run_accuracy(conn, queries, base, bits, tq_prod, fast):
    """
    Call tqflat_test_ip_accuracy for one (tq_prod, fast) configuration.
    Returns dict with bias, rmse, mean_abs_true, pearson, n_pairs.
    """
    qarr_sql = vectors_to_pg_array(queries)
    barr_sql = vectors_to_pg_array(base)
    tq_prod_sql = "true" if tq_prod else "false"
    fast_sql = "true" if fast else "false"
    sql = (
        f"SELECT tqflat_test_ip_accuracy("
        f"  {qarr_sql},"
        f"  {barr_sql},"
        f"  {bits},"
        f"  {tq_prod_sql},"
        f"  {fast_sql}"
        f")"
    )
    row = _fetchone(conn, sql)
    arr = row[0]  # psycopg2 returns a Python list for float8[]
    return {
        "bias": arr[0],
        "rmse": arr[1],
        "mean_abs_true": arr[2],
        "pearson": arr[3],
        "n_pairs": int(arr[4]),
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_table(rows):
    """Print the results as a fixed-width table."""
    header = f"{'config':<16} | {'bias':>10} | {'rmse':>10} | {'rel_rmse':>10} | {'pearson':>10}"
    sep = "-" * len(header)
    print(header)
    print(sep)
    for r in rows:
        rel = r["rmse"] / r["mean_abs_true"] if r["mean_abs_true"] > 1e-12 else float("nan")
        print(
            f"{r['config']:<16} | {r['bias']:>10.6f} | {r['rmse']:>10.6f}"
            f" | {rel:>10.6f} | {r['pearson']:>10.6f}"
        )
    print(sep)
    print(f"(n_pairs={rows[0]['n_pairs']} per config)")


def print_interpretation(rows):
    """Print a one-line qualitative note."""
    mse_dense = next(r for r in rows if r["config"] == "mse  dense")
    prod_dense = next(r for r in rows if r["config"] == "prod dense")
    if abs(prod_dense["bias"]) < abs(mse_dense["bias"]):
        note = ("dense tq_prod |bias| < dense tq_mse |bias|: "
                "QJL stage debiases inner-product estimation as expected.")
    else:
        note = ("dense tq_prod |bias| >= dense tq_mse |bias| on this sample: "
                "debiasing effect may not be visible at this sample size or data distribution.")
    print(f"\nNote: {note}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="IP-estimation accuracy benchmark for tqflat configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          # Synthetic data (no files needed):
          python3 bench/ip_accuracy.py

          # Real dataset (.npy):
          python3 bench/ip_accuracy.py --dataset-vectors base.npy --dataset-queries queries.npy

          # SIFT/GIST .fvecs:
          python3 bench/ip_accuracy.py --fvecs-vectors sift_base.fvecs --fvecs-queries sift_query.fvecs

          # ann-benchmarks .hdf5:
          python3 bench/ip_accuracy.py --hdf5 glove-200-angular.hdf5
        """),
    )

    # Data sources
    p.add_argument("--dataset-vectors", metavar="PATH",
                   help=".npy file: float32 base vectors, shape (N, dim)")
    p.add_argument("--dataset-queries", metavar="PATH",
                   help=".npy file: float32 query vectors, shape (Nq, dim)")
    p.add_argument("--fvecs-vectors", metavar="PATH",
                   help=".fvecs base vectors file")
    p.add_argument("--fvecs-queries", metavar="PATH",
                   help=".fvecs query vectors file")
    p.add_argument("--hdf5", metavar="PATH",
                   help="ann-benchmarks .hdf5 dataset (train/test keys)")

    # Sampling
    p.add_argument("--n-base", type=int, default=500,
                   help="Number of base vectors to sample (default 500)")
    p.add_argument("--n-queries", type=int, default=100,
                   help="Number of query vectors to sample (default 100)")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for synthetic data and sampling (default 42)")

    # Synthetic fallback
    p.add_argument("--dim", type=int, default=64,
                   help="Dimension for synthetic data (default 64)")
    p.add_argument("--n-synthetic", type=int, default=1000,
                   help="Total synthetic base vectors to generate (default 1000)")

    # tqflat config
    p.add_argument("--bits", type=int, default=4,
                   help="Bits per coordinate (2-4, default 4)")

    # DB connection
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=5432)
    p.add_argument("--dbname", default="tqtest")
    p.add_argument("--user", default=os.environ.get("USER", "postgres"))
    p.add_argument("--password", default=None)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- Load dataset ----
    if args.fvecs_vectors:
        print("Loading .fvecs dataset ...")
        base_all = load_fvecs(args.fvecs_vectors)
        queries_all = (load_fvecs(args.fvecs_queries)
                       if args.fvecs_queries else base_all[:args.n_queries])
    elif args.hdf5:
        print("Loading .hdf5 dataset ...")
        base_all, queries_all = load_hdf5(args.hdf5)
    elif args.dataset_vectors:
        print("Loading .npy dataset ...")
        base_all, queries_all = load_from_npy(args.dataset_vectors, args.dataset_queries)
    else:
        print(f"Generating synthetic data: N={args.n_synthetic}, dim={args.dim}, seed={args.seed}")
        base_all, queries_all = load_synthetic(args.n_synthetic, args.dim, args.n_queries, args.seed)

    # ---- Sample ----
    rng = np.random.default_rng(args.seed)
    nb = min(args.n_base, len(base_all))
    nq = min(args.n_queries, len(queries_all))
    base_idx = rng.choice(len(base_all), nb, replace=False)
    query_idx = rng.choice(len(queries_all), nq, replace=False)
    base = base_all[base_idx]
    queries = queries_all[query_idx]

    n_pairs = nb * nq
    dim = base.shape[1]
    print(f"Sampled {nq} queries x {nb} base vectors, dim={dim}, {n_pairs:,} pairs per config, bits={args.bits}")

    if n_pairs > 200_000:
        print(f"Warning: {n_pairs:,} pairs may be slow to score in a single SQL call. "
              f"Consider reducing --n-base or --n-queries.")

    # ---- Connect ----
    print(f"\nConnecting to {args.user}@{args.host}:{args.port}/{args.dbname} ...")
    try:
        import psycopg2
        conn = _connect_psycopg2(args)
    except ImportError:
        print("ERROR: psycopg2 not installed. Run: pip install psycopg2-binary", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: could not connect: {e}", file=sys.stderr)
        sys.exit(1)
    print("Connected.")

    ensure_accuracy_function(conn)

    # ---- Run the 4 configurations ----
    configs = [
        ("mse  dense", False, False),
        ("prod dense", True,  False),
        ("mse  fast ", False, True),
        ("prod fast ", True,  True),
    ]

    results = []
    for label, tq_prod, fast in configs:
        print(f"  Scoring {label} ...", end="", flush=True)
        try:
            metrics = run_accuracy(conn, queries, base, args.bits, tq_prod, fast)
            metrics["config"] = label
            results.append(metrics)
            print(f" bias={metrics['bias']:+.5f}  rmse={metrics['rmse']:.5f}  pearson={metrics['pearson']:.4f}")
        except Exception as e:
            print(f" ERROR: {e}", file=sys.stderr)
            conn.close()
            sys.exit(1)

    conn.close()

    # ---- Print table ----
    print("\n=== IP Estimation Accuracy ===\n")
    print_table(results)
    print_interpretation(results)


if __name__ == "__main__":
    main()
