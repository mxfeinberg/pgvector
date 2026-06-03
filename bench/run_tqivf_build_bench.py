#!/usr/bin/env python3
"""tqivf parallel-build timing benchmark (the-fire / x86).

Loads each dataset ONCE, then times CREATE INDEX for tqivf and ivfflat under a
serial budget (max_parallel_maintenance_workers = 0) and a parallel budget
(request = the box's core budget), on the identical loaded table.  Captures the
"using N parallel workers" DEBUG1 line to confirm engagement and records the
on-disk index size.  Goal: show tqivf's serial->parallel build speedup and
whether the build-time gap to ivfflat narrows.

Reuses tqflat_bench.py's dataset loaders + table/connection helpers so there is
no reimplementation of fvecs/hdf5/npy parsing.  Build-time only: no queries, no
ground truth.

Usage (on the-fire):
  python3 bench/run_tqivf_build_bench.py \
      --parallel-workers 8 --maintenance-work-mem 16GB \
      --csv bench/results/tqivf_build_time_x86.csv
"""
import argparse
import collections
import math
import os
import re
import sys
import time

# Import the existing harness (same dir) for loaders + helpers.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tqflat_bench as H  # noqa: E402

WORKER_RE = re.compile(r"using (\d+) parallel workers")

# Datasets: (key, loader-thunk -> np.float32 (n,dim), metric).  Only base
# vectors are needed for build timing.
def _datasets(home):
    td = os.path.join(home, "tqdata")
    return [
        ("sift1m", "l2",
         lambda: H.load_fvecs(os.path.join(td, "sift/sift/sift_base.fvecs"))),
        ("glove200", "cosine",
         lambda: H.load_hdf5(os.path.join(td, "glove/glove-200-angular.hdf5"))[0]),
        ("openai1m", "cosine",
         lambda: H.load_from_files(os.path.join(td, "openai1m/base.npy"),
                                   os.path.join(td, "openai1m/base.npy"))[0]),
    ]


def time_build(conn, ddl, workers, mwm):
    """Time one CREATE INDEX under the given worker budget.  Returns
    (seconds, launched_workers_or_None, heap_bytes)."""
    H.drop_index(conn)
    H._execute(conn, f"SET maintenance_work_mem = '{mwm}'")
    H._execute(conn, f"SET max_parallel_maintenance_workers = {workers}")
    # min_parallel_table_scan_size default (8MB) is far below 1M rows, so the
    # planner's size gate is satisfied; no need to lower it.
    H._execute(conn, "SET client_min_messages = DEBUG1")
    # Fresh, generously-sized notice buffer so the (early) worker-count line is
    # not evicted by later DEBUG noise.
    conn.notices = collections.deque(maxlen=1000)

    t0 = time.perf_counter()
    H._execute(conn, ddl)
    elapsed = time.perf_counter() - t0

    H._execute(conn, "SET client_min_messages = NOTICE")
    launched = None
    for n in conn.notices:
        m = WORKER_RE.search(n)
        if m:
            launched = int(m.group(1))
    heap_bytes, _total = H.index_size(conn)
    return elapsed, launched, heap_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="/var/run/postgresql")
    ap.add_argument("--port", type=int, default=5432)
    ap.add_argument("--dbname", default="tqtest")
    ap.add_argument("--user", default=os.environ.get("USER", "max"))
    ap.add_argument("--password", default=None)
    ap.add_argument("--parallel-workers", type=int, default=8,
                    help="worker budget for the parallel build (request; actual "
                         "is capped by max_parallel_workers/max_worker_processes)")
    ap.add_argument("--maintenance-work-mem", default="16GB")
    ap.add_argument("--datasets", default="sift1m,glove200,openai1m",
                    help="comma list subset of sift1m,glove200,openai1m")
    ap.add_argument("--csv", default="bench/results/tqivf_build_time_x86.csv")
    args = ap.parse_args()

    home = os.path.expanduser("~")
    want = set(args.datasets.split(","))
    conn = H._connect_psycopg2(args)

    rows = []
    for key, metric, load in _datasets(home):
        if key not in want:
            continue
        op, opclass = H.METRICS[metric]
        print(f"\n================ {key} ({metric}) ================", flush=True)
        vectors = load()
        n, dim = vectors.shape
        lists = max(1, int(math.isqrt(n)))
        print(f"  N={n:,} dim={dim} lists={lists}", flush=True)

        H.setup_schema(conn)
        H.create_table(conn, dim)
        H.load_vectors(conn, vectors)
        H._execute(conn, f"ANALYZE {H.TABLE}")

        ddls = {
            "tqivf": (f"CREATE INDEX {H.INDEX_NAME} ON {H.TABLE} "
                      f"USING tqivf (v {opclass}) "
                      f"WITH (lists={lists}, fast_rotation=on)"),
            "ivfflat": (f"CREATE INDEX {H.INDEX_NAME} ON {H.TABLE} "
                        f"USING ivfflat (v {opclass}) WITH (lists={lists})"),
        }

        for method, ddl in ddls.items():
            for label, workers in (("serial", 0), ("parallel", args.parallel_workers)):
                sec, launched, hb = time_build(conn, ddl, workers, args.maintenance_work_mem)
                wtxt = "serial" if workers == 0 else f"req={workers} launched={launched}"
                print(f"  {method:8s} {label:8s}: {sec:8.2f}s  "
                      f"size={H.format_bytes(hb)}  [{wtxt}]", flush=True)
                rows.append(dict(dataset=key, metric=metric, n=n, dim=dim, lists=lists,
                                 method=method, mode=label, req_workers=workers,
                                 launched_workers=(launched if launched is not None else 0),
                                 build_s=round(sec, 3), index_bytes=hb))

    conn.close()

    # ---- CSV ----
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    import csv
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.csv}")

    # ---- Summary table ----
    print("\n==================== SUMMARY ====================")
    print(f"{'dataset':10s} {'method':8s} {'serial_s':>9s} {'par_s':>8s} "
          f"{'speedup':>8s} {'workers':>8s}")
    byk = {}
    for r in rows:
        byk.setdefault((r["dataset"], r["method"]), {})[r["mode"]] = r
    for (ds, method), d in byk.items():
        s = d.get("serial", {}).get("build_s")
        p = d.get("parallel", {}).get("build_s")
        lw = d.get("parallel", {}).get("launched_workers", 0)
        sp = (s / p) if (s and p) else float("nan")
        print(f"{ds:10s} {method:8s} {s:9.2f} {p:8.2f} {sp:7.2f}x {lw:8d}")
    # tqivf/ivfflat parallel ratio
    print("\n-- parallel tqivf vs parallel ivfflat (build-time ratio) --")
    for ds in sorted({r['dataset'] for r in rows}):
        tq = byk.get((ds, "tqivf"), {}).get("parallel", {}).get("build_s")
        iv = byk.get((ds, "ivfflat"), {}).get("parallel", {}).get("build_s")
        if tq and iv:
            print(f"  {ds:10s} tqivf={tq:.2f}s  ivfflat={iv:.2f}s  "
                  f"tqivf/ivfflat={tq/iv:.2f}x")


if __name__ == "__main__":
    main()
