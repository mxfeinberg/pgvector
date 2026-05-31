# tqflat benchmark harness

`bench/tqflat_bench.py` compares the `tqflat` index access method against `hnsw`,
`ivfflat`, and exact seqscan across four metrics: **recall@k**, **QPS / latency**,
**build time**, and **index size**.

---

## Requirements

- Python 3.8+ with **psycopg2** (`pip install psycopg2-binary`)
- PostgreSQL 14 running locally with the `vector` 0.9.0 extension installed
  (including the `tqflat` AM)
- The extension must be installed in the target database:
  ```sql
  CREATE EXTENSION IF NOT EXISTS vector;
  ```

**DB driver used:** `psycopg2` — chosen because it was pip-installable
(`pip install psycopg2-binary`). If psycopg2 is unavailable the script will fail
at import time with a clear message; see the `--use-psql` note in the source
docstring for the alternative subprocess approach.

---

## Quick start (synthetic data — runs immediately, no download needed)

From the repo root:

```sh
python3 bench/tqflat_bench.py
```

Default: N=5 000 vectors, dim=64, 200 queries, k=10, seed=42.

Customize size:

```sh
python3 bench/tqflat_bench.py --n 50000 --dim 128 --queries 1000
```

Full option list:

```
--n           Number of synthetic vectors (default: 5000)
--dim         Vector dimension (default: 64)
--queries     Number of query vectors (default: 200)
--seed        RNG seed (default: 42; fixed for reproducibility)
--k           k for recall@k and ORDER BY ... LIMIT k (default: 10)
--tqflat-bits   Space-separated list of bits to test (default: 2 4)
--tqflat-reranks  Rerank candidate counts to test (default: 200 0)
--metric      Distance metric(s): l2, cosine, ip (space-separated for several; default: l2)
--maintenance-work-mem MEM   maintenance_work_mem for index builds (default 2GB; the server
              default forces pgvector's slow on-disk HNSW build path at large N)
--hnsw-ef-search N   hnsw.ef_search at query time (default 100; pgvector's default 40
              under-recalls on hard datasets like GloVe-angular)
--tqflat-fast-rotation on|off   tqflat fast_rotation reloption (default on; off = dense
              O(d^2) rotation baseline). A/B = run twice and diff the CSVs.
--tqflat-tq-prod on|off   tqflat tq_prod / QJL residual stage (default on)
--max-queries N   Cap the query set to the first N (also aligns file ground truth); for big datasets
--no-exact    Skip the exact-seqscan baseline (prohibitively slow at large N; use with a precomputed GT)
--no-hnsw     Skip HNSW benchmark
--no-ivfflat  Skip IVFFlat benchmark
--csv PATH    Output CSV path (default: bench/results.csv)
--verbose     Print EXPLAIN plans for each index type
```

### Metric selection

Each metric uses its own operator and opclass across all index types
(`l2` → `<->`/`vector_l2_ops`, `cosine` → `<=>`/`vector_cosine_ops`,
`ip` → `<#>`/`vector_ip_ops`). Pass several to benchmark them in one run; ground
truth is computed (or, for file datasets, applied) per metric:

```sh
python3 bench/tqflat_bench.py --metric l2 cosine     # both in one run
```

For angular/cosine datasets (GloVe, OpenAI) use `--metric cosine`; the shipped
ground truth is cosine-based.

---

## DB connection options

```
--host    (default: localhost)
--port    (default: 5432)
--dbname  (default: tqtest)
--user    (default: $USER)
--password (default: none)
```

Example connecting to a different DB:

```sh
python3 bench/tqflat_bench.py --dbname mydb --user pguser --port 5433
```

The script creates a scratch schema `bench_tqflat`, loads data there, runs all
benchmarks, then **drops the schema** on exit — no manual cleanup needed.

---

## Pointing at a real dataset

### NumPy .npy files

```sh
python3 bench/tqflat_bench.py \
    --dataset-vectors /path/to/base.npy \
    --dataset-queries /path/to/queries.npy \
    --dataset-gt     /path/to/gt.npy      # optional: int32 (nq, k) 0-based indices
```

If `--dataset-gt` is omitted, ground truth is computed via exact seqscan.

### SIFT/GIST .fvecs / .ivecs format (L2)

SIFT1M ships a precomputed L2 ground truth (10k queries). Use it and skip the
exact baseline, capping the query set for a faster run:

```sh
# from corpus-texmex.irisa.fr: sift.tar.gz
python3 bench/tqflat_bench.py \
    --fvecs-vectors sift/sift_base.fvecs \
    --fvecs-queries sift/sift_query.fvecs \
    --ivecs-gt      sift/sift_groundtruth.ivecs \
    --metric l2 --no-exact --max-queries 500
```

### ann-benchmarks .hdf5 format (cosine/angular)

Loaded natively (`train`/`test`/`neighbors`); the shipped `neighbors` ground
truth is angular, so run with `--metric cosine`:

```sh
# from ann-benchmarks.com: glove-200-angular.hdf5
python3 bench/tqflat_bench.py \
    --hdf5 glove-200-angular.hdf5 \
    --metric cosine --no-exact --max-queries 500
```

### Standard datasets

| Dataset       | Dim  | N         | Metric | Source / flags                                 |
|---------------|------|-----------|--------|------------------------------------------------|
| SIFT1M        | 128  | 1 000 000 | L2     | corpus-texmex.irisa.fr (`sift.tar.gz`), `--fvecs-*`/`--ivecs-gt` |
| GloVe-200     | 200  | 1 183 514 | cosine | ann-benchmarks.com `glove-200-angular.hdf5`, `--hdf5` |
| GIST1M        | 960  | 1 000 000 | L2     | corpus-texmex.irisa.fr (`gist.tar.gz`), `--fvecs-*`/`--ivecs-gt` |
| OpenAI-1536   | 1536 | varies    | cosine | ann-benchmarks `*-angular.hdf5`, `--hdf5`      |

---

## What the metrics mean

| Metric | Definition |
|--------|-----------|
| **Recall@10** | Average fraction of the exact top-10 neighbors that appear in the index's top-10 results, over all queries. 1.0 = perfect recall. |
| **Build (s)** | Wall-clock seconds to run `CREATE INDEX`. For tqflat, the same index is reused across different `rerank` values (rerank is a query-time GUC, not a storage parameter). |
| **Index size** | `pg_relation_size(index_oid)` — heap bytes only (excludes TOAST/VM/FSM). hnsw and ivfflat store full-precision vectors; tqflat stores compressed codes, so its index is substantially smaller. |
| **QPS** | Queries per second measured end-to-end from Python (includes network round-trip for each query; single-threaded). |
| **Avg latency (ms)** | 1000 / QPS — average wall time per query. |

**Note on QPS numbers:** All queries are single-threaded from Python with one round-trip per query, so absolute QPS reflects that overhead. The relative ordering between methods is meaningful; absolute values will be higher in production with connection pooling / batch queries.

---

## Benchmark results (actual run)

Config: **N=5 000, dim=64, 200 queries, k=10, seed=42**, PostgreSQL 14 on Apple M-series hardware.

```
python3 bench/tqflat_bench.py --n 5000 --dim 64 --queries 200 --dbname tqtest --user max
```

| Method                      | Build (s) | Index size | Recall@10 | QPS    | Avg latency (ms) |
|-----------------------------|-----------|------------|-----------|--------|------------------|
| exact (seqscan)             | 0.00      | —          | 1.0000    | 982.3  | 1.02             |
| tqflat (bits=2,rerank=200)  | 0.04      | 328.0 KB   | 0.9970    | 446.2  | 2.24             |
| tqflat (bits=2,rerank=0)    | 0.04      | 328.0 KB   | 0.5250    | 583.1  | 1.72             |
| tqflat (bits=4,rerank=200)  | 0.04      | 400.0 KB   | 1.0000    | 835.8  | 1.20             |
| tqflat (bits=4,rerank=0)    | 0.04      | 400.0 KB   | 0.8400    | 364.7  | 2.74             |
| hnsw (m=16,ef=64)           | 1.02      | 2.7 MB     | 0.9100    | 3071.4 | 0.33             |
| ivfflat (lists=70,probes=7) | 0.09      | 1.7 MB     | 0.3800    | 4917.6 | 0.20             |

### Key observations

- **Index size**: tqflat (328–400 KB) is **6.8–8.2× smaller** than hnsw (2.7 MB) and
  **4.2–5.2× smaller** than ivfflat (1.7 MB). Both hnsw and ivfflat store full-precision
  float32 vectors; tqflat stores only the quantized bit codes.
- **Build time**: tqflat builds in ~0.04 s (no graph construction, no k-means) vs
  hnsw 1.02 s (~25× faster) and ivfflat 0.09 s (~2.2× faster). At large N this gap grows
  super-linearly for hnsw.
- **Recall with rerank**: bits=4,rerank=200 achieves **perfect recall (1.000)**;
  bits=2,rerank=200 achieves **0.997**. Full-precision rerank restores accuracy effectively.
- **Recall without rerank**: bits=4,rerank=0 gives 0.840 (quantized scores only, no heap
  fetch). bits=2,rerank=0 gives 0.525 — significant information loss at 2 bits without correction.
- **QPS**: hnsw and ivfflat are faster per query at this scale because the flat scan
  visits all N entries; at larger N or higher dim where HNSW graph traversal becomes
  expensive, tqflat's flat SIMD scan becomes more competitive. The flat scan QPS also
  benefits from the much smaller index footprint (better cache utilization).

---

## Note on `tqflat_test_*` SQL functions

The `tqflat_test_codebook`, `tqflat_test_roundtrip`, `tqflat_test_rotation_orthogonality`,
`tqflat_test_meta`, `tqflat_test_qjl`, and `tqflat_test_ip_estimate` SQL functions are
**prototype/test-only** helpers used by the regression and unit test suites. They would be
**dropped from `sql/vector.sql` and the migration** before any upstream submission to the
pgvector project. They exist here to support TDD development of the quantizer math; the
benchmark harness does not call them.
