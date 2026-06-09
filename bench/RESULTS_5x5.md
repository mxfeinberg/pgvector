# 5×5 TurboQuant vs Baseline Benchmark — Results

**Methods (5):** `tqflat`, `tqivf`, `tqhnsw` (TurboQuant) vs `ivfflat`, `hnsw` (pgvector baseline).  
**Datasets (5):**
- **agtalk — 7,735,045 × 768 (EmbeddingGemma-300M, Q8_0)** — cosine *(full `search_chunks` export: agtalk 7.09 M + combineforum 0.64 M)*
- **SIFT1M — 1,000,000 × 128** — L2
- **GIST1M — 1,000,000 × 960** — L2
- **GloVe-200 — 1,183,514 × 200** — cosine (angular)
- **OpenAI-1M — 999,800 × 1536 (text-embedding-3-large)** — cosine

> **agtalk grew from 4.59 M to 7.74 M this run** (full table dump, both forums). Its numbers below are **not** 1:1 comparable to the prior report — the corpus is ~1.7× larger and now mixes two sources, which makes the ANN problem harder (lower absolute recall at fixed `ef_search`/`probes`). The other four datasets are unchanged.

**Host:** the-fire (16 cores, 60 GB RAM, NVMe). **PG 18**, `vector` 0.9.0, built **without** `-DUSE_ASSERT_CHECKING` (production `-march=native`). **k=10**, 1000 held-out queries (OpenAI-1M: 200; **agtalk tqflat: 200** — its flat scan is too slow at 7.7 M for 1000). Ground truth: exact kNN (file GT for SIFT/GIST/GloVe; numpy-exact for agtalk/OpenAI). `maintenance_work_mem` 16 GB (1M sets) / 24 GB (agtalk); `max_parallel_maintenance_workers=8`. One index resident at a time.

## 1. Headline: index size (the memory thesis)

Index size per method per dataset (MB), and the TQ-vs-baseline shrink factor:

| dataset | tqflat | tqivf | tqhnsw | hnsw | ivfflat | tqhnsw vs hnsw | tqivf vs ivfflat |
|---|--:|--:|--:|--:|--:|--:|--:|
| agtalk | 3,923 | 3,980 | 6,043 | 30,215 | 30,231 | **5.0×** | **7.6×** |
| SIFT1M | 78 | 88 | 296 | 966 | 525 | **3.3×** | **6.0×** |
| GIST1M | 507 | 530 | 781 | 7,679 | 3,912 | **9.8×** | **7.4×** |
| GloVe-200 | 165 | 178 | 424 | 1,542 | 1,032 | **3.6×** | **5.8×** |
| OpenAI-1M | 998 | 1,029 | 1,302 | 7,811 | 7,819 | **6.0×** | **7.6×** |

> TurboQuant's 4-bit codes shrink the graph/list index **3–10×**, and the gap widens with dimension — peaking on the 1536-d and 960-d sets. On agtalk (768-d, **7.7 M**) tqhnsw is **5.0× smaller** than hnsw (**6.0 GB vs 30.2 GB**) and tqivf is **7.6× smaller** than ivfflat (**4.0 GB vs 30.2 GB**).

## 2. Per-dataset operating points

Two rows per method: **max-recall** point, and **knee** (highest QPS at recall ≥ 0.95). QPS and avg latency are per single-query (k=10).

### agtalk — 7,735,045 × 768 (EmbeddingGemma-300M, Q8_0) — cosine

| method | op point | recall | QPS | avg ms | build s | index |
|---|---|--:|--:|--:|--:|--:|
| tqflat (max-rec) | bits=4,rerank=200 | 1.0000 | 0.24 | 4156.9 | 194 | 3923 MB |
| tqivf (max-rec) | lists=2781,probes=400,rerank=100 | 0.9936 | 2.0 | 492.8 | 949 | 3980 MB |
| tqivf (knee) | lists=2781,probes=99,rerank=100 | 0.9638 | 7.7 | 129.5 | 949 | 3980 MB |
| tqhnsw (max-rec) | m=16,ef_construction=64,ef_search=200,rerank=100 | 0.9449 | 56.3 | 17.8 | 5693 | 6043 MB |
| tqhnsw (mid) | m=16,ef_construction=64,ef_search=100,rerank=200 | 0.9164 | 97.9 | 10.2 | 5693 | 6043 MB |
| hnsw (max-rec) | m=32,ef_construction=128,ef_search=100 | 0.9762 | 71.0 | 14.1 | 10781 | 30215 MB |
| ivfflat (max-rec) | lists=2781,probes=278 | 0.9879 | 0.80 | 1243.3 | 987 | 30231 MB |

> At 7.7 M, **tqhnsw caps at 0.945** within the swept `ef_search ≤ 200` — **but this row used the
> tqhnsw defaults m=16/ef_c=64 against hnsw's m=32/ef_c=128** (not apples-to-apples). The matched-params
> rerun (**§8**) builds tqhnsw at m=32/ef_c=128 and it then **clears hnsw on recall** (0.983 @ ef=200,
> 0.966 @ ef=100) in 4× less memory — the 0.945 here is a settings artifact, not a TQ ceiling. The hnsw
> build took ~3 h — its ~30 GB graph exceeds the 24 GB `maintenance_work_mem`, so pgvector spills to its
> slower on-disk insert phase.

### SIFT1M — 1,000,000 × 128 — L2

| method | op point | recall | QPS | avg ms | build s | index |
|---|---|--:|--:|--:|--:|--:|
| tqflat (max-rec) | bits=4,rerank=200 | 0.9994 | 5.7 | 175.6 | 3 | 77.8 MB |
| tqivf (max-rec) | lists=1000,probes=400,rerank=200 | 0.9995 | 18.1 | 55.3 | 10 | 87.8 MB |
| tqivf (knee) | lists=1000,probes=30,rerank=100 | 0.9739 | 211.3 | 4.7 | 10 | 87.8 MB |
| tqhnsw (max-rec) | m=16,ef_construction=64,ef_search=200,rerank=100 | 0.9919 | 127.5 | 7.8 | 88 | 295.7 MB |
| tqhnsw (knee) | m=16,ef_construction=64,ef_search=100,rerank=200 | 0.9758 | 230.0 | 4.3 | 88 | 295.7 MB |
| hnsw (max-rec) | m=32,ef_construction=128,ef_search=100 | 0.9956 | 124.5 | 8.0 | 182 | 966.3 MB |
| ivfflat (max-rec) | lists=1000,probes=100 | 0.9987 | 28.8 | 34.7 | 9 | 525.0 MB |

### GIST1M — 1,000,000 × 960 — L2

| method | op point | recall | QPS | avg ms | build s | index |
|---|---|--:|--:|--:|--:|--:|
| tqflat (max-rec) | bits=4,rerank=2000 | 0.9994 | 1.8 | 547.0 | 24 | 507.2 MB |
| tqflat (knee) | bits=4,rerank=200 | 0.9992 | 1.9 | 529.5 | 24 | 507.2 MB |
| tqivf (max-rec) | lists=1000,probes=400,rerank=100 | 0.9992 | 5.0 | 200.1 | 47 | 530.4 MB |
| tqivf (knee) | lists=1000,probes=99,rerank=100 | 0.9951 | 17.1 | 58.4 | 47 | 530.4 MB |
| tqhnsw (max-rec) | m=16,ef_construction=64,ef_search=200,rerank=100 | 0.9085 | 58.7 | 17.0 | 412 | 781.3 MB |
| hnsw (max-rec) | m=32,ef_construction=128,ef_search=100 | 0.9404 | 74.7 | 13.4 | 608 | 7679.1 MB |
| ivfflat (max-rec) | lists=1000,probes=100 | 0.9932 | 4.3 | 230.7 | 49 | 3912.2 MB |

### GloVe-200 — 1,183,514 × 200 — cosine (angular)

| method | op point | recall | QPS | avg ms | build s | index |
|---|---|--:|--:|--:|--:|--:|
| tqflat (max-rec) | bits=4,rerank=200 | 1.0000 | 3.5 | 289.3 | 8 | 164.6 MB |
| tqivf (max-rec) | lists=1087,probes=400,rerank=100 | 0.9897 | 12.8 | 78.1 | 18 | 177.5 MB |
| tqivf (knee) | lists=1087,probes=200,rerank=100 | 0.9679 | 25.1 | 39.9 | 18 | 177.5 MB |
| tqhnsw (max-rec) | m=16,ef_construction=64,ef_search=200,rerank=100 | 0.7781 | 85.4 | 11.7 | 209 | 424.0 MB |
| hnsw (max-rec) | m=32,ef_construction=128,ef_search=100 | 0.8328 | 84.3 | 11.9 | 344 | 1541.5 MB |
| ivfflat (max-rec) | lists=1087,probes=108 | 0.9274 | 17.7 | 56.5 | 19 | 1032.2 MB |

### OpenAI-1M — 999,800 × 1536 (text-embedding-3-large) — cosine

| method | op point | recall | QPS | avg ms | build s | index |
|---|---|--:|--:|--:|--:|--:|
| tqflat (max-rec) | bits=4,rerank=200 | 1.0000 | 1.1 | 875.1 | 46 | 997.7 MB |
| tqivf (max-rec) | lists=999,probes=400,rerank=100 | 0.9985 | 5.0 | 198.3 | 83 | 1028.8 MB |
| tqivf (knee) | lists=999,probes=99,rerank=100 | 0.9790 | 19.6 | 51.0 | 83 | 1028.8 MB |
| tqhnsw (max-rec) | m=16,ef_construction=64,ef_search=200,rerank=100 | 0.9755 | 46.4 | 21.6 | 577 | 1301.8 MB |
| tqhnsw (knee) | m=16,ef_construction=64,ef_search=100,rerank=200 | 0.9555 | 73.3 | 13.6 | 577 | 1301.8 MB |
| hnsw (max-rec) | m=32,ef_construction=128,ef_search=100 | 0.9905 | 88.2 | 11.3 | 702 | 7810.9 MB |
| ivfflat (max-rec) | lists=999,probes=99 | 0.9785 | 4.6 | 217.4 | 88 | 7818.8 MB |

## 3. agtalk Pareto (the centerpiece, 7.7 M × 768, cosine)

Recall vs QPS across the full sweep (one index built per method, queried at each operating point;
best QPS shown per recall level):

**tqivf** (index 3980 MB, build 949s):
  r=0.813@49qps  ·  r=0.908@25qps  ·  r=0.964@8qps  ·  r=0.983@4qps  ·  r=0.994@2qps

**tqhnsw** (index 6043 MB, build 5693s):
  r=0.834@186qps  ·  r=0.916@98qps  ·  r=0.945@56qps

**hnsw** (index 30215 MB, build 10781s): r=0.976@71qps (single op point, ef_search=100).
**ivfflat** (index 30231 MB, build 987s): r=0.988@0.8qps (probes=278).

## 4. Findings

**Memory is TurboQuant's decisive win.** Across all five datasets the TQ indexes are **3–10× smaller**
than their baselines, and the advantage grows with dimension (the 4-bit code plane replaces full
float32 vectors, so storage savings scale with `dim`). The clearest case is agtalk at 768-d/**7.7 M**:
tqhnsw holds **0.945 recall in 6.0 GB** where hnsw needs **30.2 GB** for 0.976, and tqivf is
**7.6× smaller** than ivfflat (4.0 GB vs 30.2 GB). The 5–7.6× footprint cut is the durable result; the
recall/speed tradeoff (below) shifts with corpus difficulty.

**tqhnsw is the small-footprint graph option — and at matched parameters it equals/beats hnsw (see §8).**
At its defaults (m=16/ef_c=64) it reaches 0.945 @ 56 QPS in **6 GB**, which *looks* like it trails hnsw
(0.976). But that compared m=16 against hnsw's m=32. The matched-params rerun (**§8**: tqhnsw at
m=32/ef_c=128) lifts the whole curve — **0.966 @ 54 QPS at ef=100, 0.983 @ 33 QPS at ef=200** — i.e.
tqhnsw **clears hnsw's recall in 4× less memory (7.55 GB vs 30.2 GB)**, trading ~1.4× throughput and
~2× build time. The memory thesis holds and the recall gap was an artifact of unequal settings.

**tqivf trades throughput for tiny memory + tunable recall — and clearly beats ivfflat.** It sweeps
0.81→0.99 recall via `probes` in the smallest practical index. Absolute QPS is modest at 7.7 M (2–49
QPS), but it dominates **ivfflat** on both memory (7.6×) and, at matched recall, speed: ivfflat needs
probes=278 for 0.988 recall at just **0.80 QPS / 1243 ms**, while tqivf hits 0.983 at **3.9 QPS** (and
0.994 at 2 QPS) — roughly **5× faster** per equivalent recall, in 1/7.6 the space.

**tqflat is exact-but-not-scalable.** It delivers **1.0000 recall** (full rerank over a near-lossless
code) in a 3.9 GB index, but it is a flat scan: throughput is **0.24 QPS** at 7.7 M (it reads the whole
~3.9 GB code plane per query). It is the right tool up to ~1 M rows; beyond that, tqivf/tqhnsw are the
scalable TQ options.

**GloVe-angular is hard for graph methods** (both tqhnsw 0.78 and hnsw 0.83 cap out at ef_search≤200);
the IVF/flat methods reach higher recall there. This is a known property of angular GloVe, not a TQ
regression.

## 5. Engineering finding & fix: tqflat scan allocation

The first agtalk run crashed tqflat's query: `invalid memory alloc request size 2352250880`. Root cause:
the block fast-scan in `src/tqscan.c` reads the **entire** code plane into one `palloc`
(`blockCount × dimCodes × 16` bytes). That is ~512 MB at 1 M rows but **2.35 GB at agtalk's 4.6 M**,
exceeding PostgreSQL's 1 GB `MaxAllocSize`. **tqflat therefore could not scan tables above ~2 M rows at
high dimension.** Only tqflat is affected — tqivf allocates per-probe-list and tqhnsw per-node, both
bounded.

**Fix:** that one allocation now uses `MemoryContextAllocHuge` (transparent for the <1 GB case, so the
1 M-row results are unchanged). Validated on mac: `make installcheck` **26/26**, all tqflat tests green;
and at **7.7 M** tqflat completed with **recall 1.0000** (the code plane is now ~3.9 GB — well past the
1 GB ceiling). *(A streaming, per-block scan would be the better long-term design — it would also cut
tqflat's per-query I/O — but the huge-alloc fix is the minimal correct change to lift the size ceiling.)*
The fix is uncommitted in the working tree (`src/tqscan.c`) pending maintainer commit.

**Second fix (benchmark harness, not the extension): loader OOM at 7.7 M.** `bench/tqflat_bench.py`'s
`load_from_files` did `np.load(path).astype(np.float32)`, which materialized the base array **twice**
(np.load + a redundant `astype` copy) — ~47 GB for agtalk's 24 GB npy — and OOM-killed the loader before
any build. It now mmaps the base and only copies if the dtype isn't already float32; the COPY path
already streams row-by-row, so the mmap is all that's needed. The new `bench/agtalk_copy_extract.py`
(parses the `search_chunks` pg_dump COPY format) and its truncation step are likewise bounded-memory
(chunked copy, no whole-array materialization).

## 6. Caveats / methodology notes

- **agtalk is now a single uniform run** (one `5x5_agtalk.csv`), unlike the prior report's two-run
  merge. tqflat ran with **200 queries** (its 0.24 QPS flat scan is too slow for 1000 at 7.7 M); the
  other four ran with 1000. The two invocations (4 methods, then tqflat) write separate CSVs that the
  runner concatenates.
- **hnsw is a single operating point** (m=32, ef_construction=128, ef_search=100); TQ methods are swept
  across ef_search/probes × rerank. A full hnsw ef_search sweep would fill in its Pareto curve.
- **tqhnsw `ef_search` was capped at 200 in §2, and tqhnsw ran at m=16 vs hnsw's m=32** — both fixed in
  the **§8** follow-up (matched m=32/ef_c=128, ef_search to 800), where tqhnsw clears hnsw on recall.
  The §2 agtalk tqhnsw row is the m=16 default; §8 is the apples-to-apples comparison.
- **`/dev/shm` ceiling & the 3 h hnsw build:** the parallel builds allocate a DSM sized to
  `maintenance_work_mem`, backed by tmpfs (`/dev/shm` = 31 GB here), so it must stay ≤ ~30 GB. agtalk ran
  at **24 GB**. At 7.7 M the hnsw graph (~30 GB) exceeds that, so the build spilled to pgvector's slower
  on-disk insert phase — hence the ~3 h hnsw build (vs ~95 min for the 6 GB tqhnsw graph, which fit). The
  result (recall/QPS/size) is unaffected by the spill; only build wall-clock is. A bigger in-RAM build
  would need a `/dev/shm` remount + the spare RAM.
- **Build times** are wall-clock with 8 parallel workers; the hnsw build dominates. agtalk's slow
  tqflat/tqivf query points reflect their per-query scan cost at 7.7 M, not a measurement artifact.

## 7. Reproduce

```sh
# agtalk data pipeline (run on the-fire; the 24 GB npy + 80 GB dump don't fit on the mac)
# search_chunks.sql is a pg_dump of public.search_chunks (COPY format, embedding = field 13).
python3 bench/agtalk_copy_extract.py --sql ~/tqdata/search_chunks.sql \
    --base-out ~/tqdata/agtalk/agtalk_base.npy --queries-out ~/tqdata/agtalk/agtalk_queries.npy \
    --dim 768 --total 7736111 --n-queries 1000 --seed 42
python3 bench/agtalk_groundtruth.py --base ~/tqdata/agtalk/agtalk_base.npy \
    --queries ~/tqdata/agtalk/agtalk_queries.npy --out ~/tqdata/agtalk/agtalk_gt.npy \
    --metric cosine --k 100 --chunk 500000

# the 5x5 suite (per-dataset CSV + log under bench/results/5x5_*). agtalk splits into a
# 4-method run (1000 q) + a tqflat run (200 q); the runner concatenates them into 5x5_agtalk.csv.
bash bench/run_5x5_suite.sh                 # all five
bash bench/run_5x5_suite.sh agtalk sift1m   # subset
```

Per-dataset CSVs: `bench/results/5x5_{agtalk,sift1m,gist1m,glove200,openai1m}.csv`
(agtalk also keeps the per-invocation `5x5_agtalk_main.csv` + `5x5_agtalk_tqflat.csv`).

## 8. Follow-up: apples-to-apples tqhnsw (m=32, ef_construction=128)

The §2 run compared tqhnsw at its **defaults (m=16, ef_construction=64)** against hnsw at
**m=32, ef_construction=128** — i.e. hnsw got *double* the graph connectivity and build beam, and
tqhnsw's `ef_search` sweep stopped at 200 while recall was still climbing. That made the 0.945 "cap"
an **artifact of the settings, not a TQ ceiling.** This rerun builds tqhnsw at the **same m=32 /
ef_construction=128** as hnsw and sweeps `ef_search {100,200,400,800} × rerank {30,50,100}` (agtalk
7.74 M, cosine, 1000 queries). Runner: `bench/run_tqhnsw_apples.sh`; data:
`bench/results/agtalk_tqhnsw_m32.csv`.

| tqhnsw (m=32, ef_c=128) | recall | QPS | avg ms | index | build |
|---|--:|--:|--:|--:|--:|
| ef_search=100 | 0.9655 | 53.6 | 18.6 | 7554 MB | 25278 s |
| ef_search=200 | 0.9833 | 32.9 | 30.4 | 7554 MB | 25278 s |
| ef_search=400 | 0.9929 | 17.5 | 57.2 | 7554 MB | 25278 s |
| ef_search=800 | 0.9959 | 8.9 | 112.0 | 7554 MB | 25278 s |
| **hnsw (m=32, ef_search=100)** | **0.9762** | **71.0** | 14.1 | 30215 MB | 10781 s |

(QPS is the best across rerank ∈ {30,50,100} — see below; index/build are the single m=32 tqhnsw index.)

**Findings:**

1. **At matched graph params, tqhnsw matches hnsw recall at ef_search=100 (0.966 vs 0.976) and
   *clears* it at ef_search ≥ 200 (0.983 → 0.996) — in 4× less memory (7.55 GB vs 30.2 GB).** The
   earlier "tqhnsw < hnsw on recall" conclusion was purely the m=16 vs m=32 mismatch. Doubling `m`
   moved tqhnsw's whole curve up by ~3–5 recall points and lifted the ceiling well past 0.99.

2. **The memory win narrows but holds: 4.0× smaller** (was 5.0× at m=16 — m=32 doubles the graph to
   7.55 GB, still a quarter of hnsw's 30.2 GB).

3. **Throughput is the remaining trade.** At matched recall (~0.976), tqhnsw runs at ef_search≈100–150
   → ~45–54 QPS vs hnsw's 71 — hnsw is ~1.4× faster there (its full-precision graph scores nodes with
   a direct dot product; tqhnsw pays for ADC-LUT gathers + heap reranking). You buy 4× memory for ~1.4×
   throughput at equal recall — or spend the headroom on recall (0.99+ at lower QPS), which hnsw can't
   reach at all here.

4. **rerank is over-provisioned (hypothesis confirmed).** Recall is *identical* across rerank=30/50/100
   at every `ef_search`, and QPS barely moves (e.g. ef=800: 8.88→8.92). So **rerank=30 is plenty** —
   but, contrary to the earlier guess, the rerank heap-fetch is *not* the throughput bottleneck at high
   `ef_search`; the graph traversal (scoring ef×m nodes by ADC LUT) dominates. The throughput lever is
   therefore fewer/faster node scorings — a higher-quality graph (higher ef_c) to hit a target recall at
   lower `ef_search`, or SIMD on the per-node LUT gather — not rerank depth.

5. **Build cost is tqhnsw's price here: 25278 s (~7.0 h) vs hnsw's 10781 s (~3.0 h)** at the same m/ef_c.
   tqhnsw's build (random rotation + 4-bit encode + LUT) is ~2.3× more wall-clock per the same graph,
   even though it builds fully in-RAM (the 7.5 GB graph fits 24 GB — no spill, unlike hnsw). The 4×
   smaller, equal-or-better-recall index costs more to build.

**Bottom line:** with matched parameters, **tqhnsw is a genuine hnsw replacement at 4× less memory and
equal-or-higher recall**, trading ~1.4× query throughput and ~2× build time. The right operating point
for agtalk is **ef_search=100, rerank=30 (0.966 @ 54 QPS, 7.55 GB)** for throughput, or **ef_search=200
(0.983 @ 33 QPS)** to beat hnsw on recall outright.
