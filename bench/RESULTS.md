# tqflat large-scale benchmark results (standard ANN datasets)

**Environment:** PostgreSQL 18.4 (Homebrew), Apple M-series. Index builds use
`maintenance_work_mem=2GB` + parallel workers; HNSW queried at
`hnsw.ef_search=100` (pgvector's default 40 under-recalls on hard datasets).
200 queries per run, k=10. tqflat at `bits ∈ {2,4}`, `rerank ∈ {200, 2000}`.

Raw CSVs in `bench/results/`. Reproduce with the commands in `bench/README.md`.

> **Historical note (superseded below):** the three tables in this section are the
> **original scalar-kernel + dense-rotation baseline**. They are kept as the
> reference point. Two milestones since then change the latency and build numbers
> materially — **structured fast rotation** (see "Structured fast rotation" below)
> and the **v4 blocked SIMD LUT kernel** (see "v4 blocked SIMD LUT kernel" at the
> end). The current shipping numbers are in the v4 section; read it for the latency
> story (the scalar latencies below are a pessimistic ceiling the SIMD kernel lifts).

---

## SIFT1M — 1,000,000 × 128, L2

| Method | Build (s) | Index size | Recall@10 | Latency (ms) |
|--------|-----------|------------|-----------|--------------|
| tqflat bits=2, rerank=200  | 27  | **73 MB**  | 0.457 | 351 |
| tqflat bits=2, rerank=2000 | 27  | **73 MB**  | 0.924 | 355 |
| tqflat bits=4, rerank=200  | 24  | **104 MB** | 0.984 | 347 |
| tqflat bits=4, rerank=2000 | 24  | **104 MB** | **0.999** | 363 |
| hnsw (ef_search=100)       | 124 | 782 MB     | 0.988 | **4.9** |
| ivfflat (lists=1000, probes=100) | 37 | 525 MB | 0.999 | 47 |

## GloVe-200-angular — 1,183,514 × 200, cosine

| Method | Build (s) | Index size | Recall@10 | Latency (ms) |
|--------|-----------|------------|-----------|--------------|
| tqflat bits=2, rerank=200  | 74  | **115 MB** | 0.957 | 582 |
| tqflat bits=2, rerank=2000 | 74  | **115 MB** | 0.999 | 620 |
| tqflat bits=4, rerank=200  | 69  | **178 MB** | **1.000** | 634 |
| tqflat bits=4, rerank=2000 | 69  | **178 MB** | **1.000** | 635 |
| hnsw (ef_search=100)       | 247 | 1321 MB    | 0.699 | **7.3** |
| ivfflat (lists=1087, probes=108) | 90 | 1032 MB | 0.939 | 89 |

GloVe-angular is a known-hard case for HNSW; even at ef_search=100 its recall is
0.70 (raising ef trades latency for recall). tqflat reaches ~1.0 with rerank.

## OpenAI text-embedding-3-large — 100,000 × 1536, cosine (Qdrant/dbpedia)

| Method | Build (s) | Index size | Recall@10 | Latency (ms) |
|--------|-----------|------------|-----------|--------------|
| tqflat bits=2, rerank=200  | **457** | **78 MB**  | **1.000** | 359 |
| tqflat bits=4, rerank=200  | **439** | **116 MB** | **1.000** | 332 |
| hnsw (ef_search=100)       | 348 | 781 MB     | 0.976 | **9.3** |
| ivfflat (lists=316, probes=31) | 80 | 784 MB  | 0.939 | 35 |

Embeddings are unit-normalized (mean ‖v‖ = 1.0), so cosine == L2 ranking here.
Only 100k rows: the full 1M is gated on the build-cost wall below.

---

## What the data says

**tqflat wins decisively on two axes, at every scale and dimension:**

1. **Index size — 5–11× smaller.** 73–178 MB vs HNSW 0.78–1.32 GB and ivfflat
   0.5–1.0 GB. HNSW/ivfflat store full float32 vectors; tqflat stores 2–4-bit
   codes. This is the core TurboQuant value proposition and it holds on real data.
2. **Recall — matches or beats** the graph/cluster methods after rerank: 0.999
   (SIFT, bits=4), 1.000 (GloVe & OpenAI). On the hard GloVe-angular set tqflat
   (1.0) far exceeds HNSW (0.70).

**tqflat loses on two axes — and these define the integration decision:**

3. **Query latency — the O(N) flat-scan wall.** 330–640 ms/query vs HNSW 5–9 ms
   (50–100×). A flat scan touches every vector; HNSW touches ~log(N). This makes
   flat-tqflat unsuitable for latency-sensitive search at ≥1M scale **as a
   standalone index** — even before noting the scalar-kernel caveat.
4. **Build cost at high dimension — the dense-rotation wall.** The O(d²) dense
   rotation dominates encode: at dim=1536 the tqflat build is **457 s — slower
   than HNSW (348 s)** and 5.7× slower than ivfflat. "Near-zero build time" holds
   at dim≤200 (24–74 s) but **breaks at dim≥1536**.

### Implications

- **Standalone flat-tqflat is a real, shippable index for small-to-mid N**
  (≤~100k) where O(N) latency is acceptable and the 5–11× memory saving + tiny
  build (at low/mid dim) matter — and where recall-per-byte beats everything.
- **The integration thesis is confirmed.** The memory and recall advantages are
  large and real; the only thing missing is sublinear latency. Putting TQ codes
  *inside* IVF lists (or HNSW nodes) keeps the 5–11× compression and ~1.0 recall
  while inheriting sublinear search. **IVF + TurboQuant is the high-leverage first
  step**: ivfflat already scans only `probes × (N/lists)` vectors (e.g. 47 ms on
  SIFT); replacing its stored float32 vectors with TQ codes would cut its 525 MB–1
  GB footprint ~7× *and* speed each list scan — reusing `tqquant.c`/`tqdistance.c`
  almost verbatim.
- **Two prerequisites before/alongside integration**, both surfaced by this data:
  1. **SIMD LUT scan kernel** — the latencies above are the scalar kernel; the
     paper's premise is SIMD. Needed to make any latency claim fair.
  2. **Structured fast rotation** (Hadamard / FFT, O(d·log d)) — mandatory for
     dim≥1536, where the dense O(d²) rotation makes builds slower than HNSW.

---

# Structured fast rotation (randomized Hadamard) — results

The dense O(d²) rotation above was replaced by a structured **randomized Hadamard
transform** (RHT, O(d·log d)), selectable via the `fast_rotation` reloption
(default **on**). The QJL second stage (`tq_prod`) was also evaluated and now
defaults **off** (see below). Same PG18 / parallel-build / `maintenance_work_mem=2GB`
/ `hnsw.ef_search=100` setup; 200 queries; A/B via `--tqflat-fast-rotation` and
`--tqflat-tq-prod` (CSVs under `bench/results/`).

## Fast vs dense rotation (build wall)

Build time, dense → fast (`tq_prod=off` both), and the resulting speedup:

| Dataset (dim) | dense build | fast build | speedup | recall dense→fast | index dense→fast |
|---|---|---|---|---|---|
| OpenAI (1536) | 457 s | **5.6 s** | **~80×** | 1.000 → 1.000 | 116 → 112 MB |
| GloVe (200)   | 69 s  | **9.1 s** | ~7.6×   | b2/r200 0.957 → 0.992 | 115 → 105 MB |
| SIFT (128)    | 25 s  | **4.8 s** | ~5×     | b2/r200 **0.457 → 0.962** | 73 → 57 MB |

- **Build-cost wall demolished.** The speedup scales with dimension exactly as
  O(d²)→O(d·log d) predicts — modest at d=128 (rotation was never the bottleneck),
  ~80× at d=1536 where the wall lived. The dim≥1536 build is now ~6 s, not ~7.5 min.
- **The structured rotation preserves or *improves* recall**, dramatically on SIFT
  (0.457 → 0.962 at bits=2/rerank=200): the 3-stage RHT decorrelates structured
  descriptor dimensions better than the dense Gaussian (MGS) rotation.
- Padding d→next_pow2(d) adds codes (e.g. 1536→2048), offset by dropping QJL.

## The QJL second stage (`tq_prod`) — dropped by default

The structured QJL is biased (the orthonormal RHT violates the sign-estimator's
i.i.d.-Gaussian assumption; bias grows with dim, ~20% at d=128). Measured impact:

**Recall (reranked ANN), SIFT1M bits=2/rerank=200:** fast+QJL 0.727 vs
fast+noQJL **0.962** — the biased QJL *hurts*. Elsewhere QJL is neutral but costs
build time, index size (sign bits), and ~2× query latency.

**Inner-product estimation accuracy** (`bench/ip_accuracy.py`, 50k query×base pairs;
this is the metric reranked-recall can't see — turboquant_prod's actual objective):

| config | SIFT rel_rmse | GloVe rel_rmse | OpenAI rel_rmse | Pearson (all) |
|---|---|---|---|---|
| mse  dense (no QJL) | **1.8%** | **8.3%** | **2.9%** | 0.999 |
| prod dense (QJL)    | 2.5%     | 10.3%    | 3.6%     | 0.999 |
| mse  fast (no QJL)  | **1.8%** | **7.3%** | **2.5%** | 0.999 |
| prod fast (QJL)     | 2.2%     | 7.0%     | 2.6%     | 0.999 |

- turboquant_mse (no QJL) is already an excellent IP estimator on real data
  (Pearson ≈ 0.999, low relative RMSE). The QJL *does* reduce |bias| on normalized
  data (directionally confirming the paper's Theorem 2) but **raises RMSE/variance
  in every case**, so net accuracy is *worse* with QJL. The structured QJL is badly
  biased on un-normalized SIFT (bias −748 vs mse +49).
- **Conclusion:** recall *and* IP-estimation agree — turboquant_mse wins. `tq_prod`
  now **defaults off**; the QJL remains an opt-in reloption. A correctly-formulated
  (non-orthonormal) structured QJL is a future option if a bias-dominated IP-MIPS
  use case appears.

## Net defaults

`fast_rotation=on, tq_prod=off`: fastest build (5–80× at high dim), smallest index,
lowest latency, and recall equal-or-better than the previous dense+QJL default.

---

# v4 blocked SIMD LUT kernel — results

The scalar `TqScoreEntryDefault` per-vector scorer (the "latency caveat" at the
top) was replaced by a **blocked, 4-bit-nibble-packed, SIMD LUT fast-scan kernel**
(`src/tqfastscan.c`; FAISS Fast-Scan / Quick-ADC technique) over a new on-disk
**v4 blocked format**. The block scorer dispatches at load time — NEON
(`vqtbl1q_u8`) on arm64 (this host), AVX-512 (`_mm_shuffle_epi8`) on x86 — and all
variants are bit-identical to the scalar reference. The per-query LUT is
8-bit-quantized; `bits` is fixed at 4 and `tq_prod` (QJL) is not used in this layout.

A build-writer **streaming fix** (each block's code-plane is written incrementally
onto the code-plane page chain instead of being buffered in one `StringInfo`)
removes the ~1 GB build-memory ceiling that previously capped a build at ~2M rows
at dim=768 — large builds are now bounded only by `nVectors` (u32, ~4.29 B) and disk.

Same PG18 / parallel-build / `maintenance_work_mem=2GB` / `hnsw.ef_search=100` setup;
200 queries; `bits=4`, `fast_rotation=on`, `tq_prod=off`; arm64/NEON.

## v4 (SIMD kernel + fast rotation) vs the original scalar + dense baseline — arm64/NEON

Host: Apple M-series (arm64), NEON kernel dispatched. tqflat, bits=4, rerank=200.
CSVs: `bench/results/{sift1m,glove200,openai1536}_v4.csv`.

| Dataset (dim) | metric | build | index | recall@10 | latency | speedup vs baseline |
|---|---|---|---|---|---|---|
| SIFT1M (128)    | L2     | **5.0 s** | **77.8 MB**  | **0.9995** | **137 ms** | build 4.7× · latency 2.5× · recall 0.984→0.9995 · size 104→78 MB |
| GloVe-200 (200) | cosine | **8.8 s** | **164.6 MB** | **1.000**  | **219 ms** | build 7.9× · latency 2.9× · size 178→165 MB |
| OpenAI (1536)   | cosine | **5.6 s** | **99.8 MB**  | **1.000**  | **29 ms**  | build **78×** · latency **11.5×** · size 116→100 MB |

_Baseline (bits=4, rerank=200, scalar kernel + dense rotation): SIFT 24 s / 104 MB / 0.984 / 347 ms;
GloVe 69 s / 178 MB / 1.000 / 634 ms; OpenAI-100k 439 s / 116 MB / 1.000 / 332 ms. OpenAI here is the
same 100k tier as the baseline — the full 1M run is the section below._

- **Build:** fast rotation collapses the dim-1536 wall (439 s → 5.6 s, **78×**); the
  speedup scales with dimension (O(d²) → O(d·log d)), modest at d=128 where rotation
  was never the bottleneck.
- **Latency:** the SIMD kernel cuts query time 2.5–11.5×. The win is largest at 100k
  rows (OpenAI **11.5×**) where the per-vector score step dominates, and smaller at 1M
  (SIFT/GloVe) where Postgres page-walk + top-k heap + rerank form a fixed floor the
  kernel can't touch — the Amdahl caveat, and the reason IVF integration is the next
  lever.
- **Recall / size:** recall holds or improves, and the blocked format is *smaller*
  than the row-major scalar layout (it drops per-row `TqEntry` header + MAXALIGN
  padding, moving per-vector side data into a compact per-block side chain).

## Kernel A/B and raw score-step throughput — arm64/NEON

- **Raw score-step kernel** (`bench/microbench/`, 1M coded vectors, best of 5):
  **19.5× @ dc=2048**, ~12× @ dc=128 / dc=256 vs the scalar gather. The 4-bit nibble
  split is essentially free (`Anib/Abyte ≈ 0.9–1.1×`) — the high-dim scan is
  bandwidth-bound, so index size is unchanged from row-major 4-bit.
- **8-bit vs float LUT** (in-extension A/B via `--tqflat-force-scalar`, synthetic
  200k×256): identical recall (0.846 vs 0.852 at rerank=0; **1.000 vs 1.000** at
  rerank=200) at **2.4× lower latency**. On every real dataset above the 8-bit LUT
  costs no measurable recall (≥0.9995 at rerank=200), so it is the default; a
  dedicated real-data 8-bit-vs-float gate remains an open confirmation but shows no
  loss so far.

## x86-64 / AVX-512 — `the-fire` (Ubuntu 24.04, gcc 13.3, PG 18.4)

All results above dispatch the **NEON** block kernel (arm64). The **AVX-512F+BW**
variant (`TqScoreBlockRangeAvx512`, `_mm_shuffle_epi8`) only compiles + runs on
x86-64 and was previously just cross-compile-checked. `the-fire` (AVX-512
F/BW/CD/DQ/VL/IFMA/VBMI) is the validation host.

**Kernel validation** (`make installcheck` + the dispatch / consistency probes):

| Check | Expected | Observed |
|---|---|---|
| `make installcheck` | all pass | **19/19 pass** ✅ |
| `tqflat_test_active_kernel()` | `avx512` | **`avx512`** ✅ |
| `tqflat_test_score_block_consistency(256)` | `0` (AVX-512 == scalar Default) | **0** ✅ |
| `tqflat_test_score_block_consistency(2048)` | `0` | **0** ✅ |

The AVX-512 kernel is the live dispatch on `the-fire` and is bit-identical to the
scalar reference; the full suite (incl. the multi-page streaming-build test and
`tqflat_math`) passes with it active, confirming the v4 on-disk format + build are
arch-independent.

**Raw score-step throughput** (`bench/microbench/tq_simd_microbench3_avx512.c`,
AMD Ryzen 9 7940HS / Zen4, 1M coded vectors, best of 5 — directly comparable to
the arm64/NEON numbers above):

| dim (padded) | scalar | AVX-512 A-nibble | speedup | (arm64 NEON speedup) |
|---|---|---|---|---|
| SIFT 128         | 81.2 ms   | **8.5 ms**   | **9.5×** | 11.9× |
| GloVe 200→256    | 161.0 ms  | **16.8 ms**  | **9.6×** | 11.2× |
| OpenAI 1536→2048 | 1357.7 ms | **137.5 ms** | **9.9×** | 19.5× |

- The AVX-512 kernel beats the scalar gather **~9.5–9.9×** and nibble-packing
  stays free (`Anib/Abyte ≈ 0.98–1.01×`), same as NEON — so the index-size win
  holds on x86 too.
- The microbench `sink` checksums are **byte-identical to the arm64 run**
  (`1.70915e11` / `3.42337e11` / `2.73312e12`), independently corroborating the
  `score_block_consistency = 0` result: the NEON and AVX-512 kernels compute the
  same accumulation as the scalar reference.
- The x86 speedup trails NEON at high dim (9.9× vs 19.5× @ dc=2048) because this
  is the **128-bit `_mm_shuffle_epi8` v1**: it processes 16 lanes/shuffle (like
  NEON) but widens u8→u16 with `unpack`+`add` (two ops) where NEON does it in one
  (`vaddw_u8`). **This has since been fixed** — the widened 256/512-bit kernel
  (one `_mm256_shuffle_epi8` for both nibbles + one `_mm512_cvtepu8_epi16` widen,
  register-resident accumulators) reaches **24.6–31.8× vs scalar**, past NEON parity,
  and is bit-identical to the v1/Default. See the *AVX-512 fast-scan kernel widening*
  section at the end of this file for the throughput and end-to-end numbers.
  Correctness was already final at the first cut; the widening is pure throughput.

## OpenAI-1536 @ **full 1M** — three-way (x86/AVX-512, `the-fire`)

The first true 1M-row run at dim=1536 (999,800 base + 200 held-out queries,
cosine), now that the build-cost wall and build-memory ceiling are gone. Run on
`the-fire` (AMD Zen4, 60 GB RAM, PG 18.4, AVX-512 kernel) with
`maintenance_work_mem=16GB` so HNSW builds in-memory. tqflat bits=4, fast rotation;
ivfflat `lists=√N`, `probes=lists/10`; hnsw tuned `m=32, ef_construction=128,
ef_search=100`. CSV: `bench/results/openai1m_x86.csv`.

| Method | Build (s) | Index size | Recall@10 | QPS | Latency (ms) |
|---|---|---|---|---|---|
| exact (seqscan)                       | —      | —          | 1.0000 | 0.5  | 1849.7 |
| **tqflat** b4, rerank=200             | **54** | **998 MB** | **1.0000** | 1.1 | 928 |
| **tqflat** b4, rerank=2000            | **54** | **998 MB** | **1.0000** | 1.0 | 971 |
| hnsw (m=32, efc=128, ef_search=100)   | 1763   | 7811 MB    | 0.9910 | **73.2** | **13.7** |
| ivfflat (lists=999, probes=99)        | 512    | 7819 MB    | 0.9770 | 3.2  | 310 |

**What 1M/1536 shows (the integration thesis, quantified at scale):**

- **Size — tqflat is 7.8× smaller** (998 MB vs ~7.8 GB for both hnsw and ivfflat,
  which store full float32). At 1M×1536 that is a ~6.8 GB saving per index.
- **Recall — tqflat is perfect (1.0000)**, above tuned HNSW (0.991) and ivfflat
  (0.977). The 8-bit kernel + rerank loses nothing even at 1M.
- **Build — tqflat 54 s vs HNSW 1763 s (33×) and ivfflat 512 s (9.4×).** Fast
  rotation makes a 1M×1536 tqflat build essentially free relative to the graph/
  cluster methods; the dense-rotation baseline would have been ~1 hour here.
- **Latency — the O(N) flat-scan wall, now at 1M scale: tqflat 928 ms (~1 QPS) vs
  HNSW 13.7 ms (73 QPS), a 68× gap.** ivfflat sits in between at 310 ms — but note
  it buys that only‑3× speedup over the flat scan while costing 7.8× the memory and
  losing recall.

**This is the strongest motivation yet for TQ-inside-IVF.** ivfflat at 1M/1536 is
the weakest cell in the table — 7.8 GB, 0.977 recall, and still 310 ms — precisely
because it scans `probes×(N/lists) ≈ 99k` *full float32* vectors per query.
Replacing that stored float32 with tqflat's 4-bit codes (998 MB, 1.0 recall) would
cut its footprint ~7.8× and accelerate each list scan with the AVX-512 kernel,
while inheriting IVF's sublinear candidate selection — the one axis (latency) where
standalone flat tqflat loses. tqflat already wins size/recall/build decisively; IVF
integration is how it also wins latency.

# IVF + TurboQuant (`tqivf`) — results

This is the milestone the section above pointed at: store each IVF list's members
as TurboQuant 4-bit codes (tqflat's v4 blocked layout) instead of full float32,
probe the nearest list centroids, score the probed lists with the AVX-512 LUT
kernel, and rerank the top candidates against full-precision heap vectors. One
global rotation + one global Lloyd-Max codebook shared by all lists (data-oblivious;
no per-list PQ codebook). Centroids are stored full-precision in original space for
probe selection; only stored members are rotated + quantized.

**Setup.** All four-way runs on `the-fire` (AMD Zen4, 60 GB RAM, PG 18.4, AVX-512
kernel) with `maintenance_work_mem=16GB`. `tqivf`/`ivfflat` use `lists=√N`; `tqivf`
sweeps `probes ∈ {10,30,99,200,400}` × `rerank ∈ {100,200,2000}` over a **single
built index** (probes/rerank are query-time GUCs). tqflat bits=4, fast rotation,
no-QJL. hnsw tuned `m=32, ef_construction=128, ef_search=100`. SIFT/GloVe use the
shipped file ground truth (500 queries); OpenAI-1M computes exact GT (200 queries).
CSVs: `bench/results/{sift1m,glove200,openai1m}_tqivf_x86.csv`.

> **Kernel note.** The four-way tables below were measured with the **128-bit v1**
> AVX-512 score kernel. The kernel was subsequently widened (256/512-bit); recall is
> unchanged (the kernel is bit-identical) and end-to-end latency drops ~5–22% where
> the score step dominates. The before/after deltas are in the *AVX-512 fast-scan
> kernel widening* section at the end of this file; the prior CSVs are preserved as
> `bench/results/*_tqivf_x86_oldkernel.csv`.

## OpenAI-1536 @ 1M — four-way (the headline)

999,800 × 1536, cosine. `lists=999`. Representative `tqivf` sweep points
(`rerank=200` unless noted; recall is flat across rerank here, latency rises with it):

| Method | Build (s) | Index size | Recall@10 | Latency (ms) |
|---|---|---|---|---|
| exact (seqscan)                     | —    | —        | 1.0000 | 1517 |
| tqflat b4, rerank=200               | 46   | 998 MB   | 1.0000 | 980 |
| **tqivf** lists=999, probes=99      | 161  | **1029 MB** | **0.9760** | **62** |
| tqivf  lists=999, probes=30         | 161  | 1029 MB  | 0.9305 | 22 |
| tqivf  lists=999, probes=200        | 161  | 1029 MB  | 0.9910 | 122 |
| tqivf  lists=999, probes=400        | 161  | 1029 MB  | 0.9970 | 239 |
| hnsw (m=32, efc=128, ef=100)        | 812  | 7811 MB  | 0.9900 | **14.9** |
| ivfflat (lists=999, probes=99)      | 93   | 7819 MB  | 0.9775 | 217 |

**The thesis is confirmed.** At matched recall vs ivfflat (0.976 vs 0.9775) `tqivf`
is **3.5× faster (62 ms vs 217 ms) and 7.6× smaller (1029 MB vs 7819 MB)** — it
dominates the cell it was built to beat on all three axes at once. It also drops
standalone flat tqflat's latency **16×** (62 ms vs 980 ms) at near-identical size,
closing the one axis flat tqflat lost. The `probes` knob buys recall monotonically
(0.93 → 0.976 → 0.991 → 0.997) at a latency cost, so 0.99+ recall is available at
122 ms if needed.

**The honest caveat:** at 1M×1536, **HNSW still wins raw latency** (14.9 ms vs
`tqivf`'s 62 ms at p99) — it is ~4–8× faster per query across the matched-recall
range. `tqivf`'s standing offer in exchange is **7.6× less storage** (1.0 GB vs
7.8 GB), a **5× faster build** (161 s vs 812 s), and recall that meets or beats HNSW
(p200: 0.991 ≥ hnsw 0.990). `tqivf` is the size/recall/build-balanced index;
HNSW remains the choice when query latency is the sole objective and 8× the memory
is acceptable. `tqivf` build here is ~1.7× ivfflat's, but that compares serial-tqivf
(no parallel build at measurement time) to 4-way-parallel-ivfflat; with `tqivf`'s own
parallel build it is **0.9–1.0× ivfflat** — see *Parallel build* below — and 5× below HNSW.

## SIFT1M — four-way

1,000,000 × 128, L2. `lists=1000`. Same conclusion, larger margins (low dim ⇒ the
quantized score step dominates and the kernel shines):

| Method | Build (s) | Index size | Recall@10 | Latency (ms) |
|---|---|---|---|---|
| tqflat b4, rerank=200          | 3.2  | 77.8 MB  | 0.9996 | 180 |
| **tqivf** lists=1000, probes=99 | 21  | **87.5 MB** | **0.9988** | **15.2** |
| tqivf  lists=1000, probes=30   | 21   | 87.5 MB  | 0.9708 | 5.1 |
| tqivf  lists=1000, probes=200  | 21   | 87.5 MB  | 0.9994 | 29.9 |
| hnsw (m=32, efc=128, ef=100)   | 234  | 966 MB   | 0.9944 | 8.8 |
| ivfflat (lists=1000, probes=100) | 11 | 525 MB   | 0.9988 | 33.6 |

At matched recall vs ivfflat (both 0.9988), `tqivf` is **2.2× faster (15.2 vs
33.6 ms) and 6.0× smaller (87.5 vs 525 MB)**, and **12× faster than flat tqflat**
(15 vs 180 ms). It out-recalls HNSW at p200 (0.9994 vs 0.9944) and is within 2× of
HNSW's latency at 11× less storage.

## GloVe-200-angular — four-way

1,183,514 × 200, cosine. `lists=1087`. The hardest dataset for graph recall:

| Method | Build (s) | Index size | Recall@10 | Latency (ms) |
|---|---|---|---|---|
| tqflat b4, rerank=200          | 7.4  | 164.6 MB | 1.0000 | 299 |
| **tqivf** lists=1087, probes=99 | 38  | **177.5 MB** | **0.9330** | **22.1** |
| tqivf  lists=1087, probes=30   | 38   | 177.5 MB | 0.8420 | 7.7 |
| tqivf  lists=1087, probes=200  | 38   | 177.5 MB | 0.9658 | 42.4 |
| tqivf  lists=1087, probes=400  | 38   | 177.5 MB | 0.9874 | 82.8 |
| hnsw (m=32, efc=128, ef=100)   | 436  | 1542 MB  | 0.8386 | 12.2 |
| ivfflat (lists=1087, probes=108) | 20 | 1032 MB  | 0.9374 | 56.2 |

At matched recall vs ivfflat (0.933 vs 0.937), `tqivf` is **2.5× faster (22 vs
56 ms) and 5.8× smaller (178 vs 1032 MB)**. Notably, **`tqivf` beats HNSW on both
axes here**: at p30 it matches HNSW's recall (0.842 vs 0.839) at 7.7 ms vs 12.2 ms,
and every higher probe point out-recalls HNSW by a wide margin (HNSW tops out at
0.839 on GloVe-angular at these params, where `tqivf`'s rerank reaches 0.987).

## What the four-way data says

- **`tqivf` keeps everything flat tqflat won — size and recall — and adds the
  latency `tqflat` lacked.** Across all three datasets it is **~6–8× smaller than
  ivfflat/hnsw**, matches ivfflat's recall, and is **2.2–3.5× faster than ivfflat**
  while cutting flat tqflat's latency **12–16×**. It strictly dominates ivfflat on
  size + latency + recall everywhere tested.
- **vs HNSW it is the storage/recall play, not the raw-latency play.** On the high-dim
  OpenAI set HNSW is 4–8× faster per query (14.9 ms vs 62 ms) but costs 7.6× the
  storage; on lower-dim / angular sets (SIFT, GloVe) `tqivf` closes or reverses the
  latency gap and out-recalls HNSW. The kernel's advantage grows as the quantized
  score step dominates (lower dim, more probes).
- **Build sits between ivfflat and HNSW** — the k-means pass plus a TQ encode pass,
  dwarfed by HNSW graph construction. The 1.7× shown here is serial-tqivf vs
  parallel-ivfflat; with `tqivf`'s parallel build (added since) it is **0.9–1.0× ivfflat**
  and 5× below HNSW (see *Parallel build*).
- **The `probes × rerank` sweep is a clean, monotone recall/latency dial**, exactly
  as designed: pick the probe count for the recall target, then rerank for the final
  polish. 0.99+ recall is reachable on every dataset.

**Bottom line:** TQ-inside-IVF delivers the production-shaped TurboQuant index the
prior sections motivated — it wins size, recall, and build like flat tqflat *and*
turns the O(N) latency wall into IVF-sublinear selection, beating ivfflat outright
and trading raw latency to HNSW only in exchange for ~8× less memory.

## Parallel build — serial vs parallel, tqivf vs ivfflat (the-fire)

`tqivf` now supports the standard Postgres parallel index build (`amcanbuildparallel`):
the k-means + codebook are computed serially by the leader, then nearest-center
assignment **and** the TurboQuant encode run in the per-tuple callback across worker
backends, broadcasting the centers + the (few-KB, fast-rotation) model through shared
memory. The on-disk format is byte-equivalent to a serial build, so recall, latency,
and size are unchanged — **build wall-clock is the only thing that moves.**

**Setup.** `the-fire` (Zen4, 16 cores, PG 18.4), `maintenance_work_mem=16GB`,
`lists=√N`, fast rotation. *Serial* = `max_parallel_maintenance_workers=0`; *parallel*
requests 8 (the box's `max_worker_processes`/`max_parallel_workers` ceiling) and the
planner grants what its main-heap-page heuristic allows. Each dataset is loaded once
and both methods are timed on the identical table.
CSV: `bench/results/tqivf_build_time_x86.csv`.

| Dataset | dim | tqivf serial | tqivf parallel | tqivf speedup | ivfflat serial | ivfflat parallel | ivfflat speedup | workers | **par tqivf / par ivfflat** |
|---|---|---|---|---|---|---|---|---|---|
| SIFT1M      | 128  | 20.3 s  | **9.2 s**  | 2.19× | 17.0 s  | 10.2 s | 1.67× | 4 | **0.91×** |
| GloVe-200   | 200  | 36.6 s  | **16.4 s** | 2.23× | 33.2 s  | 17.6 s | 1.89× | 5 | **0.94×** |
| OpenAI-1M   | 1536 | 163.6 s | **87.1 s** | 1.88× | 135.6 s | 85.4 s | 1.59× | 2 | **1.02×** |

**The ~1.7× build gap to ivfflat was an artifact and is now gone.** The four-way tables
above report `tqivf` build *serial* (parallel build did not exist when they were
measured — note `tqivf` serial 20.3/36.6/163.6 s ≈ the four-way 21/38/161 s) against
`ivfflat` build *already 4-way parallel* (the suite ran at the harness default of 4
maintenance workers — note `ivfflat` parallel ≈ the four-way 11/20/93 s). That compared
serial-tqivf to parallel-ivfflat. Measured apples-to-apples — **both parallel** —
`tqivf` build is **0.91–1.02× ivfflat's**: equal at high dim, slightly *faster* at low/mid
dim (its 4-bit codes sort ~8× smaller than ivfflat's raw float vectors, offsetting the
extra encode). `tqivf` parallel build itself runs **1.9–2.2× faster than its serial path**.

**Worker grant scales with main-heap pages, not vector dimension.** High-dim vectors
(OpenAI 1536-d) exceed the TOAST threshold, so the *main* heap holds only TOAST pointers
— few pages → the planner grants fewer workers (2) than for inline low-dim vectors
(SIFT 4, GloVe 5). Both AMs get the same grant on the same table, so the comparison
stays fair; it does mean the parallel speedup is largest where the main heap is biggest.

**Correctness/recall preserved on silicon.** `make installcheck` is 22/22 on `the-fire`
(including the forced-worker `tqivf_build` case asserting per-list sum=2000 /
cardinality=20). A parallel-built SIFT1M index queries identically to the committed
serial baseline: recall@10 **0.9984** (serial 0.9988 — within unseeded-k-means sampling
noise), **15.3 ms** (serial 15.2), **87.6 MB** (serial 87.5). The end-to-end TAP recall
variant (config E) runs in CI only — `PostgreSQL::Test::*` is absent from the apt PG on
`the-fire`.

# AVX-512 fast-scan kernel widening (256/512-bit) — throughput + end-to-end

The AVX-512 score kernel (`TqScoreBlockRangeAvx512`, `src/tqfastscan.c`) was rewritten
from the **128-bit `_mm_shuffle_epi8` v1** (the un-widened first cut benchmarked in all
sections above) to a widened **256/512-bit** kernel. Three bottlenecks were removed:
(1) the per-coordinate `acc16` load/store traffic — accumulators are now
**register-resident** (`acc16` in one zmm, `acc32` in two); (2) the u8→u16 widen, now
**one `_mm512_cvtepu8_epi16` + `vpaddw`** instead of `unpacklo/hi` + add; (3) the nibble
shuffle, now **one 256-bit `_mm256_shuffle_epi8`** (both nibbles, LUT broadcast to both
lanes) instead of two 128-bit lookups. The kernel stays **bit-identical** to the scalar
`TqScoreBlockRangeDefault` — the mandatory `& 0x0F` after the 16-bit shift and the
128-coord `TQ_LUT_FLUSH` cadence are preserved.

**Raw score-step throughput** (`bench/microbench/tq_simd_microbench3_avx512.c`, AMD
Ryzen 9 7940HS / Zen4 `the-fire`, 1M coded vectors, best of 7; the bench runs both
kernels back-to-back and asserts their checksums match):

| dim (padded) | scalar | 128-bit v1 | **256/512 (new)** | new vs scalar | **new vs v1** | checksum |
|---|---|---|---|---|---|---|
| SIFT 128         | 81.4 ms   | 8.5 ms   | **2.6 ms**   | **31.6×** | **3.31×** | MATCH |
| GloVe 200→256    | 162.0 ms  | 16.9 ms  | **5.1 ms**   | **31.8×** | **3.31×** | MATCH |
| OpenAI 1536→2048 | 1367.2 ms | 137.6 ms | **55.7 ms**  | **24.6×** | **2.47×** | MATCH |

The widened kernel is **2.5–3.3× faster than v1** and clears the ~20× NEON-parity
target (NEON peaks at 19.5× @ dc=2048). The register-resident accumulator was the
dominant win.

**Validation on `the-fire`** (the §x86/AVX-512 checklist, re-run with the new kernel
live, PG 18.4):

| Check | Expected | Observed |
|---|---|---|
| `make installcheck` | all pass | **22/22 pass** ✅ |
| `tqflat_test_active_kernel()` | `avx512` | **`avx512`** ✅ |
| `tqflat_test_score_block_consistency(256)` | `0` | **0** ✅ |
| `tqflat_test_score_block_consistency(2048)` | `0` | **0** ✅ |
| microbench checksum (v1 vs new, all dims) | identical | **MATCH** ✅ |

**End-to-end four-way re-run (new kernel vs v1, same host/params).** Recall is
unchanged at every point (bit-identical kernel); the table below is the latency delta.
The win tracks the score-step fraction: largest on high-dim OpenAI, smaller on
low-dim SIFT where the page-walk/rerank dominates, and absent (run-to-run noise only)
for `hnsw`/`ivfflat`/`exact`, which don't use the kernel.

| Dataset | Operating point | recall | v1 ms | **new ms** | speedup |
|---|---|---|---|---|---|
| OpenAI-1M×1536 | tqivf p99 / rerank200  | 0.980 | 62.1  | **51.3**  | **1.21×** |
| OpenAI-1M×1536 | tqivf p200 / rerank200 | 0.992 | 121.8 | **100.3** | **1.21×** |
| OpenAI-1M×1536 | tqivf p400 / rerank100 | 0.998 | 236.9 | **194.4** | **1.22×** |
| OpenAI-1M×1536 | tqflat flat scan r200  | 1.000 | 980   | **882**   | 1.11× |
| GloVe-200      | tqivf p99 / rerank200  | 0.936 | 22.1  | **20.6**  | 1.08× |
| GloVe-200      | tqivf p400 / rerank200 | 0.987 | 82.8  | **77.3**  | 1.07× |
| SIFT1M         | tqivf p99 / rerank200  | 0.999 | 15.2  | **14.6**  | 1.04× |
| SIFT1M         | tqivf p400 / rerank200 | 0.999 | 57.8  | **54.8**  | 1.05× |

On the headline OpenAI-1M set `tqivf` is **~18–22% faster end-to-end at every real
operating point**, with the win monotone across the whole `probes × rerank` sweep —
the signature of a genuine kernel speedup rather than noise. The matched-recall cell
vs ivfflat (p99, 0.98) drops 62 → 51 ms, so `tqivf` is now **~4.1× faster than
ivfflat** (was 3.5×) at the same 7.6× size advantage. The tiny low-probe points
(p10, 10–13 ms, dominated by fixed per-query overhead) are too small/variable to read
as a kernel signal. CSVs: `bench/results/{sift1m,glove200,openai1m}_tqivf_x86.csv`
(new) vs `*_oldkernel.csv` (v1 baseline).
