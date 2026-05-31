# tqflat large-scale benchmark results (standard ANN datasets)

**Environment:** PostgreSQL 18.4 (Homebrew), Apple M-series. Index builds use
`maintenance_work_mem=2GB` + parallel workers; HNSW queried at
`hnsw.ef_search=100` (pgvector's default 40 under-recalls on hard datasets).
200 queries per run, k=10. tqflat at `bits ∈ {2,4}`, `rerank ∈ {200, 2000}`.

Raw CSVs in `bench/results/`. Reproduce with the commands in `bench/README.md`.

> **Latency caveat:** tqflat distance scoring is the **scalar** `TqScoreEntryDefault`
> kernel — no SIMD variant exists yet. The paper's whole performance premise is a
> SIMD LUT scan (`pshufb`/`vqtbl`); the tqflat latencies below are therefore a
> pessimistic ceiling. Build/size/recall are unaffected.

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
