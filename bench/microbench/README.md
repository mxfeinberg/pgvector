# tqflat SIMD-kernel layout microbenches

Throwaway diagnostics that drove the **data-layout decision** for the tqflat
SIMD LUT scan kernel (FAISS "Fast Scan" / Quick-ADC style). Not part of the
extension build; arm64/NEON only. They score N stored 4-bit-coded vectors the way
a per-query scan does, comparing layouts.

```sh
cc -O3 -ffast-math -o /tmp/mb1 tq_simd_microbench1_transpose.c -lm && /tmp/mb1 2048 1048576 5
cc -O3 -ffast-math -o /tmp/mb2 tq_simd_microbench2_nibble.c    -lm && /tmp/mb2 2048 1048576 5
# args: <dc padded-dim> <N vectors> <reps>; reports best-of-reps ms per layout.
```

## What they compare

- **scalar** — faithful copy of `TqScoreEntryDefault`: per-coordinate unpack-4-bit
  + float-LUT gather. The current production path.
- **B (microbench 1)** — codes row-major on disk (today's format); transpose+unpack
  a 32-vector block into a scratch buffer *at scan time*, then NEON `vqtbl1q_u8`.
- **A-byte (microbench 1 & 2)** — codes stored blocked/transposed, 1 byte/code
  (2× index size); time only the NEON kernel. Pure-kernel ceiling.
- **A-nibble (microbench 2)** — codes stored blocked AND 4-bit nibble-packed
  (**same index size as today**), nibble split inside the kernel. The shippable
  Option A.

## Result (Apple M-series, 1M vectors, best of 5)

| dim (padded) | scalar | B transpose-on-scan | A-nibble (shippable) |
|---|---|---|---|
| OpenAI 1536→2048 | 1037 ms | ~2.1× | 50 ms (**20.7×**) |
| GloVe 200→256    |  103 ms | ~1.7× | 6.3 ms (**16.3×**) |
| SIFT 128         |   44 ms | ~1.5× | 3.2 ms (**14.2×**) |

## Conclusion → Option A (blocked, nibble-packed, on-disk)

1. **B is a dead end (~1.5–2.1×).** The transpose is *query-independent* work; doing
   it per-query inside the scan re-pays a scalar-cost unpack every query, capping the
   win near 2×. It belongs at build time — which *is* Option A. (This independently
   rediscovers why FAISS stores blocked on disk.)
2. **A delivers 14–21× on the score step**, an order of magnitude over B.
3. **The nibble split is free** (`Anib/Abyte ≈ 0.97–1.00×`): the 4-bit-packed blocks
   are half the bytes, and the high-dim scan is bandwidth-bound, so halved memory
   traffic hides the mask+shift. **Index size is unchanged from today** — the 5–11×
   compression advantage is fully preserved.

### Caveats (carried into the build milestone)
- These time only the **score step**. Flat tqflat end-to-end is also bounded by
  Postgres page-walk + top-k + rerank, so a perfect kernel won't make *flat* tqflat
  HNSW-competitive alone. The kernel's full payoff is in **IVF + TurboQuant**, where
  the score step dominates.
- Real kernel uses an **8-bit-quantized LUT** (as fast-scan does); recall impact vs
  the float LUT needs an A/B during the build (rerank recovers the remainder; the
  `tqflat_math` self-score tests guard numeric consistency).
