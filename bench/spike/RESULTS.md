# TQ + HNSW viability spike - results

200k subset per dataset; HNSW M=32 efC=128. Verdict bands are advisory (spec section 7); raw numbers below are the deliverable. Quantized distance uses distance-to-reconstruction, a conservative proxy for the asymmetric ADC.

## sift1m (128d, l2) - **GREEN**

- numpy-TQ gate (rerank=200 brute force): **1.0000**
- near-neighbor rank fidelity: Kendall-tau=0.730, Spearman=0.894
- over-fetch curve recall@10: C=0:0.830, C=10:0.830, C=20:0.984, C=50:1.000, C=100:1.000, C=200:1.000, C=500:1.000
- full-precision baseline recall@10 (ef=100): 0.9965
- min efSearch reaching 0.98xbaseline (best of A/B): 50

| efSearch | full | A all-quant | B build-full | C faiss-SQ |
|---|---|---|---|---|
| 50 | 0.9865 | 0.9840 | 0.9810 | 0.9745 |
| 100 | 0.9965 | 0.9980 | 0.9970 | 0.9865 |
| 200 | 0.9995 | 0.9990 | 0.9990 | 0.9880 |
| 400 | 0.9995 | 1.0000 | 1.0000 | 0.9885 |
| 800 | 1.0000 | 1.0000 | 1.0000 | 0.9885 |

## glove200 (200d, cosine) - **GREEN**

- numpy-TQ gate (rerank=200 brute force): **1.0000**
- near-neighbor rank fidelity: Kendall-tau=0.768, Spearman=0.916
- over-fetch curve recall@10: C=0:0.880, C=10:0.880, C=20:0.993, C=50:1.000, C=100:1.000, C=200:1.000, C=500:1.000
- full-precision baseline recall@10 (ef=100): 0.8315
- min efSearch reaching 0.98xbaseline (best of A/B): 100

| efSearch | full | A all-quant | B build-full | C faiss-SQ |
|---|---|---|---|---|
| 50 | 0.7590 | 0.7645 | 0.7615 | 0.7585 |
| 100 | 0.8315 | 0.8220 | 0.8390 | 0.8280 |
| 200 | 0.8950 | 0.8940 | 0.8880 | 0.8830 |
| 400 | 0.9460 | 0.9400 | 0.9390 | 0.9225 |
| 800 | 0.9765 | 0.9680 | 0.9700 | 0.9560 |

## openai1m (1536d, cosine) - **GREEN**

- numpy-TQ gate (rerank=200 brute force): **1.0000**
- near-neighbor rank fidelity: Kendall-tau=0.935, Spearman=0.992
- over-fetch curve recall@10: C=0:0.966, C=10:0.966, C=20:1.000, C=50:1.000, C=100:1.000, C=200:1.000, C=500:1.000
- full-precision baseline recall@10 (ef=100): 0.9900
- min efSearch reaching 0.98xbaseline (best of A/B): 50

| efSearch | full | A all-quant | B build-full | C faiss-SQ |
|---|---|---|---|---|
| 50 | 0.9810 | 0.9805 | 0.9795 | 0.9780 |
| 100 | 0.9900 | 0.9905 | 0.9900 | 0.9855 |
| 200 | 0.9930 | 0.9935 | 0.9935 | 0.9895 |
| 400 | 0.9955 | 0.9970 | 0.9970 | 0.9940 |
| 800 | 0.9965 | 0.9985 | 0.9970 | 0.9950 |
