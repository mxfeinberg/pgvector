import numpy as np
import datasets as D
import tq_numpy as TQ
import scoring as S

# C tqflat rerank=200 recall@10 (full 1M, bench/RESULTS.md). Subset 200k is easier or
# comparable; require numpy-TQ to be within 0.03 below these and >= 0.97 absolute.
C_TQFLAT = {"sift1m": 0.9996, "glove200": 1.0000, "openai1m": 1.0000}


def _gate(key, n_queries=200):
    d = D.load(key, n_queries=n_queries)
    m = TQ.TQModel(d["base"].shape[1])
    rec = S.quantized_search_vectors(m.reconstruct(d["base"]), d["metric"])
    r = S.bruteforce_rerank_recall(d["base"], rec, d["queries"], d["gt"],
                                   d["metric"], rerank=200, k=10)
    assert r >= min(0.97, C_TQFLAT[key] - 0.03), f"{key}: numpy-TQ recall {r:.4f}"
    return r


def test_gate_sift():    print("sift",   _gate("sift1m"))
def test_gate_glove():   print("glove",  _gate("glove200"))
def test_gate_openai():  print("openai", _gate("openai1m"))
