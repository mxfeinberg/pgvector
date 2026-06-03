import datasets as D
import tq_numpy as TQ
import traversal as T
import scoring as S


def test_variant_full_and_A_sift():
    d = D.load("sift1m", n_queries=100)
    m = TQ.TQModel(d["base"].shape[1])
    rec = S.quantized_search_vectors(m.reconstruct(d["base"]), d["metric"])
    full = T.variant_full(d["base"], d["queries"], d["gt"], d["metric"], efs=(100,))
    A = T.variant_A_all_quantized(d["base"], rec, d["queries"], d["gt"], d["metric"], efs=(100,))
    assert 0.0 <= full[100] <= 1.0 and 0.0 <= A[100] <= 1.0
    # full-precision HNSW at ef=100 on 200k SIFT should be strong
    assert full[100] >= 0.95, full


def test_variant_C_sift():
    d = D.load("sift1m", n_queries=100)
    C = T.variant_C_faiss_sq(d["base"], d["queries"], d["gt"], d["metric"], efs=(100,))
    assert 0.0 <= C[100] <= 1.0
