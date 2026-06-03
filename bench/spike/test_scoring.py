import datasets as D
import tq_numpy as TQ
import scoring as S


def test_scoring_smoke_sift():
    d = D.load("sift1m", n_queries=50)
    m = TQ.TQModel(d["base"].shape[1])
    rec = S.quantized_search_vectors(m.reconstruct(d["base"]), d["metric"])
    curve = S.overfetch_curve(d["base"], rec, d["queries"], d["gt"], d["metric"])
    # monotone non-decreasing in C, and rerank>=200 should be strong
    vals = [curve[c] for c in (0, 10, 50, 200)]
    assert vals == sorted(vals), curve
    assert curve[200] >= 0.95, curve
    fid = S.near_neighbor_rank_fidelity(d["base"], rec, d["queries"], d["metric"])
    assert -1.0 <= fid["kendall_tau"] <= 1.0
