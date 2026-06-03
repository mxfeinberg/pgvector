import datasets as D
import tq_numpy as TQ
import traversal as T
import scoring as S


def test_variant_B_runs_and_is_reasonable():
    d = D.load("sift1m", n_queries=100)
    m = TQ.TQModel(d["base"].shape[1])
    rec = S.quantized_search_vectors(m.reconstruct(d["base"]), d["metric"])
    full = T.variant_full(d["base"], d["queries"], d["gt"], d["metric"], efs=(200,))
    B = T.variant_B_build_full_traverse_quant(d["base"], rec, d["queries"], d["gt"],
                                              d["metric"], efs=(200,))
    assert 0.0 <= B[200] <= 1.0, B
    # B reranks from true vectors over a full-precision-built graph, so on low-dim SIFT
    # it should land within striking distance of full (sanity, not the verdict).
    assert B[200] >= 0.85 * full[200], (B, full)
