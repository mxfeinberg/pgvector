import numpy as np
import datasets as D


def test_sift_subset_shapes_and_gt():
    d = D.load("sift1m", n_queries=50)
    assert d["base"].shape == (200_000, 128)
    assert d["queries"].shape == (50, 128)
    assert d["gt"].shape == (50, 10)
    # GT row 0's nearest must be a valid index and distinct
    assert d["gt"][0].min() >= 0 and d["gt"][0].max() < 200_000
    assert len(set(d["gt"][0].tolist())) == 10
