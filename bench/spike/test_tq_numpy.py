import numpy as np
import tq_numpy as TQ


def _unit(n, d, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_rotation_orthogonal():
    R = TQ.random_rotation(64)
    assert np.allclose(R @ R.T, np.eye(64), atol=1e-4)


def test_roundtrip_shapes_and_codes():
    m = TQ.TQModel(128)
    X = _unit(500, 128)
    codes, norm, scale = m.encode(X)
    assert codes.shape == (500, 128)
    assert codes.max() < TQ.LEVELS and codes.min() >= 0
    rec = m.dequantize(codes, norm, scale)
    assert rec.shape == X.shape


def test_reconstruction_mse_near_paper_b4():
    # Paper: per-coordinate MSE ~0.009 for b=4 on unit vectors (coords ~ N(0,1/d)).
    m = TQ.TQModel(256)
    X = _unit(2000, 256)
    rec = m.reconstruct(X)
    # total squared error per vector ~ d * 0.009 * (1/d) scale; check normalized recon error
    err = np.linalg.norm(X - rec, axis=1) ** 2
    assert 0.002 < err.mean() < 0.05, err.mean()


def test_quantized_distance_rank_correlates():
    # On a unit-vector set, quantized L2 ranking should broadly track true ranking.
    m = TQ.TQModel(128)
    base = _unit(3000, 128, seed=1)
    rec = m.reconstruct(base)
    q = _unit(1, 128, seed=2)[0]
    dt = np.linalg.norm(base - q, axis=1)
    dq = np.linalg.norm(rec - q, axis=1)
    # top-10 by quantized should overlap true top-10 substantially
    overlap = len(set(np.argsort(dt)[:10]) & set(np.argsort(dq)[:10]))
    assert overlap >= 5, overlap
