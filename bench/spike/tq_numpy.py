"""TurboQuant-b4 quantizer in numpy (rotation + 4-bit scalar codebook + norm/scale).

Distortion is rotation-invariant in distribution, so a generic random orthogonal
rotation is statistically equivalent to the shipped seeded fast-Hadamard for aggregate
recall.  Per-vector storage mirrors the C layout: an L2 norm, a per-vector scale, and
b=4 codes against a unit-variance Lloyd-Max codebook.
"""
import numpy as np

BITS = 4
LEVELS = 1 << BITS  # 16
SEED = 1234


def random_rotation(dim, seed=SEED):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((dim, dim)).astype(np.float64)
    q, r = np.linalg.qr(a)
    q *= np.sign(np.diag(r))  # make deterministic
    return q.astype(np.float32)


def fit_codebook(samples, levels=LEVELS, iters=50):
    """1-D Lloyd-Max on unit-variance samples -> (centroids[levels], edges[levels-1])."""
    s = np.sort(samples.astype(np.float64))
    # init centroids at quantiles
    cents = np.quantile(s, (np.arange(levels) + 0.5) / levels)
    for _ in range(iters):
        edges = (cents[:-1] + cents[1:]) / 2.0
        idx = np.searchsorted(edges, s)
        for j in range(levels):
            m = idx == j
            if m.any():
                cents[j] = s[m].mean()
    edges = (cents[:-1] + cents[1:]) / 2.0
    return cents.astype(np.float32), edges.astype(np.float32)


class TQModel:
    def __init__(self, dim, seed=SEED):
        self.dim = dim
        self.R = random_rotation(dim, seed)            # (d,d) orthogonal
        rng = np.random.default_rng(seed + 1)
        # fit codebook on the post-rotation unit-variance coordinate distribution
        probe = rng.standard_normal((4000, dim)).astype(np.float32)
        probe /= np.linalg.norm(probe, axis=1, keepdims=True)
        rot = probe @ self.R.T
        rot /= rot.std()
        self.cents, self.edges = fit_codebook(rot.ravel())

    def encode(self, X):
        """X: (n,d).  Returns (codes uint8 (n,d), norm (n,), scale (n,))."""
        norm = np.linalg.norm(X, axis=1)
        safe = np.where(norm == 0, 1.0, norm)
        u = X / safe[:, None]
        rot = u @ self.R.T
        scale = rot.std(axis=1)
        scale = np.where(scale == 0, 1.0, scale)
        z = rot / scale[:, None]
        codes = np.searchsorted(self.edges, z).astype(np.uint8)
        return codes, norm, scale

    def dequantize(self, codes, norm, scale):
        """Inverse of encode: returns reconstructed vectors (n,d)."""
        z = self.cents[codes]                  # (n,d)
        rot = z * scale[:, None]
        u = rot @ self.R                        # R orthogonal => inverse is R^T applied as @R
        return u * norm[:, None]

    def reconstruct(self, X):
        return self.dequantize(*self.encode(X))
