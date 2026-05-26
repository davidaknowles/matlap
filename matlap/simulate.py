"""Simulation utilities for generating test data.

Provides :func:`sample_nnd` for sampling matrices from the Nuclear Norm
Distribution (NND) via its SVD representation.
"""

from __future__ import annotations

import numpy as np


def sample_nnd(
    rng: np.random.Generator,
    m: int,
    n: int,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a matrix from the Nuclear Norm Distribution via the SVD representation.

    The NND is defined by p(X) ∝ exp(-λ‖X‖_*).  Under the SVD X = U diag(σ) Vᵀ
    with m ≥ n, the joint density of the singular values is (see Appendix of the
    paper):

        p(σ₁,…,σₙ) ∝ ∏ᵢ σᵢ^(m-n) exp(-λσᵢ) · ∏_{i<j} (σᵢ²-σⱼ²)

    This sampler uses the **independent marginal approximation**: each σᵢ is
    drawn i.i.d. from Gamma(m-n+1, rate=λ) and sorted descending.  The
    Coulomb repulsion term ∏_{i<j}(σᵢ²-σⱼ²) is omitted, which is accurate
    for the individual marginals but ignores inter-singular-value correlations.

    For m = n the Gamma shape equals 1, giving σᵢ ~ Exp(λ): the maximum-spread
    case where singular values vary widely (coefficient of variation = 1).

    Left singular vectors U are drawn from the Stiefel manifold St(n, m) and
    right singular vectors V from the orthogonal group O(n), both via QR
    decomposition of random Gaussian matrices (Haar measure).

    Args:
        rng:  NumPy random generator.
        m:    Number of rows; must satisfy m ≥ n.
        n:    Number of columns.
        lam:  NND regularisation strength λ > 0.
              Mean singular value = (m-n+1)/λ.

    Returns:
        X:     Sampled matrix, shape (m, n).
        sigma: Singular values, shape (n,), in descending order.

    Raises:
        ValueError: If m < n or lam ≤ 0.

    Example::

        rng = np.random.default_rng(0)
        X, sigma = sample_nnd(rng, m=100, n=100, lam=0.05)
        # Square case: sigma ~ Exp(0.05), mean ≈ 20
    """
    if m < n:
        raise ValueError(f"Require m >= n, got m={m}, n={n}.")
    if lam <= 0:
        raise ValueError(f"lam must be positive, got {lam}.")

    k = n
    shape = m - n + 1  # Gamma shape; equals 1 for m=n (Exponential)

    sigma = rng.gamma(shape=shape, scale=1.0 / lam, size=k)
    sigma = np.sort(sigma)[::-1]  # descending

    # U: m×k orthonormal columns (Stiefel manifold St(n,m))
    U, _ = np.linalg.qr(rng.standard_normal((m, k)))
    # V: k×k orthogonal (O(n))
    V, _ = np.linalg.qr(rng.standard_normal((k, k)))

    X = (U * sigma[None, :]) @ V.T  # = U @ diag(sigma) @ V^T
    return X, sigma
