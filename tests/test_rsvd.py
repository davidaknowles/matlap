"""Tests for rsvd and approx_nuclear_norm."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from matlap.linalg import rsvd, approx_nuclear_norm

jax.config.update("jax_enable_x64", False)

RNG = np.random.default_rng(99)


def make_low_rank_matrix(m, n, rank, rng):
    U = rng.standard_normal((m, rank)).astype(np.float32)
    V = rng.standard_normal((n, rank)).astype(np.float32)
    return jnp.asarray(U @ V.T, dtype=jnp.float32)


# ---------------------------------------------------------------------------
# rsvd
# ---------------------------------------------------------------------------


def test_rsvd_shapes():
    """rsvd should return arrays with the right shapes."""
    rng = np.random.default_rng(0)
    A = jnp.asarray(rng.standard_normal((20, 15)), dtype=jnp.float32)
    rank = 5
    U_r, s_r, Vt_r = rsvd(A, rank)
    assert U_r.shape == (20, rank)
    assert s_r.shape == (rank,)
    assert Vt_r.shape == (rank, 15)


def test_rsvd_singular_values_descending():
    """Approximate singular values should be roughly descending."""
    A = make_low_rank_matrix(30, 20, rank=5, rng=RNG)
    _, s_r, _ = rsvd(A, rank=8)
    diffs = jnp.diff(s_r)
    assert jnp.all(diffs <= 0.5), f"Singular values not descending: {s_r}"


def test_rsvd_approximation_quality():
    """For a low-rank matrix, rsvd should recover top singular values accurately."""
    rng = np.random.default_rng(1)
    rank = 4
    A = make_low_rank_matrix(40, 30, rank=rank, rng=rng)

    # Full SVD for reference
    s_full = jnp.linalg.svd(A, compute_uv=False)

    _, s_approx, _ = rsvd(A, rank=rank, n_iter=6)
    # Top-rank singular values should be close
    np.testing.assert_allclose(s_approx, s_full[:rank], rtol=0.05, atol=0.1)


def test_rsvd_reconstruction_quality():
    """A @ Vt^T ≈ U Sigma should hold for a low-rank matrix."""
    rng = np.random.default_rng(2)
    rank = 3
    A = make_low_rank_matrix(25, 20, rank=rank, rng=rng)
    U_r, s_r, Vt_r = rsvd(A, rank=rank, n_iter=6)

    A_approx = (U_r * s_r) @ Vt_r
    rel_err = float(jnp.linalg.norm(A - A_approx) / jnp.linalg.norm(A))
    assert rel_err < 0.1, f"Reconstruction relative error too high: {rel_err:.4f}"


def test_rsvd_orthonormal_U():
    """Left singular vectors should be approximately orthonormal."""
    rng = np.random.default_rng(3)
    A = jnp.asarray(rng.standard_normal((30, 20)), dtype=jnp.float32)
    U_r, _, _ = rsvd(A, rank=5)
    UtU = U_r.T @ U_r
    np.testing.assert_allclose(UtU, jnp.eye(5), atol=1e-3)


def test_rsvd_orthonormal_V():
    """Right singular vectors should be approximately orthonormal."""
    rng = np.random.default_rng(4)
    A = jnp.asarray(rng.standard_normal((30, 20)), dtype=jnp.float32)
    _, _, Vt_r = rsvd(A, rank=5)
    VVt = Vt_r @ Vt_r.T
    np.testing.assert_allclose(VVt, jnp.eye(5), atol=1e-3)


# ---------------------------------------------------------------------------
# approx_nuclear_norm
# ---------------------------------------------------------------------------


def test_approx_nuclear_norm_scalar():
    """approx_nuclear_norm should return a scalar."""
    rng = np.random.default_rng(5)
    A = jnp.asarray(rng.standard_normal((10, 8)), dtype=jnp.float32)
    nn = approx_nuclear_norm(A, rank=4)
    assert nn.shape == ()
    assert float(nn) > 0


def test_approx_nuclear_norm_lower_bound():
    """approx_nuclear_norm(rank=r) <= full nuclear norm."""
    rng = np.random.default_rng(6)
    A = jnp.asarray(rng.standard_normal((15, 10)), dtype=jnp.float32)
    nn_approx = float(approx_nuclear_norm(A, rank=5))
    nn_full = float(jnp.linalg.svd(A, compute_uv=False).sum())
    assert nn_approx <= nn_full + 0.5, (
        f"approx_nuclear_norm ({nn_approx:.4f}) > full nuclear norm ({nn_full:.4f})"
    )


def test_approx_nuclear_norm_full_rank_close():
    """approx_nuclear_norm with rank=min(m,n) should be close to true nuclear norm."""
    rng = np.random.default_rng(7)
    A = jnp.asarray(rng.standard_normal((10, 8)), dtype=jnp.float32)
    rank = 8  # full rank
    nn_approx = float(approx_nuclear_norm(A, rank=rank, n_iter=8))
    nn_full = float(jnp.linalg.svd(A, compute_uv=False).sum())
    np.testing.assert_allclose(nn_approx, nn_full, rtol=0.05)


def test_approx_nuclear_norm_gradient_exists():
    """approx_nuclear_norm should be differentiable via custom_vjp."""
    rng = np.random.default_rng(8)
    A = jnp.asarray(rng.standard_normal((10, 8)), dtype=jnp.float32)

    def f(A):
        return approx_nuclear_norm(A, rank=4)

    grad = jax.grad(f)(A)
    assert grad.shape == A.shape
    assert jnp.all(jnp.isfinite(grad))


def test_approx_nuclear_norm_gradient_direction():
    """Gradient should be approximately U_r V_r^T (subgradient of nuclear norm)."""
    rng = np.random.default_rng(9)
    rank = 3
    A = make_low_rank_matrix(12, 8, rank=rank, rng=rng)

    def f(A):
        return approx_nuclear_norm(A, rank=rank, n_iter=8)

    grad = jax.grad(f)(A)

    # For low-rank A, nuclear-norm subgradient = U_r V_r^T should have
    # singular values all <= 1
    sv_grad = jnp.linalg.svd(grad, compute_uv=False)
    assert jnp.all(sv_grad <= 1.0 + 1e-4), f"Gradient svs > 1: {sv_grad}"


def test_approx_nuclear_norm_positive():
    """Nuclear norm should be positive for a non-zero matrix."""
    A = jnp.ones((5, 4), dtype=jnp.float32)
    nn = approx_nuclear_norm(A, rank=2)
    assert float(nn) > 0
