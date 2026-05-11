"""Tests for matlap.linalg — matrix square root and row-update routines."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from matlap.linalg import matrix_sqrt_eigh, trace_sqrt, update_row, update_rows

jax.config.update("jax_enable_x64", False)


# ---------------------------------------------------------------------------
# matrix_sqrt_eigh
# ---------------------------------------------------------------------------


def test_matrix_sqrt_recovers_original():
    """Q = Psi^{1/2} should satisfy Q @ Q ≈ Psi."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 10))
    Psi = jnp.asarray(A @ A.T, dtype=jnp.float32)  # 5x5 PSD

    decomp = matrix_sqrt_eigh(Psi)
    Q = decomp.vecs @ jnp.diag(decomp.sqrt_vals) @ decomp.vecs.T
    np.testing.assert_allclose(Q @ Q, Psi, atol=1e-4)


def test_matrix_sqrt_symmetric():
    """Q should be symmetric."""
    rng = np.random.default_rng(1)
    A = rng.standard_normal((4, 8))
    Psi = jnp.asarray(A @ A.T, dtype=jnp.float32)

    decomp = matrix_sqrt_eigh(Psi)
    Q = decomp.vecs @ jnp.diag(decomp.sqrt_vals) @ decomp.vecs.T
    np.testing.assert_allclose(Q, Q.T, atol=1e-5)


def test_trace_sqrt():
    """Tr(Q) = sum of sqrt eigenvalues."""
    rng = np.random.default_rng(2)
    A = rng.standard_normal((3, 6))
    Psi = jnp.asarray(A @ A.T + np.eye(3), dtype=jnp.float32)

    decomp = matrix_sqrt_eigh(Psi)
    Q = decomp.vecs @ jnp.diag(decomp.sqrt_vals) @ decomp.vecs.T
    expected_trace = jnp.trace(Q)

    np.testing.assert_allclose(float(trace_sqrt(decomp)), float(expected_trace), rtol=1e-4)


def test_sqrt_nonneg_eigenvalues():
    """sqrt_vals should be non-negative (numerical safety clamp)."""
    # Nearly-singular matrix
    Psi = jnp.asarray([[1.0, 1.0], [1.0, 1.0]], dtype=jnp.float32)
    decomp = matrix_sqrt_eigh(Psi)
    assert jnp.all(decomp.sqrt_vals >= 0)


# ---------------------------------------------------------------------------
# update_row
# ---------------------------------------------------------------------------


def test_update_row_diagonal_case():
    """For diagonal Q and diagonal S, result should match analytic formula."""
    n = 4
    s2 = jnp.array([0.5, 1.0, 2.0, 4.0], dtype=jnp.float32)
    y = jnp.array([1.0, -2.0, 3.0, 0.5], dtype=jnp.float32)
    lambda_bar = jnp.asarray(2.0, dtype=jnp.float32)

    # Diagonal Q: Psi = diag(q_vals^2)
    q_sqrt_vals = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)  # sqrt eigenvalues of Psi
    q_vecs = jnp.eye(n, dtype=jnp.float32)

    mu, Sigma, log_det = update_row(y, s2, q_sqrt_vals, q_vecs, lambda_bar)

    # Analytic: A_i = diag(1/s2 + lambda_bar / q_sqrt_vals)
    a_diag = 1.0 / s2 + lambda_bar / q_sqrt_vals
    sigma_diag_expected = 1.0 / a_diag
    mu_expected = sigma_diag_expected * (y / s2)
    log_det_expected = jnp.sum(jnp.log(sigma_diag_expected))

    np.testing.assert_allclose(jnp.diag(Sigma), sigma_diag_expected, rtol=1e-4)
    np.testing.assert_allclose(mu, mu_expected, rtol=1e-4)
    np.testing.assert_allclose(float(log_det), float(log_det_expected), rtol=1e-3)


def test_update_row_missing_data_ignored():
    """Missing observations (s2=inf) should not affect posterior mean."""
    n = 3
    y = jnp.array([1.0, float("nan"), 3.0], dtype=jnp.float32)
    s2 = jnp.array([0.5, float("inf"), 1.0], dtype=jnp.float32)
    lambda_bar = jnp.asarray(1.0, dtype=jnp.float32)

    q_sqrt_vals = jnp.ones(n, dtype=jnp.float32)
    q_vecs = jnp.eye(n, dtype=jnp.float32)

    mu, Sigma, log_det = update_row(y, s2, q_sqrt_vals, q_vecs, lambda_bar)

    # Result should be finite
    assert jnp.all(jnp.isfinite(mu)), f"mu has non-finite: {mu}"
    assert jnp.all(jnp.isfinite(Sigma)), "Sigma has non-finite"

    # Compare against version with no observation for the missing entry
    y_alt = jnp.array([1.0, 0.0, 3.0], dtype=jnp.float32)
    s2_alt = jnp.array([0.5, float("inf"), 1.0], dtype=jnp.float32)
    mu_alt, Sigma_alt, _ = update_row(y_alt, s2_alt, q_sqrt_vals, q_vecs, lambda_bar)

    np.testing.assert_allclose(mu, mu_alt, atol=1e-5)


def test_update_row_all_missing_shrinks_to_zero():
    """With all observations missing, posterior mean should be zero (prior mean)."""
    n = 4
    y = jnp.full(n, float("nan"), dtype=jnp.float32)
    s2 = jnp.full(n, float("inf"), dtype=jnp.float32)
    lambda_bar = jnp.asarray(1.0, dtype=jnp.float32)

    q_sqrt_vals = jnp.ones(n, dtype=jnp.float32)
    q_vecs = jnp.eye(n, dtype=jnp.float32)

    mu, Sigma, log_det = update_row(y, s2, q_sqrt_vals, q_vecs, lambda_bar)

    np.testing.assert_allclose(mu, jnp.zeros(n), atol=1e-6)
    assert jnp.all(jnp.isfinite(Sigma))


def test_update_row_positive_definite_sigma():
    """Sigma_i should be symmetric positive definite."""
    rng = np.random.default_rng(42)
    n = 5
    y = jnp.asarray(rng.standard_normal(n), dtype=jnp.float32)
    s2 = jnp.asarray(rng.uniform(0.1, 2.0, n), dtype=jnp.float32)

    A = rng.standard_normal((n, n + 2))
    Psi = jnp.asarray(A @ A.T + np.eye(n), dtype=jnp.float32)
    decomp = matrix_sqrt_eigh(Psi)

    lambda_bar = jnp.asarray(1.0, dtype=jnp.float32)
    _, Sigma, _ = update_row(y, s2, decomp.sqrt_vals, decomp.vecs, lambda_bar)

    # Check symmetry
    np.testing.assert_allclose(Sigma, Sigma.T, atol=1e-5)
    # Check positive eigenvalues
    eigs = jnp.linalg.eigvalsh(Sigma)
    assert jnp.all(eigs > 0), f"Sigma not PD; eigenvalues: {eigs}"


def test_update_rows_vmap_matches_single():
    """update_rows (vmapped) should match individual update_row calls."""
    rng = np.random.default_rng(7)
    m, n = 6, 4
    Y = jnp.asarray(rng.standard_normal((m, n)), dtype=jnp.float32)
    S2 = jnp.asarray(rng.uniform(0.1, 1.0, (m, n)), dtype=jnp.float32)
    lambda_bar = jnp.asarray(2.0, dtype=jnp.float32)

    A = rng.standard_normal((n, n + 2))
    Psi = jnp.asarray(A @ A.T + np.eye(n), dtype=jnp.float32)
    decomp = matrix_sqrt_eigh(Psi)

    mus_v, sigmas_v, logdets_v = update_rows(Y, S2, decomp.sqrt_vals, decomp.vecs, lambda_bar)

    for i in range(m):
        mu_i, sigma_i, ld_i = update_row(Y[i], S2[i], decomp.sqrt_vals, decomp.vecs, lambda_bar)
        np.testing.assert_allclose(mus_v[i], mu_i, atol=1e-5)
        np.testing.assert_allclose(sigmas_v[i], sigma_i, atol=1e-5)
        np.testing.assert_allclose(float(logdets_v[i]), float(ld_i), atol=1e-4)
