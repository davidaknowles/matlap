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


# ---------------------------------------------------------------------------
# ldlt_cuda: CUDA LDL^T kernel
# ---------------------------------------------------------------------------

def _has_gpu():
    """Check whether a CUDA GPU is available."""
    try:
        import jax
        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        return False


@pytest.mark.skipif(not _has_gpu(), reason="no GPU available")
def test_ldlt_kernel_factorisation_accuracy():
    """LDL^T CUDA kernel: A ≈ L D L^T, L unit lower-tri (PSD matrices for stability)."""
    from matlap.ldlt_cuda import ldlt_batched

    rng = np.random.default_rng(42)
    m, r = 20, 8
    # Use PSD matrices: A = B B^T + I (unpivoted LDL^T is stable for PD matrices)
    B = rng.standard_normal((m, r, r)).astype(np.float32)
    A_np = (B @ B.transpose(0, 2, 1)) / r + np.eye(r, dtype=np.float32)
    A = jnp.array(A_np)

    L, d = ldlt_batched(A)

    for i in range(m):
        A_rec = L[i] @ jnp.diag(d[i]) @ L[i].T
        np.testing.assert_allclose(A_rec, A[i], atol=1e-3,
                                   err_msg=f"LDL^T reconstruction failed for matrix {i}")
        np.testing.assert_allclose(jnp.diag(L[i]), jnp.ones(r), atol=1e-6,
                                   err_msg="L diagonal must be 1 (unit lower triangular)")
        np.testing.assert_allclose(jnp.triu(L[i], 1), jnp.zeros((r, r)), atol=1e-6,
                                   err_msg="L upper triangle must be zero")


@pytest.mark.skipif(not _has_gpu(), reason="no GPU available")
def test_ldlt_matches_eigh_row_update():
    """update_rows_lowrank_isotropic_ldlt matches eigh path to float32 tolerance."""
    from matlap.linalg import update_rows_lowrank_isotropic
    from matlap.ldlt_cuda import update_rows_lowrank_isotropic_ldlt

    rng = np.random.default_rng(7)
    m, n, r = 50, 80, 6
    Y = rng.standard_normal((m, n)).astype(np.float32)
    S2 = (rng.uniform(0.1, 1.0, (m, n)) ** 2).astype(np.float32)
    # 15% missing data
    Y.ravel()[rng.choice(m * n, m * n // 7, replace=False)] = np.nan
    S2.ravel()[rng.choice(m * n, m * n // 7, replace=False)] = np.inf

    V_np = rng.standard_normal((n, r)).astype(np.float32)
    V_np, _ = np.linalg.qr(V_np)
    d_r = np.abs(rng.standard_normal(r)).astype(np.float32) + 1.0

    Y_j, S2_j = jnp.array(Y), jnp.array(S2)
    V_j, d_j = jnp.array(V_np), jnp.array(d_r)
    lb, gm = jnp.array(0.5), jnp.array(0.5)

    ref = update_rows_lowrank_isotropic(Y_j, S2_j, V_j, d_j, lb, gm)
    out = update_rows_lowrank_isotropic_ldlt(Y_j, S2_j, V_j, d_j, lb, gm)

    names = ["mu", "z_tilde", "VtSigmaV", "log_det", "diag_sig"]
    for name, a, b in zip(names, ref, out):
        np.testing.assert_allclose(np.array(a), np.array(b), atol=1e-3,
                                   err_msg=f"LDL^T vs eigh mismatch in {name}")


@pytest.mark.skipif(not _has_gpu(), reason="no GPU available")
def test_xla_ldlt_factorisation_accuracy():
    """XLA FFI LDL^T kernel: A ≈ L D L^T, L unit lower-tri."""
    from matlap.xla_ext.ldlt_xla import ldlt_xla

    rng = np.random.default_rng(42)
    m, r = 20, 8
    B = rng.standard_normal((m, r, r)).astype(np.float32)
    A_np = (B @ B.transpose(0, 2, 1)) / r + np.eye(r, dtype=np.float32)
    A = jnp.array(A_np)

    L, d = ldlt_xla(A)

    for i in range(m):
        A_rec = L[i] @ jnp.diag(d[i]) @ L[i].T
        np.testing.assert_allclose(A_rec, A[i], atol=1e-3,
                                   err_msg=f"XLA LDL^T reconstruction failed for matrix {i}")
        np.testing.assert_allclose(jnp.diag(L[i]), jnp.ones(r), atol=1e-6,
                                   err_msg="L diagonal must be 1")
        np.testing.assert_allclose(jnp.triu(L[i], 1), jnp.zeros((r, r)), atol=1e-6,
                                   err_msg="L upper triangle must be zero")


@pytest.mark.skipif(not _has_gpu(), reason="no GPU available")
def test_xla_ldlt_matches_eigh_row_update():
    """update_rows_lowrank_isotropic_xla_ldlt matches eigh path to float32 tolerance."""
    from matlap.linalg import update_rows_lowrank_isotropic
    from matlap.ldlt_cuda import update_rows_lowrank_isotropic_xla_ldlt

    rng = np.random.default_rng(7)
    m, n, r = 50, 80, 6
    Y = rng.standard_normal((m, n)).astype(np.float32)
    S2 = (rng.uniform(0.1, 1.0, (m, n)) ** 2).astype(np.float32)
    Y.ravel()[rng.choice(m * n, m * n // 7, replace=False)] = np.nan
    S2.ravel()[rng.choice(m * n, m * n // 7, replace=False)] = np.inf

    V_np = rng.standard_normal((n, r)).astype(np.float32)
    V_np, _ = np.linalg.qr(V_np)
    d_r = np.abs(rng.standard_normal(r)).astype(np.float32) + 1.0

    Y_j, S2_j = jnp.array(Y), jnp.array(S2)
    V_j, d_j = jnp.array(V_np), jnp.array(d_r)
    lb, gm = jnp.array(0.5), jnp.array(0.5)

    ref = update_rows_lowrank_isotropic(Y_j, S2_j, V_j, d_j, lb, gm)
    out = update_rows_lowrank_isotropic_xla_ldlt(Y_j, S2_j, V_j, d_j, lb, gm)

    names = ["mu", "z_tilde", "VtSigmaV", "log_det", "diag_sig"]
    for name, a, b in zip(names, ref, out):
        np.testing.assert_allclose(np.array(a), np.array(b), atol=1e-3,
                                   err_msg=f"XLA LDL^T vs eigh mismatch in {name}")
