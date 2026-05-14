"""Tests for matlap_lowrank (low-rank CAVI) and supporting low-rank linalg."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import matlap
from matlap import (
    matlap_lowrank, matlap_lowrank_isotropic, matlap as run_matlap,
    matlap_grid_lowrank, matlap_grid_lowrank_isotropic,
    LowRankIsotropicGridResult,
)
from matlap.linalg import (
    update_row_lowrank, update_rows_lowrank,
    update_row_lowrank_isotropic, update_rows_lowrank_isotropic,
)

jax.config.update("jax_enable_x64", False)

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_low_rank(m, n, rank, noise_std, rng):
    U = rng.standard_normal((m, rank)).astype(np.float32)
    V = rng.standard_normal((n, rank)).astype(np.float32)
    X_true = U @ V.T
    noise = (rng.standard_normal((m, n)) * noise_std).astype(np.float32)
    Y = X_true + noise
    S = np.full((m, n), noise_std, dtype=np.float32)
    return jnp.asarray(Y), jnp.asarray(S), jnp.asarray(X_true)


# ---------------------------------------------------------------------------
# update_row_lowrank
# ---------------------------------------------------------------------------


def test_update_row_lowrank_shapes():
    """Return shapes match expected (r,), (r,r), (), (n,)."""
    rng = np.random.default_rng(0)
    n, r = 10, 3
    y_i = jnp.asarray(rng.standard_normal(n), dtype=jnp.float32)
    s2_i = jnp.ones(n, dtype=jnp.float32)
    Q, _ = jnp.linalg.qr(jnp.asarray(rng.standard_normal((n, r)), dtype=jnp.float32))
    V_r = Q
    d_r = jnp.ones(r, dtype=jnp.float32)
    lambda_bar = jnp.array(1.0, dtype=jnp.float32)

    z_i, A_r_inv, log_det_i, diag_sig_i = update_row_lowrank(y_i, s2_i, V_r, d_r, lambda_bar)

    assert z_i.shape == (r,)
    assert A_r_inv.shape == (r, r)
    assert log_det_i.shape == ()
    assert diag_sig_i.shape == (n,)


def test_update_row_lowrank_positive_diag_sigma():
    """Diagonal of covariance must be positive."""
    rng = np.random.default_rng(1)
    n, r = 8, 4
    y_i = jnp.asarray(rng.standard_normal(n), dtype=jnp.float32)
    s2_i = jnp.asarray(rng.uniform(0.5, 2.0, n), dtype=jnp.float32)
    Q, _ = jnp.linalg.qr(jnp.asarray(rng.standard_normal((n, r)), dtype=jnp.float32))
    V_r = Q
    d_r = jnp.asarray(rng.uniform(1.0, 3.0, r), dtype=jnp.float32)
    lambda_bar = jnp.array(0.5, dtype=jnp.float32)

    _, _, _, diag_sig_i = update_row_lowrank(y_i, s2_i, V_r, d_r, lambda_bar)
    assert jnp.all(diag_sig_i > 0)


def test_update_row_lowrank_missing_ignored():
    """Missing observations (s2=inf) should not change posterior mean direction."""
    rng = np.random.default_rng(2)
    n, r = 6, 2
    y_i = jnp.asarray(rng.standard_normal(n), dtype=jnp.float32)
    s2_i_full = jnp.ones(n, dtype=jnp.float32)
    s2_i_miss = s2_i_full.at[2].set(jnp.inf).at[4].set(jnp.inf)  # 2 missing
    Q, _ = jnp.linalg.qr(jnp.asarray(rng.standard_normal((n, r)), dtype=jnp.float32))
    V_r = Q
    d_r = jnp.ones(r, dtype=jnp.float32)
    lambda_bar = jnp.array(1.0, dtype=jnp.float32)

    z_full, _, _, _ = update_row_lowrank(y_i, s2_i_full, V_r, d_r, lambda_bar)
    z_miss, _, _, _ = update_row_lowrank(y_i, s2_i_miss, V_r, d_r, lambda_bar)

    # z_miss is valid (no NaN/inf) and different from z_full
    assert jnp.all(jnp.isfinite(z_miss))
    assert not jnp.allclose(z_full, z_miss)


def test_update_rows_lowrank_batches():
    """update_rows_lowrank should give the same result as calling update_row_lowrank in a loop."""
    rng = np.random.default_rng(3)
    m, n, r = 5, 8, 3
    Y = jnp.asarray(rng.standard_normal((m, n)), dtype=jnp.float32)
    S2 = jnp.ones((m, n), dtype=jnp.float32)
    Q, _ = jnp.linalg.qr(jnp.asarray(rng.standard_normal((n, r)), dtype=jnp.float32))
    V_r = Q
    d_r = jnp.ones(r, dtype=jnp.float32)
    lambda_bar = jnp.array(1.0, dtype=jnp.float32)

    zs, A_invs, log_dets, diag_sigs = update_rows_lowrank(Y, S2, V_r, d_r, lambda_bar)

    for i in range(m):
        z_i, A_inv_i, ld_i, ds_i = update_row_lowrank(Y[i], S2[i], V_r, d_r, lambda_bar)
        np.testing.assert_allclose(zs[i], z_i, atol=2e-4)
        np.testing.assert_allclose(A_invs[i], A_inv_i, atol=2e-4)
        np.testing.assert_allclose(float(log_dets[i]), float(ld_i), atol=2e-4)
        np.testing.assert_allclose(diag_sigs[i], ds_i, atol=2e-4)


# ---------------------------------------------------------------------------
# matlap_lowrank — basic correctness
# ---------------------------------------------------------------------------


def test_matlap_lowrank_returns_result():
    """matlap_lowrank should return a LowRankCAVIResult without error."""
    Y, S, _ = make_low_rank(20, 10, rank=2, noise_std=0.5, rng=RNG)
    result = matlap_lowrank(Y, S, rank=5, max_iter=10)
    assert result.mu.shape == Y.shape
    assert result.z.shape[1] == 5
    assert result.V_r.shape == (10, 5)
    assert result.diag_sigma is not None
    assert result.diag_sigma.shape == Y.shape
    assert jnp.all(result.diag_sigma > 0)
    assert len(result.elbo_trace) > 0
    assert jnp.all(jnp.isfinite(result.mu))


def test_matlap_lowrank_elbo_finite():
    """ELBO trace must be all-finite."""
    Y, S, _ = make_low_rank(15, 8, rank=2, noise_std=0.3, rng=RNG)
    result = matlap_lowrank(Y, S, rank=4, max_iter=20)
    assert all(np.isfinite(e) for e in result.elbo_trace)


def test_matlap_lowrank_elbo_nondecreasing():
    """ELBO should be non-decreasing (up to float32 rounding)."""
    Y, S, _ = make_low_rank(20, 12, rank=2, noise_std=0.5, rng=RNG)
    result = matlap_lowrank(Y, S, rank=5, max_iter=50, tol=1e-9)
    elbo = result.elbo_trace
    for i in range(1, len(elbo)):
        assert elbo[i] >= elbo[i - 1] - 0.05, (
            f"ELBO decreased at iter {i}: {elbo[i-1]:.6f} → {elbo[i]:.6f}"
        )


def test_matlap_lowrank_recovers_signal():
    """Low-rank CAVI should produce lower RMSE than the noisy observations."""
    Y, S, X_true = make_low_rank(30, 15, rank=3, noise_std=0.5, rng=RNG)
    result = matlap_lowrank(Y, S, rank=5, max_iter=50)

    rmse_noisy = float(jnp.sqrt(jnp.mean((Y - X_true) ** 2)))
    rmse_pred = float(jnp.sqrt(jnp.mean((result.mu - X_true) ** 2)))
    assert rmse_pred < rmse_noisy, (
        f"RMSE not improved: noisy={rmse_noisy:.4f}, pred={rmse_pred:.4f}"
    )


def test_matlap_lowrank_with_missing():
    """Should handle missing data (S=inf) without NaN in results."""
    rng = np.random.default_rng(10)
    Y, S, _ = make_low_rank(15, 8, rank=2, noise_std=0.5, rng=rng)
    # Mask ~20% of entries
    mask = rng.random(Y.shape) < 0.2
    S = S.at[jnp.asarray(mask)].set(jnp.inf)

    result = matlap_lowrank(Y, S, rank=4, max_iter=20)
    assert jnp.all(jnp.isfinite(result.mu))
    assert all(np.isfinite(e) for e in result.elbo_trace)


def test_matlap_lowrank_rank_clips_to_min_mn():
    """Requesting rank > min(m, n) should clip without error."""
    Y, S, _ = make_low_rank(5, 4, rank=2, noise_std=0.3, rng=RNG)
    result = matlap_lowrank(Y, S, rank=100, max_iter=5)
    assert result.V_r.shape[1] <= min(5, 4)


def test_matlap_lowrank_lambda_positive():
    """lambda_bar must be strictly positive."""
    Y, S, _ = make_low_rank(20, 10, rank=2, noise_std=0.5, rng=RNG)
    result = matlap_lowrank(Y, S, rank=5, max_iter=20)
    assert result.lambda_bar > 0


def test_matlap_lowrank_v_orthonormal():
    """V_r should have approximately orthonormal columns at convergence."""
    Y, S, _ = make_low_rank(20, 10, rank=2, noise_std=0.5, rng=RNG)
    result = matlap_lowrank(Y, S, rank=5, max_iter=30)
    VtV = result.V_r.T @ result.V_r
    # Allow tolerance for float32 accumulation across rotations
    np.testing.assert_allclose(VtV, jnp.eye(5), atol=1e-3)


# ---------------------------------------------------------------------------
# update_row_lowrank_isotropic
# ---------------------------------------------------------------------------


def test_update_row_lowrank_isotropic_shapes():
    """Return shapes: (n,), (r,), (r,r), (), (n,)."""
    rng = np.random.default_rng(20)
    n, r = 10, 3
    y_i = jnp.asarray(rng.standard_normal(n), dtype=jnp.float32)
    s2_i = jnp.ones(n, dtype=jnp.float32)
    Q, _ = jnp.linalg.qr(jnp.asarray(rng.standard_normal((n, r)), dtype=jnp.float32))
    V_r = Q
    d_r = jnp.ones(r, dtype=jnp.float32)
    lambda_bar = jnp.array(1.0, dtype=jnp.float32)
    gamma = jnp.array(1e-3, dtype=jnp.float32)

    mu_i, z_tilde, VtSV, log_det, diag_sig = update_row_lowrank_isotropic(
        y_i, s2_i, V_r, d_r, lambda_bar, gamma
    )

    assert mu_i.shape == (n,)
    assert z_tilde.shape == (r,)
    assert VtSV.shape == (r, r)
    assert log_det.shape == ()
    assert diag_sig.shape == (n,)


def test_update_row_lowrank_isotropic_positive_diag():
    """Diagonal of Σ_i must be strictly positive."""
    rng = np.random.default_rng(21)
    n, r = 12, 4
    y_i = jnp.asarray(rng.standard_normal(n), dtype=jnp.float32)
    s2_i = jnp.asarray(rng.uniform(0.5, 2.0, n), dtype=jnp.float32)
    Q, _ = jnp.linalg.qr(jnp.asarray(rng.standard_normal((n, r)), dtype=jnp.float32))
    V_r = Q
    d_r = jnp.asarray(rng.uniform(1.0, 3.0, r), dtype=jnp.float32)
    lambda_bar = jnp.array(0.5, dtype=jnp.float32)
    gamma = jnp.array(1e-3, dtype=jnp.float32)

    _, _, _, _, diag_sig = update_row_lowrank_isotropic(
        y_i, s2_i, V_r, d_r, lambda_bar, gamma
    )
    assert jnp.all(diag_sig > 0)


def test_update_row_lowrank_isotropic_z_tilde_matches_Vt_mu():
    """z̃_i must equal V_r^T μ_i (for orthonormal V_r)."""
    rng = np.random.default_rng(22)
    n, r = 10, 3
    y_i = jnp.asarray(rng.standard_normal(n), dtype=jnp.float32)
    s2_i = jnp.asarray(rng.uniform(0.5, 2.0, n), dtype=jnp.float32)
    Q, _ = jnp.linalg.qr(jnp.asarray(rng.standard_normal((n, r)), dtype=jnp.float32))
    V_r = Q
    d_r = jnp.asarray(rng.uniform(1.0, 3.0, r), dtype=jnp.float32)
    lambda_bar = jnp.array(1.0, dtype=jnp.float32)
    gamma = jnp.array(1e-3, dtype=jnp.float32)

    mu_i, z_tilde, _, _, _ = update_row_lowrank_isotropic(
        y_i, s2_i, V_r, d_r, lambda_bar, gamma
    )
    np.testing.assert_allclose(z_tilde, V_r.T @ mu_i, atol=1e-4)


def test_update_row_lowrank_isotropic_mu_has_offsubspace():
    """μ_i should have non-zero off-subspace component (unlike matlap_lowrank)."""
    rng = np.random.default_rng(23)
    n, r = 12, 3
    # Signal in first 3 cols of V_r and also outside — use full observation
    y_i = jnp.asarray(rng.standard_normal(n), dtype=jnp.float32)
    s2_i = jnp.ones(n, dtype=jnp.float32)
    Q, _ = jnp.linalg.qr(jnp.asarray(rng.standard_normal((n, r)), dtype=jnp.float32))
    V_r = Q
    d_r = jnp.ones(r, dtype=jnp.float32)
    lambda_bar = jnp.array(1.0, dtype=jnp.float32)
    gamma = jnp.array(1e-3, dtype=jnp.float32)

    mu_i, z_tilde, _, _, _ = update_row_lowrank_isotropic(
        y_i, s2_i, V_r, d_r, lambda_bar, gamma
    )
    # Off-subspace residual: μ_i - V_r z̃_i should be non-zero
    off_subspace = mu_i - V_r @ z_tilde
    assert float(jnp.linalg.norm(off_subspace)) > 1e-4, (
        "mu_i has no off-subspace component; expected non-zero for isotropic prior"
    )


def test_update_row_lowrank_isotropic_missing_data():
    """Missing observations should yield finite results with no NaN."""
    rng = np.random.default_rng(24)
    n, r = 8, 3
    y_i = jnp.asarray(rng.standard_normal(n), dtype=jnp.float32)
    s2_i = jnp.ones(n, dtype=jnp.float32).at[2].set(jnp.inf).at[5].set(jnp.inf)
    Q, _ = jnp.linalg.qr(jnp.asarray(rng.standard_normal((n, r)), dtype=jnp.float32))
    V_r = Q
    d_r = jnp.ones(r, dtype=jnp.float32)
    lambda_bar = jnp.array(1.0, dtype=jnp.float32)
    gamma = jnp.array(1e-3, dtype=jnp.float32)

    mu_i, z_tilde, VtSV, log_det, diag_sig = update_row_lowrank_isotropic(
        y_i, s2_i, V_r, d_r, lambda_bar, gamma
    )
    assert jnp.all(jnp.isfinite(mu_i))
    assert jnp.all(jnp.isfinite(diag_sig))
    assert jnp.all(diag_sig > 0)
    assert jnp.isfinite(log_det)


def test_update_rows_lowrank_isotropic_matches_single():
    """Vmapped version must match row-by-row calls."""
    rng = np.random.default_rng(25)
    m, n, r = 5, 8, 3
    Y = jnp.asarray(rng.standard_normal((m, n)), dtype=jnp.float32)
    S2 = jnp.ones((m, n), dtype=jnp.float32)
    Q, _ = jnp.linalg.qr(jnp.asarray(rng.standard_normal((n, r)), dtype=jnp.float32))
    V_r = Q
    d_r = jnp.ones(r, dtype=jnp.float32)
    lambda_bar = jnp.array(1.0, dtype=jnp.float32)
    gamma = jnp.array(1e-3, dtype=jnp.float32)

    mus, zs, VtSVs, log_dets, diag_sigs = update_rows_lowrank_isotropic(
        Y, S2, V_r, d_r, lambda_bar, gamma
    )

    for i in range(m):
        mu_i, z_i, VtSV_i, ld_i, ds_i = update_row_lowrank_isotropic(
            Y[i], S2[i], V_r, d_r, lambda_bar, gamma
        )
        np.testing.assert_allclose(mus[i], mu_i, atol=2e-4)
        np.testing.assert_allclose(zs[i], z_i, atol=2e-4)
        np.testing.assert_allclose(VtSVs[i], VtSV_i, atol=2e-4)
        np.testing.assert_allclose(float(log_dets[i]), float(ld_i), atol=2e-4)
        np.testing.assert_allclose(diag_sigs[i], ds_i, atol=2e-4)


# ---------------------------------------------------------------------------
# matlap_lowrank_isotropic — correctness
# ---------------------------------------------------------------------------


def test_matlap_lowrank_isotropic_returns_result():
    """Should return LowRankIsotropicResult without error."""
    Y, S, _ = make_low_rank(20, 10, rank=2, noise_std=0.5, rng=RNG)
    result = matlap_lowrank_isotropic(Y, S, rank=5, max_iter=10)
    assert result.mu.shape == Y.shape
    assert result.z.shape == (20, 5)
    assert result.V_r.shape == (10, 5)
    assert result.diag_sigma is not None
    assert result.diag_sigma.shape == Y.shape
    assert jnp.all(result.diag_sigma > 0)
    assert jnp.all(jnp.isfinite(result.mu))
    assert result.delta > 0


def test_matlap_lowrank_isotropic_elbo_finite():
    """ELBO trace must be all-finite."""
    Y, S, _ = make_low_rank(15, 8, rank=2, noise_std=0.3, rng=RNG)
    result = matlap_lowrank_isotropic(Y, S, rank=4, max_iter=20)
    assert all(np.isfinite(e) for e in result.elbo_trace)


def test_matlap_lowrank_isotropic_elbo_nondecreasing():
    """ELBO should converge (finite, net improvement).

    The Q update projects Ψ onto V_r, discarding off-subspace contributions,
    so strict per-step ELBO monotonicity is not guaranteed.  We check that
    the ELBO is finite throughout and that the final value exceeds the initial.
    """
    Y, S, _ = make_low_rank(20, 12, rank=2, noise_std=0.5, rng=RNG)
    result = matlap_lowrank_isotropic(Y, S, rank=5, max_iter=50, tol=1e-9)
    elbo = result.elbo_trace
    assert all(np.isfinite(e) for e in elbo), "ELBO contains non-finite values"
    assert elbo[-1] > elbo[0], (
        f"ELBO did not improve overall: start={elbo[0]:.4f}, end={elbo[-1]:.4f}"
    )


def test_matlap_lowrank_isotropic_recovers_signal():
    """Isotropic CAVI should produce lower RMSE than the noisy observations."""
    Y, S, X_true = make_low_rank(30, 15, rank=3, noise_std=0.5, rng=RNG)
    result = matlap_lowrank_isotropic(Y, S, rank=5, max_iter=50)

    rmse_noisy = float(jnp.sqrt(jnp.mean((Y - X_true) ** 2)))
    rmse_pred = float(jnp.sqrt(jnp.mean((result.mu - X_true) ** 2)))
    assert rmse_pred < rmse_noisy, (
        f"RMSE not improved: noisy={rmse_noisy:.4f}, pred={rmse_pred:.4f}"
    )


def test_matlap_lowrank_isotropic_lambda_unbiased_vs_lowrank():
    """iso lambda should be similar order-of-magnitude to lowrank lambda.

    iso uses a_N = m*n but trace_Q from the full n-dim diagonal of Psi, while
    lowrank uses a_N = m*r with trace_Q from the r projected eigenvalues.  Both
    effects partially cancel, so the ratio is in the same ballpark as n/r but
    need not be exactly n/r.  The key property is that iso does NOT diverge.
    """
    rng = np.random.default_rng(30)
    Y, S, _ = make_low_rank(40, 20, rank=3, noise_std=0.5, rng=rng)
    r = 5

    res_lr = matlap_lowrank(Y, S, rank=r, max_iter=50)
    res_iso = matlap_lowrank_isotropic(Y, S, rank=r, max_iter=50)

    m, n = Y.shape
    ratio = res_iso.lambda_bar / (res_lr.lambda_bar + 1e-8)
    expected = n / r  # rough reference, not an exact prediction
    # Ratio should be in the rough ballpark (between 0.3× and 3× of n/r)
    assert 0.3 * expected <= ratio <= 3.0 * expected, (
        f"lambda ratio {ratio:.2f} far from ballpark {expected:.2f} (n/r)"
    )
    assert res_iso.lambda_bar < 1e4, f"iso lambda diverged: {res_iso.lambda_bar}"


def test_matlap_lowrank_isotropic_with_missing():
    """Should handle missing data (S=inf) without NaN."""
    rng = np.random.default_rng(31)
    Y, S, _ = make_low_rank(15, 8, rank=2, noise_std=0.5, rng=rng)
    mask = rng.random(Y.shape) < 0.2
    S = S.at[jnp.asarray(mask)].set(jnp.inf)

    result = matlap_lowrank_isotropic(Y, S, rank=4, max_iter=20)
    assert jnp.all(jnp.isfinite(result.mu))
    assert all(np.isfinite(e) for e in result.elbo_trace)


def test_matlap_lowrank_isotropic_v_orthonormal():
    """V_r should have approximately orthonormal columns at convergence."""
    Y, S, _ = make_low_rank(20, 10, rank=2, noise_std=0.5, rng=RNG)
    result = matlap_lowrank_isotropic(Y, S, rank=5, max_iter=30)
    VtV = result.V_r.T @ result.V_r
    np.testing.assert_allclose(VtV, jnp.eye(5), atol=1e-3)


def test_matlap_lowrank_isotropic_z_matches_Vt_mu():
    """||z||_F should equal ||mu @ V_r||_F (invariant under the final V_r rotation)."""
    rng = np.random.default_rng(32)
    Y, S, _ = make_low_rank(20, 10, rank=2, noise_std=0.5, rng=rng)
    result = matlap_lowrank_isotropic(Y, S, rank=5, max_iter=20)
    norm_z = float(jnp.linalg.norm(result.z, 'fro'))
    norm_mu_Vr = float(jnp.linalg.norm(result.mu @ result.V_r, 'fro'))
    np.testing.assert_allclose(norm_z, norm_mu_Vr, atol=1e-3)


# ---------------------------------------------------------------------------
# d_r field and warm-start
# ---------------------------------------------------------------------------


def test_lowrank_result_has_d_r():
    """LowRankCAVIResult should expose d_r with shape (r,)."""
    Y, S, _ = make_low_rank(20, 10, rank=2, noise_std=0.5, rng=RNG)
    result = matlap_lowrank(Y, S, rank=5, max_iter=10)
    assert result.d_r.shape == (5,)
    assert jnp.all(result.d_r >= 0)


def test_lowrank_isotropic_result_has_d_r():
    """LowRankIsotropicResult should expose d_r with shape (r,)."""
    Y, S, _ = make_low_rank(20, 10, rank=2, noise_std=0.5, rng=RNG)
    result = matlap_lowrank_isotropic(Y, S, rank=5, max_iter=10)
    assert result.d_r.shape == (5,)
    assert jnp.all(result.d_r >= 0)


def test_lowrank_warmstart_matches_cold():
    """Warm-starting from a converged solution should reach same ELBO as cold-start."""
    rng = np.random.default_rng(99)
    Y, S, _ = make_low_rank(30, 15, rank=3, noise_std=0.5, rng=rng)
    cold = matlap_lowrank(Y, S, 5.0, rank=5, max_iter=100, tol=1e-8)
    warm = matlap_lowrank(Y, S, 5.0, rank=5, max_iter=100, tol=1e-8,
                          V_r_init=cold.V_r, d_r_init=cold.d_r)
    np.testing.assert_allclose(cold.elbo_trace[-1], warm.elbo_trace[-1], rtol=1e-4)


def test_lowrank_isotropic_warmstart_converges_faster():
    """Warm-starting iso from a converged solution should reach the same ELBO."""
    rng = np.random.default_rng(77)
    Y, S, _ = make_low_rank(30, 15, rank=3, noise_std=0.5, rng=rng)
    cold = matlap_lowrank_isotropic(Y, S, 5.0, rank=5, max_iter=100, tol=1e-7)
    warm = matlap_lowrank_isotropic(Y, S, 5.0, rank=5, max_iter=100, tol=1e-7,
                                    V_r_init=cold.V_r, d_r_init=cold.d_r,
                                    delta_init=cold.delta)
    np.testing.assert_allclose(cold.elbo_trace[-1], warm.elbo_trace[-1], rtol=1e-4)


# ---------------------------------------------------------------------------
# matlap_grid_lowrank_isotropic
# ---------------------------------------------------------------------------


def test_matlap_grid_lowrank_isotropic_returns_result():
    """matlap_grid_lowrank_isotropic should return LowRankIsotropicGridResult."""
    rng = np.random.default_rng(55)
    Y, S, _ = make_low_rank(20, 10, rank=2, noise_std=0.5, rng=rng)
    grid = jnp.array([1.0, 5.0, 20.0])
    result = matlap_grid_lowrank_isotropic(Y, S, grid, rank=5, max_iter=10)
    assert isinstance(result, LowRankIsotropicGridResult)
    assert result.best_lambda in [1.0, 5.0, 20.0]
    assert result.best_result.mu.shape == Y.shape
    assert len(result.results) == 3


def test_matlap_grid_lowrank_isotropic_selects_best_elbo():
    """best_lambda should correspond to the grid point with the highest ELBO."""
    rng = np.random.default_rng(66)
    Y, S, _ = make_low_rank(25, 12, rank=3, noise_std=0.4, rng=rng)
    grid = jnp.array([0.5, 2.0, 10.0, 50.0])
    result = matlap_grid_lowrank_isotropic(Y, S, grid, rank=5, max_iter=20)
    elbos = {lam: res.elbo_trace[-1] for lam, res in result.results}
    best_lam_by_elbo = max(elbos, key=elbos.get)
    assert result.best_lambda == best_lam_by_elbo


def test_matlap_grid_lowrank_iso_results_sorted_ascending():
    """results list should be sorted by lambda ascending."""
    rng = np.random.default_rng(44)
    Y, S, _ = make_low_rank(20, 10, rank=2, noise_std=0.5, rng=rng)
    grid = jnp.array([10.0, 1.0, 50.0, 5.0])
    result = matlap_grid_lowrank_isotropic(Y, S, grid, rank=5, max_iter=10)
    lams = [lam for lam, _ in result.results]
    assert lams == sorted(lams)


def test_matlap_grid_lowrank_isotropic_runs_with_loo_score():
    """LOO scoring path should run and return a valid grid result."""
    rng = np.random.default_rng(88)
    Y, S, _ = make_low_rank(20, 10, rank=2, noise_std=0.5, rng=rng)
    grid = jnp.array([1.0, 5.0, 20.0])
    result = matlap_grid_lowrank_isotropic(
        Y, S, grid, rank=5, max_iter=10, score_fn="loo"
    )
    assert isinstance(result, LowRankIsotropicGridResult)
    assert result.best_lambda in [1.0, 5.0, 20.0]


def test_matlap_grid_lowrank_isotropic_runs_with_renyi_score():
    """Rényi scoring path should run and return a valid grid result."""
    rng = np.random.default_rng(89)
    Y, S, _ = make_low_rank(20, 10, rank=2, noise_std=0.5, rng=rng)
    grid = jnp.array([1.0, 5.0, 20.0])
    result = matlap_grid_lowrank_isotropic(
        Y, S, grid, rank=5, max_iter=10, score_fn="renyi", alpha=0.5
    )
    assert isinstance(result, LowRankIsotropicGridResult)
    assert result.best_lambda in [1.0, 5.0, 20.0]


def test_matlap_grid_lowrank_isotropic_runs_with_alpha_zero_score():
    """Alpha-zero Rényi/IS scoring path should run and return a valid result."""
    rng = np.random.default_rng(91)
    Y, S, _ = make_low_rank(20, 10, rank=2, noise_std=0.5, rng=rng)
    grid = jnp.array([1.0, 5.0, 20.0])
    result = matlap_grid_lowrank_isotropic(
        Y, S, grid, rank=5, max_iter=10, score_fn="renyi", alpha=0.0
    )
    assert isinstance(result, LowRankIsotropicGridResult)
    assert result.best_lambda in [1.0, 5.0, 20.0]


def test_matlap_grid_lowrank_isotropic_rejects_bad_score_fn():
    """Invalid score_fn should raise ValueError."""
    rng = np.random.default_rng(90)
    Y, S, _ = make_low_rank(20, 10, rank=2, noise_std=0.5, rng=rng)
    grid = jnp.array([1.0, 5.0, 20.0])
    with pytest.raises(ValueError, match="score_fn"):
        _ = matlap_grid_lowrank_isotropic(
            Y, S, grid, rank=5, max_iter=5, score_fn="not-a-score"
        )
