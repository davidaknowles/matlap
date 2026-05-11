"""Tests for matlap_lowrank (low-rank CAVI) and supporting low-rank linalg."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import matlap
from matlap import matlap_lowrank, matlap as run_matlap
from matlap.linalg import update_row_lowrank, update_rows_lowrank

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
