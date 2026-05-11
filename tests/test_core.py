"""Tests for matlap.core — full CAVI algorithm."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import matlap
from matlap import matlap as run_matlap
from matlap import matlap_grid

jax.config.update("jax_enable_x64", False)

RNG = np.random.default_rng(2025)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_low_rank(m: int, n: int, rank: int, noise_std: float, rng):
    """Generate rank-r matrix plus Gaussian noise."""
    U = rng.standard_normal((m, rank)).astype(np.float32)
    V = rng.standard_normal((n, rank)).astype(np.float32)
    X_true = U @ V.T
    noise = (rng.standard_normal((m, n)) * noise_std).astype(np.float32)
    Y = X_true + noise
    S = np.full((m, n), noise_std, dtype=np.float32)
    return jnp.asarray(Y), jnp.asarray(S), jnp.asarray(X_true)


# ---------------------------------------------------------------------------
# Basic smoke test
# ---------------------------------------------------------------------------


def test_matlap_runs_and_returns_correct_shapes():
    m, n = 12, 8
    Y, S, _ = make_low_rank(m, n, rank=2, noise_std=0.5, rng=RNG)
    result = run_matlap(Y, S, max_iter=20)

    assert result.mu.shape == (m, n)
    assert result.sigma.shape == (m, n, n)
    assert len(result.elbo_trace) > 0
    assert np.isfinite(result.lambda_bar)
    assert result.lambda_bar > 0


# ---------------------------------------------------------------------------
# Low-rank recovery
# ---------------------------------------------------------------------------


def test_low_rank_recovery():
    """Posterior mean should be closer to X_true than noisy Y."""
    m, n, rank = 20, 15, 2
    noise_std = 0.3
    Y, S, X_true = make_low_rank(m, n, rank, noise_std, RNG)

    result = run_matlap(Y, S, max_iter=100, tol=1e-5)

    err_mu = float(jnp.mean((result.mu - X_true) ** 2))
    err_Y = float(jnp.mean((Y - X_true) ** 2))

    assert err_mu < err_Y, (
        f"Posterior mean (MSE={err_mu:.4f}) should beat noisy Y (MSE={err_Y:.4f})"
    )


# ---------------------------------------------------------------------------
# ELBO monotonicity
# ---------------------------------------------------------------------------


def test_elbo_nondecreasing():
    """ELBO must be non-decreasing across CAVI iterations."""
    Y, S, _ = make_low_rank(15, 10, rank=2, noise_std=0.5, rng=RNG)
    result = run_matlap(Y, S, max_iter=50, tol=1e-9)

    elbo = result.elbo_trace
    for i in range(1, len(elbo)):
        assert elbo[i] >= elbo[i - 1] - 1e-2, (
            f"ELBO decreased at iter {i}: {elbo[i-1]:.6f} → {elbo[i]:.6f}"
        )


# ---------------------------------------------------------------------------
# Missing data
# ---------------------------------------------------------------------------


def test_missing_data_finite_result():
    """Result should be finite when ~30% of entries are missing."""
    m, n = 15, 10
    Y, S, X_true = make_low_rank(m, n, rank=2, noise_std=0.5, rng=RNG)

    # Randomly mask ~30% of entries
    mask = RNG.random((m, n)) < 0.3
    Y_obs = np.array(Y).copy()
    Y_obs[mask] = np.nan
    S_obs = np.array(S).copy()
    S_obs[mask] = np.inf

    result = run_matlap(jnp.asarray(Y_obs), jnp.asarray(S_obs), max_iter=80)

    assert jnp.all(jnp.isfinite(result.mu)), "mu has non-finite values"
    assert jnp.all(jnp.isfinite(result.sigma)), "sigma has non-finite values"


def test_missing_data_improves_over_zero_imputation():
    """CAVI reconstruction should beat naive zero-imputation for missing entries."""
    m, n, rank = 20, 15, 2
    noise_std = 0.3
    Y, S, X_true = make_low_rank(m, n, rank, noise_std, RNG)

    mask = RNG.random((m, n)) < 0.25
    Y_obs = np.array(Y).copy()
    Y_obs[mask] = np.nan
    S_obs = np.array(S).copy()
    S_obs[mask] = np.inf

    result = run_matlap(jnp.asarray(Y_obs), jnp.asarray(S_obs), max_iter=100)

    X_true_np = np.array(X_true)
    mu_np = np.array(result.mu)

    # Only evaluate on missing entries
    mse_matlap = np.mean((mu_np[mask] - X_true_np[mask]) ** 2)
    mse_zero = np.mean(X_true_np[mask] ** 2)  # zero-imputation MSE

    assert mse_matlap < mse_zero, (
        f"matlap MSE on missing ({mse_matlap:.4f}) >= zero-imputation ({mse_zero:.4f})"
    )


# ---------------------------------------------------------------------------
# Lambda estimation
# ---------------------------------------------------------------------------


def test_lambda_is_positive_and_finite():
    Y, S, _ = make_low_rank(12, 8, rank=2, noise_std=0.5, rng=RNG)
    result = run_matlap(Y, S, max_iter=50)
    assert result.lambda_bar > 0
    assert np.isfinite(result.lambda_bar)


def test_lambda_scales_with_signal_strength():
    """Stronger signal (larger scale) should lead to lower lambda (less regularisation)."""
    m, n, rank = 15, 10, 1
    noise_std = 0.1
    rng = np.random.default_rng(99)

    u = rng.standard_normal((m, rank)).astype(np.float32)
    v = rng.standard_normal((n, rank)).astype(np.float32)
    noise = (rng.standard_normal((m, n)) * noise_std).astype(np.float32)
    S = jnp.full((m, n), noise_std, dtype=jnp.float32)

    # Weak signal
    Y_weak = jnp.asarray(0.5 * u @ v.T + noise)
    res_weak = run_matlap(Y_weak, S, max_iter=100)

    # Strong signal (same structure, bigger scale)
    Y_strong = jnp.asarray(5.0 * u @ v.T + noise)
    res_strong = run_matlap(Y_strong, S, max_iter=100)

    assert res_strong.lambda_bar < res_weak.lambda_bar, (
        f"Stronger signal should give lower lambda: strong={res_strong.lambda_bar:.4f} "
        f"vs weak={res_weak.lambda_bar:.4f}"
    )


# ---------------------------------------------------------------------------
# Convergence flag
# ---------------------------------------------------------------------------


def test_convergence_flag():
    """Tight tol should converge; very short runs should not."""
    Y, S, _ = make_low_rank(15, 10, rank=2, noise_std=0.5, rng=RNG)

    result_long = run_matlap(Y, S, max_iter=300, tol=1e-6)
    assert result_long.converged, "Should converge with 300 iterations and tol=1e-6"

    result_short = run_matlap(Y, S, max_iter=1, tol=1e-10)
    assert not result_short.converged, "Should NOT converge with 1 iteration"


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------


def test_matlap_grid_returns_correct_structure():
    m, n = 12, 8
    Y, S, _ = make_low_rank(m, n, rank=2, noise_std=0.5, rng=RNG)

    grid = jnp.array([0.1, 1.0, 10.0], dtype=jnp.float32)
    gr = matlap_grid(Y, S, lambda_grid=grid, max_iter=30)

    assert gr.best_lambda in [0.1, 1.0, 10.0] or any(
        abs(gr.best_lambda - v) < 1e-4 for v in [0.1, 1.0, 10.0]
    ), f"best_lambda={gr.best_lambda} not in grid"
    assert len(gr.results) == 3
    assert gr.best_result.mu.shape == (m, n)


def test_matlap_grid_elbo_monotone_per_lambda():
    """Each fixed-lambda run should also have non-decreasing ELBO."""
    Y, S, _ = make_low_rank(12, 8, rank=2, noise_std=0.5, rng=RNG)
    grid = jnp.logspace(-1, 1, 5, dtype=jnp.float32)
    gr = matlap_grid(Y, S, lambda_grid=grid, max_iter=50, tol=1e-9)

    for lam, res in gr.results:
        elbo = res.elbo_trace
        for i in range(1, len(elbo)):
            assert elbo[i] >= elbo[i - 1] - 1e-2, (
                f"lambda={lam:.4f}: ELBO decreased at iter {i}: {elbo[i-1]:.6f} → {elbo[i]:.6f}"
            )


def test_matlap_grid_best_has_max_elbo():
    """The reported best_lambda should correspond to the highest final ELBO."""
    Y, S, _ = make_low_rank(12, 8, rank=2, noise_std=0.5, rng=RNG)
    grid = jnp.logspace(-1, 1, 5, dtype=jnp.float32)
    gr = matlap_grid(Y, S, lambda_grid=grid, max_iter=50)

    elbos = {lam: res.elbo_trace[-1] for lam, res in gr.results}
    expected_best = max(elbos, key=elbos.get)
    assert abs(gr.best_lambda - expected_best) < 1e-4, (
        f"best_lambda={gr.best_lambda:.4f} but max ELBO at {expected_best:.4f}"
    )


def test_matlap_grid_vs_auto_lambda_similar_reconstruction():
    """Grid search best solution should give similar reconstruction to auto-lambda."""
    m, n, rank = 20, 15, 2
    noise_std = 0.3
    Y, S, X_true = make_low_rank(m, n, rank, noise_std, RNG)

    result_auto = run_matlap(Y, S, max_iter=150, tol=1e-6)
    grid = jnp.logspace(-2, 3, 20, dtype=jnp.float32)
    grid_res = matlap_grid(Y, S, lambda_grid=grid, max_iter=100, tol=1e-6)

    mse_auto = float(jnp.mean((result_auto.mu - X_true) ** 2))
    mse_grid = float(jnp.mean((grid_res.best_result.mu - X_true) ** 2))

    # Grid result should be within 3x of auto (generous tolerance due to grid resolution)
    assert mse_grid < mse_auto * 3.0, (
        f"Grid MSE ({mse_grid:.4f}) much worse than auto ({mse_auto:.4f})"
    )
