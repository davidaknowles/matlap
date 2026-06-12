"""Tests for MCMC samplers in matlap.mcmc."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import matlap
from matlap.mcmc import MCMCResult, mcmc_gsm_gibbs, mcmc_proximal_mala


# ---------------------------------------------------------------------------
# Shared test fixture
# ---------------------------------------------------------------------------


def _make_data(m=30, n=15, rank=3, seed=0):
    """Generate a small low-rank + heteroscedastic-noise matrix."""
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((m, rank))
    V = rng.standard_normal((n, rank))
    X_true = U @ V.T / np.sqrt(rank)
    sigma = rng.uniform(0.3, 0.7, size=(m, n))
    Y_full = X_true + rng.standard_normal((m, n)) * sigma

    # 20% missing
    mask = rng.uniform(size=(m, n)) < 0.20
    Y = np.where(~mask, Y_full, np.nan)
    S = np.where(~mask, sigma, np.inf)

    return (
        jnp.array(Y, dtype=jnp.float32),
        jnp.array(S, dtype=jnp.float32),
        X_true,
    )


# ---------------------------------------------------------------------------
# MCMCResult type checks
# ---------------------------------------------------------------------------


def test_mala_result_type():
    Y, S, _ = _make_data()
    r = mcmc_proximal_mala(Y, S, lambda_val=1.0, n_warmup=10, n_samples=20,
                           key=jax.random.PRNGKey(0))
    assert isinstance(r, MCMCResult)
    assert r.mu.shape == Y.shape
    assert 0.0 <= r.accept_rate <= 1.0
    assert r.n_samples == 20
    assert r.lambda_bar == pytest.approx(1.0)


def test_gibbs_result_type():
    Y, S, _ = _make_data()
    r = mcmc_gsm_gibbs(Y, S, n_warmup=10, n_samples=20,
                       key=jax.random.PRNGKey(1))
    assert isinstance(r, MCMCResult)
    assert r.mu.shape == Y.shape
    assert 0.0 < r.accept_rate <= 1.0, f"lambda MH accept_rate {r.accept_rate:.2f} out of range"
    assert r.n_samples == 20
    assert r.lambda_bar > 0.0


# ---------------------------------------------------------------------------
# No NaN / Inf in outputs
# ---------------------------------------------------------------------------


def test_mala_no_nan():
    Y, S, _ = _make_data()
    r = mcmc_proximal_mala(Y, S, n_warmup=20, n_samples=50,
                           key=jax.random.PRNGKey(2))
    assert np.all(np.isfinite(np.array(r.mu))), "MALA mu contains NaN/Inf"


def test_gibbs_no_nan():
    Y, S, _ = _make_data()
    r = mcmc_gsm_gibbs(Y, S, n_warmup=20, n_samples=50,
                       key=jax.random.PRNGKey(3))
    assert np.all(np.isfinite(np.array(r.mu))), "Gibbs mu contains NaN/Inf"
    assert np.isfinite(r.lambda_bar), "Gibbs lambda_bar is NaN/Inf"


def test_mala_lowrank_proposal_no_nan():
    Y, S, _ = _make_data(m=16, n=10, rank=2)
    r = mcmc_proximal_mala(
        Y, S, lambda_val=1.0, n_warmup=8, n_samples=12,
        proposal_svd_rank=3, proposal_svd_n_iter=1, proposal_svd_oversample=1,
        key=jax.random.PRNGKey(31),
    )
    assert np.all(np.isfinite(np.array(r.mu))), "low-rank MALA mu contains NaN/Inf"
    assert 0.0 <= r.accept_rate <= 1.0


def test_gibbs_lowrank_proposal_no_nan():
    Y, S, _ = _make_data(m=16, n=10, rank=2)
    r = mcmc_gsm_gibbs(
        Y, S, n_warmup=8, n_samples=12,
        proposal_svd_rank=3, proposal_svd_n_iter=1, proposal_svd_oversample=1,
        key=jax.random.PRNGKey(32),
    )
    assert np.all(np.isfinite(np.array(r.mu))), "low-rank Gibbs mu contains NaN/Inf"
    assert np.isfinite(r.lambda_bar)


def test_mala_can_sample_lambda():
    Y, S, _ = _make_data(m=16, n=10, rank=2)
    r = mcmc_proximal_mala(
        Y, S, lambda_val=1.0, sample_lambda=True, n_warmup=8, n_samples=12,
        proposal_svd_rank=3, proposal_svd_n_iter=1, proposal_svd_oversample=1,
        key=jax.random.PRNGKey(33),
    )
    assert np.all(np.isfinite(np.array(r.mu))), "lambda-sampling MALA mu contains NaN/Inf"
    assert np.isfinite(r.lambda_bar)
    assert r.lambda_bar > 0.0


def test_mala_return_trace_shapes():
    Y, S, _ = _make_data(m=12, n=8, rank=2)
    r = mcmc_proximal_mala(
        Y, S, lambda_val=1.0, n_warmup=7, n_samples=11,
        return_trace=True, key=jax.random.PRNGKey(34),
    )
    assert r.lambda_trace.shape == (11,)
    assert r.accept_trace.shape == (11,)
    assert r.nuclear_trace.shape == (11,)
    assert r.logpost_trace.shape == (11,)
    assert r.step_size_trace.shape == (11,)
    assert r.warmup_lambda_trace.shape == (7,)
    assert r.warmup_accept_trace.shape == (7,)
    assert jnp.all(jnp.isfinite(r.lambda_trace))
    assert jnp.all(jnp.isfinite(r.nuclear_trace))
    assert jnp.all(jnp.isfinite(r.logpost_trace))


def test_lambda_sampling_mala_return_trace_shapes():
    Y, S, _ = _make_data(m=12, n=8, rank=2)
    r = mcmc_proximal_mala(
        Y, S, lambda_val=1.0, sample_lambda=True,
        n_warmup=7, n_samples=11, return_trace=True,
        proposal_svd_rank=3, proposal_svd_oversample=1,
        key=jax.random.PRNGKey(35),
    )
    assert r.lambda_trace.shape == (11,)
    assert r.lambda_accept_trace.shape == (11,)
    assert r.warmup_lambda_trace.shape == (7,)
    assert r.warmup_lambda_accept_trace.shape == (7,)
    assert jnp.all(jnp.isfinite(r.lambda_trace))
    assert jnp.all(jnp.isfinite(r.lambda_accept_trace))


def test_lowrank_proposal_requires_positive_rank():
    Y, S, _ = _make_data(m=12, n=8)
    with pytest.raises(ValueError, match="proposal_svd_rank"):
        mcmc_proximal_mala(Y, S, proposal_svd_rank=0)


# ---------------------------------------------------------------------------
# MALA: different random keys give different results
# ---------------------------------------------------------------------------


def test_mala_randomness():
    Y, S, _ = _make_data()
    r1 = mcmc_proximal_mala(Y, S, lambda_val=1.5, n_warmup=20, n_samples=50,
                            key=jax.random.PRNGKey(4))
    r2 = mcmc_proximal_mala(Y, S, lambda_val=1.5, n_warmup=20, n_samples=50,
                            key=jax.random.PRNGKey(5))
    assert not np.allclose(np.array(r1.mu), np.array(r2.mu)), \
        "MALA with different keys should give different results"


# ---------------------------------------------------------------------------
# Gibbs: different random keys give different results
# ---------------------------------------------------------------------------


def test_gibbs_randomness():
    Y, S, _ = _make_data()
    r1 = mcmc_gsm_gibbs(Y, S, n_warmup=20, n_samples=50, key=jax.random.PRNGKey(6))
    r2 = mcmc_gsm_gibbs(Y, S, n_warmup=20, n_samples=50, key=jax.random.PRNGKey(7))
    assert not np.allclose(np.array(r1.mu), np.array(r2.mu)), \
        "Gibbs with different keys should give different results"


# ---------------------------------------------------------------------------
# Convergence: posterior mean close to truth with dense observations
# ---------------------------------------------------------------------------


def test_mala_recovers_signal():
    """With many observations and low noise, posterior mean ≈ truth."""
    Y, S, X_true = _make_data(m=40, n=20, rank=2, seed=10)
    # Use iso_auto lambda as a reasonable value
    r_iso = matlap.matlap_lowrank_isotropic(Y, S, rank=5, max_iter=100)
    r = mcmc_proximal_mala(Y, S, float(r_iso.lambda_bar),
                           n_warmup=50, n_samples=200,
                           key=jax.random.PRNGKey(8))
    rmse = float(np.sqrt(np.mean((np.array(r.mu) - X_true) ** 2)))
    # RMSE should be substantially better than the noise level (0.5)
    assert rmse < 0.5, f"MALA RMSE {rmse:.3f} too large"


def test_gibbs_recovers_signal():
    """GSM Gibbs posterior mean should recover the low-rank signal."""
    Y, S, X_true = _make_data(m=40, n=20, rank=2, seed=11)
    r = mcmc_gsm_gibbs(Y, S, n_warmup=50, n_samples=200,
                       key=jax.random.PRNGKey(9))
    rmse = float(np.sqrt(np.mean((np.array(r.mu) - X_true) ** 2)))
    assert rmse < 0.5, f"Gibbs RMSE {rmse:.3f} too large"


# ---------------------------------------------------------------------------
# Missing data: fully missing matrix handled gracefully
# ---------------------------------------------------------------------------


def test_mala_fully_observed():
    """MALA handles fully observed matrix (no missing entries)."""
    Y, S, _ = _make_data(m=20, n=10, seed=12)
    S_full = jnp.where(jnp.isfinite(S), S, 0.5)  # fill in missing with σ=0.5
    Y_full = jnp.where(jnp.isfinite(S), Y, 0.0)
    r = mcmc_proximal_mala(Y_full, S_full, lambda_val=1.0,
                           n_warmup=10, n_samples=30,
                           key=jax.random.PRNGKey(13))
    assert np.all(np.isfinite(np.array(r.mu)))


def test_gibbs_fully_observed():
    """Gibbs handles fully observed matrix."""
    Y, S, _ = _make_data(m=20, n=10, seed=14)
    S_full = jnp.where(jnp.isfinite(S), S, 0.5)
    Y_full = jnp.where(jnp.isfinite(S), Y, 0.0)
    r = mcmc_gsm_gibbs(Y_full, S_full, n_warmup=10, n_samples=30,
                       key=jax.random.PRNGKey(15))
    assert np.all(np.isfinite(np.array(r.mu)))


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


def test_mcmc_exported():
    """MCMCResult and both samplers are accessible via the top-level package."""
    assert hasattr(matlap, "MCMCResult")
    assert hasattr(matlap, "mcmc_proximal_mala")
    assert hasattr(matlap, "mcmc_gsm_gibbs")
