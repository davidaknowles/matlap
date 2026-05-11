"""Tests for new scalable VI guides: row_lowrank and matrix_factor."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import matlap
from matlap import fit_vi

jax.config.update("jax_enable_x64", False)

RNG = np.random.default_rng(77)


def make_low_rank(m, n, rank, noise_std, rng):
    U = rng.standard_normal((m, rank)).astype(np.float32)
    V = rng.standard_normal((n, rank)).astype(np.float32)
    X_true = U @ V.T
    noise = (rng.standard_normal((m, n)) * noise_std).astype(np.float32)
    Y = X_true + noise
    S = np.full((m, n), noise_std, dtype=np.float32)
    return jnp.asarray(Y), jnp.asarray(S), jnp.asarray(X_true)


# ---------------------------------------------------------------------------
# row_lowrank guide
# ---------------------------------------------------------------------------


def test_vi_row_lowrank_runs():
    """row_lowrank guide should run without error."""
    Y, S, _ = make_low_rank(10, 6, rank=2, noise_std=0.5, rng=RNG)
    result = fit_vi(Y, S, guide_type="row_lowrank", n_steps=50, guide_rank=3)
    assert result.mu.shape == Y.shape
    assert jnp.all(jnp.isfinite(result.mu))


def test_vi_row_lowrank_elbo_finite():
    """ELBO trace should be finite."""
    Y, S, _ = make_low_rank(10, 6, rank=2, noise_std=0.5, rng=RNG)
    result = fit_vi(Y, S, guide_type="row_lowrank", n_steps=100, guide_rank=3)
    assert all(np.isfinite(e) for e in result.elbo_trace)


def test_vi_row_lowrank_elbo_improves():
    """ELBO should trend upward: mean of last half > mean of first half."""
    rng = np.random.default_rng(55)
    Y, S, _ = make_low_rank(12, 7, rank=2, noise_std=0.5, rng=rng)
    result = fit_vi(Y, S, guide_type="row_lowrank", n_steps=500, guide_rank=3, record_every=25)
    elbo = result.elbo_trace
    mid = len(elbo) // 2
    assert np.mean(elbo[mid:]) > np.mean(elbo[:mid]), (
        f"ELBO did not trend up: first_half_mean={np.mean(elbo[:mid]):.4f}, "
        f"last_half_mean={np.mean(elbo[mid:]):.4f}"
    )


def test_vi_row_lowrank_fixed_lambda():
    """row_lowrank with fixed lambda should work."""
    Y, S, _ = make_low_rank(10, 6, rank=2, noise_std=0.5, rng=RNG)
    result = fit_vi(Y, S, guide_type="row_lowrank", n_steps=50, guide_rank=3, lambda_val=1.0)
    assert result.lambda_bar == pytest.approx(1.0)
    assert jnp.all(jnp.isfinite(result.mu))


def test_vi_row_lowrank_with_missing():
    """row_lowrank should handle missing data."""
    rng = np.random.default_rng(80)
    Y, S, _ = make_low_rank(10, 6, rank=2, noise_std=0.5, rng=rng)
    mask = rng.random(Y.shape) < 0.2
    S = S.at[jnp.asarray(mask)].set(jnp.inf)
    result = fit_vi(Y, S, guide_type="row_lowrank", n_steps=50, guide_rank=3)
    assert jnp.all(jnp.isfinite(result.mu))


# ---------------------------------------------------------------------------
# matrix_factor guide
# ---------------------------------------------------------------------------


def test_vi_matrix_factor_runs():
    """matrix_factor guide should run without error."""
    Y, S, _ = make_low_rank(10, 6, rank=2, noise_std=0.5, rng=RNG)
    result = fit_vi(Y, S, guide_type="matrix_factor", n_steps=50, guide_rank=3)
    assert result.mu.shape == Y.shape
    assert jnp.all(jnp.isfinite(result.mu))


def test_vi_matrix_factor_elbo_finite():
    """ELBO trace should be finite."""
    Y, S, _ = make_low_rank(10, 6, rank=2, noise_std=0.5, rng=RNG)
    result = fit_vi(Y, S, guide_type="matrix_factor", n_steps=100, guide_rank=3)
    assert all(np.isfinite(e) for e in result.elbo_trace)


def test_vi_matrix_factor_elbo_improves():
    """ELBO should trend upward: mean of last half > mean of first half."""
    rng = np.random.default_rng(56)
    Y, S, _ = make_low_rank(12, 7, rank=2, noise_std=0.5, rng=rng)
    result = fit_vi(Y, S, guide_type="matrix_factor", n_steps=500, guide_rank=3, record_every=25)
    elbo = result.elbo_trace
    mid = len(elbo) // 2
    assert np.mean(elbo[mid:]) > np.mean(elbo[:mid]), (
        f"ELBO did not trend up: first_half_mean={np.mean(elbo[:mid]):.4f}, "
        f"last_half_mean={np.mean(elbo[mid:]):.4f}"
    )


def test_vi_matrix_factor_fixed_lambda():
    """matrix_factor with fixed lambda should work."""
    Y, S, _ = make_low_rank(10, 6, rank=2, noise_std=0.5, rng=RNG)
    result = fit_vi(Y, S, guide_type="matrix_factor", n_steps=50, guide_rank=3, lambda_val=0.5)
    assert result.lambda_bar == pytest.approx(0.5)


def test_vi_matrix_factor_with_missing():
    """matrix_factor should handle missing data."""
    rng = np.random.default_rng(81)
    Y, S, _ = make_low_rank(10, 6, rank=2, noise_std=0.5, rng=rng)
    mask = rng.random(Y.shape) < 0.2
    S = S.at[jnp.asarray(mask)].set(jnp.inf)
    result = fit_vi(Y, S, guide_type="matrix_factor", n_steps=50, guide_rank=3)
    assert jnp.all(jnp.isfinite(result.mu))


# ---------------------------------------------------------------------------
# approx_rank in model (rSVD nuclear norm)
# ---------------------------------------------------------------------------


def test_vi_diagonal_approx_rank_runs():
    """approx_rank > 0 should activate rSVD nuclear norm in the model."""
    Y, S, _ = make_low_rank(10, 6, rank=2, noise_std=0.5, rng=RNG)
    result = fit_vi(Y, S, guide_type="diagonal", n_steps=50, approx_rank=4)
    assert jnp.all(jnp.isfinite(result.mu))


def test_vi_approx_rank_elbo_finite():
    """approx_rank model should produce finite ELBO values."""
    Y, S, _ = make_low_rank(10, 6, rank=2, noise_std=0.5, rng=RNG)
    result = fit_vi(Y, S, guide_type="diagonal", n_steps=100, approx_rank=4, record_every=20)
    assert all(np.isfinite(e) for e in result.elbo_trace)


# ---------------------------------------------------------------------------
# Existing guide types still work with new approx_rank arg
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("guide_type", ["diagonal", "row_mvn", "matrix_normal"])
def test_existing_guides_still_work(guide_type):
    """Existing guides should still work after adding approx_rank parameter."""
    rng = np.random.default_rng(90)
    Y, S, _ = make_low_rank(8, 5, rank=2, noise_std=0.5, rng=rng)
    result = fit_vi(Y, S, guide_type=guide_type, n_steps=30)
    assert jnp.all(jnp.isfinite(result.mu))


def test_invalid_guide_type_raises():
    """Invalid guide_type should raise ValueError."""
    Y, S, _ = make_low_rank(6, 4, rank=1, noise_std=0.5, rng=RNG)
    with pytest.raises(ValueError, match="guide_type"):
        fit_vi(Y, S, guide_type="nonexistent", n_steps=5)
