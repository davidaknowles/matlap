"""Tests for matlap.vi (Numpyro SVI)."""

import jax
import jax.numpy as jnp
import pytest

from matlap.vi import VIResult, fit_vi
from matlap.cv import cv_lambda


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _low_rank_data(m=12, n=8, rank=2, noise=0.3, seed=0):
    key = jax.random.PRNGKey(seed)
    U = jax.random.normal(key, (m, rank))
    V = jax.random.normal(jax.random.fold_in(key, 1), (n, rank))
    X_true = U @ V.T
    Y = X_true + noise * jax.random.normal(jax.random.fold_in(key, 2), (m, n))
    S = jnp.full((m, n), noise)
    return Y, S, X_true


# ---------------------------------------------------------------------------
# Basic shape / API tests (all three guides)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("guide_type", ["diagonal", "row_mvn", "matrix_normal"])
def test_vi_shape(guide_type):
    Y, S, _ = _low_rank_data()
    r = fit_vi(Y, S, lambda_val=1.0, guide_type=guide_type, n_steps=100)
    assert isinstance(r, VIResult)
    assert r.mu.shape == Y.shape


@pytest.mark.parametrize("guide_type", ["diagonal", "row_mvn", "matrix_normal"])
def test_vi_finite_output(guide_type):
    """Posterior mean and ELBO should be finite."""
    Y, S, _ = _low_rank_data()
    r = fit_vi(Y, S, lambda_val=1.0, guide_type=guide_type, n_steps=200)
    assert jnp.all(jnp.isfinite(r.mu)), f"{guide_type}: mu has non-finite values"
    assert all(jnp.isfinite(jnp.array(r.elbo_trace))), f"{guide_type}: ELBO trace has non-finite values"


@pytest.mark.parametrize("guide_type", ["diagonal", "row_mvn", "matrix_normal"])
def test_vi_elbo_increases(guide_type):
    """ELBO should trend upward over SVI steps (allow small jitter)."""
    Y, S, _ = _low_rank_data()
    r = fit_vi(Y, S, lambda_val=1.0, guide_type=guide_type,
               n_steps=500, record_every=50)
    trace = r.elbo_trace
    assert len(trace) >= 3
    # Final ELBO should be higher than the initial ELBO
    assert trace[-1] > trace[0], (
        f"{guide_type}: ELBO did not increase: {trace[0]:.2f} -> {trace[-1]:.2f}"
    )


# ---------------------------------------------------------------------------
# Lambda estimation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("guide_type", ["diagonal", "row_mvn", "matrix_normal"])
def test_vi_auto_lambda_positive(guide_type):
    """When lambda_val=None, estimated lambda should be positive and finite."""
    Y, S, _ = _low_rank_data()
    r = fit_vi(Y, S, lambda_val=None, guide_type=guide_type, n_steps=200)
    assert r.lambda_bar > 0
    assert jnp.isfinite(r.lambda_bar)


def test_vi_fixed_lambda_respected():
    """When lambda_val is provided, lambda_bar must equal it."""
    Y, S, _ = _low_rank_data()
    for lam in [0.5, 2.0]:
        r = fit_vi(Y, S, lambda_val=lam, guide_type="diagonal", n_steps=50)
        assert abs(r.lambda_bar - lam) < 1e-5, f"Expected {lam}, got {r.lambda_bar}"


# ---------------------------------------------------------------------------
# Missing data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("guide_type", ["diagonal", "row_mvn", "matrix_normal"])
def test_vi_missing_data_finite(guide_type):
    """Missing entries (S=inf) should not cause NaN in the posterior mean."""
    Y, S, _ = _low_rank_data()
    S = S.at[0, :].set(jnp.inf)
    S = S.at[:, 0].set(jnp.inf)
    r = fit_vi(Y, S, lambda_val=1.0, guide_type=guide_type, n_steps=100)
    assert jnp.all(jnp.isfinite(r.mu))


# ---------------------------------------------------------------------------
# Bad guide_type raises
# ---------------------------------------------------------------------------


def test_vi_bad_guide_type_raises():
    Y, S, _ = _low_rank_data()
    with pytest.raises(ValueError, match="guide_type"):
        fit_vi(Y, S, lambda_val=1.0, guide_type="nonexistent")


# ---------------------------------------------------------------------------
# CV integration
# ---------------------------------------------------------------------------


def test_vi_cv_lambda():
    """cv_lambda should work with fit_vi (auto get_mu via .mu attribute)."""
    Y, S, _ = _low_rank_data()
    grid = [0.5, 1.0, 2.0]

    def _fit(Y, S, lam):
        return fit_vi(Y, S, lambda_val=lam, guide_type="diagonal",
                      n_steps=150, lr=5e-3)

    best_lam, res = cv_lambda(Y, S, grid, _fit, n_folds=3)
    assert any(abs(best_lam - g) < 1e-5 for g in grid)
    assert res.mu.shape == Y.shape
