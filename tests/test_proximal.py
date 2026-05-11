"""Tests for matlap.proximal and matlap.cv."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from matlap.proximal import ProximalResult, proximal_cv, proximal_gradient
from matlap.cv import cv_lambda


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _low_rank_data(m=20, n=15, rank=2, noise=0.3, seed=0):
    key = jax.random.PRNGKey(seed)
    U = jax.random.normal(key, (m, rank))
    V = jax.random.normal(jax.random.fold_in(key, 1), (n, rank))
    X_true = U @ V.T
    Y = X_true + noise * jax.random.normal(jax.random.fold_in(key, 2), (m, n))
    S = jnp.full((m, n), noise)
    return Y, S, X_true


# ---------------------------------------------------------------------------
# proximal_gradient tests
# ---------------------------------------------------------------------------


def test_proximal_gradient_shape():
    Y, S, _ = _low_rank_data()
    r = proximal_gradient(Y, S, 1.0)
    assert isinstance(r, ProximalResult)
    assert r.X.shape == Y.shape


def test_proximal_gradient_reduces_objective():
    """Solver should produce a lower penalized objective than the X=Y baseline.

    At X=Y, the smooth (data-fit) term is zero but the nuclear norm penalty is
    at its maximum.  FISTA should trade off some data fit for lower nuclear norm
    and arrive at a strictly better composite objective.
    """
    Y, S, _ = _low_rank_data()
    lambda_val = 2.0
    r = proximal_gradient(Y, S, lambda_val, max_iter=500, tol=1e-8)
    obj_Y = lambda_val * float(jnp.linalg.svd(Y, compute_uv=False).sum())
    assert r.loss_trace[-1] < obj_Y - 1e-4, (
        f"Optimized obj={r.loss_trace[-1]:.4f} not less than X=Y baseline={obj_Y:.4f}"
    )


def test_proximal_gradient_low_rank_recovery():
    """With low noise, the estimate should be close to X_true."""
    Y, S, X_true = _low_rank_data(noise=0.1)
    r = proximal_gradient(Y, S, 0.2, max_iter=500)
    rmse = float(jnp.sqrt(jnp.mean((r.X - X_true) ** 2)))
    assert rmse < 1.0, f"RMSE={rmse:.4f} too large"


def test_proximal_gradient_missing_data():
    """Setting S=inf for some entries should not cause NaN."""
    Y, S, _ = _low_rank_data()
    S = S.at[0, :].set(jnp.inf)   # entire first row missing
    S = S.at[:, 0].set(jnp.inf)   # entire first column missing
    r = proximal_gradient(Y, S, 0.5)
    assert jnp.all(jnp.isfinite(r.X))


def test_proximal_gradient_high_lambda_gives_low_rank():
    """Very large lambda should drive most singular values to zero."""
    Y, S, _ = _low_rank_data()
    r = proximal_gradient(Y, S, 100.0, max_iter=500)
    sv = jnp.linalg.svd(r.X, compute_uv=False)
    n_nonzero = int(jnp.sum(sv > 1e-3))
    assert n_nonzero <= 3, f"Expected low rank, got {n_nonzero} non-zero singular values"


def test_proximal_gradient_zero_lambda_recovers_obs():
    """lambda=0 with no missing data should fit the observations exactly."""
    Y, S, _ = _low_rank_data()
    r = proximal_gradient(Y, S, 0.0, max_iter=500)
    # With lambda=0 and fully observed data, optimal X = Y
    residual = float(jnp.max(jnp.abs(r.X - Y)))
    assert residual < 0.1, f"Max residual={residual:.4f}"


# ---------------------------------------------------------------------------
# cv_lambda tests
# ---------------------------------------------------------------------------


def test_cv_lambda_returns_observed_lambda():
    """Selected lambda must be one of the supplied grid values."""
    Y, S, _ = _low_rank_data()
    grid = jnp.array([0.1, 0.5, 1.0, 2.0, 5.0])
    best_lam, _ = cv_lambda(Y, S, grid, proximal_gradient, lambda r: r.X, n_folds=3)
    assert any(abs(best_lam - float(g)) < 1e-5 for g in grid)


def test_cv_lambda_result_shape():
    Y, S, _ = _low_rank_data()
    grid = [0.5, 1.0, 2.0]
    _, res = cv_lambda(Y, S, grid, proximal_gradient, lambda r: r.X, n_folds=3)
    assert res.X.shape == Y.shape


def test_cv_lambda_beats_bad_lambda():
    """CV-selected lambda should fit held-out entries better than a clearly wrong one."""
    Y, S, X_true = _low_rank_data(noise=0.2, seed=7)
    grid = jnp.logspace(-1, 1, 8)
    best_lam, res_cv = cv_lambda(Y, S, grid, proximal_gradient, lambda r: r.X, n_folds=4)

    # Compare against a very large lambda (over-regularised)
    res_bad = proximal_gradient(Y, S, 50.0)
    rmse_cv = float(jnp.sqrt(jnp.mean((res_cv.X - X_true) ** 2)))
    rmse_bad = float(jnp.sqrt(jnp.mean((res_bad.X - X_true) ** 2)))
    assert rmse_cv < rmse_bad, f"CV RMSE={rmse_cv:.4f} not better than bad RMSE={rmse_bad:.4f}"


def test_cv_lambda_with_missing_data():
    """CV should work when some entries are missing (S=inf)."""
    Y, S, _ = _low_rank_data()
    # Mark ~15% of entries missing
    key = jax.random.PRNGKey(99)
    mask = jax.random.uniform(key, Y.shape) < 0.15
    S = jnp.where(mask, jnp.inf, S)
    grid = [0.5, 1.0, 2.0]
    best_lam, res = cv_lambda(Y, S, grid, proximal_gradient, lambda r: r.X, n_folds=3)
    assert jnp.all(jnp.isfinite(res.X))


def test_cv_lambda_auto_get_mu_via_fit_vi():
    """cv_lambda auto-detects .mu attribute (VIResult) without explicit get_mu."""
    from matlap.vi import fit_vi  # noqa: PLC0415

    Y, S, _ = _low_rank_data(m=10, n=8)
    grid = [0.5, 1.0, 2.0]

    def _fit(Y, S, lam):
        return fit_vi(Y, S, lambda_val=lam, guide_type="diagonal",
                      n_steps=100, lr=5e-3)

    best_lam, res = cv_lambda(Y, S, grid, _fit, n_folds=3)
    assert any(abs(best_lam - g) < 1e-5 for g in grid)
    assert res.mu.shape == Y.shape


def test_cv_lambda_proximal_cv_wrapper():
    """proximal_cv should give the same result as cv_lambda directly."""
    Y, S, _ = _low_rank_data()
    grid = jnp.array([0.5, 1.0, 2.0])
    lam1, _ = proximal_cv(Y, S, grid, n_folds=3)
    lam2, _ = cv_lambda(Y, S, grid, proximal_gradient, lambda r: r.X, n_folds=3)
    assert lam1 == lam2
