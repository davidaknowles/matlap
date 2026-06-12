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


def test_proximal_gradient_accepts_warm_start():
    """Warm-started proximal solves should preserve shape and finite values."""
    Y, S, _ = _low_rank_data()
    high_lam = proximal_gradient(Y, S, 5.0, max_iter=20)
    warm = proximal_gradient(Y, S, 1.0, max_iter=20, init_X=high_lam.X)
    assert warm.X.shape == Y.shape
    assert jnp.all(jnp.isfinite(warm.X))


def test_proximal_gradient_randomized_svd_path():
    """Randomized SVT should run and return a reusable right basis."""
    Y, S, _ = _low_rank_data()
    r = proximal_gradient(
        Y, S, 1.0, max_iter=10, svd_rank=5, svd_n_iter=1, svd_oversample=2
    )
    assert r.X.shape == Y.shape
    assert r.svd_basis is not None
    assert r.svd_basis.shape[0] == Y.shape[1]
    assert jnp.all(jnp.isfinite(r.X))


def test_proximal_gradient_randomized_svd_warmstart_basis():
    """Randomized SVT should accept a basis from a previous solve."""
    Y, S, _ = _low_rank_data()
    r1 = proximal_gradient(
        Y, S, 3.0, max_iter=8, svd_rank=5, svd_n_iter=1, svd_oversample=2
    )
    r2 = proximal_gradient(
        Y,
        S,
        1.0,
        max_iter=8,
        init_X=r1.X,
        svd_rank=5,
        svd_n_iter=1,
        svd_oversample=2,
        init_svd_basis=r1.svd_basis,
    )
    assert r2.svd_basis is not None
    assert r2.svd_basis.shape[0] == Y.shape[1]
    assert jnp.all(jnp.isfinite(r2.X))


def test_proximal_gradient_randomized_svd_fixed_iter():
    """The scanned fixed-iteration path should support randomized SVT."""
    Y, S, _ = _low_rank_data()
    r = proximal_gradient(
        Y,
        S,
        1.0,
        max_iter=6,
        fixed_iter=True,
        svd_rank=5,
        svd_n_iter=1,
        svd_oversample=2,
    )
    assert r.n_iter == 6
    assert r.svd_basis is not None
    assert jnp.all(jnp.isfinite(r.X))


def test_proximal_gradient_adaptive_svd_rank_grows():
    """Adaptive rSVD should increase rank when all captured SVs survive."""
    Y, S, _ = _low_rank_data()
    r = proximal_gradient(
        Y,
        S,
        0.01,
        max_iter=4,
        svd_rank=2,
        svd_rank_adaptive=True,
        svd_rank_min=2,
        svd_rank_max=8,
        svd_rank_step=2,
        svd_n_iter=1,
        svd_oversample=2,
    )
    assert r.svd_rank_trace
    assert max(r.svd_rank_trace) > 2
    assert r.svd_rank is not None and r.svd_rank <= 8


def test_proximal_gradient_adaptive_svd_rank_shrinks():
    """Adaptive rSVD should decrease rank when many captured SVs are removed."""
    Y, S, _ = _low_rank_data()
    r = proximal_gradient(
        Y,
        S,
        100.0,
        max_iter=4,
        svd_rank=10,
        svd_rank_adaptive=True,
        svd_rank_min=2,
        svd_rank_max=10,
        svd_rank_step=2,
        svd_rank_shrink_fraction=0.5,
        svd_n_iter=1,
        svd_oversample=2,
    )
    assert r.svd_rank_trace
    assert min(r.svd_rank_trace) < 10
    assert r.svd_rank is not None and r.svd_rank >= 2


def test_proximal_gradient_adaptive_svd_rejects_fixed_iter():
    Y, S, _ = _low_rank_data()
    with pytest.raises(ValueError, match="fixed_iter"):
        proximal_gradient(
            Y,
            S,
            1.0,
            fixed_iter=True,
            svd_rank=5,
            svd_rank_adaptive=True,
        )


def test_proximal_gradient_monotone_fista_nonincreasing_objective():
    """Monotone FISTA should reject objective-increasing accelerated steps."""
    Y, S, _ = _low_rank_data()
    r = proximal_gradient(
        Y,
        S,
        1.0,
        max_iter=50,
        solver="monotone_fista",
        obj_tol=1e-8,
    )
    diffs = np.diff(np.asarray(r.loss_trace))
    assert np.all(diffs <= 1e-4)


def test_proximal_gradient_rejects_unknown_solver():
    Y, S, _ = _low_rank_data()
    with pytest.raises(ValueError, match="solver"):
        proximal_gradient(Y, S, 1.0, solver="not-a-solver")


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
