"""Tests for matlap.taylor."""

import jax
import jax.numpy as jnp
import pytest

from matlap.scoring import make_elbo_scorer, make_renyi_scorer
from matlap.taylor import TaylorResult, taylor_cv, taylor_gradient


def _low_rank_data(m=12, n=8, rank=2, noise=0.3, seed=0):
    key = jax.random.PRNGKey(seed)
    U = jax.random.normal(key, (m, rank))
    V = jax.random.normal(jax.random.fold_in(key, 1), (n, rank))
    X_true = U @ V.T
    Y = X_true + noise * jax.random.normal(jax.random.fold_in(key, 2), (m, n))
    S = jnp.full((m, n), noise)
    return Y, S, X_true


def test_taylor_gradient_shape_and_sigma():
    Y, S, _ = _low_rank_data()
    r = taylor_gradient(Y, S, 1.0, max_iter=20)
    assert isinstance(r, TaylorResult)
    assert r.mu.shape == Y.shape
    assert r.sigma is not None
    assert r.sigma.shape == (Y.shape[0], Y.shape[1], Y.shape[1])


def test_taylor_gradient_reduces_loss():
    Y, S, _ = _low_rank_data()
    r = taylor_gradient(Y, S, 1.0, max_iter=50, tol=1e-8, recover_sigma=False)
    assert len(r.loss_trace) >= 2
    assert r.loss_trace[-1] <= r.loss_trace[0]


def test_taylor_gradient_finite_with_missing_data():
    Y, S, _ = _low_rank_data()
    S = S.at[0, :].set(jnp.inf)
    S = S.at[:, 0].set(jnp.inf)
    r = taylor_gradient(Y, S, 1.0, max_iter=20)
    assert jnp.all(jnp.isfinite(r.mu))
    assert r.sigma is not None
    assert jnp.all(jnp.isfinite(r.sigma))


def test_taylor_covariances_are_positive_definite():
    Y, S, _ = _low_rank_data()
    r = taylor_gradient(Y, S, 1.0, max_iter=20)
    eigvals = jnp.linalg.eigvalsh(r.sigma)
    assert float(jnp.min(eigvals)) > 0.0


def test_taylor_randomized_svd_warmstart_path():
    Y, S, _ = _low_rank_data(m=14, n=9)
    r1 = taylor_gradient(
        Y, S, 1.0, max_iter=10, svd_rank=4, recover_sigma=False,
    )
    assert r1.svd_basis is not None
    assert r1.svd_basis.shape[0] == Y.shape[1]
    assert jnp.all(jnp.isfinite(r1.mu))

    r2 = taylor_gradient(
        Y, S, 1.0, max_iter=5, init_mu=r1.mu,
        svd_rank=4, init_svd_basis=r1.svd_basis, recover_sigma=False,
    )
    assert r2.svd_basis is not None
    assert jnp.all(jnp.isfinite(r2.mu))




def test_taylor_returns_lambda_selection_scores():
    Y, S, _ = _low_rank_data()
    r = taylor_gradient(Y, S, 1.0, max_iter=20, recover_sigma=False)
    assert r.sigma is None
    assert r.sigma_diag is not None
    assert r.sigma_diag.shape == Y.shape
    assert r.prior_var is not None
    assert r.prior_var.shape == (Y.shape[1],)
    assert jnp.all(jnp.isfinite(r.sigma_diag))
    assert jnp.all(jnp.isfinite(r.prior_var))
    assert jnp.isfinite(jnp.asarray(r.elbo))
    assert jnp.isfinite(jnp.asarray(r.renyi_elbo))
    assert len(r.elbo_trace) > 0
    assert abs(r.elbo_trace[-1] - r.elbo) < 1e-4


def test_taylor_scores_work_with_scorer_factories():
    Y, S, _ = _low_rank_data()
    r = taylor_gradient(Y, S, 1.0, max_iter=20, recover_sigma=False)
    elbo_score = make_elbo_scorer()(r, Y, S, 1.0)
    renyi_score = make_renyi_scorer(alpha=0.5)(r, Y, S, 1.0)
    assert abs(elbo_score - r.elbo) < 1e-4
    assert jnp.isfinite(jnp.asarray(renyi_score))
    assert abs(float(renyi_score) - r.renyi_elbo) < 1e-4

def test_taylor_cv_returns_observed_lambda():
    Y, S, _ = _low_rank_data(m=10, n=6)
    grid = jnp.array([0.5, 1.0, 2.0])
    best_lam, res = taylor_cv(
        Y, S, grid, n_folds=3, max_iter=15, recover_sigma=False,
    )
    assert any(abs(best_lam - float(g)) < 1e-5 for g in grid)
    assert res.mu.shape == Y.shape


def test_taylor_requires_positive_lambda():
    Y, S, _ = _low_rank_data()
    with pytest.raises(ValueError, match="lambda_val"):
        taylor_gradient(Y, S, 0.0)


def test_taylor_requires_valid_renyi_alpha():
    Y, S, _ = _low_rank_data()
    with pytest.raises(ValueError, match="renyi_alpha"):
        taylor_gradient(Y, S, 1.0, renyi_alpha=1.0)
