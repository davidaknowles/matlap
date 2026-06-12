import jax.numpy as jnp
import numpy as np

from matlap.prox_noise import (
    proximal_noise_eb,
    proximal_noise_lambda_grid,
    taylor_diag_variance_at_prox,
)


def _toy_data(seed=0, m=12, n=8, g_true=0.25):
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((m, 3))
    V = rng.standard_normal((n, 3))
    X = (U @ V.T) / 3.0
    S = rng.uniform(0.3, 0.8, size=(m, n)).astype(np.float32)
    Y = X + rng.standard_normal((m, n)) * np.sqrt(S ** 2 + g_true)
    return jnp.asarray(Y, dtype=jnp.float32), jnp.asarray(S, dtype=jnp.float32), jnp.asarray(X, dtype=jnp.float32)


def test_taylor_diag_variance_at_prox_is_finite_and_correct_shape():
    Y, S, _ = _toy_data()
    sigma_diag = taylor_diag_variance_at_prox(Y, S, 1.0)
    assert sigma_diag.shape == Y.shape
    assert bool(jnp.all(jnp.isfinite(sigma_diag)))
    assert bool(jnp.all(sigma_diag > 0.0))


def test_proximal_noise_fixed_lambda_estimates_positive_g():
    Y, S, _ = _toy_data()
    res = proximal_noise_eb(
        Y,
        S,
        lambda_val=2.0,
        update_lambda=False,
        max_outer=4,
        prox_max_iter=8,
        gamma_max_iter=8,
    )
    assert res.X.shape == Y.shape
    assert res.sigma_diag.shape == Y.shape
    assert res.lambda_val == 2.0
    assert res.lambda_eff > 0.0
    assert res.g > 0.0
    assert len(res.g_trace) == res.n_iter


def test_proximal_noise_effective_lambda_is_default_parameterization():
    Y, S, _ = _toy_data()
    res = proximal_noise_eb(
        Y,
        S,
        lambda_val=2.0,
        update_lambda=False,
        max_outer=2,
        prox_max_iter=4,
        gamma_max_iter=4,
    )
    assert res.lambda_parameterization == "effective"
    assert res.lambda_val == 2.0
    assert res.lambda_eff == 2.0


def test_proximal_noise_base_lambda_parameterization_still_available():
    Y, S, _ = _toy_data()
    res = proximal_noise_eb(
        Y,
        S,
        lambda_val=2.0,
        update_lambda=False,
        max_outer=2,
        prox_max_iter=4,
        gamma_max_iter=4,
        lambda_parameterization="base",
    )
    assert res.lambda_parameterization == "base"
    assert res.lambda_val == 2.0
    assert res.lambda_eff > 0.0
    assert res.lambda_eff != res.lambda_val


def test_proximal_noise_joint_updates_lambda():
    Y, S, _ = _toy_data()
    res = proximal_noise_eb(
        Y,
        S,
        lambda_val=1.0,
        update_lambda=True,
        max_outer=4,
        prox_max_iter=8,
        gamma_max_iter=8,
    )
    assert res.X.shape == Y.shape
    assert res.lambda_val > 0.0
    assert res.lambda_eff > 0.0
    assert res.g > 0.0
    assert len(res.lambda_trace) == res.n_iter


def test_proximal_noise_lambda_grid_selects_result():
    Y, S, _ = _toy_data()
    grid = proximal_noise_lambda_grid(
        Y,
        S,
        [0.5, 1.0, 2.0],
        max_outer=3,
        prox_max_iter=6,
        gamma_max_iter=6,
    )
    assert len(grid.grid_results) == 3
    assert len(grid.rows) == 3
    assert grid.best.lambda_val in {0.5, 1.0, 2.0}


def test_proximal_noise_hutchinson_gamma_update_runs():
    Y, S, _ = _toy_data()
    res = proximal_noise_eb(
        Y,
        S,
        lambda_val=1.0,
        update_lambda=False,
        max_outer=3,
        prox_max_iter=6,
        gamma_max_iter=2,
        gamma_update="hutchinson",
        hutchinson_probes=2,
        hutchinson_cg_maxiter=10,
        hutchinson_lr=1e-4,
        score_exact_objective=False,
    )
    assert res.X.shape == Y.shape
    assert res.sigma_diag.shape == Y.shape
    assert res.gamma_update == "hutchinson"
    assert res.g > 0.0


def test_proximal_noise_transposes_wide_inputs_back():
    Y, S, _ = _toy_data(m=8, n=12)
    res = proximal_noise_eb(
        Y,
        S,
        lambda_val=1.0,
        update_lambda=False,
        max_outer=3,
        prox_max_iter=6,
        gamma_max_iter=6,
    )
    assert res.X.shape == Y.shape
    assert res.sigma_diag.shape == Y.shape
