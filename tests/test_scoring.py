"""Tests for analytical scoring functions used in lambda grid selection."""

import jax.numpy as jnp
import numpy as np

from matlap.scoring import closed_form_loo, compute_iso_prior_var, renyi_elbo


def _diag_elbo(mu, sigma2, prior_var, Y, S):
    """Reference diagonal-Gaussian ELBO used for a Rényi sanity check."""
    obs = jnp.isfinite(Y) & jnp.isfinite(S)
    inv_s2 = jnp.where(obs, 1.0 / (S ** 2), 0.0)
    e = jnp.where(obs, Y - mu, 0.0)
    sigma2 = jnp.maximum(sigma2, 1e-12)
    prior_var = jnp.maximum(prior_var, 1e-12)

    like = jnp.where(
        obs,
        -0.5 * ((e ** 2 + sigma2) * inv_s2 + jnp.log(2.0 * jnp.pi * (S ** 2))),
        0.0,
    ).sum()
    prior = -0.5 * (
        (mu ** 2 + sigma2) / prior_var[None, :]
        + jnp.log(2.0 * jnp.pi * prior_var[None, :])
    ).sum()
    entropy = (0.5 * jnp.log(2.0 * jnp.pi * jnp.e * sigma2)).sum()
    return float(like + prior + entropy)


def test_compute_iso_prior_var_shape_and_positive():
    n, r = 7, 3
    V_r = jnp.eye(n, r)
    d_r = jnp.array([2.0, 1.0, 0.5], dtype=jnp.float32)
    prior_var = compute_iso_prior_var(V_r, d_r, delta=0.25, lambda_bar=3.0)
    assert prior_var.shape == (n,)
    assert jnp.all(prior_var > 0)


def test_closed_form_loo_prefers_better_fit():
    Y = jnp.array([[1.0, -2.0], [0.5, 1.5]], dtype=jnp.float32)
    S = jnp.full_like(Y, 1.0)
    diag_sigma = jnp.full_like(Y, 0.1)

    good_mu = Y
    bad_mu = jnp.zeros_like(Y)

    loo_good = closed_form_loo(good_mu, diag_sigma, Y, S)
    loo_bad = closed_form_loo(bad_mu, diag_sigma, Y, S)
    assert loo_good > loo_bad


def test_renyi_elbo_tighter_than_elbo():
    rng = np.random.default_rng(0)
    Y = jnp.asarray(rng.normal(size=(4, 3)), dtype=jnp.float32)
    S = jnp.asarray(0.5 + rng.random(size=(4, 3)), dtype=jnp.float32)
    mu = 0.8 * Y
    sigma2 = jnp.full_like(Y, 0.05)
    prior_var = jnp.full((3,), 1.5, dtype=jnp.float32)

    elbo = _diag_elbo(mu, sigma2, prior_var, Y, S)
    renyi = renyi_elbo(mu, sigma2, prior_var, Y, S, alpha=0.5)

    assert np.isfinite(renyi)
    assert renyi >= elbo - 1e-6


def test_renyi_elbo_alpha_zero_is_finite():
    rng = np.random.default_rng(1)
    Y = jnp.asarray(rng.normal(size=(4, 3)), dtype=jnp.float32)
    S = jnp.asarray(0.5 + rng.random(size=(4, 3)), dtype=jnp.float32)
    mu = 0.8 * Y
    sigma2 = jnp.full_like(Y, 0.05)
    prior_var = jnp.full((3,), 1.5, dtype=jnp.float32)

    score = renyi_elbo(mu, sigma2, prior_var, Y, S, alpha=0.0)
    assert np.isfinite(score)
