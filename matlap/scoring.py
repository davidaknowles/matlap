"""Post-hoc scores for lambda selection on fixed-lambda CAVI runs.

All scores are O(mn) and analytical (no Monte Carlo):

1. ``closed_form_loo``: exact Gaussian leave-one-out log predictive density.
2. ``renyi_elbo``: analytical Rényi α-ELBO for factored Gaussian q.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def compute_iso_prior_var(
    V_r: jax.Array,
    d_r: jax.Array,
    delta: float,
    lambda_bar: float,
) -> jax.Array:
    """Diagonal of the prior covariance Ψ/λ² for the low-rank+isotropic model.

    The row prior is x_i ~ N(0, Ψ/λ²) where
    Ψ = V_r diag(d_r²) V_r^T + δ² (I − V_r V_r^T).

    The per-column marginal variance (Ψ_jj / λ²) is:
    v_j = (Σ_k d_r[k]² V_r[j,k]² + δ²) / λ².

    For the pure low-rank model (no isotropic component) set ``delta`` to a
    small positive value (e.g. 1e-6) so no column is zero.

    Args:
        V_r:        Factor loadings, shape (n, r); orthonormal columns.
        d_r:        Sqrt-eigenvalues of Ψ_r, shape (r,).
        delta:      Off-subspace scale δ (scalar > 0).
        lambda_bar: E_q[λ], scalar.

    Returns:
        prior_var: shape (n,), Ψ_jj / λ² for each column j.
    """
    lam = jnp.maximum(jnp.asarray(lambda_bar, dtype=jnp.float32), 1e-12)
    psi_diag = ((V_r * d_r) ** 2).sum(axis=1) + float(delta) ** 2  # (n,)
    return jnp.maximum(psi_diag / (lam ** 2), 1e-12)


# Backward-compatible alias
iso_prior_var = compute_iso_prior_var


def closed_form_loo(
    mu: jax.Array,
    diag_sigma: jax.Array,
    Y: jax.Array,
    S: jax.Array,
) -> float:
    """Analytical LOO cross-validation score for a diagonal Gaussian posterior.

    For the Gaussian model y_ij = x_ij + ε_ij (ε_ij ~ N(0, s²_ij)), the
    leave-one-entry-out predictive log-density under the diagonal variational
    posterior q(x_ij) = N(μ_ij, σ²_ij) is exact::

        log p^loo(y_ij) = -e²/(2 s² (1−r)) − ½ log(2π s²) + ½ log(1−r)

    where e_ij = y_ij − μ_ij and r_ij = σ²_ij / s²_ij ∈ (0, 1).

    This follows from the Gaussian LOO identity: removing observation (i,j)
    gives a posterior q^{−ij}(x_ij) ∝ q(x_ij) / p(y_ij|x_ij), whose
    marginal predictive density for y_ij is the formula above.

    Args:
        mu:          Posterior mean, shape (m, n).
        diag_sigma:  Diagonal posterior variances σ²_ij, shape (m, n).
        Y:           Observations, shape (m, n).  NaN or any value where missing.
        S:           Noise standard deviations, shape (m, n).  ``inf`` where missing.

    Returns:
        Sum of log LOO predictive densities over observed entries (higher = better).
    """
    obs_mask = jnp.isfinite(Y) & jnp.isfinite(S)
    S2 = jnp.where(obs_mask, S ** 2, 1.0)
    sigma2 = jnp.maximum(diag_sigma, 1e-12)
    e = jnp.where(obs_mask, Y - mu, 0.0)
    # Leverage ratio r = σ²/s²; clip away from 1 to avoid division by zero
    r = jnp.clip(sigma2 / S2, 0.0, 1.0 - 1e-6)
    log_loo = (
        -0.5 * e ** 2 / (S2 * (1.0 - r))
        - 0.5 * jnp.log(2.0 * jnp.pi * S2)
        + 0.5 * jnp.log1p(-r)
    )
    return float(jnp.where(obs_mask, log_loo, 0.0).sum())


def renyi_elbo(
    mu: jax.Array,
    diag_sigma: jax.Array,
    prior_var: jax.Array,
    Y: jax.Array,
    S: jax.Array,
    alpha: float = 0.5,
) -> float:
    """Analytical Rényi α-ELBO for a diagonal Gaussian posterior.

    Uses the closed-form correction to the standard diagonal-Gaussian ELBO:

        L_α = ELBO + (1/β) Σ_ij [ β² f_ij² / (2(1 - 2β g_ij)) - ½ log(1 - 2β g_ij) ]
        β = 1 - α
        f_ij = sqrt(σ²_ij) * (e_ij/s²_ij - μ_ij/v_j)
        g_ij = 0.5 * (1 - σ²_ij/s²_ij - σ²_ij/v_j)

    where e_ij = y_ij - μ_ij and v_j is the per-column prior variance.
    As α→1, the correction vanishes and L_α→ELBO.
    At α=0, this is the importance-weighted (IS/IWAE-style) objective.
    """
    if not (0.0 <= alpha < 1.0):
        raise ValueError(f"alpha must be in [0, 1), got {alpha}")

    beta = 1.0 - alpha
    obs_mask = jnp.isfinite(Y) & jnp.isfinite(S)
    sigma2 = jnp.maximum(diag_sigma, 1e-12)
    v = jnp.maximum(prior_var, 1e-12)  # (n,); broadcast over rows
    inv_s2 = jnp.where(obs_mask, 1.0 / (S ** 2), 0.0)
    e_obs = jnp.where(obs_mask, Y - mu, 0.0)

    # Standard ELBO for diagonal Gaussian q
    elbo_lik = jnp.where(
        obs_mask,
        -0.5 * ((e_obs ** 2 + sigma2) * inv_s2 + jnp.log(2.0 * jnp.pi * (S ** 2))),
        0.0,
    )
    elbo_prior = -0.5 * ((mu ** 2 + sigma2) / v[None, :] + jnp.log(2.0 * jnp.pi * v[None, :]))
    entropy = 0.5 * jnp.log(2.0 * jnp.pi * jnp.e * sigma2)
    elbo = elbo_lik.sum() + elbo_prior.sum() + entropy.sum()

    # Analytical Rényi correction (beta = 1 - alpha)
    sigma = jnp.sqrt(sigma2)
    f = sigma * (e_obs * inv_s2 - mu / v[None, :])

    # For missing entries inv_s2=0, so likelihood contribution drops out naturally
    g = 0.5 * (1.0 - sigma2 * inv_s2 - sigma2 / v[None, :])
    denom = jnp.maximum(1.0 - 2.0 * beta * g, 1e-12)
    correction = beta ** 2 * f ** 2 / (2.0 * denom) - 0.5 * jnp.log(denom)

    return float(elbo + (1.0 / beta) * correction.sum())
