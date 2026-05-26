"""Post-hoc scores for lambda selection on fixed-lambda CAVI runs.

All scores are O(mn) and analytical (no Monte Carlo):

1. ``closed_form_loo``: exact Gaussian leave-one-out log predictive density.
2. ``renyi_elbo``: analytical Rényi α-ELBO for factored Gaussian q (JAX array).
3. ``renyi_lambda_opt``: argmax_λ R_α(λ | q) via BFGS on log λ.

Score factories (for use with :func:`~matlap.adaptive.adaptive_lambda_search`):

4. ``make_elbo_scorer``: returns a scorer that reads ``result.elbo_trace[-1]``.
5. ``make_loo_scorer``: returns a scorer using ``closed_form_loo``.
6. ``make_renyi_scorer``: returns a scorer using ``renyi_elbo``.

All scorer factories handle both low-rank results (``diag_sigma``, ``V_r``,
``d_r``) and full-rank batched results (``sigma_diag`` field name; prior
variance estimated as ``diag(Ψ)/λ²`` with Ψ = Σᵢ(μᵢμᵢᵀ + Σᵢ)).
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp


def compute_iso_prior_var(
    V_r: jax.Array,
    d_r: jax.Array,
    delta: float,
    lambda_bar: float,
) -> jax.Array:
    """Diagonal of the prior covariance Ψ/λ² for the low-rank+isotropic model.

    Used as a heuristic prior-variance proxy for Rényi α-ELBO and LOO scoring.
    Ψ = Q² = V_r diag(d_r²) V_r^T + δ² (I − V_r V_r^T) is the posterior
    second-moment matrix; Ψ_jj/λ² decays as ~1/λ⁴ at large λ (since d_r → 0),
    which gives the Rényi score a well-defined peak at a finite λ.

    The per-column proxy (Ψ_jj / λ²) is:
        v_j = (Σ_k d_r[k]² V_r[j,k]² + δ²) / λ².

    For the pure low-rank model (no isotropic component) set ``delta`` to a
    small positive value (e.g. 1e-6) so no column is zero.

    Args:
        V_r:        Factor loadings, shape (n, r); orthonormal columns.
        d_r:        Sqrt-eigenvalues of Ψ_r (= Q-eigenvalues in-subspace), shape (r,).
        delta:      Off-subspace posterior δ (scalar > 0).
        lambda_bar: E_q[λ], scalar.

    Returns:
        prior_var: shape (n,), Ψ_jj / λ² for each column j.
    """
    lam = jnp.maximum(jnp.asarray(lambda_bar, dtype=jnp.float32), 1e-12)
    psi_diag = ((V_r * d_r) ** 2).sum(axis=1) + float(delta) ** 2  # (n,)
    return jnp.maximum(psi_diag / (lam ** 2), 1e-12)


# Backward-compatible alias
iso_prior_var = compute_iso_prior_var


def _get_diag_sigma(result: Any) -> jax.Array:
    """Return diagonal posterior variance from any result type.

    Handles both low-rank results (field ``diag_sigma``) and full-rank batched
    results (field ``sigma_diag``).
    """
    sigma = getattr(result, "diag_sigma", None)
    if sigma is None:
        sigma = result.sigma_diag
    return sigma


def _get_prior_var(result: Any, lam: float, delta_fallback: float) -> jax.Array:
    """Return per-column prior variance proxy for any result type.

    * Low-rank results (have ``V_r``): ``compute_iso_prior_var`` with ``d_r``
      and ``delta`` (falls back to ``delta_fallback`` if absent).
    * Full-rank batched results: ``diag(√Ψ)/λ`` where Ψ = Σᵢ E[xᵢxᵢᵀ] and
      ``psi_sqrt_diag = diag(√Ψ)`` is stored in the result.  This matches the
      actual prior covariance diagonal Q_jj/λ (Q = √Ψ = prior covariance matrix
      times λ).
    """
    if hasattr(result, "V_r"):
        delta = getattr(result, "delta", delta_fallback)
        return compute_iso_prior_var(result.V_r, result.d_r, delta=delta, lambda_bar=lam)
    lam_safe = jnp.maximum(jnp.asarray(lam, dtype=jnp.float32), 1e-12)
    return jnp.maximum(result.psi_sqrt_diag / lam_safe, 1e-12)


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

    return elbo + (1.0 / beta) * correction.sum()

def renyi_lambda_opt(
    V_r: jax.Array,
    d_r: jax.Array,
    delta: float,
    mu: jax.Array,
    diag_sigma: jax.Array,
    Y: jax.Array,
    S: jax.Array,
    alpha: float = 0.5,
    lambda_init: float = 1.0,
) -> float:
    """Find λ* = argmax_λ R_α(λ | q) by BFGS on log λ.

    Given a fixed variational posterior q(X) = N(mu, diag_sigma) from the
    low-rank+isotropic CAVI, finds the λ that maximises the Rényi α-ELBO.
    Optimises in log space to keep λ > 0.

    Args:
        V_r:          Factor loadings, shape (n, r); orthonormal columns.
        d_r:          Sqrt-eigenvalues of Ψ_r, shape (r,).
        delta:        Off-subspace scale δ (scalar > 0).
        mu:           Posterior mean, shape (m, n).
        diag_sigma:   Diagonal posterior variances, shape (m, n).
        Y:            Observations, shape (m, n).  NaN where missing.
        S:            Noise std devs, shape (m, n).  inf where missing.
        alpha:        Rényi order; must satisfy 0 ≤ α < 1 (default 0.5).
        lambda_init:  Starting λ for the optimiser (default 1.0).

    Returns:
        Optimal λ* as a Python float.
    """
    V_r = jnp.asarray(V_r, dtype=jnp.float32)
    d_r = jnp.asarray(d_r, dtype=jnp.float32)
    mu = jnp.asarray(mu, dtype=jnp.float32)
    diag_sigma = jnp.asarray(diag_sigma, dtype=jnp.float32)
    Y = jnp.asarray(Y, dtype=jnp.float32)
    S = jnp.asarray(S, dtype=jnp.float32)
    delta_f = float(delta)

    def neg_score(log_lam_arr: jax.Array) -> jax.Array:
        lam = jnp.exp(log_lam_arr[0])
        prior_var = compute_iso_prior_var(V_r, d_r, delta=delta_f, lambda_bar=lam)
        return -renyi_elbo(mu, diag_sigma, prior_var, Y, S, alpha=alpha)

    log_lam_min = jnp.log(jnp.asarray(1e-3))   # λ ≥ 1e-3
    log_lam_max = jnp.log(jnp.asarray(1e5))    # λ ≤ 1e5
    x0 = jnp.array([jnp.clip(jnp.log(jnp.maximum(jnp.asarray(lambda_init), 1e-6)),
                              log_lam_min, log_lam_max)])
    opt = jax.scipy.optimize.minimize(neg_score, x0, method="BFGS")
    log_lam_opt = jnp.clip(opt.x[0], log_lam_min, log_lam_max)
    return float(jnp.exp(log_lam_opt))




def make_elbo_scorer() -> Callable[[Any, jax.Array, jax.Array, float], float]:
    """Return a scorer that reads ``result.elbo_trace[-1]``.

    Works with any result object that has an ``elbo_trace`` attribute.

    Returns:
        ``score_fn(result, Y, S, lam) -> float``
    """
    def score(result: Any, Y: jax.Array, S: jax.Array, lam: float) -> float:
        return float(result.elbo_trace[-1])
    return score


def make_loo_scorer() -> Callable[[Any, jax.Array, jax.Array, float], float]:
    """Return a scorer that computes the analytical closed-form LOO score.

    Works with low-rank results (field ``diag_sigma``) and full-rank batched
    results (field ``sigma_diag``).

    Returns:
        ``score_fn(result, Y, S, lam) -> float``
    """
    def score(result: Any, Y: jax.Array, S: jax.Array, lam: float) -> float:
        return closed_form_loo(result.mu, _get_diag_sigma(result), Y, S)
    return score


def make_renyi_scorer(
    alpha: float = 0.5,
    delta_fallback: float = 1e-6,
) -> Callable[[Any, jax.Array, jax.Array, float], float]:
    """Return a scorer that computes the analytical Rényi α-ELBO.

    Works with:

    * :class:`~matlap.core.LowRankIsotropicResult` — uses ``result.delta``,
      ``result.V_r``, ``result.d_r`` via :func:`compute_iso_prior_var`.
    * :class:`~matlap.core.LowRankCAVIResult` — falls back to
      ``delta_fallback`` for the off-subspace component.
    * :class:`~matlap.core.BatchedCAVIResult` — prior variance estimated as
      ``diag(Ψ)/λ²`` where ``Ψ_jj = Σᵢ(μᵢⱼ² + σᵢⱼ²)``.

    Args:
        alpha:          Rényi order; must satisfy 0 ≤ α < 1 (default 0.5).
        delta_fallback: Off-subspace scale used when ``result`` has no ``.delta``
                        attribute (e.g. plain low-rank model).

    Returns:
        ``score_fn(result, Y, S, lam) -> float``
    """
    if not (0.0 <= alpha < 1.0):
        raise ValueError(f"alpha must be in [0, 1), got {alpha}")

    def score(result: Any, Y: jax.Array, S: jax.Array, lam: float) -> float:
        sigma = _get_diag_sigma(result)
        prior_var = _get_prior_var(result, lam, delta_fallback)
        return renyi_elbo(result.mu, sigma, prior_var, Y, S, alpha=alpha)
    return score
