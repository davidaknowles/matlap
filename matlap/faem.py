"""
Factor Analysis EM and Gradient Marginal Likelihood for matrix denoising.

Two alternative optimisation approaches compared to CAVI:

1. ``matlap_faem``: Factor Analysis EM.  The E-step is identical to the
   low-rank CAVI E-step (exact posterior given W_r, λ).  The M-step updates
   W_r via per-column weighted least squares, allowing the **column space**
   of W_r to evolve freely — unlike the CAVI implementation, which freezes
   col(V_r) at the initial rSVD and only rotates within it.

2. ``matlap_gradml``: Direct gradient ascent on the marginal log-likelihood
   via JAX autodiff + optax Adam.  No closed-form M-step; W_r and log λ are
   treated as free parameters and optimised jointly.

Both methods optimise the same objective:

    log p(Y_Ω | W_r, λ) = Σ_i log N(y_i_obs; 0, W_r W_r^T/λ + S_i_obs)

i.e. the Gaussian marginal likelihood obtained by integrating out the
latent rows z_i ~ N(0, I/λ).  For the Gaussian model this marginal LL
*equals* the ELBO with an exact (non-mean-field) posterior, so both
methods directly maximise the model evidence.

The parameter λ in these methods uses the **isotropic** prior
z_i ~ N(0, I/λ), where the per-factor scale is absorbed into the columns
of W_r.  This differs from the current CAVI parameterisation
(z_ik ~ N(0, d_{r,k}/λ)) but is equivalent: W_r ≡ V_r @ diag(d_r).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla
import optax

from .linalg import rsvd


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class FAEMResult:
    """Result of :func:`matlap_faem` Factor Analysis EM optimisation.

    Attributes:
        mu:          Posterior mean of X = Z W_r^T, shape (m, n).
        W_r:         Loading matrix at convergence, shape (n, r).
        lambda_bar:  Estimated λ (E_q[λ] via Gamma posterior), scalar.
        a_N:         Gamma posterior shape (= a0 + m·r/2), scalar.
        b_N:         Gamma posterior rate, scalar.
        ll_trace:    Marginal log-likelihood at each iteration.
        converged:   True if relative |ΔLL| < tol before max_iter.
        n_iter:      Number of iterations executed.
    """

    mu: jax.Array
    W_r: jax.Array
    lambda_bar: float
    a_N: float
    b_N: float
    ll_trace: list[float] = field(default_factory=list)
    converged: bool = False
    n_iter: int = 0


@dataclass
class GradMLResult:
    """Result of :func:`matlap_gradml` gradient marginal likelihood optimisation.

    Attributes:
        mu:          Posterior mean of X (from E-step at converged W_r, λ), shape (m, n).
        W_r:         Loading matrix at convergence, shape (n, r).
        lambda_bar:  Estimated λ, scalar.
        ll_trace:    Marginal log-likelihood (+ log prior) at each iteration.
        converged:   True if relative |ΔLL| < tol before max_iter.
        n_iter:      Number of optimiser steps executed.
    """

    mu: jax.Array
    W_r: jax.Array
    lambda_bar: float
    ll_trace: list[float] = field(default_factory=list)
    converged: bool = False
    n_iter: int = 0


# ---------------------------------------------------------------------------
# Core E-step and marginal LL (shared by both methods)
# ---------------------------------------------------------------------------


def _estep_row(
    y_i: jax.Array,
    prec_i: jax.Array,
    W_r: jax.Array,
    lambda_val: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Exact E-step for one row: posterior moments of z_hat_i.

    Model: x_i = W_r z_hat_i,  z_hat_i ~ N(0, I/λ),  y_ij | x_ij ~ N(x_ij, s_ij²).

    Returns:
        z_hat:  Posterior mean E[z_hat_i], shape (r,).
        A_inv:  Posterior covariance, shape (r, r).
    """
    r = W_r.shape[1]
    # Posterior precision: A_i = λI + W_r^T diag(prec_i) W_r
    A_i = lambda_val * jnp.eye(r) + (W_r.T * prec_i) @ W_r  # (r, r)
    rhs_i = W_r.T @ (prec_i * jnp.where(jnp.isfinite(y_i), y_i, 0.0))  # (r,)
    cho = jsla.cho_factor(A_i)
    z_hat = jsla.cho_solve(cho, rhs_i)  # (r,)
    A_inv = jsla.cho_solve(cho, jnp.eye(r))  # (r, r)
    return z_hat, A_inv


_estep_rows = jax.jit(
    jax.vmap(_estep_row, in_axes=(0, 0, None, None))
)


def _marginal_ll_row(
    y_i: jax.Array,
    prec_i: jax.Array,
    W_r: jax.Array,
    lambda_val: jax.Array,
) -> jax.Array:
    """Marginal log-likelihood for one row via Woodbury identity.

    log p(y_i_obs | W_r, λ) = -½ [yᵀΛy - rhsᵀ A⁻¹ rhs
                                   - Σ_obs log(prec) + log|A| - r log λ
                                   + n_obs log(2π)]
    """
    r = W_r.shape[1]
    A_i = lambda_val * jnp.eye(r) + (W_r.T * prec_i) @ W_r  # (r, r)
    y_obs = jnp.where(jnp.isfinite(y_i), y_i, 0.0)
    rhs_i = W_r.T @ (prec_i * y_obs)  # (r,)

    quad = jnp.dot(y_obs * prec_i, y_obs) - rhs_i @ jnp.linalg.solve(A_i, rhs_i)
    _, log_det_A = jnp.linalg.slogdet(A_i)
    log_prec_sum = jnp.sum(jnp.where(prec_i > 0, jnp.log(prec_i), 0.0))
    n_obs = jnp.sum(prec_i > 0).astype(jnp.float32)

    return -0.5 * (
        quad - log_prec_sum + log_det_A - r * jnp.log(lambda_val)
        + n_obs * jnp.log(2.0 * jnp.pi)
    )


_marginal_ll_rows = jax.jit(
    jax.vmap(_marginal_ll_row, in_axes=(0, 0, None, None))
)


def _total_marginal_ll(
    Y: jax.Array,
    prec: jax.Array,
    W_r: jax.Array,
    lambda_val: jax.Array,
) -> jax.Array:
    """Sum marginal log-likelihood over all rows."""
    return _marginal_ll_rows(Y, prec, W_r, lambda_val).sum()


# ---------------------------------------------------------------------------
# Factor Analysis EM
# ---------------------------------------------------------------------------


def matlap_faem(
    Y: jax.Array,
    S: jax.Array,
    *,
    rank: int = 50,
    a0: float = 1e-3,
    b0: float = 1e-3,
    max_iter: int = 300,
    tol: float = 1e-6,
    verbose: bool = False,
    W_r_init: jax.Array | None = None,
    lambda_init: float | None = None,
) -> FAEMResult:
    """Low-rank Bayesian matrix denoising via Factor Analysis EM.

    Optimises the same marginal likelihood as ``matlap_lowrank`` but uses a
    proper FA M-step that allows the column space of W_r to change at each
    iteration.  The CAVI-lowrank implementation only rotates within the column
    space fixed by the initial rSVD, while FA EM freely updates W_r via
    per-column weighted least squares.

    Model::

        x_i = W_r z_i,   z_i ~ N(0, I/λ)
        y_ij | x_ij ~ N(x_ij, s_ij²)   for observed (i,j)

    The loading matrix W_r ∈ R^{n×r} is unconstrained (columns not normalised),
    absorbing the per-factor scale.  This is equivalent to V_r @ diag(d_r) in
    the CAVI parameterisation.

    Args:
        Y:            Observed matrix, shape (m, n). NaN/any value where missing.
        S:            Known standard errors, shape (m, n). ``jnp.inf`` where missing.
        rank:         Rank r of the latent subspace (default 50).
        a0:           Gamma prior shape for λ (default: 1e-3).
        b0:           Gamma prior rate for λ (default: 1e-3).
        max_iter:     Maximum EM iterations.
        tol:          Convergence tolerance on relative |ΔLL| change.
        verbose:      Print LL at each iteration.
        W_r_init:     Warm-start loading matrix, shape (n, r). If None, uses rSVD init.
        lambda_init:  Warm-start λ value. If None, initialised from W_r_init.

    Returns:
        :class:`FAEMResult` with posterior mean, loading matrix, and diagnostics.
    """
    Y = jnp.asarray(Y, dtype=jnp.float32)
    S = jnp.asarray(S, dtype=jnp.float32)
    m, n = Y.shape
    r = min(rank, n, m)

    prec = jnp.where(jnp.isfinite(S) & jnp.isfinite(Y), 1.0 / S ** 2, 0.0)  # (m, n)

    # Initialise W_r from rSVD of observed data
    if W_r_init is not None:
        W_r = jnp.asarray(W_r_init, dtype=jnp.float32)
    else:
        Y_obs = jnp.where(jnp.isfinite(Y) & jnp.isfinite(S), Y, 0.0)
        _, _, Vt_r = rsvd(Y_obs, r)
        W_r = Vt_r.T  # (n, r)

    # Initialise λ
    if lambda_init is not None:
        lambda_val = jnp.asarray(lambda_init, dtype=jnp.float32)
    else:
        lambda_val = jnp.asarray(1.0, dtype=jnp.float32)

    ll_trace: list[float] = []
    converged = False
    z_hats: jax.Array | None = None

    for i in range(max_iter):
        # --- E-step: posterior moments of z_i for all rows ---
        z_hats, A_invs = _estep_rows(Y, prec, W_r, lambda_val)
        # z_hats: (m, r), A_invs: (m, r, r)

        # E[z_i z_i^T] = A_inv_i + z_hat_i z_hat_i^T
        E_zzT = A_invs + z_hats[:, :, None] * z_hats[:, None, :]  # (m, r, r)

        # --- M-step for W_r: per-column weighted least squares ---
        # S_j = Σ_i prec_ij * E[z_i z_i^T]   (n, r, r)
        S_j = jnp.einsum('ij,ikl->jkl', prec, E_zzT)  # (n, r, r)
        S_j = S_j + 1e-6 * jnp.eye(r)[None, :, :]     # ridge for numerics

        # c_j = Σ_i prec_ij * y_ij * z_hat_i  (n, r)
        y_obs = jnp.where(jnp.isfinite(Y), Y, 0.0)    # (m, n)
        c = (prec * y_obs).T @ z_hats                  # (n, r)

        # w_j = S_j^{-1} c_j  — vmapped over n columns
        W_r = jax.vmap(jnp.linalg.solve)(S_j, c)       # (n, r)

        # --- M-step for λ: Gamma posterior mean ---
        tr_E_zzT = jnp.sum(jnp.diagonal(E_zzT, axis1=1, axis2=2))  # Σ_i Tr(E[z_i z_i^T])
        a_N = jnp.asarray(a0 + m * r / 2.0, dtype=jnp.float32)
        b_N = jnp.asarray(b0 + 0.5 * float(tr_E_zzT), dtype=jnp.float32)
        lambda_val = a_N / b_N

        # --- Track marginal LL ---
        ll = float(_total_marginal_ll(Y, prec, W_r, lambda_val))
        ll_trace.append(ll)

        if verbose:
            print(f"  iter {i+1:4d}  LL={ll:.4f}  λ={float(lambda_val):.6f}")

        if i > 0:
            prev = ll_trace[-2]
            denom = max(abs(prev), 1e-10)
            if abs(ll - prev) / denom < tol:
                converged = True
                break

    # Posterior means: μ_i = W_r z_hat_i
    if z_hats is None:
        z_hats, _ = _estep_rows(Y, prec, W_r, lambda_val)
    mu = z_hats @ W_r.T  # (m, n)

    return FAEMResult(
        mu=mu,
        W_r=W_r,
        lambda_bar=float(lambda_val),
        a_N=float(a_N),
        b_N=float(b_N),
        ll_trace=ll_trace,
        converged=converged,
        n_iter=len(ll_trace),
    )


# ---------------------------------------------------------------------------
# Gradient marginal likelihood
# ---------------------------------------------------------------------------


@jax.jit
def _negative_objective(
    W_r: jax.Array,
    log_lambda: jax.Array,
    Y: jax.Array,
    prec: jax.Array,
    a0: float,
    b0: float,
) -> jax.Array:
    """Negative marginal log-likelihood + log prior (to minimise)."""
    lambda_val = jnp.exp(log_lambda)
    ll = _total_marginal_ll(Y, prec, W_r, lambda_val)
    log_prior = (a0 - 1.0) * log_lambda - b0 * lambda_val  # Gamma(a0, b0) log prior
    return -(ll + log_prior)


_neg_obj_and_grad = jax.jit(jax.value_and_grad(_negative_objective, argnums=(0, 1)))


def matlap_gradml(
    Y: jax.Array,
    S: jax.Array,
    *,
    rank: int = 50,
    a0: float = 1e-3,
    b0: float = 1e-3,
    max_iter: int = 2000,
    tol: float = 1e-6,
    learning_rate: float = 1e-2,
    verbose: bool = False,
    W_r_init: jax.Array | None = None,
    lambda_init: float | None = None,
) -> GradMLResult:
    """Low-rank Bayesian matrix denoising via gradient marginal likelihood.

    Directly maximises the marginal log-likelihood plus log prior:

        log p(Y_Ω | W_r, λ) + log p(λ)

    with respect to the unconstrained loading matrix W_r ∈ R^{n×r} and
    log λ, using the Adam optimiser (via optax).  Gradients are computed
    by JAX autodiff through the Woodbury marginal LL formula.

    After optimisation the posterior mean is recovered via the E-step at the
    converged (W_r, λ).

    Args:
        Y:             Observed matrix, shape (m, n). NaN/any value where missing.
        S:             Known standard errors, shape (m, n). ``jnp.inf`` where missing.
        rank:          Rank r of the latent subspace (default 50).
        a0:            Gamma prior shape for λ (default: 1e-3).
        b0:            Gamma prior rate for λ (default: 1e-3).
        max_iter:      Maximum Adam steps (default 2000).
        tol:           Convergence tolerance on relative |ΔLL|.
        learning_rate: Adam step size (default 1e-2).
        verbose:       Print LL every 100 steps.
        W_r_init:      Warm-start loading matrix, shape (n, r). If None, uses rSVD init.
        lambda_init:   Warm-start λ. If None, initialised to 1.0.

    Returns:
        :class:`GradMLResult` with posterior mean, loading matrix, and diagnostics.
    """
    Y = jnp.asarray(Y, dtype=jnp.float32)
    S = jnp.asarray(S, dtype=jnp.float32)
    m, n = Y.shape
    r = min(rank, n, m)

    prec = jnp.where(jnp.isfinite(S) & jnp.isfinite(Y), 1.0 / S ** 2, 0.0)  # (m, n)

    # Initialise W_r
    if W_r_init is not None:
        W_r = jnp.asarray(W_r_init, dtype=jnp.float32)
    else:
        Y_obs = jnp.where(jnp.isfinite(Y) & jnp.isfinite(S), Y, 0.0)
        _, _, Vt_r = rsvd(Y_obs, r)
        W_r = Vt_r.T  # (n, r)

    # Initialise log_lambda
    lambda_init_val = lambda_init if lambda_init is not None else 1.0
    log_lambda = jnp.asarray(float(jnp.log(jnp.asarray(lambda_init_val))), dtype=jnp.float32)

    # Set up optax Adam optimiser
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init((W_r, log_lambda))

    ll_trace: list[float] = []
    converged = False

    for i in range(max_iter):
        neg_obj, (grad_W, grad_log_lam) = _neg_obj_and_grad(
            W_r, log_lambda, Y, prec, a0, b0
        )

        updates, opt_state = optimizer.update(
            (grad_W, grad_log_lam), opt_state, (W_r, log_lambda)
        )
        W_r = optax.apply_updates(W_r, updates[0])
        log_lambda = optax.apply_updates(log_lambda, updates[1])

        ll = -float(neg_obj)
        ll_trace.append(ll)

        if verbose and (i % 100 == 0 or i == max_iter - 1):
            print(f"  step {i+1:5d}  LL={ll:.4f}  λ={float(jnp.exp(log_lambda)):.6f}")

        if i > 0:
            prev = ll_trace[-2]
            denom = max(abs(prev), 1e-10)
            if abs(ll - prev) / denom < tol:
                converged = True
                break

    lambda_val = jnp.exp(log_lambda)

    # Posterior means: E-step at converged (W_r, λ)
    z_hats, _ = _estep_rows(Y, prec, W_r, lambda_val)
    mu = z_hats @ W_r.T  # (m, n)

    return GradMLResult(
        mu=mu,
        W_r=W_r,
        lambda_bar=float(lambda_val),
        ll_trace=ll_trace,
        converged=converged,
        n_iter=len(ll_trace),
    )
