"""
Taylor-delta approximation for the Matrix Laplace variational objective.

The paper's row-factorised variational approximation gives the collapsed
objective

    min_mu  0.5 * sum_obs (Y_ij - mu_ij)^2 / s_ij^2
          + 0.5 * sum_i log |diag(p_i) + lambda * (mu.T mu)^(-1/2)|
          + lambda * ||mu||_*

where p_i is the row of observation precisions.  The smooth part is handled by
autodiff, and the nuclear-norm penalty is handled with singular value
soft-thresholding.  Exact SVT is the default; a randomized, warm-startable SVT
path is available for larger matrices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
import math

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla
import jax.scipy.sparse.linalg as jspla

from .cv import cv_lambda
from .proximal import proximal_gradient
from .scoring import renyi_elbo as _diag_renyi_elbo


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class TaylorResult:
    """Result of Taylor-delta proximal-gradient optimisation.

    Attributes:
        mu:          Posterior mean estimate, shape (m, n).
        sigma:       Closed-form row covariances, shape (m, n, n), or ``None``
                     when ``recover_sigma=False``.
        lambda_val:  Regularisation strength used.
        loss_trace:        Collapsed minimisation objective at each iteration.
        elbo_trace:        Approximate Taylor ELBO at each iteration.
        elbo:              Final approximate Taylor ELBO.
        renyi_elbo:        Final diagonal-projection Rényi α-ELBO.
        sigma_diag:        Diagonal of the closed-form row covariances, shape
                           (m, n), always returned for scoring.
        prior_var:         Diagonal prior-variance proxy used by ``renyi_elbo``.
        renyi_alpha:       Rényi order used for ``renyi_elbo``.
        converged:         True if relative change in ``mu`` is below ``tol``.
        n_iter:            Number of iterations executed.
        step_size:         Final proximal-gradient step size.
        svd_basis:         Right singular-vector basis from the last accepted
                           randomized SVT step, or ``None`` for exact SVD.
    """

    mu: jax.Array
    sigma: jax.Array | None = None
    lambda_val: float = 0.0
    loss_trace: list[float] = field(default_factory=list)
    elbo_trace: list[float] = field(default_factory=list)
    elbo: float = float("nan")
    renyi_elbo: float = float("nan")
    sigma_diag: jax.Array | None = None
    prior_var: jax.Array | None = None
    renyi_alpha: float = 0.5
    converged: bool = False
    n_iter: int = 0
    step_size: float = 0.0
    svd_basis: jax.Array | None = None


@dataclass
class TaylorHomoskedasticResult:
    """Result for the homoskedastic Taylor approximation with unknown noise.

    Attributes:
        mu:           Posterior mean estimate, shape (m, n).
        sigma:        Closed-form row covariances, shape (m, n, n), or ``None``
                      when ``recover_sigma=False``.
        lambda_val:   Base Matrix Laplace regularisation strength.
        gamma2:       Estimated homoskedastic noise variance.
        lambda_eff:   Effective penalty ``lambda_val / sqrt(gamma2)``.
        loss_trace:   Collapsed objective at each accepted outer iteration.
        gamma2_trace: Noise variance at each accepted outer iteration.
        mu_update:    Method used for the matrix mean update.
        gamma_update: Method used for the scalar gamma2 update.
        sigma_diag:   Diagonal of the closed-form row covariances.
        prior_var:    Diagonal prior-variance proxy used by scoring utilities.
        converged:    True if both ``mu`` and ``gamma2`` satisfy ``tol``.
        n_iter:       Number of accepted outer iterations.
        step_size:    Final proximal-gradient step size for ``mu``.
        gamma_step_size: Final gradient step size for ``log(gamma2)``.
    """

    mu: jax.Array
    sigma: jax.Array | None = None
    lambda_val: float = 0.0
    gamma2: float = 1.0
    lambda_eff: float = 0.0
    loss_trace: list[float] = field(default_factory=list)
    gamma2_trace: list[float] = field(default_factory=list)
    mu_update: str = "proximal"
    gamma_update: str = "exact"
    sigma_diag: jax.Array | None = None
    prior_var: jax.Array | None = None
    converged: bool = False
    n_iter: int = 0
    step_size: float = 0.0
    gamma_step_size: float = 0.0


# ---------------------------------------------------------------------------
# JIT-compiled helpers
# ---------------------------------------------------------------------------


def _inv_sqrt_gram(
    mu: jax.Array,
    ridge: jax.Array,
    max_inv_sqrt: jax.Array,
) -> jax.Array:
    """Regularised ``(mu.T @ mu)^(-1/2)`` via symmetric eigendecomposition."""
    n = mu.shape[1]
    gram = mu.T @ mu
    scale = ridge * jnp.maximum(jnp.trace(gram) / jnp.asarray(n, mu.dtype), 1.0)

    # A tiny deterministic diagonal ramp avoids undefined eigenvector
    # derivatives in the common rank-deficient case.
    ramp = 1.0 + 1e-3 * jnp.arange(n, dtype=mu.dtype) / jnp.maximum(n - 1, 1)
    gram_reg = gram + scale * jnp.diag(ramp)

    vals, vecs = jnp.linalg.eigh(gram_reg)
    vals = jnp.maximum(vals, jnp.finfo(mu.dtype).tiny)
    inv_sqrt_vals = jnp.minimum(1.0 / jnp.sqrt(vals), max_inv_sqrt)
    return (vecs * inv_sqrt_vals) @ vecs.T


def _sqrt_gram_diag(mu: jax.Array, ridge: jax.Array) -> jax.Array:
    """Diagonal of the same regularised ``(mu.T @ mu)^(1/2)`` used in scores."""
    n = mu.shape[1]
    gram = mu.T @ mu
    scale = ridge * jnp.maximum(jnp.trace(gram) / jnp.asarray(n, mu.dtype), 1.0)
    ramp = 1.0 + 1e-3 * jnp.arange(n, dtype=mu.dtype) / jnp.maximum(n - 1, 1)
    gram_reg = gram + scale * jnp.diag(ramp)

    vals, vecs = jnp.linalg.eigh(gram_reg)
    vals = jnp.maximum(vals, 0.0)
    sqrt_vals = jnp.sqrt(vals)
    return jnp.sum((vecs ** 2) * sqrt_vals[None, :], axis=1)


def _elbo_constant(Y: jax.Array, S2: jax.Array, lambda_val: jax.Array) -> jax.Array:
    """Terms in the approximate Taylor ELBO not present in ``_full_loss``."""
    m, n = Y.shape
    obs_mask = jnp.isfinite(S2) & jnp.isfinite(Y)
    likelihood_const = -0.5 * jnp.sum(
        jnp.where(obs_mask, jnp.log(2.0 * jnp.pi * S2), 0.0),
    )
    entropy_trace_const = 0.5 * m * n * jnp.log(2.0 * jnp.pi)
    prior_norm = m * n * jnp.log(jnp.maximum(lambda_val, 1e-12))
    return likelihood_const + entropy_trace_const + prior_norm


def _loss_to_elbo(loss: jax.Array, Y: jax.Array, S2: jax.Array, lambda_val: jax.Array) -> jax.Array:
    """Convert the collapsed minimisation loss to the approximate Taylor ELBO."""
    return -loss + _elbo_constant(Y, S2, lambda_val)


def _stabilize_spd(A: jax.Array, jitter: jax.Array) -> jax.Array:
    """Add scale-aware diagonal jitter to an intended SPD matrix."""
    n = A.shape[0]
    scale = jnp.maximum(jnp.trace(A) / jnp.asarray(n, A.dtype), 1.0)
    return A + (jitter * scale) * jnp.eye(n, dtype=A.dtype)


def _logdet_spd(A: jax.Array, jitter: jax.Array) -> jax.Array:
    """Log determinant of an SPD matrix using Cholesky."""
    L = jnp.linalg.cholesky(_stabilize_spd(A, jitter))
    return 2.0 * jnp.sum(jnp.log(jnp.diag(L)))


def _smooth_loss(
    mu: jax.Array,
    Y: jax.Array,
    prec: jax.Array,
    lambda_val: jax.Array,
    gram_ridge: jax.Array,
    max_inv_sqrt: jax.Array,
    precision_jitter: jax.Array,
) -> jax.Array:
    """Smooth part of the collapsed Taylor objective."""
    resid = jnp.where(jnp.isfinite(Y), mu - Y, 0.0)
    data_loss = 0.5 * jnp.sum(prec * resid ** 2)

    def _with_logdet(_: None) -> jax.Array:
        B = _inv_sqrt_gram(mu, gram_ridge, max_inv_sqrt)

        def row_logdet(prec_i: jax.Array) -> jax.Array:
            A_i = jnp.diag(prec_i) + lambda_val * B
            return _logdet_spd(A_i, precision_jitter)

        return 0.5 * jnp.sum(jax.vmap(row_logdet)(prec))

    logdet_loss = jax.lax.cond(
        lambda_val > 0.0,
        _with_logdet,
        lambda _: jnp.asarray(0.0, dtype=mu.dtype),
        operand=None,
    )
    return data_loss + logdet_loss


@jax.jit
def _full_loss(
    mu: jax.Array,
    Y: jax.Array,
    prec: jax.Array,
    lambda_val: jax.Array,
    gram_ridge: jax.Array,
    max_inv_sqrt: jax.Array,
    precision_jitter: jax.Array,
) -> jax.Array:
    smooth = _smooth_loss(
        mu, Y, prec, lambda_val, gram_ridge, max_inv_sqrt, precision_jitter,
    )
    nuclear = jnp.linalg.svd(mu, compute_uv=False).sum()
    return smooth + lambda_val * nuclear


_smooth_value_and_grad = jax.jit(jax.value_and_grad(_smooth_loss))


def _homoskedastic_smooth_loss(
    mu: jax.Array,
    Y: jax.Array,
    obs_mask: jax.Array,
    lambda_val: jax.Array,
    log_gamma2: jax.Array,
    a0: jax.Array,
    b0: jax.Array,
    gram_ridge: jax.Array,
    max_inv_sqrt: jax.Array,
    precision_jitter: jax.Array,
) -> jax.Array:
    """Smooth part of the homoskedastic unknown-noise Taylor objective."""
    gamma2 = jnp.exp(log_gamma2)
    gamma = jnp.sqrt(gamma2)
    lambda_eff = lambda_val / gamma
    obs_float = obs_mask.astype(mu.dtype)
    resid = jnp.where(obs_mask, mu - Y, 0.0)
    data_loss = 0.5 * jnp.sum(resid ** 2) / gamma2

    def _with_logdet(_: None) -> jax.Array:
        B = _inv_sqrt_gram(mu, gram_ridge, max_inv_sqrt)

        def row_logdet(obs_i: jax.Array) -> jax.Array:
            A_i = jnp.diag(obs_i / gamma2) + lambda_eff * B
            return _logdet_spd(A_i, precision_jitter)

        return 0.5 * jnp.sum(jax.vmap(row_logdet)(obs_float))

    logdet_loss = jax.lax.cond(
        lambda_val > 0.0,
        _with_logdet,
        lambda _: jnp.asarray(0.0, dtype=mu.dtype),
        operand=None,
    )
    m, n = Y.shape
    obs_count = jnp.sum(obs_float)
    parameter_count = jnp.asarray(m * n, dtype=mu.dtype)
    gamma_log_coef = 0.5 * obs_count + 0.5 * parameter_count + a0 + 1.0
    gamma_loss = gamma_log_coef * log_gamma2 + b0 / gamma2
    return data_loss + logdet_loss + gamma_loss


@jax.jit
def _homoskedastic_full_loss(
    mu: jax.Array,
    Y: jax.Array,
    obs_mask: jax.Array,
    lambda_val: jax.Array,
    log_gamma2: jax.Array,
    a0: jax.Array,
    b0: jax.Array,
    gram_ridge: jax.Array,
    max_inv_sqrt: jax.Array,
    precision_jitter: jax.Array,
) -> jax.Array:
    smooth = _homoskedastic_smooth_loss(
        mu, Y, obs_mask, lambda_val, log_gamma2, a0, b0,
        gram_ridge, max_inv_sqrt, precision_jitter,
    )
    gamma = jnp.sqrt(jnp.exp(log_gamma2))
    nuclear = jnp.linalg.svd(mu, compute_uv=False).sum()
    return smooth + (lambda_val / gamma) * nuclear


_homoskedastic_mu_value_and_grad = jax.jit(
    jax.value_and_grad(_homoskedastic_smooth_loss, argnums=0),
)
_homoskedastic_log_gamma_value_and_grad = jax.jit(
    jax.value_and_grad(_homoskedastic_full_loss, argnums=4),
)


def _make_homoskedastic_hutchinson_probes(
    m: int,
    n: int,
    num_probes: int,
    *,
    seed: int,
    dtype: jnp.dtype,
) -> jax.Array:
    key = jax.random.PRNGKey(seed)
    probes = jax.random.bernoulli(key, 0.5, shape=(m, num_probes, n))
    return jnp.asarray(2.0 * probes - 1.0, dtype=dtype)


@partial(jax.jit, static_argnames=("cg_maxiter",))
def _homoskedastic_hutch_log_gamma_grad(
    mu: jax.Array,
    Y: jax.Array,
    obs_mask: jax.Array,
    lambda_val: jax.Array,
    log_gamma2: jax.Array,
    a0: jax.Array,
    b0: jax.Array,
    inv_sqrt: jax.Array,
    probes: jax.Array,
    cg_x0: jax.Array,
    *,
    cg_tol: float,
    cg_maxiter: int,
) -> tuple[jax.Array, jax.Array]:
    """Stochastic gradient of the homoskedastic full loss wrt log(gamma2)."""
    gamma2 = jnp.exp(log_gamma2)
    lambda_eff = lambda_val / jnp.sqrt(gamma2)
    d_lambda_eff = -0.5 * lambda_eff
    obs_float = obs_mask.astype(mu.dtype)
    resid = jnp.where(obs_mask, mu - Y, 0.0)

    data_grad = -0.5 * jnp.sum(resid ** 2) / gamma2
    nuclear = jnp.linalg.svd(mu, compute_uv=False).sum()
    nuclear_grad = d_lambda_eff * nuclear
    m, n = Y.shape
    obs_count = jnp.sum(obs_float)
    parameter_count = jnp.asarray(m * n, dtype=mu.dtype)
    gamma_log_coef = 0.5 * obs_count + 0.5 * parameter_count + a0 + 1.0
    gamma_prior_grad = gamma_log_coef - b0 / gamma2

    prec = obs_float / gamma2
    dprec = -obs_float / gamma2
    trace_B = jnp.trace(inv_sqrt)
    n_arr = jnp.asarray(n, dtype=mu.dtype)

    def row_probe_solve(prec_i, dprec_i, z, x0):
        mean_diag = (jnp.sum(prec_i) + lambda_eff * trace_B) / n_arr
        jitter_diag = 1e-6 * jnp.maximum(mean_diag, 1.0)

        def matvec(x):
            return prec_i * x + lambda_eff * (inv_sqrt @ x) + jitter_diag * x

        rhs = dprec_i * z + d_lambda_eff * (inv_sqrt @ z)
        sol, _ = jspla.cg(matvec, rhs, x0=x0, tol=cg_tol, atol=0.0, maxiter=cg_maxiter)
        return jnp.dot(z, sol), sol

    def row_estimate(prec_i, dprec_i, probes_i, x0_i):
        vals, sols = jax.vmap(row_probe_solve, in_axes=(None, None, 0, 0))(
            prec_i, dprec_i, probes_i, x0_i,
        )
        return jnp.mean(vals), sols

    row_vals, new_x0 = jax.vmap(row_estimate)(prec, dprec, probes, cg_x0)
    logdet_grad = 0.5 * jnp.sum(row_vals)
    grad = data_grad + logdet_grad + nuclear_grad + gamma_prior_grad
    return grad, new_x0


def _homoskedastic_update_log_gamma_hutchinson(
    mu: jax.Array,
    Y: jax.Array,
    obs_mask: jax.Array,
    lambda_arr: jax.Array,
    log_gamma2: jax.Array,
    a0_arr: jax.Array,
    b0_arr: jax.Array,
    inv_sqrt: jax.Array,
    probes: jax.Array,
    cg_state: jax.Array,
    *,
    lr: float,
    steps: int,
    log_gamma2_min: float,
    log_gamma2_max: float,
    cg_tol: float,
    cg_maxiter: int,
    grad_clip: float,
) -> tuple[jax.Array, jax.Array]:
    lo = jnp.asarray(log_gamma2_min, dtype=jnp.float32)
    hi = jnp.asarray(log_gamma2_max, dtype=jnp.float32)
    log_gamma2_new = log_gamma2
    for _ in range(steps):
        grad, cg_state = _homoskedastic_hutch_log_gamma_grad(
            mu,
            Y,
            obs_mask,
            lambda_arr,
            log_gamma2_new,
            a0_arr,
            b0_arr,
            inv_sqrt,
            probes,
            cg_state,
            cg_tol=cg_tol,
            cg_maxiter=cg_maxiter,
        )
        grad = jnp.clip(grad, -grad_clip, grad_clip)
        log_gamma2_new = jnp.clip(log_gamma2_new - lr * grad, lo, hi)
    return log_gamma2_new, cg_state


@jax.jit
def _svt(Z: jax.Array, threshold: jax.Array) -> jax.Array:
    """Singular value soft-thresholding."""
    U, sv, Vt = jnp.linalg.svd(Z, full_matrices=False)
    sv_thresh = jnp.maximum(sv - threshold, 0.0)
    return (U * sv_thresh) @ Vt


@jax.jit
def _prox_candidate(
    mu: jax.Array,
    grad: jax.Array,
    step: jax.Array,
    lambda_val: jax.Array,
) -> jax.Array:
    return _svt(mu - step * grad, step * lambda_val)


@jax.jit
def _prox_candidate_with_basis(
    mu: jax.Array,
    grad: jax.Array,
    step: jax.Array,
    lambda_val: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    Z = mu - step * grad
    U, sv, Vt = jnp.linalg.svd(Z, full_matrices=False)
    sv_thresh = jnp.maximum(sv - step * lambda_val, 0.0)
    return (U * sv_thresh) @ Vt, Vt.T


def _orthonormalize(A: jax.Array) -> jax.Array:
    Q, _ = jnp.linalg.qr(A)
    return Q


def _randomized_svd(
    A: jax.Array,
    rank: int,
    *,
    n_iter: int,
    oversample: int,
    key: jax.Array,
    init_basis: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Randomized SVD with an optional warm-started right subspace."""
    m, n = A.shape
    max_rank = min(m, n)
    if rank >= max_rank:
        U, sv, Vt = jnp.linalg.svd(A, full_matrices=False)
        return U, sv, Vt, Vt.T

    k = min(max_rank, rank + oversample)
    if init_basis is not None:
        basis = init_basis[:, :min(init_basis.shape[1], k)]
        n_random = k - basis.shape[1]
        if n_random > 0:
            random_basis = jax.random.normal(key, (n, n_random), dtype=A.dtype)
            Omega = jnp.concatenate([basis, random_basis], axis=1)
        else:
            Omega = basis
    else:
        Omega = jax.random.normal(key, (n, k), dtype=A.dtype)
    Omega = _orthonormalize(Omega)

    Q = _orthonormalize(A @ Omega)
    for _ in range(n_iter):
        Q = _orthonormalize(A @ (A.T @ Q))

    B = Q.T @ A
    Ub, sv, Vt = jnp.linalg.svd(B, full_matrices=False)
    U = Q @ Ub
    U = U[:, :rank]
    sv = sv[:rank]
    Vt = Vt[:rank, :]
    return U, sv, Vt, Vt.T


def _randomized_svt(
    Z: jax.Array,
    threshold: jax.Array,
    rank: int,
    *,
    n_iter: int,
    oversample: int,
    key: jax.Array,
    init_basis: jax.Array | None,
) -> tuple[jax.Array, jax.Array]:
    U, sv, Vt, basis = _randomized_svd(
        Z, rank, n_iter=n_iter, oversample=oversample,
        key=key, init_basis=init_basis,
    )
    sv_thresh = jnp.maximum(sv - threshold, 0.0)
    return (U * sv_thresh) @ Vt, basis


def _recover_row_sigma(
    prec_i: jax.Array,
    inv_sqrt: jax.Array,
    lambda_val: jax.Array,
    precision_jitter: jax.Array,
) -> jax.Array:
    n = prec_i.shape[0]
    A_i = jnp.diag(prec_i) + lambda_val * inv_sqrt
    cho = jsla.cho_factor(_stabilize_spd(A_i, precision_jitter))
    sigma_i = jsla.cho_solve(cho, jnp.eye(n, dtype=prec_i.dtype))
    return 0.5 * (sigma_i + sigma_i.T)


def _recover_row_sigma_diag(
    prec_i: jax.Array,
    inv_sqrt: jax.Array,
    lambda_val: jax.Array,
    precision_jitter: jax.Array,
) -> jax.Array:
    n = prec_i.shape[0]
    A_i = jnp.diag(prec_i) + lambda_val * inv_sqrt
    cho = jsla.cho_factor(_stabilize_spd(A_i, precision_jitter))
    sigma_i = jsla.cho_solve(cho, jnp.eye(n, dtype=prec_i.dtype))
    return jnp.maximum(jnp.diag(sigma_i), 1e-12)


_recover_sigmas_vmap = jax.jit(
    jax.vmap(_recover_row_sigma, in_axes=(0, None, None, None)),
)
_recover_sigma_diag_vmap = jax.jit(
    jax.vmap(_recover_row_sigma_diag, in_axes=(0, None, None, None)),
)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def taylor_gradient(
    Y: jax.Array,
    S: jax.Array,
    lambda_val: float,
    *,
    max_iter: int = 500,
    tol: float = 1e-6,
    lr: float | None = None,
    gram_ridge: float = 1e-5,
    max_inv_sqrt: float = 1e4,
    precision_jitter: float = 1e-6,
    max_backtracking: int = 20,
    init_mu: jax.Array | None = None,
    svd_rank: int | None = None,
    svd_n_iter: int = 2,
    svd_oversample: int = 10,
    init_svd_basis: jax.Array | None = None,
    random_seed: int = 0,
    renyi_alpha: float = 0.5,
    recover_sigma: bool = True,
    verbose: bool = False,
) -> TaylorResult:
    """Fit the Taylor-delta approximation via proximal gradient.

    Missing entries are indicated by ``S[i, j] = jnp.inf``; the corresponding
    ``Y[i, j]`` value is ignored.

    Args:
        Y:                Observations, shape (m, n).
        S:                Known standard errors, shape (m, n); ``inf`` where
                          missing.
        lambda_val:       Positive Matrix Laplace regularisation strength.
        max_iter:         Maximum proximal-gradient iterations.
        tol:              Convergence tolerance on relative change in ``mu``.
        lr:               Initial step size.  Defaults to ``1 / max(precision)``.
        gram_ridge:       Relative ridge used for ``(mu.T @ mu)^(-1/2)``.
        max_inv_sqrt:     Upper bound on eigenvalues of the inverse square root.
        precision_jitter: Relative diagonal jitter for row precision Cholesky.
        max_backtracking: Number of step halvings allowed per iteration.
        init_mu:          Optional warm start for ``mu``.
        svd_rank:         If provided, use rank-``svd_rank`` randomized SVT.
                          The exact SVT is used by default.
        svd_n_iter:       Power iterations for randomized SVT.
        svd_oversample:   Oversampling columns for randomized SVT.
        init_svd_basis:   Optional right singular-vector warm start, shape
                          ``(n, k)``.
        random_seed:      Seed for randomized SVT.
        renyi_alpha:      Rényi order for the returned diagonal approximate
                          ``renyi_elbo``; must satisfy 0 ≤ α < 1.
        recover_sigma:    Recover full row covariance matrices at the optimum.
        verbose:          Print loss at each accepted iteration.

    Returns:
        :class:`TaylorResult` with posterior mean, optional covariances, and
        diagnostics.
    """
    if lambda_val <= 0:
        raise ValueError("lambda_val must be positive for the Taylor approximation.")
    if gram_ridge <= 0:
        raise ValueError("gram_ridge must be positive.")
    if max_inv_sqrt <= 0:
        raise ValueError("max_inv_sqrt must be positive.")
    if precision_jitter < 0:
        raise ValueError("precision_jitter must be non-negative.")
    if max_backtracking < 0:
        raise ValueError("max_backtracking must be non-negative.")
    if svd_rank is not None and svd_rank <= 0:
        raise ValueError("svd_rank must be positive when provided.")
    if svd_n_iter < 0:
        raise ValueError("svd_n_iter must be non-negative.")
    if svd_oversample < 0:
        raise ValueError("svd_oversample must be non-negative.")
    if not (0.0 <= renyi_alpha < 1.0):
        raise ValueError("renyi_alpha must be in [0, 1).")

    Y = jnp.asarray(Y, dtype=jnp.float32)
    S = jnp.asarray(S, dtype=jnp.float32)
    S2 = S ** 2
    obs_mask = jnp.isfinite(S2) & jnp.isfinite(Y)
    prec = jnp.where(obs_mask, 1.0 / S2, 0.0)

    max_prec = float(jnp.max(prec))
    if max_prec <= 0:
        raise ValueError("No observed entries (all S are inf or Y is non-finite).")

    step = float(lr) if lr is not None else 1.0 / max(max_prec, 1.0)
    if step <= 0:
        raise ValueError("lr must be positive.")

    if init_mu is None:
        mu = jnp.where(obs_mask, Y, 0.0)
    else:
        mu = jnp.asarray(init_mu, dtype=jnp.float32)
        if mu.shape != Y.shape:
            raise ValueError(f"init_mu must have shape {Y.shape}; got {mu.shape}.")

    if init_svd_basis is not None and init_svd_basis.shape[0] != Y.shape[1]:
        raise ValueError(
            f"init_svd_basis must have {Y.shape[1]} rows; got {init_svd_basis.shape[0]}."
        )

    lambda_arr = jnp.asarray(lambda_val, dtype=jnp.float32)
    ridge_arr = jnp.asarray(gram_ridge, dtype=jnp.float32)
    max_inv_sqrt_arr = jnp.asarray(max_inv_sqrt, dtype=jnp.float32)
    precision_jitter_arr = jnp.asarray(precision_jitter, dtype=jnp.float32)
    svd_basis = None if init_svd_basis is None else jnp.asarray(init_svd_basis, dtype=jnp.float32)

    loss_trace: list[float] = []
    elbo_trace: list[float] = []
    converged = False

    current_loss = float(_full_loss(
        mu, Y, prec, lambda_arr, ridge_arr, max_inv_sqrt_arr, precision_jitter_arr,
    ))
    for i in range(max_iter):
        _, grad = _smooth_value_and_grad(
            mu, Y, prec, lambda_arr, ridge_arr, max_inv_sqrt_arr, precision_jitter_arr,
        )
        if not bool(jnp.all(jnp.isfinite(grad))):
            break

        accepted = False
        trial_step = step
        mu_new = mu
        new_loss = current_loss
        basis_new = svd_basis

        for _ in range(max_backtracking + 1):
            step_arr = jnp.asarray(trial_step, dtype=jnp.float32)
            if svd_rank is None:
                candidate = _prox_candidate(mu, grad, step_arr, lambda_arr)
                candidate_basis = None
            else:
                key = jax.random.fold_in(jax.random.PRNGKey(random_seed), i)
                candidate, candidate_basis = _randomized_svt(
                    mu - step_arr * grad,
                    step_arr * lambda_arr,
                    svd_rank,
                    n_iter=svd_n_iter,
                    oversample=svd_oversample,
                    key=key,
                    init_basis=svd_basis,
                )
            candidate_loss = float(_full_loss(
                candidate, Y, prec, lambda_arr, ridge_arr,
                max_inv_sqrt_arr, precision_jitter_arr,
            ))

            if math.isfinite(candidate_loss) and candidate_loss <= current_loss:
                mu_new = candidate
                new_loss = candidate_loss
                basis_new = candidate_basis
                accepted = True
                break
            trial_step *= 0.5

        if not accepted and svd_rank is not None:
            trial_step = step
            for _ in range(max_backtracking + 1):
                step_arr = jnp.asarray(trial_step, dtype=jnp.float32)
                candidate, candidate_basis = _prox_candidate_with_basis(
                    mu, grad, step_arr, lambda_arr,
                )
                candidate_loss = float(_full_loss(
                    candidate, Y, prec, lambda_arr, ridge_arr,
                    max_inv_sqrt_arr, precision_jitter_arr,
                ))
                if math.isfinite(candidate_loss) and candidate_loss <= current_loss:
                    mu_new = candidate
                    new_loss = candidate_loss
                    basis_new = candidate_basis
                    accepted = True
                    break
                trial_step *= 0.5

        if not accepted:
            # The current iterate is the last monotone point we trust.
            break

        dx = float(jnp.linalg.norm(mu_new - mu, ord="fro"))
        mu_norm = float(jnp.linalg.norm(mu_new, ord="fro"))
        rel_change = dx / max(mu_norm, 1.0)

        mu = mu_new
        svd_basis = basis_new
        step = trial_step
        current_loss = new_loss
        loss_trace.append(new_loss)
        elbo_trace.append(float(_loss_to_elbo(
            jnp.asarray(new_loss, dtype=jnp.float32), S2=S2, Y=Y, lambda_val=lambda_arr,
        )))

        if verbose:
            print(f"  iter {i + 1:4d}  loss={new_loss:.4f}  step={step:.3g}")

        if rel_change < tol:
            converged = True
            break

    inv_sqrt = _inv_sqrt_gram(mu, ridge_arr, max_inv_sqrt_arr)
    sigma_diag = _recover_sigma_diag_vmap(prec, inv_sqrt, lambda_arr, precision_jitter_arr)
    prior_var = jnp.maximum(_sqrt_gram_diag(mu, ridge_arr) / lambda_arr, 1e-12)
    renyi_score = float(_diag_renyi_elbo(
        mu, sigma_diag, prior_var, Y, S, alpha=renyi_alpha,
    ))

    sigma = None
    if recover_sigma:
        sigma = _recover_sigmas_vmap(prec, inv_sqrt, lambda_arr, precision_jitter_arr)

    final_elbo = float(_loss_to_elbo(
        jnp.asarray(current_loss, dtype=jnp.float32), S2=S2, Y=Y, lambda_val=lambda_arr,
    ))
    if not elbo_trace:
        elbo_trace.append(final_elbo)

    return TaylorResult(
        mu=mu,
        sigma=sigma,
        lambda_val=float(lambda_val),
        loss_trace=loss_trace,
        elbo_trace=elbo_trace,
        elbo=final_elbo,
        renyi_elbo=renyi_score,
        sigma_diag=sigma_diag,
        prior_var=prior_var,
        renyi_alpha=float(renyi_alpha),
        converged=converged,
        n_iter=len(loss_trace),
        step_size=float(step),
        svd_basis=svd_basis,
    )


def taylor_proximal_homoskedastic_unknown_noise(
    Y: jax.Array,
    lambda_val: float,
    *,
    observed_mask: jax.Array | None = None,
    max_iter: int = 500,
    tol: float = 1e-6,
    lr: float | None = None,
    mu_update: str = "proximal",
    mu_prox_max_iter: int = 50,
    mu_prox_tol: float = 1e-5,
    mu_prox_fixed_iter: bool = True,
    score_exact_loss: bool = True,
    gamma_lr: float = 0.1,
    gamma_steps: int = 1,
    gamma_update: str = "exact",
    hutchinson_lr: float | None = None,
    hutchinson_probes: int = 4,
    hutchinson_cg_tol: float = 1e-3,
    hutchinson_cg_maxiter: int = 50,
    hutchinson_grad_clip: float = 1e3,
    hutchinson_seed: int = 0,
    gamma2_prior_a: float = 1e-3,
    gamma2_prior_b: float = 1e-3,
    init_mu: jax.Array | None = None,
    init_gamma2: float | None = None,
    gram_ridge: float = 1e-5,
    max_inv_sqrt: float = 1e4,
    precision_jitter: float = 1e-6,
    max_backtracking: int = 20,
    log_gamma2_bounds: tuple[float, float] = (-20.0, 20.0),
    recover_sigma: bool = True,
    transpose_if_wide: bool = True,
    verbose: bool = False,
) -> TaylorHomoskedasticResult:
    """Fit a homoskedastic Taylor approximation with unknown noise variance.

    The likelihood is ``Y_ij | X_ij, gamma2 ~ N(X_ij, gamma2)`` on observed
    entries.  The nuclear-norm prior is scaled as
    ``NND(lambda_val / sqrt(gamma2))`` and ``gamma2`` has an inverse-gamma
    prior with shape ``gamma2_prior_a`` and scale ``gamma2_prior_b``.

    Missing entries are selected with ``observed_mask=False``.  If no mask is
    supplied, finite entries of ``Y`` are treated as observed.
    """
    mu_update = mu_update.lower()
    if mu_update not in {"proximal", "taylor"}:
        raise ValueError("mu_update must be one of 'proximal' or 'taylor'.")
    gamma_update = gamma_update.lower()
    if gamma_update not in {"exact", "hutchinson"}:
        raise ValueError("gamma_update must be one of 'exact' or 'hutchinson'.")
    if lambda_val <= 0:
        raise ValueError("lambda_val must be positive.")
    if gamma_lr <= 0:
        raise ValueError("gamma_lr must be positive.")
    if hutchinson_lr is not None and hutchinson_lr <= 0:
        raise ValueError("hutchinson_lr must be positive when provided.")
    if gamma_steps < 0:
        raise ValueError("gamma_steps must be non-negative.")
    if mu_prox_max_iter < 1:
        raise ValueError("mu_prox_max_iter must be positive.")
    if hutchinson_probes < 1:
        raise ValueError("hutchinson_probes must be positive.")
    if hutchinson_cg_maxiter < 1:
        raise ValueError("hutchinson_cg_maxiter must be positive.")
    if gamma2_prior_a < 0:
        raise ValueError("gamma2_prior_a must be non-negative.")
    if gamma2_prior_b < 0:
        raise ValueError("gamma2_prior_b must be non-negative.")
    if gram_ridge <= 0:
        raise ValueError("gram_ridge must be positive.")
    if max_inv_sqrt <= 0:
        raise ValueError("max_inv_sqrt must be positive.")
    if precision_jitter < 0:
        raise ValueError("precision_jitter must be non-negative.")
    if max_backtracking < 0:
        raise ValueError("max_backtracking must be non-negative.")
    if log_gamma2_bounds[0] >= log_gamma2_bounds[1]:
        raise ValueError("log_gamma2_bounds must be increasing.")

    Y = jnp.asarray(Y, dtype=jnp.float32)
    if transpose_if_wide and Y.shape[0] < Y.shape[1]:
        res = taylor_proximal_homoskedastic_unknown_noise(
            Y.T,
            lambda_val,
            observed_mask=None if observed_mask is None else jnp.asarray(observed_mask, dtype=bool).T,
            max_iter=max_iter,
            tol=tol,
            lr=lr,
            mu_update=mu_update,
            mu_prox_max_iter=mu_prox_max_iter,
            mu_prox_tol=mu_prox_tol,
            mu_prox_fixed_iter=mu_prox_fixed_iter,
            score_exact_loss=score_exact_loss,
            gamma_lr=gamma_lr,
            gamma_steps=gamma_steps,
            gamma_update=gamma_update,
            hutchinson_lr=hutchinson_lr,
            hutchinson_probes=hutchinson_probes,
            hutchinson_cg_tol=hutchinson_cg_tol,
            hutchinson_cg_maxiter=hutchinson_cg_maxiter,
            hutchinson_grad_clip=hutchinson_grad_clip,
            hutchinson_seed=hutchinson_seed,
            gamma2_prior_a=gamma2_prior_a,
            gamma2_prior_b=gamma2_prior_b,
            init_mu=None if init_mu is None else jnp.asarray(init_mu, dtype=jnp.float32).T,
            init_gamma2=init_gamma2,
            gram_ridge=gram_ridge,
            max_inv_sqrt=max_inv_sqrt,
            precision_jitter=precision_jitter,
            max_backtracking=max_backtracking,
            log_gamma2_bounds=log_gamma2_bounds,
            recover_sigma=False,
            transpose_if_wide=False,
            verbose=verbose,
        )
        return TaylorHomoskedasticResult(
            mu=res.mu.T,
            sigma=None,
            lambda_val=res.lambda_val,
            gamma2=res.gamma2,
            lambda_eff=res.lambda_eff,
            loss_trace=res.loss_trace,
            gamma2_trace=res.gamma2_trace,
            mu_update=res.mu_update,
            gamma_update=res.gamma_update,
            sigma_diag=None if res.sigma_diag is None else res.sigma_diag.T,
            prior_var=None,
            converged=res.converged,
            n_iter=res.n_iter,
            step_size=res.step_size,
            gamma_step_size=res.gamma_step_size,
        )

    finite_y = jnp.isfinite(Y)
    if observed_mask is None:
        obs_mask = finite_y
    else:
        obs_mask = jnp.asarray(observed_mask, dtype=bool)
        if obs_mask.shape != Y.shape:
            raise ValueError(f"observed_mask must have shape {Y.shape}; got {obs_mask.shape}.")
        obs_mask = obs_mask & finite_y

    obs_count = float(jnp.sum(obs_mask))
    if obs_count <= 0:
        raise ValueError("No observed finite entries.")

    Y_clean = jnp.where(obs_mask, Y, 0.0)
    if init_mu is None:
        mu = Y_clean
    else:
        mu = jnp.asarray(init_mu, dtype=jnp.float32)
        if mu.shape != Y.shape:
            raise ValueError(f"init_mu must have shape {Y.shape}; got {mu.shape}.")

    if init_gamma2 is None:
        mean_obs = jnp.sum(Y_clean) / jnp.asarray(obs_count, dtype=Y.dtype)
        centered = jnp.where(obs_mask, Y - mean_obs, 0.0)
        gamma2_init = float(jnp.maximum(
            jnp.sum(centered ** 2) / jnp.asarray(obs_count, dtype=Y.dtype),
            1e-6,
        ))
    else:
        gamma2_init = float(init_gamma2)
        if gamma2_init <= 0:
            raise ValueError("init_gamma2 must be positive.")

    log_gamma2_min, log_gamma2_max = log_gamma2_bounds
    log_gamma2 = jnp.asarray(
        min(max(math.log(gamma2_init), log_gamma2_min), log_gamma2_max),
        dtype=jnp.float32,
    )
    step = float(lr) if lr is not None else max(float(jnp.exp(log_gamma2)), 1e-6)
    if step <= 0:
        raise ValueError("lr must be positive.")

    lambda_arr = jnp.asarray(lambda_val, dtype=jnp.float32)
    a0_arr = jnp.asarray(gamma2_prior_a, dtype=jnp.float32)
    b0_arr = jnp.asarray(gamma2_prior_b, dtype=jnp.float32)
    ridge_arr = jnp.asarray(gram_ridge, dtype=jnp.float32)
    max_inv_sqrt_arr = jnp.asarray(max_inv_sqrt, dtype=jnp.float32)
    precision_jitter_arr = jnp.asarray(precision_jitter, dtype=jnp.float32)

    loss_trace: list[float] = []
    gamma2_trace: list[float] = []
    converged = False
    hutchinson_probe_arr = None
    hutchinson_cg_state = None
    if gamma_update == "hutchinson":
        hutchinson_probe_arr = _make_homoskedastic_hutchinson_probes(
            Y.shape[0],
            Y.shape[1],
            hutchinson_probes,
            seed=hutchinson_seed,
            dtype=Y.dtype,
        )
        hutchinson_cg_state = jnp.zeros_like(hutchinson_probe_arr)
    needs_exact_loss = score_exact_loss or mu_update == "taylor" or gamma_update == "exact"
    current_loss = (
        float(_homoskedastic_full_loss(
            mu, Y_clean, obs_mask, lambda_arr, log_gamma2, a0_arr, b0_arr,
            ridge_arr, max_inv_sqrt_arr, precision_jitter_arr,
        ))
        if needs_exact_loss else float("nan")
    )

    gamma_step = float(gamma_lr)
    hutchinson_step = min(gamma_step, 1e-3) if hutchinson_lr is None else float(hutchinson_lr)
    for i in range(max_iter):
        old_gamma2 = float(jnp.exp(log_gamma2))
        lambda_eff_arr = lambda_arr / jnp.sqrt(jnp.exp(log_gamma2))
        if mu_update == "taylor":
            _, grad = _homoskedastic_mu_value_and_grad(
                mu, Y_clean, obs_mask, lambda_arr, log_gamma2, a0_arr, b0_arr,
                ridge_arr, max_inv_sqrt_arr, precision_jitter_arr,
            )
            if not bool(jnp.all(jnp.isfinite(grad))):
                break

            accepted = False
            trial_step = step
            mu_new = mu
            new_loss = current_loss
            for _ in range(max_backtracking + 1):
                step_arr = jnp.asarray(trial_step, dtype=jnp.float32)
                candidate = _prox_candidate(mu, grad, step_arr, lambda_eff_arr)
                candidate_loss = float(_homoskedastic_full_loss(
                    candidate, Y_clean, obs_mask, lambda_arr, log_gamma2,
                    a0_arr, b0_arr, ridge_arr, max_inv_sqrt_arr,
                    precision_jitter_arr,
                ))
                if math.isfinite(candidate_loss) and candidate_loss <= current_loss:
                    mu_new = candidate
                    new_loss = candidate_loss
                    accepted = True
                    break
                trial_step *= 0.5

            if not accepted:
                break
            step = trial_step
            current_loss = new_loss
        else:
            gamma2_arr_for_mu = jnp.exp(log_gamma2)
            S_eff = jnp.where(
                obs_mask,
                jnp.sqrt(gamma2_arr_for_mu),
                jnp.inf,
            )
            prox = proximal_gradient(
                Y_clean,
                S_eff,
                float(lambda_eff_arr),
                max_iter=mu_prox_max_iter,
                tol=mu_prox_tol,
                init_X=mu,
                fixed_iter=mu_prox_fixed_iter,
            )
            mu_new = prox.X
            if needs_exact_loss:
                current_loss = float(_homoskedastic_full_loss(
                    mu_new, Y_clean, obs_mask, lambda_arr, log_gamma2,
                    a0_arr, b0_arr, ridge_arr, max_inv_sqrt_arr,
                    precision_jitter_arr,
                ))

        dx = float(jnp.linalg.norm(mu_new - mu, ord="fro"))
        mu_norm = float(jnp.linalg.norm(mu_new, ord="fro"))
        rel_mu = dx / max(mu_norm, 1.0)
        mu = mu_new

        rel_gamma = 0.0
        if gamma_update == "exact":
            for _ in range(gamma_steps):
                _, grad_log_gamma2 = _homoskedastic_log_gamma_value_and_grad(
                    mu, Y_clean, obs_mask, lambda_arr, log_gamma2, a0_arr, b0_arr,
                    ridge_arr, max_inv_sqrt_arr, precision_jitter_arr,
                )
                grad_log_gamma2_f = float(grad_log_gamma2)
                if not math.isfinite(grad_log_gamma2_f):
                    break

                accepted_gamma = False
                trial_gamma_step = gamma_step
                log_gamma2_new = log_gamma2
                gamma_loss_new = current_loss
                for _ in range(max_backtracking + 1):
                    candidate_log = float(log_gamma2) - trial_gamma_step * grad_log_gamma2_f
                    candidate_log = min(max(candidate_log, log_gamma2_min), log_gamma2_max)
                    candidate_log_arr = jnp.asarray(candidate_log, dtype=jnp.float32)
                    candidate_loss = float(_homoskedastic_full_loss(
                        mu, Y_clean, obs_mask, lambda_arr, candidate_log_arr,
                        a0_arr, b0_arr, ridge_arr, max_inv_sqrt_arr,
                        precision_jitter_arr,
                    ))
                    if math.isfinite(candidate_loss) and candidate_loss <= current_loss:
                        log_gamma2_new = candidate_log_arr
                        gamma_loss_new = candidate_loss
                        accepted_gamma = True
                        break
                    trial_gamma_step *= 0.5

                if not accepted_gamma:
                    break

                gamma_step = trial_gamma_step
                log_gamma2 = log_gamma2_new
                current_loss = gamma_loss_new
        elif gamma_steps > 0:
            inv_sqrt_for_gamma = _inv_sqrt_gram(mu, ridge_arr, max_inv_sqrt_arr)
            log_gamma2, hutchinson_cg_state = _homoskedastic_update_log_gamma_hutchinson(
                mu,
                Y_clean,
                obs_mask,
                lambda_arr,
                log_gamma2,
                a0_arr,
                b0_arr,
                inv_sqrt_for_gamma,
                hutchinson_probe_arr,
                hutchinson_cg_state,
                lr=hutchinson_step,
                steps=gamma_steps,
                log_gamma2_min=log_gamma2_min,
                log_gamma2_max=log_gamma2_max,
                cg_tol=hutchinson_cg_tol,
                cg_maxiter=hutchinson_cg_maxiter,
                grad_clip=hutchinson_grad_clip,
            )
            current_loss = float(_homoskedastic_full_loss(
                mu, Y_clean, obs_mask, lambda_arr, log_gamma2,
                a0_arr, b0_arr, ridge_arr, max_inv_sqrt_arr,
                precision_jitter_arr,
            )) if (score_exact_loss or mu_update == "taylor") else float("nan")

        new_gamma2 = float(jnp.exp(log_gamma2))
        rel_gamma = abs(new_gamma2 - old_gamma2) / max(new_gamma2, 1.0)
        loss_trace.append(current_loss)
        gamma2_trace.append(new_gamma2)

        if verbose:
            print(
                f"  iter {i + 1:4d}  loss={current_loss:.4f}  "
                f"gamma2={new_gamma2:.4g}  step={step:.3g}"
            )

        if rel_mu < tol and rel_gamma < tol:
            converged = True
            break

    gamma2 = float(jnp.exp(log_gamma2))
    lambda_eff = float(lambda_val / math.sqrt(gamma2))
    gamma2_arr = jnp.asarray(gamma2, dtype=jnp.float32)
    lambda_eff_arr = jnp.asarray(lambda_eff, dtype=jnp.float32)
    prec = jnp.where(obs_mask, 1.0 / gamma2_arr, 0.0)
    inv_sqrt = _inv_sqrt_gram(mu, ridge_arr, max_inv_sqrt_arr)
    sigma_diag = _recover_sigma_diag_vmap(prec, inv_sqrt, lambda_eff_arr, precision_jitter_arr)
    prior_var = jnp.maximum(_sqrt_gram_diag(mu, ridge_arr) / lambda_eff_arr, 1e-12)

    sigma = None
    if recover_sigma:
        sigma = _recover_sigmas_vmap(prec, inv_sqrt, lambda_eff_arr, precision_jitter_arr)

    return TaylorHomoskedasticResult(
        mu=mu,
        sigma=sigma,
        lambda_val=float(lambda_val),
        gamma2=gamma2,
        lambda_eff=lambda_eff,
        loss_trace=loss_trace,
        gamma2_trace=gamma2_trace,
        mu_update=mu_update,
        gamma_update=gamma_update,
        sigma_diag=sigma_diag,
        prior_var=prior_var,
        converged=converged,
        n_iter=len(loss_trace),
        step_size=float(step),
        gamma_step_size=float(gamma_step),
    )


def taylor_cv(
    Y: jax.Array,
    S: jax.Array,
    lambda_grid: jax.Array,
    *,
    n_folds: int = 5,
    max_iter: int = 500,
    tol: float = 1e-6,
    lr: float | None = None,
    gram_ridge: float = 1e-5,
    max_inv_sqrt: float = 1e4,
    precision_jitter: float = 1e-6,
    max_backtracking: int = 20,
    svd_rank: int | None = None,
    svd_n_iter: int = 2,
    svd_oversample: int = 10,
    random_seed: int = 0,
    renyi_alpha: float = 0.5,
    recover_sigma: bool = True,
    verbose: bool = False,
) -> tuple[float, TaylorResult]:
    """Select lambda by entry-wise K-fold CV, then refit the Taylor model."""
    return cv_lambda(
        Y, S, lambda_grid,
        fit_fn=taylor_gradient,
        n_folds=n_folds,
        verbose=verbose,
        max_iter=max_iter,
        tol=tol,
        lr=lr,
        gram_ridge=gram_ridge,
        max_inv_sqrt=max_inv_sqrt,
        precision_jitter=precision_jitter,
        max_backtracking=max_backtracking,
        svd_rank=svd_rank,
        svd_n_iter=svd_n_iter,
        svd_oversample=svd_oversample,
        random_seed=random_seed,
        renyi_alpha=renyi_alpha,
        recover_sigma=recover_sigma,
    )
