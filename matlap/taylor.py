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
import math

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla

from .cv import cv_lambda
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
