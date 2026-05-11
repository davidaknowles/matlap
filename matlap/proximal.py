"""
Nuclear-norm penalized matrix regression via FISTA proximal gradient.

Solves:
    min_{X}  0.5 * sum_{obs(i,j)} (Y_ij - X_ij)^2 / s_ij^2  +  lambda * ||X||_*

The proximal operator of (t * lambda) * ||X||_* is singular value
soft-thresholding (SVT):
    SVT(Z, threshold)[U, Sigma, V] = U @ diag(max(sigma_k - threshold, 0)) @ V.T

Lambda can be supplied directly via ``proximal_gradient``, or selected by
entry-wise K-fold cross-validation via ``proximal_cv`` (a thin convenience
wrapper around the general :func:`matlap.cv.cv_lambda`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from .cv import cv_lambda


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class ProximalResult:
    """Result of proximal gradient optimisation.

    Attributes:
        X:           Denoised matrix estimate, shape (m, n).
        loss_trace:  Objective value at the end of each iteration.
        lambda_val:  Regularisation strength used.
        converged:   True if ||DeltaX||_F / max(||X||_F, 1) < tol.
        n_iter:      Number of iterations executed.
    """

    X: jax.Array
    loss_trace: list[float] = field(default_factory=list)
    lambda_val: float = 0.0
    converged: bool = False
    n_iter: int = 0


# ---------------------------------------------------------------------------
# JIT-compiled helpers
# ---------------------------------------------------------------------------


@jax.jit
def _svt(Z: jax.Array, threshold: jax.Array) -> jax.Array:
    """Singular value soft-thresholding: proximal op of threshold * ||.||_*."""
    U, sv, Vt = jnp.linalg.svd(Z, full_matrices=False)
    sv_thresh = jnp.maximum(sv - threshold, 0.0)
    return (U * sv_thresh) @ Vt


@jax.jit
def _grad_and_loss(
    X: jax.Array,
    Y: jax.Array,
    prec: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Gradient and value of 0.5 * sum_{obs} (X - Y)^2 / s^2."""
    resid = jnp.where(jnp.isfinite(Y), X - Y, 0.0)
    grad = prec * resid
    loss = 0.5 * jnp.sum(prec * resid ** 2)
    return grad, loss


@jax.jit
def _nuclear_norm(X: jax.Array) -> jax.Array:
    return jnp.linalg.svd(X, compute_uv=False).sum()


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def proximal_gradient(
    Y: jax.Array,
    S: jax.Array,
    lambda_val: float,
    *,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> ProximalResult:
    """Nuclear-norm penalized matrix denoising via FISTA.

    Solves:
        min_{X}  0.5 * sum_{obs} (Y_ij - X_ij)^2 / s_ij^2  +  lambda * ||X||_*

    Missing entries are indicated by ``S[i, j] = jnp.inf`` (and the
    corresponding ``Y[i, j]`` value is ignored).

    Args:
        Y:           Observations, shape (m, n).
        S:           Known standard errors, shape (m, n); ``jnp.inf`` where missing.
        lambda_val:  Nuclear norm regularisation strength.
        max_iter:    Maximum FISTA iterations.
        tol:         Convergence tolerance on relative change in X.

    Returns:
        ProximalResult with denoised matrix and diagnostics.
    """
    Y = jnp.asarray(Y, dtype=jnp.float32)
    S = jnp.asarray(S, dtype=jnp.float32)
    S2 = S ** 2
    obs_mask = jnp.isfinite(S2) & jnp.isfinite(Y)
    prec = jnp.where(obs_mask, 1.0 / S2, 0.0)

    L = float(jnp.max(prec))
    if L <= 0:
        return ProximalResult(X=jnp.zeros_like(Y), lambda_val=float(lambda_val),
                              converged=True, n_iter=0)
    step = 1.0 / L

    X = jnp.where(obs_mask, Y, 0.0)
    X_prev = X
    momentum = 1.0

    loss_trace: list[float] = []
    converged = False

    for _ in range(max_iter):
        t_new = 0.5 * (1.0 + (1.0 + 4.0 * momentum ** 2) ** 0.5)
        Y_fista = X + ((momentum - 1.0) / t_new) * (X - X_prev)

        grad, _ = _grad_and_loss(Y_fista, Y, prec)
        Z = Y_fista - step * grad
        X_new = _svt(Z, step * lambda_val)

        _, loss_smooth = _grad_and_loss(X_new, Y, prec)
        obj = float(loss_smooth) + lambda_val * float(_nuclear_norm(X_new))
        loss_trace.append(obj)

        dx = float(jnp.linalg.norm(X_new - X, ord='fro'))
        x_nrm = float(jnp.linalg.norm(X_new, ord='fro'))
        if dx / max(x_nrm, 1.0) < tol:
            X_prev = X
            X = X_new
            momentum = t_new
            converged = True
            break

        X_prev = X
        X = X_new
        momentum = t_new

    return ProximalResult(
        X=X,
        loss_trace=loss_trace,
        lambda_val=float(lambda_val),
        converged=converged,
        n_iter=len(loss_trace),
    )


def proximal_cv(
    Y: jax.Array,
    S: jax.Array,
    lambda_grid: jax.Array,
    *,
    n_folds: int = 5,
    max_iter: int = 500,
    tol: float = 1e-6,
    verbose: bool = False,
) -> tuple[float, ProximalResult]:
    """Select lambda by entry-wise K-fold CV, then refit on all observed entries.

    Thin wrapper around :func:`matlap.cv.cv_lambda` using
    :func:`proximal_gradient` as the fitting function.

    Args:
        Y:            Observations, shape (m, n).
        S:            Known standard errors; ``jnp.inf`` where missing.
        lambda_grid:  1-D array of lambda values to evaluate.
        n_folds:      Number of CV folds (default 5).
        max_iter:     Maximum FISTA iterations per fit.
        tol:          Convergence tolerance.
        verbose:      Print CV progress.

    Returns:
        ``(best_lambda, ProximalResult)`` fitted on all observed entries.
    """
    return cv_lambda(
        Y, S, lambda_grid,
        fit_fn=proximal_gradient,
        get_mu=lambda r: r.X,
        n_folds=n_folds,
        verbose=verbose,
        max_iter=max_iter,
        tol=tol,
    )
