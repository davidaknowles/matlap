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
def _fista_step(
    X: jax.Array,
    X_prev: jax.Array,
    Y: jax.Array,
    prec: jax.Array,
    step: jax.Array,
    lambda_val: jax.Array,
    momentum: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """One complete FISTA step.

    Returns (X_new, t_new, obj, rel_change) where obj is the full objective
    value and rel_change = ||X_new - X||_F / max(||X_new||_F, 1).

    Fusing all scalar outputs into a single JIT call means only one
    host-device sync per iteration, keeping GPU pipelines full.
    """
    t_new = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * momentum ** 2))
    Y_fista = X + ((momentum - 1.0) / t_new) * (X - X_prev)

    resid_f = jnp.where(jnp.isfinite(Y), Y_fista - Y, 0.0)
    grad = prec * resid_f
    Z = Y_fista - step * grad

    X_new = _svt(Z, step * lambda_val)

    resid_n = jnp.where(jnp.isfinite(Y), X_new - Y, 0.0)
    loss_smooth = 0.5 * jnp.sum(prec * resid_n ** 2)
    nuc = jnp.linalg.svd(X_new, compute_uv=False).sum()
    obj = loss_smooth + lambda_val * nuc

    dx = jnp.linalg.norm(X_new - X, ord='fro')
    x_nrm = jnp.linalg.norm(X_new, ord='fro')
    rel_change = dx / jnp.maximum(x_nrm, 1.0)

    return X_new, t_new, obj, rel_change


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
    step_arr = jnp.asarray(step, dtype=jnp.float32)
    lambda_arr = jnp.asarray(lambda_val, dtype=jnp.float32)
    mom_arr = jnp.asarray(momentum, dtype=jnp.float32)

    for _ in range(max_iter):
        X_new, t_arr, obj, rel_change = _fista_step(
            X, X_prev, Y, prec, step_arr, lambda_arr, mom_arr,
        )
        # Single host-device sync for both scalars
        obj_val, rel_val = jax.device_get((obj, rel_change))
        loss_trace.append(float(obj_val))

        X_prev = X
        X = X_new
        mom_arr = t_arr

        if float(rel_val) < tol:
            converged = True
            break

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
