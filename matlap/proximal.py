"""
Nuclear-norm penalized matrix regression via FISTA proximal gradient.

Solves::

    min_{X}  0.5 * sum_{obs(i,j)} (Y_ij - X_ij)^2 / s_ij^2  +  lambda * ||X||_*

The proximal operator of (t * lambda) * ||X||_* is singular value
soft-thresholding (SVT)::

    SVT(Z, threshold)[U, Sigma, V] = U @ diag(max(sigma_k - threshold, 0)) @ V.T

Lambda can be supplied directly via ``proximal_gradient``, or selected by
entry-wise K-fold cross-validation via ``proximal_cv`` (a thin convenience
wrapper around the general :func:`matlap.cv.cv_lambda`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

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
        converged:   True if a configured convergence criterion was met.
        n_iter:      Number of iterations executed.
        convergence_reason: Criterion that triggered convergence, or "".
        n_restarts:  Number of monotone-FISTA restarts.
        svd_basis:   Right singular-vector basis from the final randomized SVT
                     step, or ``None`` for exact SVD.
        svd_rank:    Final randomized SVT rank, or ``None`` for exact SVD.
        svd_kept_rank:
                     Number of captured singular values above threshold in the
                     final randomized SVT step, or ``None`` for exact SVD.
        svd_rank_trace:
                     Per-iteration randomized SVT ranks. Empty for exact SVD.
    """

    X: jax.Array
    loss_trace: list[float] = field(default_factory=list)
    lambda_val: float = 0.0
    converged: bool = False
    n_iter: int = 0
    convergence_reason: str = ""
    n_restarts: int = 0
    svd_basis: jax.Array | None = None
    svd_rank: int | None = None
    svd_kept_rank: int | None = None
    svd_rank_trace: list[int] = field(default_factory=list)


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
def _svt_with_nuc(Z: jax.Array, threshold: jax.Array) -> tuple[jax.Array, jax.Array]:
    """SVT and nuclear norm of the thresholded matrix in one SVD call."""
    U, sv, Vt = jnp.linalg.svd(Z, full_matrices=False)
    sv_thresh = jnp.maximum(sv - threshold, 0.0)
    return (U * sv_thresh) @ Vt, sv_thresh.sum()


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
    basis = Vt.T
    U = U[:, :rank]
    sv = sv[:rank]
    Vt = Vt[:rank, :]
    return U, sv, Vt, basis


def _randomized_svt_with_nuc(
    Z: jax.Array,
    threshold: jax.Array,
    rank: int,
    *,
    n_iter: int,
    oversample: int,
    key: jax.Array,
    init_basis: jax.Array | None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    U, sv, Vt, basis = _randomized_svd(
        Z, rank, n_iter=n_iter, oversample=oversample,
        key=key, init_basis=init_basis,
    )
    sv_thresh = jnp.maximum(sv - threshold, 0.0)
    kept_rank = jnp.sum(sv > threshold).astype(jnp.int32)
    return (U * sv_thresh) @ Vt, sv_thresh.sum(), basis, kept_rank


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

    X_new, nuc = _svt_with_nuc(Z, step * lambda_val)

    resid_n = jnp.where(jnp.isfinite(Y), X_new - Y, 0.0)
    loss_smooth = 0.5 * jnp.sum(prec * resid_n ** 2)
    obj = loss_smooth + lambda_val * nuc

    dx = jnp.linalg.norm(X_new - X, ord='fro')
    x_nrm = jnp.linalg.norm(X_new, ord='fro')
    rel_change = dx / jnp.maximum(x_nrm, 1.0)

    return X_new, t_new, obj, rel_change


@partial(
    jax.jit,
    static_argnames=("svd_rank", "svd_n_iter", "svd_oversample"),
)
def _fista_step_randomized(
    X: jax.Array,
    X_prev: jax.Array,
    Y: jax.Array,
    prec: jax.Array,
    step: jax.Array,
    lambda_val: jax.Array,
    momentum: jax.Array,
    basis: jax.Array,
    key: jax.Array,
    *,
    svd_rank: int,
    svd_n_iter: int,
    svd_oversample: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """One FISTA step using warm-started randomized SVT."""
    t_new = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * momentum ** 2))
    Y_fista = X + ((momentum - 1.0) / t_new) * (X - X_prev)

    resid_f = jnp.where(jnp.isfinite(Y), Y_fista - Y, 0.0)
    grad = prec * resid_f
    Z = Y_fista - step * grad

    X_new, nuc, basis_new, kept_rank = _randomized_svt_with_nuc(
        Z,
        step * lambda_val,
        svd_rank,
        n_iter=svd_n_iter,
        oversample=svd_oversample,
        key=key,
        init_basis=basis,
    )

    resid_n = jnp.where(jnp.isfinite(Y), X_new - Y, 0.0)
    loss_smooth = 0.5 * jnp.sum(prec * resid_n ** 2)
    obj = loss_smooth + lambda_val * nuc

    dx = jnp.linalg.norm(X_new - X, ord='fro')
    x_nrm = jnp.linalg.norm(X_new, ord='fro')
    rel_change = dx / jnp.maximum(x_nrm, 1.0)

    return X_new, t_new, obj, rel_change, basis_new, kept_rank


@partial(jax.jit, static_argnames=("max_iter",))
def _fista_run_fixed(
    X0: jax.Array,
    Y: jax.Array,
    prec: jax.Array,
    step: jax.Array,
    lambda_val: jax.Array,
    max_iter: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Run vanilla FISTA for a fixed number of iterations inside one JIT call."""

    def body(carry, _):
        X, X_prev, momentum = carry
        X_new, t_new, obj, rel_change = _fista_step(
            X, X_prev, Y, prec, step, lambda_val, momentum
        )
        return (X_new, X, t_new), (obj, rel_change)

    (X_final, _, _), trace = jax.lax.scan(
        body,
        (X0, X0, jnp.asarray(1.0, dtype=jnp.float32)),
        None,
        length=max_iter,
    )
    obj_trace, rel_trace = trace
    return X_final, obj_trace, rel_trace


@partial(
    jax.jit,
    static_argnames=("max_iter", "svd_rank", "svd_n_iter", "svd_oversample"),
)
def _fista_run_fixed_randomized(
    X0: jax.Array,
    Y: jax.Array,
    prec: jax.Array,
    step: jax.Array,
    lambda_val: jax.Array,
    basis0: jax.Array,
    keys: jax.Array,
    max_iter: int,
    *,
    svd_rank: int,
    svd_n_iter: int,
    svd_oversample: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Fixed-iteration FISTA scan using warm-started randomized SVT."""

    def body(carry, key):
        X, X_prev, momentum, basis = carry
        X_new, t_new, obj, rel_change, basis_new, kept_rank = _fista_step_randomized(
            X,
            X_prev,
            Y,
            prec,
            step,
            lambda_val,
            momentum,
            basis,
            key,
            svd_rank=svd_rank,
            svd_n_iter=svd_n_iter,
            svd_oversample=svd_oversample,
        )
        return (X_new, X, t_new, basis_new), (obj, rel_change, kept_rank)

    (X_final, _, _, basis_final), trace = jax.lax.scan(
        body,
        (X0, X0, jnp.asarray(1.0, dtype=jnp.float32), basis0),
        keys,
        length=max_iter,
    )
    obj_trace, rel_trace, kept_trace = trace
    return X_final, obj_trace, rel_trace, basis_final, kept_trace


def _initial_svd_basis(
    n_cols: int,
    max_rank: int,
    svd_rank: int,
    svd_oversample: int,
    key: jax.Array,
    dtype: jnp.dtype,
    init_svd_basis: jax.Array | None,
) -> jax.Array:
    k = min(max_rank, svd_rank + svd_oversample)
    if init_svd_basis is not None:
        basis = jnp.asarray(init_svd_basis, dtype=dtype)
        basis = basis[:, :min(basis.shape[1], k)]
        n_random = k - basis.shape[1]
        if n_random > 0:
            random_basis = jax.random.normal(key, (n_cols, n_random), dtype=dtype)
            basis = jnp.concatenate([basis, random_basis], axis=1)
    else:
        basis = jax.random.normal(key, (n_cols, k), dtype=dtype)
    return _orthonormalize(basis)


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
    init_X: jax.Array | None = None,
    solver: str = "fista",
    obj_tol: float | None = None,
    obj_patience: int = 5,
    fixed_iter: bool = False,
    svd_rank: int | None = None,
    svd_n_iter: int = 2,
    svd_oversample: int = 10,
    init_svd_basis: jax.Array | None = None,
    random_seed: int = 0,
    svd_rank_adaptive: bool = False,
    svd_rank_min: int | None = None,
    svd_rank_max: int | None = None,
    svd_rank_step: int = 5,
    svd_rank_shrink_fraction: float = 0.5,
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
        init_X:      Optional warm-start estimate.  If omitted, observed
                     entries are initialized from Y and missing entries to 0.
        solver:      One of {"fista", "monotone_fista", "pgd"}.
        obj_tol:     Optional relative objective-change tolerance.  When set,
                     convergence is declared after obj_patience consecutive
                     iterations below this threshold.
        obj_patience: Number of consecutive objective-stable iterations needed.
        fixed_iter:  If True, run vanilla FISTA for exactly ``max_iter`` steps
                     in one JIT-compiled scan.  This skips per-iteration host
                     convergence checks and is much faster for benchmark grid
                     sweeps, but only supports ``solver="fista"``.
        svd_rank:    If provided, use rank-``svd_rank`` randomized SVT instead
                     of exact SVT.  This is exact only when all singular values
                     above the threshold are captured in the randomized
                     subspace.
        svd_n_iter:  Power iterations for randomized SVT.
        svd_oversample:
                     Oversampling columns for randomized SVT.
        init_svd_basis:
                     Optional right singular-vector warm start, shape
                     ``(n, k)``.
        random_seed: Seed for randomized SVT.
        svd_rank_adaptive:
                     If True, adapt ``svd_rank`` between iterations: increase
                     when all captured singular values survive thresholding,
                     and decrease when many are thresholded away.
        svd_rank_min:
                     Minimum adaptive rank. Defaults to ``svd_rank``.
        svd_rank_max:
                     Maximum adaptive rank. Defaults to ``min(m, n)``.
        svd_rank_step:
                     Rank increment/decrement for adaptive rSVD.
        svd_rank_shrink_fraction:
                     Decrease rank when kept_rank <= this fraction of the
                     current rank.

    Returns:
        ProximalResult with denoised matrix and diagnostics.
    """
    solver = solver.lower()
    if solver not in {"fista", "monotone_fista", "pgd"}:
        raise ValueError("solver must be one of 'fista', 'monotone_fista', or 'pgd'.")
    if obj_patience < 1:
        raise ValueError("obj_patience must be >= 1.")
    if fixed_iter and solver != "fista":
        raise ValueError("fixed_iter=True currently supports only solver='fista'.")
    if fixed_iter and obj_tol is not None:
        raise ValueError("fixed_iter=True does not support obj_tol.")
    if svd_rank is not None and svd_rank <= 0:
        raise ValueError("svd_rank must be positive when provided.")
    if svd_n_iter < 0:
        raise ValueError("svd_n_iter must be non-negative.")
    if svd_oversample < 0:
        raise ValueError("svd_oversample must be non-negative.")
    if svd_rank is not None and solver != "fista":
        raise ValueError("svd_rank currently supports only solver='fista'.")
    if svd_rank_adaptive and svd_rank is None:
        raise ValueError("svd_rank_adaptive=True requires svd_rank.")
    if svd_rank_adaptive and fixed_iter:
        raise ValueError("svd_rank_adaptive=True is not supported with fixed_iter=True.")
    if svd_rank_step < 1:
        raise ValueError("svd_rank_step must be positive.")
    if not (0.0 <= svd_rank_shrink_fraction <= 1.0):
        raise ValueError("svd_rank_shrink_fraction must be in [0, 1].")

    Y = jnp.asarray(Y, dtype=jnp.float32)
    S = jnp.asarray(S, dtype=jnp.float32)
    if init_svd_basis is not None and init_svd_basis.shape[0] != Y.shape[1]:
        raise ValueError(
            f"init_svd_basis must have {Y.shape[1]} rows; got {init_svd_basis.shape[0]}."
        )
    max_svd_rank = min(Y.shape)
    if svd_rank is not None and svd_rank > max_svd_rank:
        svd_rank = max_svd_rank
    if svd_rank_min is None:
        svd_rank_min = svd_rank if svd_rank is not None else 1
    if svd_rank_max is None:
        svd_rank_max = max_svd_rank
    svd_rank_min = max(1, min(int(svd_rank_min), max_svd_rank))
    svd_rank_max = max(svd_rank_min, min(int(svd_rank_max), max_svd_rank))
    S2 = S ** 2
    obs_mask = jnp.isfinite(S2) & jnp.isfinite(Y)
    prec = jnp.where(obs_mask, 1.0 / S2, 0.0)

    L = float(jnp.max(prec))
    if L <= 0:
        return ProximalResult(X=jnp.zeros_like(Y), lambda_val=float(lambda_val),
                              converged=True, n_iter=0)
    step = 1.0 / L

    if init_X is None:
        X = jnp.where(obs_mask, Y, 0.0)
    else:
        X = jnp.asarray(init_X, dtype=jnp.float32)
        if X.shape != Y.shape:
            raise ValueError(f"init_X shape {X.shape} does not match Y shape {Y.shape}.")
    X_prev = X
    momentum = 1.0

    loss_trace: list[float] = []
    converged = False
    convergence_reason = ""
    n_restarts = 0
    stable_obj_count = 0
    step_arr = jnp.asarray(step, dtype=jnp.float32)
    lambda_arr = jnp.asarray(lambda_val, dtype=jnp.float32)
    svd_basis = None
    current_svd_rank = svd_rank
    last_kept_rank = None
    svd_rank_trace: list[int] = []
    if svd_rank is not None:
        svd_basis = _initial_svd_basis(
            Y.shape[1],
            min(Y.shape),
            svd_rank,
            svd_oversample,
            jax.random.PRNGKey(random_seed),
            Y.dtype,
            init_svd_basis,
        )

    if fixed_iter:
        if svd_rank is None:
            X_final, obj_trace, rel_trace = _fista_run_fixed(
                X, Y, prec, step_arr, lambda_arr, max_iter
            )
            basis_final = None
        else:
            keys = jax.random.split(jax.random.PRNGKey(random_seed), max_iter)
            (
                X_final, obj_trace, rel_trace, basis_final, kept_trace,
            ) = _fista_run_fixed_randomized(
                X,
                Y,
                prec,
                step_arr,
                lambda_arr,
                svd_basis,
                keys,
                max_iter,
                svd_rank=svd_rank,
                svd_n_iter=svd_n_iter,
                svd_oversample=svd_oversample,
            )
        obj_vals, rel_vals = jax.device_get((obj_trace, rel_trace))
        loss_trace = [float(v) for v in obj_vals]
        final_rel = float(rel_vals[-1]) if len(rel_vals) else float("inf")
        converged = final_rel < tol
        return ProximalResult(
            X=X_final,
            loss_trace=loss_trace,
            lambda_val=float(lambda_val),
            converged=converged,
            n_iter=max_iter,
            convergence_reason="rel_change" if converged else "",
            n_restarts=0,
            svd_basis=basis_final,
            svd_rank=svd_rank,
            svd_kept_rank=(
                int(jax.device_get(kept_trace[-1])) if svd_rank is not None else None
            ),
            svd_rank_trace=[int(svd_rank)] * max_iter if svd_rank is not None else [],
        )

    mom_arr = jnp.asarray(momentum, dtype=jnp.float32)

    for _ in range(max_iter):
        if solver == "pgd":
            X_step_prev = X
            mom_step = jnp.asarray(1.0, dtype=jnp.float32)
        else:
            X_step_prev = X_prev
            mom_step = mom_arr
        if current_svd_rank is None:
            X_new, t_arr, obj, rel_change = _fista_step(
                X, X_step_prev, Y, prec, step_arr, lambda_arr, mom_step,
            )
            basis_new = None
            kept_rank = None
        else:
            key = jax.random.fold_in(jax.random.PRNGKey(random_seed), len(loss_trace))
            rank_this_iter = int(current_svd_rank)
            svd_rank_trace.append(rank_this_iter)
            X_new, t_arr, obj, rel_change, basis_new, kept_rank_arr = _fista_step_randomized(
                X,
                X_step_prev,
                Y,
                prec,
                step_arr,
                lambda_arr,
                mom_step,
                svd_basis,
                key,
                svd_rank=rank_this_iter,
                svd_n_iter=svd_n_iter,
                svd_oversample=svd_oversample,
            )
            kept_rank = kept_rank_arr
        # Single host-device sync for both scalars
        if kept_rank is None:
            obj_val, rel_val = jax.device_get((obj, rel_change))
        else:
            obj_val, rel_val, kept_val = jax.device_get((obj, rel_change, kept_rank_arr))
            kept_rank = int(kept_val)
            last_kept_rank = kept_rank

        if (
            solver == "monotone_fista"
            and loss_trace
            and float(obj_val) > loss_trace[-1]
        ):
            X_new, _, obj, rel_change = _fista_step(
                X,
                X,
                Y,
                prec,
                step_arr,
                lambda_arr,
                jnp.asarray(1.0, dtype=jnp.float32),
            )
            obj_val, rel_val = jax.device_get((obj, rel_change))
            t_arr = jnp.asarray(1.0, dtype=jnp.float32)
            n_restarts += 1

        if obj_tol is not None and loss_trace:
            rel_obj = abs(loss_trace[-1] - float(obj_val)) / max(abs(float(obj_val)), 1.0)
            if rel_obj < obj_tol:
                stable_obj_count += 1
            else:
                stable_obj_count = 0

        loss_trace.append(float(obj_val))

        X_prev = X
        X = X_new
        svd_basis = basis_new
        mom_arr = jnp.asarray(1.0, dtype=jnp.float32) if solver == "pgd" else t_arr

        if svd_rank_adaptive and current_svd_rank is not None and kept_rank is not None:
            if kept_rank >= current_svd_rank and current_svd_rank < svd_rank_max:
                current_svd_rank = min(svd_rank_max, current_svd_rank + svd_rank_step)
            elif (
                kept_rank <= int(current_svd_rank * svd_rank_shrink_fraction)
                and current_svd_rank > svd_rank_min
            ):
                current_svd_rank = max(
                    svd_rank_min,
                    min(current_svd_rank - svd_rank_step, kept_rank + svd_rank_step),
                )

        if float(rel_val) < tol:
            converged = True
            convergence_reason = "rel_change"
            break
        if obj_tol is not None and stable_obj_count >= obj_patience:
            converged = True
            convergence_reason = "objective"
            break

    return ProximalResult(
        X=X,
        loss_trace=loss_trace,
        lambda_val=float(lambda_val),
        converged=converged,
        n_iter=len(loss_trace),
        convergence_reason=convergence_reason,
        n_restarts=n_restarts,
        svd_basis=svd_basis,
        svd_rank=current_svd_rank,
        svd_kept_rank=last_kept_rank,
        svd_rank_trace=svd_rank_trace,
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
    warm_start: bool = False,
    fixed_iter: bool = False,
    svd_rank: int | None = None,
    svd_n_iter: int = 2,
    svd_oversample: int = 10,
    random_seed: int = 0,
    svd_rank_adaptive: bool = False,
    svd_rank_min: int | None = None,
    svd_rank_max: int | None = None,
    svd_rank_step: int = 5,
    svd_rank_shrink_fraction: float = 0.5,
) -> tuple[float, ProximalResult]:
    """Select lambda by entry-wise K-fold CV, then refit on all observed entries.

    By default this is a thin wrapper around :func:`matlap.cv.cv_lambda`.
    Passing ``warm_start=True`` uses a descending lambda path inside each fold.
    The warm-started path is faster but can select a different lambda under a
    finite iteration budget, so it is opt-in.

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
    if not warm_start:
        return cv_lambda(
            Y, S, lambda_grid,
            fit_fn=proximal_gradient,
            get_mu=lambda r: r.X,
            n_folds=n_folds,
            verbose=verbose,
            max_iter=max_iter,
            tol=tol,
            fixed_iter=fixed_iter,
            svd_rank=svd_rank,
            svd_n_iter=svd_n_iter,
            svd_oversample=svd_oversample,
            random_seed=random_seed,
            svd_rank_adaptive=svd_rank_adaptive,
            svd_rank_min=svd_rank_min,
            svd_rank_max=svd_rank_max,
            svd_rank_step=svd_rank_step,
            svd_rank_shrink_fraction=svd_rank_shrink_fraction,
        )

    Y = jnp.asarray(Y, dtype=jnp.float32)
    S = jnp.asarray(S, dtype=jnp.float32)
    lambdas = sorted([float(lam) for lam in list(lambda_grid)], reverse=True)

    obs_mask_np = np.array(jnp.isfinite(S) & jnp.isfinite(Y))
    obs_idx = np.argwhere(obs_mask_np)
    n_obs = len(obs_idx)
    if n_obs == 0:
        raise ValueError("No observed entries (all S are inf).")
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2.")
    if n_obs < n_folds:
        raise ValueError(
            f"Fewer observed entries ({n_obs}) than folds ({n_folds})."
        )

    rng = np.random.default_rng(42)
    perm = rng.permutation(n_obs)
    fold_ids = perm % n_folds
    cv_mse = {lam: 0.0 for lam in lambdas}

    for fold in range(n_folds):
        val_pos = obs_idx[perm[fold_ids == fold]]
        S_train = S.at[val_pos[:, 0], val_pos[:, 1]].set(jnp.inf)
        init_X = None
        init_svd_basis = None

        for lam in lambdas:
            res = proximal_gradient(
                Y, S_train, lam, max_iter=max_iter, tol=tol, init_X=init_X,
                fixed_iter=fixed_iter, svd_rank=svd_rank,
                svd_n_iter=svd_n_iter, svd_oversample=svd_oversample,
                init_svd_basis=init_svd_basis, random_seed=random_seed + fold,
                svd_rank_adaptive=svd_rank_adaptive,
                svd_rank_min=svd_rank_min,
                svd_rank_max=svd_rank_max,
                svd_rank_step=svd_rank_step,
                svd_rank_shrink_fraction=svd_rank_shrink_fraction,
            )
            init_X = res.X
            init_svd_basis = res.svd_basis

            i_val, j_val = val_pos[:, 0], val_pos[:, 1]
            mse = float(jnp.mean(((res.X[i_val, j_val] - Y[i_val, j_val]) / S[i_val, j_val]) ** 2))
            cv_mse[lam] += mse / n_folds

        if verbose:
            print(f"  fold {fold + 1}/{n_folds} done")

    best_lambda = min(lambdas, key=lambda lam: cv_mse[lam])

    if verbose:
        for lam in lambdas:
            marker = " <-- best" if lam == best_lambda else ""
            print(f"  lambda={lam:.4g}  cv_mse={cv_mse[lam]:.6f}{marker}")

    init_X = None
    init_svd_basis = None
    final_result = None
    for lam in lambdas:
        res = proximal_gradient(
            Y, S, lam, max_iter=max_iter, tol=tol, init_X=init_X,
            fixed_iter=fixed_iter, svd_rank=svd_rank,
            svd_n_iter=svd_n_iter, svd_oversample=svd_oversample,
            init_svd_basis=init_svd_basis, random_seed=random_seed,
            svd_rank_adaptive=svd_rank_adaptive,
            svd_rank_min=svd_rank_min,
            svd_rank_max=svd_rank_max,
            svd_rank_step=svd_rank_step,
            svd_rank_shrink_fraction=svd_rank_shrink_fraction,
        )
        init_X = res.X
        init_svd_basis = res.svd_basis
        if lam == best_lambda:
            final_result = res
            break

    assert final_result is not None
    return best_lambda, final_result
