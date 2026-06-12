"""Proximal NND fitting with an extra homoskedastic noise component.

This module keeps the expensive Taylor term out of the X update:

    min_X 0.5 * sum_obs (Y_ij - X_ij)^2 / (S_ij^2 + g)
          + lambda_eff * ||X||_*

Then it updates the scalar extra variance ``g`` using a Taylor diagonal
variance approximation for uncertainty in X.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any
import warnings

import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg as jspla

from .proximal import proximal_gradient
from .taylor import _inv_sqrt_gram, _logdet_spd, _recover_sigma_diag_vmap


_JOINT_LAMBDA_G_WARNING = (
    "Joint estimation of lambda and g in proximal_noise_eb is not recommended; "
    "they can chase each other under the scaled NND prior. Prefer a fixed/grid "
    "effective lambda with g refit, selected by CV, Taylor ELBO, or LOO."
)


@dataclass
class ProximalNoiseResult:
    """Result for proximal X updates with empirical-Bayes noise fitting.

    Attributes:
        X:            Denoised matrix estimate, shape (m, n).
        sigma_diag:   Taylor diagonal variance approximation for X.
        lambda_val:   Grid/fitted lambda parameter.  This is ``lambda_eff``
                      when ``lambda_parameterization='effective'`` and the
                      base NND rate when ``lambda_parameterization='base'``.
        lambda_eff:   Effective proximal penalty used in the final X update.
        lambda_parameterization:
                      Either ``'effective'`` or ``'base'``.
        g:            Extra homoskedastic variance added to S**2.
        objective_trace:
                      Joint approximate negative log-posterior trace.
        lambda_trace: Lambda values after each outer update.
        g_trace:      Extra-variance values after each outer update.
        gamma_update: Method used to update g.
        converged:    True when both X/lambda and g stabilize.
        n_iter:       Number of outer iterations.
    """

    X: jax.Array
    sigma_diag: jax.Array | None = None
    lambda_val: float = 0.0
    lambda_eff: float = 0.0
    lambda_parameterization: str = "effective"
    g: float = 0.0
    objective_trace: list[float] = field(default_factory=list)
    lambda_trace: list[float] = field(default_factory=list)
    g_trace: list[float] = field(default_factory=list)
    gamma_update: str = "exact"
    converged: bool = False
    n_iter: int = 0


@dataclass
class ProximalNoiseGridResult:
    """Result for grid selection over lambda with g refit at each grid point."""

    best: ProximalNoiseResult
    grid_results: list[ProximalNoiseResult] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)


def _obs_mask(Y: jax.Array, S: jax.Array) -> jax.Array:
    return jnp.isfinite(Y) & jnp.isfinite(S)


def _effective_s(S: jax.Array, g: float | jax.Array) -> jax.Array:
    return jnp.sqrt(S ** 2 + jnp.asarray(g, dtype=S.dtype))


def _effective_lambda(lambda_val: float, g: float, scale_prior_by_g: bool) -> float:
    if not scale_prior_by_g:
        return float(lambda_val)
    return float(lambda_val / max(g ** 0.5, 1e-12))


def _resolve_lambda_parameterization(
    lambda_parameterization: str,
    scale_prior_by_g: bool | None,
) -> tuple[str, bool]:
    mode = lambda_parameterization.lower()
    if mode not in {"effective", "base"}:
        raise ValueError("lambda_parameterization must be 'effective' or 'base'.")
    implied = mode == "base"
    if scale_prior_by_g is None:
        return mode, implied
    explicit = bool(scale_prior_by_g)
    explicit_mode = "base" if explicit else "effective"
    if explicit != implied:
        raise ValueError(
            "scale_prior_by_g conflicts with lambda_parameterization; "
            f"got scale_prior_by_g={explicit} and "
            f"lambda_parameterization={lambda_parameterization!r}."
        )
    return explicit_mode, explicit


def taylor_diag_variance_at_prox(
    X: jax.Array,
    S_eff: jax.Array,
    lambda_val: float,
    *,
    gram_ridge: float = 1e-5,
    max_inv_sqrt: float = 1e4,
    precision_jitter: float = 1e-6,
) -> jax.Array:
    """Approximate ``diag Var(X_ij)`` at a proximal mode.

    The row Hessian approximation is

        diag(1 / S_eff_i^2) + lambda * (X.T X)^(-1/2).

    Only the diagonal of each row covariance is returned.
    """
    X = jnp.asarray(X, dtype=jnp.float32)
    S_eff = jnp.asarray(S_eff, dtype=jnp.float32)
    obs = jnp.isfinite(S_eff)
    prec = jnp.where(obs, 1.0 / (S_eff ** 2), 0.0)
    lambda_arr = jnp.asarray(lambda_val, dtype=X.dtype)
    inv_sqrt = _inv_sqrt_gram(
        X,
        jnp.asarray(gram_ridge, dtype=X.dtype),
        jnp.asarray(max_inv_sqrt, dtype=X.dtype),
    )
    return _recover_sigma_diag_vmap(
        prec,
        inv_sqrt,
        lambda_arr,
        jnp.asarray(precision_jitter, dtype=X.dtype),
    )


def _collapsed_noise_nll_log_g(
    log_g: jax.Array,
    residual2: jax.Array,
    S: jax.Array,
    obs: jax.Array,
    lambda_val: jax.Array,
    nuclear_norm: jax.Array,
    lambda_power: jax.Array,
    scale_prior_by_g: jax.Array,
    inv_sqrt: jax.Array,
    precision_jitter: jax.Array,
    gamma_log_coeff: jax.Array,
    gamma_inv_coeff: jax.Array,
) -> jax.Array:
    g = jnp.exp(log_g)
    lambda_eff = jnp.where(scale_prior_by_g, lambda_val / jnp.sqrt(g), lambda_val)
    base_var = jnp.where(obs, S ** 2, 1.0)
    resid = jnp.where(obs, residual2, 0.0)
    v = base_var + g
    prec = jnp.where(obs, 1.0 / v, 0.0)
    likelihood = 0.5 * jnp.sum(
        jnp.where(obs, jnp.log(v) + resid / v, 0.0)
    )

    def row_logdet(prec_i: jax.Array) -> jax.Array:
        A_i = jnp.diag(prec_i) + lambda_eff * inv_sqrt
        return _logdet_spd(A_i, precision_jitter)

    logdet = 0.5 * jnp.sum(jax.vmap(row_logdet)(prec))
    scaled_prior = jnp.where(
        scale_prior_by_g,
        lambda_eff * nuclear_norm + 0.5 * lambda_power * log_g,
        0.0,
    )
    return likelihood + logdet + scaled_prior + gamma_log_coeff * log_g + gamma_inv_coeff / g


_collapsed_noise_nll_value_and_grad = jax.jit(
    jax.value_and_grad(_collapsed_noise_nll_log_g)
)


def _update_g(
    g: float,
    residual2: jax.Array,
    S: jax.Array,
    obs: jax.Array,
    *,
    lambda_val: float,
    nuclear_norm: float,
    lambda_power: float,
    scale_prior_by_g: bool,
    inv_sqrt: jax.Array,
    precision_jitter: float,
    gamma_log_coeff: float,
    gamma_inv_coeff: float,
    min_g: float,
    max_g: float,
    lr: float,
    max_iter: int,
) -> tuple[float, float]:
    log_g = jnp.asarray(jnp.log(max(g, min_g)), dtype=jnp.float32)
    lo = jnp.asarray(jnp.log(min_g), dtype=jnp.float32)
    hi = jnp.asarray(jnp.log(max_g), dtype=jnp.float32)
    log_coeff = jnp.asarray(gamma_log_coeff, dtype=jnp.float32)
    inv_coeff = jnp.asarray(gamma_inv_coeff, dtype=jnp.float32)
    lambda_arr = jnp.asarray(lambda_val, dtype=jnp.float32)
    nuclear_arr = jnp.asarray(nuclear_norm, dtype=jnp.float32)
    lambda_power_arr = jnp.asarray(lambda_power, dtype=jnp.float32)
    scale_arr = jnp.asarray(scale_prior_by_g)
    jitter_arr = jnp.asarray(precision_jitter, dtype=jnp.float32)
    last_step = 0.0

    for _ in range(max_iter):
        val, grad = _collapsed_noise_nll_value_and_grad(
            log_g,
            residual2,
            S,
            obs,
            lambda_arr,
            nuclear_arr,
            lambda_power_arr,
            scale_arr,
            inv_sqrt,
            jitter_arr,
            log_coeff,
            inv_coeff,
        )
        grad_val = float(jax.device_get(grad))
        if abs(grad_val) < 1e-5:
            break

        step = lr
        accepted = False
        for _ in range(20):
            cand = jnp.clip(log_g - step * grad, lo, hi)
            cand_val = _collapsed_noise_nll_log_g(
                cand,
                residual2,
                S,
                obs,
                lambda_arr,
                nuclear_arr,
                lambda_power_arr,
                scale_arr,
                inv_sqrt,
                jitter_arr,
                log_coeff,
                inv_coeff,
            )
            cand_val_f, val_f = jax.device_get((cand_val, val))
            if float(cand_val_f) <= float(val_f) + 1e-8:
                log_g = cand
                last_step = step
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break

    return float(jnp.exp(log_g)), last_step


def _make_hutchinson_probes(
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
def _hutch_log_g_grad(
    log_g: jax.Array,
    residual2: jax.Array,
    S: jax.Array,
    obs: jax.Array,
    lambda_val: jax.Array,
    nuclear_norm: jax.Array,
    lambda_power: jax.Array,
    scale_prior_by_g: jax.Array,
    inv_sqrt: jax.Array,
    gamma_log_coeff: jax.Array,
    gamma_inv_coeff: jax.Array,
    probes: jax.Array,
    cg_x0: jax.Array,
    *,
    cg_tol: float,
    cg_maxiter: int,
) -> tuple[jax.Array, jax.Array]:
    """Gradient wrt log(g) using Hutchinson probes for the logdet term."""
    g = jnp.exp(log_g)
    lambda_eff = jnp.where(scale_prior_by_g, lambda_val / jnp.sqrt(g), lambda_val)
    d_lambda_eff = jnp.where(scale_prior_by_g, -0.5 * lambda_eff, 0.0)

    base_var = jnp.where(obs, S ** 2, 1.0)
    resid = jnp.where(obs, residual2, 0.0)
    v = base_var + g
    prec = jnp.where(obs, 1.0 / v, 0.0)
    dprec = jnp.where(obs, -g / (v ** 2), 0.0)

    likelihood_grad = 0.5 * jnp.sum(
        jnp.where(obs, g / v - resid * g / (v ** 2), 0.0)
    )
    scaled_prior_grad = jnp.where(
        scale_prior_by_g,
        d_lambda_eff * nuclear_norm + 0.5 * lambda_power,
        0.0,
    )
    gamma_prior_grad = gamma_log_coeff - gamma_inv_coeff / g
    trace_B = jnp.trace(inv_sqrt)
    n = inv_sqrt.shape[0]

    def row_probe_solve(prec_i, dprec_i, z, x0):
        mean_diag = (jnp.sum(prec_i) + lambda_eff * trace_B) / jnp.asarray(n, dtype=prec_i.dtype)
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
    grad = likelihood_grad + logdet_grad + scaled_prior_grad + gamma_prior_grad
    return grad, new_x0


def _update_g_hutchinson(
    g: float,
    residual2: jax.Array,
    S: jax.Array,
    obs: jax.Array,
    *,
    lambda_val: float,
    nuclear_norm: float,
    lambda_power: float,
    scale_prior_by_g: bool,
    inv_sqrt: jax.Array,
    gamma_log_coeff: float,
    gamma_inv_coeff: float,
    min_g: float,
    max_g: float,
    lr: float,
    max_iter: int,
    probes: jax.Array,
    cg_x0: jax.Array,
    cg_tol: float,
    cg_maxiter: int,
    grad_clip: float,
) -> tuple[float, jax.Array]:
    log_g = jnp.asarray(jnp.log(max(g, min_g)), dtype=jnp.float32)
    lo = jnp.asarray(jnp.log(min_g), dtype=jnp.float32)
    hi = jnp.asarray(jnp.log(max_g), dtype=jnp.float32)
    for _ in range(max_iter):
        grad, cg_x0 = _hutch_log_g_grad(
            log_g,
            residual2,
            S,
            obs,
            jnp.asarray(lambda_val, dtype=jnp.float32),
            jnp.asarray(nuclear_norm, dtype=jnp.float32),
            jnp.asarray(lambda_power, dtype=jnp.float32),
            jnp.asarray(scale_prior_by_g),
            inv_sqrt,
            jnp.asarray(gamma_log_coeff, dtype=jnp.float32),
            jnp.asarray(gamma_inv_coeff, dtype=jnp.float32),
            probes,
            cg_x0,
            cg_tol=cg_tol,
            cg_maxiter=cg_maxiter,
        )
        grad = jnp.clip(grad, -grad_clip, grad_clip)
        log_g = jnp.clip(log_g - lr * grad, lo, hi)
    return float(jnp.exp(log_g)), cg_x0


def _joint_objective(
    X: jax.Array,
    Y: jax.Array,
    S: jax.Array,
    obs: jax.Array,
    lambda_val: float,
    g: float,
    *,
    scale_prior_by_g: bool,
    inv_sqrt: jax.Array,
    precision_jitter: float,
    lambda_power: float,
    lambda_prior_a: float,
    lambda_prior_b: float,
    gamma_log_coeff: float,
    gamma_inv_coeff: float,
) -> float:
    residual2 = (Y - X) ** 2
    nuc = jnp.linalg.svd(X, compute_uv=False).sum()
    noise_nll = _collapsed_noise_nll_log_g(
        jnp.asarray(jnp.log(g), dtype=jnp.float32),
        residual2,
        S,
        obs,
        jnp.asarray(lambda_val, dtype=jnp.float32),
        nuc,
        jnp.asarray(lambda_power, dtype=jnp.float32),
        jnp.asarray(scale_prior_by_g),
        inv_sqrt,
        jnp.asarray(precision_jitter, dtype=jnp.float32),
        jnp.asarray(gamma_log_coeff, dtype=jnp.float32),
        jnp.asarray(gamma_inv_coeff, dtype=jnp.float32),
    )
    lambda_arr = jnp.asarray(lambda_val, dtype=jnp.float32)
    lambda_nlp = -(
        lambda_power + lambda_prior_a - 1.0
    ) * jnp.log(lambda_arr) + lambda_prior_b * lambda_arr
    if not scale_prior_by_g:
        lambda_nlp = lambda_nlp + lambda_arr * nuc
    return float(jax.device_get(noise_nll + lambda_nlp))


def _lambda_mode(
    X: jax.Array,
    *,
    lambda_power: float,
    lambda_prior_a: float,
    lambda_prior_b: float,
    min_lambda: float,
    max_lambda: float,
    g: float = 1.0,
    scale_prior_by_g: bool = False,
) -> float:
    nuc = float(jax.device_get(jnp.linalg.svd(X, compute_uv=False).sum()))
    numerator = max(lambda_power + lambda_prior_a - 1.0, 1e-12)
    denom = nuc / max(g ** 0.5, 1e-12) if scale_prior_by_g else nuc
    lam = numerator / max(denom + lambda_prior_b, 1e-12)
    return float(min(max(lam, min_lambda), max_lambda))


def proximal_noise_eb(
    Y: jax.Array,
    S: jax.Array,
    *,
    lambda_val: float | None = None,
    update_lambda: bool = True,
    init_g: float | None = None,
    max_outer: int = 20,
    tol: float = 1e-4,
    prox_max_iter: int = 80,
    prox_tol: float = 1e-5,
    prox_solver: str = "fista",
    prox_fixed_iter: bool = True,
    svd_rank: int | None = None,
    svd_n_iter: int = 2,
    svd_oversample: int = 10,
    random_seed: int = 0,
    gamma_lr: float = 0.5,
    gamma_max_iter: int = 25,
    gamma_update: str = "exact",
    hutchinson_probes: int = 4,
    hutchinson_lr: float | None = None,
    hutchinson_cg_tol: float = 1e-3,
    hutchinson_cg_maxiter: int = 50,
    hutchinson_grad_clip: float = 1e3,
    hutchinson_seed: int | None = None,
    score_exact_objective: bool = True,
    recover_sigma_diag: bool = True,
    min_g: float = 1e-8,
    max_g: float = 1e4,
    min_lambda: float = 1e-6,
    max_lambda: float = 1e4,
    lambda_power: float | None = None,
    lambda_prior_a: float = 1.0,
    lambda_prior_b: float = 0.0,
    gamma_prior_a: float | None = None,
    gamma_prior_b: float = 0.0,
    lambda_parameterization: str = "effective",
    scale_prior_by_g: bool | None = None,
    gram_ridge: float = 1e-5,
    max_inv_sqrt: float = 1e4,
    precision_jitter: float = 1e-6,
    transpose_if_wide: bool = True,
) -> ProximalNoiseResult:
    """Fit X by proximal gradient and estimate extra variance ``g``.

    If ``update_lambda=True``, lambda is updated by the conditional mode under
    the NND normalizer and a Gamma(lambda_prior_a, lambda_prior_b) prior.  If
    ``update_lambda=False``, ``lambda_val`` is held fixed.  Joint lambda/g
    estimation is experimental and not recommended; use a lambda grid in
    practice.

    The recommended parameterization is ``lambda_parameterization='effective'``:
    ``lambda_val`` is the effective proximal penalty and the NND rate is fixed
    while fitting ``g``.  With ``lambda_parameterization='base'``, ``lambda_val``
    is the base rate and the NND/proximal rate is ``lambda_val / sqrt(g)``.
    """
    gamma_update = gamma_update.lower()
    if gamma_update not in {"exact", "hutchinson"}:
        raise ValueError("gamma_update must be one of 'exact' or 'hutchinson'.")
    lambda_parameterization, scale_prior_by_g = _resolve_lambda_parameterization(
        lambda_parameterization,
        scale_prior_by_g,
    )
    if hutchinson_probes < 1:
        raise ValueError("hutchinson_probes must be positive.")
    if hutchinson_cg_maxiter < 1:
        raise ValueError("hutchinson_cg_maxiter must be positive.")
    Y = jnp.asarray(Y, dtype=jnp.float32)
    S = jnp.asarray(S, dtype=jnp.float32)
    if Y.shape != S.shape:
        raise ValueError(f"Y shape {Y.shape} does not match S shape {S.shape}.")
    if transpose_if_wide and Y.shape[0] < Y.shape[1]:
        res = proximal_noise_eb(
            Y.T,
            S.T,
            lambda_val=lambda_val,
            update_lambda=update_lambda,
            init_g=init_g,
            max_outer=max_outer,
            tol=tol,
            prox_max_iter=prox_max_iter,
            prox_tol=prox_tol,
            prox_solver=prox_solver,
            prox_fixed_iter=prox_fixed_iter,
            svd_rank=svd_rank,
            svd_n_iter=svd_n_iter,
            svd_oversample=svd_oversample,
            random_seed=random_seed,
            gamma_lr=gamma_lr,
            gamma_max_iter=gamma_max_iter,
            gamma_update=gamma_update,
            hutchinson_probes=hutchinson_probes,
            hutchinson_lr=hutchinson_lr,
            hutchinson_cg_tol=hutchinson_cg_tol,
            hutchinson_cg_maxiter=hutchinson_cg_maxiter,
            hutchinson_grad_clip=hutchinson_grad_clip,
            hutchinson_seed=hutchinson_seed,
            score_exact_objective=score_exact_objective,
            recover_sigma_diag=recover_sigma_diag,
            min_g=min_g,
            max_g=max_g,
            min_lambda=min_lambda,
            max_lambda=max_lambda,
            lambda_power=lambda_power,
            lambda_prior_a=lambda_prior_a,
            lambda_prior_b=lambda_prior_b,
            gamma_prior_a=gamma_prior_a,
            gamma_prior_b=gamma_prior_b,
            lambda_parameterization=lambda_parameterization,
            scale_prior_by_g=scale_prior_by_g,
            gram_ridge=gram_ridge,
            max_inv_sqrt=max_inv_sqrt,
            precision_jitter=precision_jitter,
            transpose_if_wide=False,
        )
        return ProximalNoiseResult(
            X=res.X.T,
            sigma_diag=None if res.sigma_diag is None else res.sigma_diag.T,
            lambda_val=res.lambda_val,
            lambda_eff=res.lambda_eff,
            lambda_parameterization=res.lambda_parameterization,
            g=res.g,
            objective_trace=res.objective_trace,
            lambda_trace=res.lambda_trace,
            g_trace=res.g_trace,
            gamma_update=res.gamma_update,
            converged=res.converged,
            n_iter=res.n_iter,
        )

    obs = _obs_mask(Y, S)
    if not bool(jax.device_get(jnp.any(obs))):
        raise ValueError("At least one observed entry is required.")
    if update_lambda:
        # Empirically this joint conditional-mode update is unstable: with the
        # NND scaling lambda/sqrt(g), lambda and g can reinforce each other and
        # run to boundary values. Keep it available for diagnostics, but warn
        # loudly so benchmark results are not mistaken for a recommended fit.
        warnings.warn(_JOINT_LAMBDA_G_WARNING, RuntimeWarning, stacklevel=2)
        print(f"WARNING: {_JOINT_LAMBDA_G_WARNING}")
    if lambda_power is None:
        lambda_power = float(Y.shape[0] * Y.shape[1])

    finite_s2 = jnp.where(obs, S ** 2, jnp.nan)
    median_s2 = float(jax.device_get(jnp.nanmedian(finite_s2)))
    g = float(init_g if init_g is not None else max(0.1 * median_s2, min_g))
    g = min(max(g, min_g), max_g)

    if lambda_val is None:
        lam = max(1.0 / max(float(jax.device_get(jnp.nanstd(jnp.where(obs, Y, jnp.nan)))), 1e-6), min_lambda)
    else:
        lam = float(lambda_val)
    lam = min(max(lam, min_lambda), max_lambda)
    if not update_lambda and lambda_val is None:
        raise ValueError("lambda_val must be provided when update_lambda=False.")

    gamma_log_coeff = 0.0 if gamma_prior_a is None else gamma_prior_a + 1.0
    gamma_inv_coeff = 0.0 if gamma_prior_a is None else gamma_prior_b

    X = None
    sigma_diag = None
    objective_trace: list[float] = []
    lambda_trace: list[float] = []
    g_trace: list[float] = []
    converged = False
    cg_state = None
    probes = None
    if gamma_update == "hutchinson":
        probes = _make_hutchinson_probes(
            Y.shape[0],
            Y.shape[1],
            hutchinson_probes,
            seed=random_seed + 1729 if hutchinson_seed is None else hutchinson_seed,
            dtype=Y.dtype,
        )
        cg_state = jnp.zeros_like(probes)
    hutch_lr = min(gamma_lr, 1e-3) if hutchinson_lr is None else hutchinson_lr

    for outer in range(max_outer):
        prev_g = g
        prev_lam = lam
        prev_X = X

        S_eff = _effective_s(S, g)
        lam_eff = _effective_lambda(lam, g, scale_prior_by_g)
        prox = proximal_gradient(
            Y,
            S_eff,
            lam_eff,
            max_iter=prox_max_iter,
            tol=prox_tol,
            init_X=X,
            solver=prox_solver,
            fixed_iter=prox_fixed_iter,
            svd_rank=svd_rank,
            svd_n_iter=svd_n_iter,
            svd_oversample=svd_oversample,
            init_svd_basis=None,
            random_seed=random_seed + outer,
        )
        X = prox.X
        nuclear_norm = float(jax.device_get(jnp.linalg.svd(X, compute_uv=False).sum()))
        inv_sqrt = _inv_sqrt_gram(
            X,
            jnp.asarray(gram_ridge, dtype=X.dtype),
            jnp.asarray(max_inv_sqrt, dtype=X.dtype),
        )
        residual2 = (Y - X) ** 2
        if gamma_update == "exact":
            g, _ = _update_g(
                g,
                residual2,
                S,
                obs,
                lambda_val=lam,
                nuclear_norm=nuclear_norm,
                lambda_power=lambda_power,
                scale_prior_by_g=scale_prior_by_g,
                inv_sqrt=inv_sqrt,
                precision_jitter=precision_jitter,
                gamma_log_coeff=gamma_log_coeff,
                gamma_inv_coeff=gamma_inv_coeff,
                min_g=min_g,
                max_g=max_g,
                lr=gamma_lr,
                max_iter=gamma_max_iter,
            )
        else:
            g, cg_state = _update_g_hutchinson(
                g,
                residual2,
                S,
                obs,
                lambda_val=lam,
                nuclear_norm=nuclear_norm,
                lambda_power=lambda_power,
                scale_prior_by_g=scale_prior_by_g,
                inv_sqrt=inv_sqrt,
                gamma_log_coeff=gamma_log_coeff,
                gamma_inv_coeff=gamma_inv_coeff,
                min_g=min_g,
                max_g=max_g,
                lr=hutch_lr,
                max_iter=gamma_max_iter,
                probes=probes,
                cg_x0=cg_state,
                cg_tol=hutchinson_cg_tol,
                cg_maxiter=hutchinson_cg_maxiter,
                grad_clip=hutchinson_grad_clip,
            )
        S_eff = _effective_s(S, g)
        lam_eff = _effective_lambda(lam, g, scale_prior_by_g)
        if update_lambda:
            lam = _lambda_mode(
                X,
                lambda_power=lambda_power,
                lambda_prior_a=lambda_prior_a,
                lambda_prior_b=lambda_prior_b,
                min_lambda=min_lambda,
                max_lambda=max_lambda,
                g=g,
                scale_prior_by_g=scale_prior_by_g,
            )
            lam_eff = _effective_lambda(lam, g, scale_prior_by_g)

        if score_exact_objective:
            obj = _joint_objective(
                X,
                Y,
                S,
                obs,
                lam,
                g,
                scale_prior_by_g=scale_prior_by_g,
                inv_sqrt=inv_sqrt,
                precision_jitter=precision_jitter,
                lambda_power=lambda_power,
                lambda_prior_a=lambda_prior_a,
                lambda_prior_b=lambda_prior_b,
                gamma_log_coeff=gamma_log_coeff,
                gamma_inv_coeff=gamma_inv_coeff,
            )
        else:
            obj = float("nan")
        objective_trace.append(obj)
        lambda_trace.append(lam)
        g_trace.append(g)

        if prev_X is not None:
            dx = float(jax.device_get(
                jnp.linalg.norm(X - prev_X) / jnp.maximum(jnp.linalg.norm(X), 1.0)
            ))
            dg = abs(g - prev_g) / max(abs(g), 1.0)
            dl = abs(lam - prev_lam) / max(abs(lam), 1.0)
            if max(dx, dg, dl) < tol:
                converged = True
                break

    if X is not None and recover_sigma_diag:
        S_eff = _effective_s(S, g)
        lam_eff = _effective_lambda(lam, g, scale_prior_by_g)
        sigma_diag = taylor_diag_variance_at_prox(
            X,
            S_eff,
            lam_eff,
            gram_ridge=gram_ridge,
            max_inv_sqrt=max_inv_sqrt,
            precision_jitter=precision_jitter,
        )

    return ProximalNoiseResult(
        X=X,
        sigma_diag=sigma_diag,
        lambda_val=lam,
        lambda_eff=_effective_lambda(lam, g, scale_prior_by_g),
        g=g,
        objective_trace=objective_trace,
        lambda_trace=lambda_trace,
        g_trace=g_trace,
        gamma_update=gamma_update,
        converged=converged,
        n_iter=len(objective_trace),
        lambda_parameterization=lambda_parameterization,
    )


def proximal_noise_lambda_grid(
    Y: jax.Array,
    S: jax.Array,
    lambda_grid: list[float] | tuple[float, ...] | jax.Array,
    **kwargs: Any,
) -> ProximalNoiseGridResult:
    """Fit ``g`` for each fixed lambda and select by approximate objective.

    By default, grid values are effective proximal penalties.  Pass
    ``lambda_parameterization='base'`` to use the older base-rate grid where
    the effective penalty is ``lambda / sqrt(g)``.
    """
    grid = [float(x) for x in list(lambda_grid)]
    if not grid:
        raise ValueError("lambda_grid must contain at least one value.")

    grid_results: list[ProximalNoiseResult] = []
    rows: list[dict[str, Any]] = []
    for lam in grid:
        res = proximal_noise_eb(
            Y,
            S,
            lambda_val=lam,
            update_lambda=False,
            **kwargs,
        )
        grid_results.append(res)
        rows.append({
            "lambda": lam,
            "lambda_eff": res.lambda_eff,
            "lambda_parameterization": res.lambda_parameterization,
            "g": res.g,
            "objective": res.objective_trace[-1],
            "n_iter": res.n_iter,
            "converged": res.converged,
        })

    best_idx = min(range(len(grid_results)), key=lambda i: grid_results[i].objective_trace[-1])
    return ProximalNoiseGridResult(
        best=grid_results[best_idx],
        grid_results=grid_results,
        rows=rows,
    )
