"""
MCMC samplers for the matrix Laplace (Nuclear-Norm Distribution) posterior.

Two samplers adapted from Segert & Wycoff (2025), arXiv:2510.05447:

1. ``mcmc_proximal_mala`` – Proximal Metropolis-Adjusted Langevin (MALA)
   targeting p(X | Y, S, lambda).  Uses singular-value soft-thresholding as a
   proximal gradient step; accepts/rejects with the full MH ratio.  Step size
   is adapted via Robbins–Monro to target the optimal acceptance rate 0.574.

2. ``mcmc_gsm_gibbs`` – Joint (X, lambda) Gibbs sampler.  Uses MALA for the
   X | lambda step (exact, with MH correction) and log-normal MH for lambda
   given X, with a half-Cauchy prior on lambda via an IG(1/2) auxiliary zeta.
   Marginalizes the GSM nu variables analytically so no GIG sampling is
   needed; avoids the shrinkage-amplification divergence of the fully-augmented
   chain.

Both samplers run warmup and sampling phases inside ``jax.lax.scan`` for
efficient JIT compilation.  The returned ``mu`` is the posterior mean over
the post-warmup samples.

See ``refs/nnd_mcmc_notes.tex`` for full algorithmic derivations.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class MCMCResult:
    """Result of an MCMC sampler.

    Attributes:
        mu:           Posterior mean of X (average over post-warmup samples),
                      shape (m, n).
        lambda_bar:   Final lambda value (fixed for MALA; sampled mean for
                      Gibbs).
        accept_rate:  Mean acceptance rate over all post-warmup steps.
        n_samples:    Number of post-warmup samples collected.
        lambda_trace: Post-warmup lambda samples, or ``None`` unless
                      ``return_trace=True``.
        accept_trace: Post-warmup MALA accept indicators.
        lambda_accept_trace:
                      Post-warmup lambda MH accept indicators for lambda
                      sampling; ``None`` for fixed-lambda MALA.
        nuclear_trace:
                      Post-warmup nuclear norm trace.
        logpost_trace:
                      Post-warmup log posterior trace for ``X | lambda``.
        step_size_trace:
                      MALA step-size trace.
        warmup_*:     Matching scalar traces from warmup.
    """

    mu: jax.Array
    lambda_bar: float
    accept_rate: float
    n_samples: int
    lambda_trace: jax.Array | None = None
    accept_trace: jax.Array | None = None
    lambda_accept_trace: jax.Array | None = None
    nuclear_trace: jax.Array | None = None
    logpost_trace: jax.Array | None = None
    step_size_trace: jax.Array | None = None
    warmup_lambda_trace: jax.Array | None = None
    warmup_accept_trace: jax.Array | None = None
    warmup_lambda_accept_trace: jax.Array | None = None
    warmup_nuclear_trace: jax.Array | None = None
    warmup_logpost_trace: jax.Array | None = None
    warmup_step_size_trace: jax.Array | None = None


# ---------------------------------------------------------------------------
# Proximal MALA helpers
# ---------------------------------------------------------------------------


def _svt(Z: jax.Array, threshold: jax.Array) -> jax.Array:
    """Singular value soft-thresholding (proximal op of threshold * ||.||_*)."""
    U, sv, Vt = jnp.linalg.svd(Z, full_matrices=False)
    return (U * jnp.maximum(sv - threshold, 0.0)) @ Vt


def _svt_with_nuc(Z: jax.Array, threshold: jax.Array) -> tuple[jax.Array, jax.Array]:
    """SVT and nuclear norm of Z in one SVD call."""
    U, sv, Vt = jnp.linalg.svd(Z, full_matrices=False)
    return (U * jnp.maximum(sv - threshold, 0.0)) @ Vt, sv.sum()


def _orthonormalize(A: jax.Array) -> jax.Array:
    Q, _ = jnp.linalg.qr(A)
    return Q


def _lowrank_svt_with_basis(
    Z: jax.Array,
    threshold: jax.Array,
    basis: jax.Array,
    n_iter: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Approximate SVT using a fixed/warm-started right singular subspace.

    This applies the SVT shrinkage only in the captured subspace and leaves the
    residual untouched.  Returning only the low-rank reconstruction would
    project away the residual even when the threshold is tiny, which produces
    unusably distant MALA proposal means and near-zero acceptance.
    """
    basis = _orthonormalize(basis)
    Q = _orthonormalize(Z @ basis)

    def power_body(_: int, Q_curr: jax.Array) -> jax.Array:
        return _orthonormalize(Z @ (Z.T @ Q_curr))

    Q = jax.lax.fori_loop(0, n_iter, power_body, Q)
    B = Q.T @ Z
    Ub, sv, Vt = jnp.linalg.svd(B, full_matrices=False)
    U = Q @ Ub
    shrink = jnp.minimum(sv, threshold)
    return Z - (U * shrink) @ Vt, Vt.T


@jax.jit
def _mala_step(
    X: jax.Array,
    nuc_X: jax.Array,
    key: jax.Array,
    Y: jax.Array,
    obs_mask: jax.Array,
    prec_noise: jax.Array,
    lambda_val: jax.Array,
    step_size: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """One proximal MALA step with MH acceptance.

    Returns:
        X_new:     Updated state (or X if rejected).
        nuc_new:   Nuclear norm of X_new.
        accepted:  1.0 if the proposal was accepted, else 0.0.
    """
    key, k1, k2 = jax.random.split(key, 3)
    half_step = 0.5 * step_size

    # --- forward proposal ---
    resid = jnp.where(obs_mask, Y - X, 0.0)
    grad = resid * prec_noise
    X_mean, _ = _svt_with_nuc(X + half_step * grad, half_step * lambda_val)
    X_star = X_mean + jnp.sqrt(step_size) * jax.random.normal(k1, X.shape)

    # --- reverse proposal mean (two more SVDs) ---
    resid_star = jnp.where(obs_mask, Y - X_star, 0.0)
    grad_star = resid_star * prec_noise
    X_mean_rev, _ = _svt_with_nuc(
        X_star + half_step * grad_star, half_step * lambda_val
    )

    # nuclear norm of X_star (one extra SVD)
    nuc_star = jnp.linalg.svd(X_star, compute_uv=False).sum()

    # --- log p(X) and log p(X*) ---
    log_lik_X = -0.5 * jnp.sum(resid**2 * prec_noise)
    log_lik_star = -0.5 * jnp.sum(resid_star**2 * prec_noise)

    # --- log q (forward and reverse) ---
    log_q_fwd = -jnp.sum((X_star - X_mean) ** 2) / (2.0 * step_size)
    log_q_rev = -jnp.sum((X - X_mean_rev) ** 2) / (2.0 * step_size)

    log_alpha = (
        (log_lik_star - lambda_val * nuc_star)
        - (log_lik_X - lambda_val * nuc_X)
        + log_q_rev
        - log_q_fwd
    )

    accepted = (jnp.log(jax.random.uniform(k2)) < log_alpha).astype(jnp.float32)
    X_new = jnp.where(accepted.astype(bool), X_star, X)
    nuc_new = jnp.where(accepted.astype(bool), nuc_star, nuc_X)
    return X_new, nuc_new, accepted


@jax.jit
def _mala_step_lowrank(
    X: jax.Array,
    nuc_X: jax.Array,
    basis: jax.Array,
    key: jax.Array,
    Y: jax.Array,
    obs_mask: jax.Array,
    prec_noise: jax.Array,
    lambda_val: jax.Array,
    step_size: jax.Array,
    proposal_svd_n_iter: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """One MALA step using an approximate low-rank SVT proposal mean.

    The target density still uses the exact nuclear norm of ``X_star``.  The
    returned basis is the forward proposal's right subspace and is intended for
    warmup adaptation; sampling freezes the basis to keep the proposal kernel
    fixed.
    """
    key, k1, k2 = jax.random.split(key, 3)
    half_step = 0.5 * step_size

    resid = jnp.where(obs_mask, Y - X, 0.0)
    grad = resid * prec_noise
    X_mean, basis_prop = _lowrank_svt_with_basis(
        X + half_step * grad,
        half_step * lambda_val,
        basis,
        proposal_svd_n_iter,
    )
    X_star = X_mean + jnp.sqrt(step_size) * jax.random.normal(k1, X.shape)

    resid_star = jnp.where(obs_mask, Y - X_star, 0.0)
    grad_star = resid_star * prec_noise
    X_mean_rev, _ = _lowrank_svt_with_basis(
        X_star + half_step * grad_star,
        half_step * lambda_val,
        basis,
        proposal_svd_n_iter,
    )

    nuc_star = jnp.linalg.svd(X_star, compute_uv=False).sum()
    log_lik_X = -0.5 * jnp.sum(resid**2 * prec_noise)
    log_lik_star = -0.5 * jnp.sum(resid_star**2 * prec_noise)
    log_q_fwd = -jnp.sum((X_star - X_mean) ** 2) / (2.0 * step_size)
    log_q_rev = -jnp.sum((X - X_mean_rev) ** 2) / (2.0 * step_size)

    log_alpha = (
        (log_lik_star - lambda_val * nuc_star)
        - (log_lik_X - lambda_val * nuc_X)
        + log_q_rev
        - log_q_fwd
    )

    accepted = (jnp.log(jax.random.uniform(k2)) < log_alpha).astype(jnp.float32)
    X_new = jnp.where(accepted.astype(bool), X_star, X)
    nuc_new = jnp.where(accepted.astype(bool), nuc_star, nuc_X)
    basis_new = jnp.where(accepted.astype(bool), basis_prop, basis)
    return X_new, nuc_new, basis_new, accepted


def _initial_right_basis(
    X: jax.Array,
    proposal_svd_rank: int | None,
    proposal_svd_oversample: int,
) -> jax.Array | None:
    """Initial right basis for the low-rank proposal path."""
    if proposal_svd_rank is None:
        return None
    if proposal_svd_rank <= 0:
        raise ValueError("proposal_svd_rank must be positive when provided.")
    if proposal_svd_oversample < 0:
        raise ValueError("proposal_svd_oversample must be non-negative.")

    max_rank = min(X.shape)
    k = min(max_rank, proposal_svd_rank + proposal_svd_oversample)
    if k >= max_rank:
        return None
    _, _, Vt = jnp.linalg.svd(X, full_matrices=False)
    return Vt[:k, :].T


@jax.jit
def _x_logpost(
    X: jax.Array,
    nuc_X: jax.Array,
    Y: jax.Array,
    obs_mask: jax.Array,
    prec_noise: jax.Array,
    lambda_val: jax.Array,
) -> jax.Array:
    """Unnormalized log posterior for ``X | lambda``."""
    resid = jnp.where(obs_mask, Y - X, 0.0)
    return -0.5 * jnp.sum(resid ** 2 * prec_noise) - lambda_val * nuc_X


def _empty_trace() -> dict[str, None]:
    return {
        "lambda_trace": None,
        "accept_trace": None,
        "lambda_accept_trace": None,
        "nuclear_trace": None,
        "logpost_trace": None,
        "step_size_trace": None,
        "warmup_lambda_trace": None,
        "warmup_accept_trace": None,
        "warmup_lambda_accept_trace": None,
        "warmup_nuclear_trace": None,
        "warmup_logpost_trace": None,
        "warmup_step_size_trace": None,
    }





# ---------------------------------------------------------------------------
# Public: Proximal MALA
# ---------------------------------------------------------------------------


def mcmc_proximal_mala(
    Y: jax.Array,
    S: jax.Array,
    lambda_val: float | None = None,
    *,
    x_init: jax.Array | None = None,
    n_warmup: int = 100,
    n_samples: int = 300,
    step_size_init: float | None = None,
    rm_step: float = 0.5,
    sample_lambda: bool = False,
    return_trace: bool = False,
    lambda_log_step: float | None = None,
    proposal_svd_rank: int | None = None,
    proposal_svd_n_iter: int = 1,
    proposal_svd_oversample: int = 5,
    key: jax.Array | None = None,
) -> MCMCResult:
    """Proximal MALA for the NND posterior p(X | Y, S, lambda).

    Uses singular-value soft-thresholding as the proximal gradient proposal,
    with Metropolis–Hastings correction.  Step size is adapted via
    Robbins–Monro to target the optimal acceptance rate 0.574.

    Args:
        Y:              Observations, shape (m, n); NaN or any value where S=inf.
        S:              Standard errors, shape (m, n); inf = missing.
        lambda_val:     Nuclear-norm regularisation weight.  When None, the
                        heuristic ``sqrt(max(m,n)) * median(1/s)`` is used.
        x_init:         Initial X matrix, shape (m, n).  Defaults to the
                        observed values (zeros for missing entries).  Passing
                        a posterior-mode estimate (e.g. from CAVI) improves
                        mixing speed significantly.
        n_warmup:       Number of burn-in steps (step size adaptation).
        n_samples:      Number of post-warmup samples to average.
        step_size_init: Initial MALA step size.  Defaults to an observed
                        dimension-scaled precision-adjusted value.
        rm_step:        Robbins–Monro learning-rate scale (default 0.5).
        sample_lambda:  If True, sample lambda with the GSM Gibbs/MH update
                        instead of keeping ``lambda_val`` fixed.  In that case
                        ``lambda_val`` is used as ``lambda_init`` when supplied.
        return_trace:   Return scalar warmup and sampling traces for diagnostics.
        lambda_log_step:
                        Log-scale random-walk proposal SD for lambda sampling.
                        Defaults to ``1 / sqrt(m*n)`` when ``sample_lambda=True``.
        proposal_svd_rank:
                        Optional rank for an approximate low-rank SVT proposal.
                        The exact full-SVD proposal is used when ``None``.
        proposal_svd_n_iter:
                        Power iterations for the approximate proposal SVT.
        proposal_svd_oversample:
                        Extra right-basis vectors for the approximate proposal.
        key:            JAX random key.  A fixed seed is used when None.

    Returns:
        MCMCResult with ``mu`` = posterior mean of X, ``lambda_bar`` =
        input lambda_val (fixed), ``accept_rate``, ``n_samples``.
    """
    if sample_lambda:
        return mcmc_gsm_gibbs(
            Y,
            S,
            lambda_init=lambda_val,
            x_init=x_init,
            n_warmup=n_warmup,
            n_samples=n_samples,
            step_size_init=step_size_init,
            rm_step=rm_step,
            proposal_svd_rank=proposal_svd_rank,
            proposal_svd_n_iter=proposal_svd_n_iter,
            proposal_svd_oversample=proposal_svd_oversample,
            return_trace=return_trace,
            lambda_log_step=lambda_log_step,
            key=key,
        )

    if key is None:
        key = jax.random.PRNGKey(42)
    if proposal_svd_n_iter < 0:
        raise ValueError("proposal_svd_n_iter must be non-negative.")

    obs_mask = jnp.isfinite(S)
    prec_noise = jnp.where(obs_mask, 1.0 / jnp.where(obs_mask, S, 1.0) ** 2, 0.0)
    m, n = Y.shape

    if lambda_val is None:
        max_prec = jnp.max(prec_noise)
        lambda_val = float(jnp.sqrt(max(m, n)) / jnp.sqrt(jnp.maximum(max_prec, 1e-10)))

    lambda_val_j = jnp.array(lambda_val, dtype=Y.dtype)

    if step_size_init is None:
        step_size_init = _default_step_size_init(prec_noise, obs_mask)

    # Initialize X: use provided x_init, else fall back to observed values
    if x_init is not None:
        X0 = x_init.astype(Y.dtype)
    else:
        X0 = jnp.where(obs_mask, Y, 0.0)
    nuc_X0 = jnp.linalg.svd(X0, compute_uv=False).sum()
    log_step_init = jnp.log(jnp.array(step_size_init, dtype=Y.dtype))
    basis0 = _initial_right_basis(X0, proposal_svd_rank, proposal_svd_oversample)

    # --- Warmup phase with Robbins–Monro step size adaptation ---
    key, key_warmup, key_sample = jax.random.split(key, 3)
    if basis0 is None:
        def warmup_body(carry, xs):
            key, iteration = xs
            X, nuc_X, log_step = carry
            step_size = jnp.exp(log_step)
            X_new, nuc_new, accepted = _mala_step(
                X, nuc_X, key, Y, obs_mask, prec_noise, lambda_val_j, step_size
            )
            log_step_new = _adapt_log_step(log_step, accepted, iteration, rm_step)
            logpost = _x_logpost(X_new, nuc_new, Y, obs_mask, prec_noise, lambda_val_j)
            trace = (lambda_val_j, accepted, jnp.nan, nuc_new, logpost, step_size)
            return (X_new, nuc_new, log_step_new), trace

        (X_warmed, nuc_warmed, log_step_final), warmup_trace = jax.lax.scan(
            warmup_body,
            (X0, nuc_X0, log_step_init),
            (jax.random.split(key_warmup, n_warmup), jnp.arange(n_warmup)),
        )

        step_size_final = jnp.exp(log_step_final)

        def sample_body(carry, key):
            X, nuc_X, sum_X, n_accept = carry
            X_new, nuc_new, accepted = _mala_step(
                X, nuc_X, key, Y, obs_mask, prec_noise, lambda_val_j, step_size_final
            )
            logpost = _x_logpost(X_new, nuc_new, Y, obs_mask, prec_noise, lambda_val_j)
            trace = (lambda_val_j, accepted, jnp.nan, nuc_new, logpost, step_size_final)
            return (X_new, nuc_new, sum_X + X_new, n_accept + accepted), trace

        (_, _, sum_X, total_accepts), sample_trace = jax.lax.scan(
            sample_body,
            (X_warmed, nuc_warmed, jnp.zeros_like(X_warmed), jnp.array(0.0)),
            jax.random.split(key_sample, n_samples),
        )
    else:
        n_iter_arr = jnp.asarray(proposal_svd_n_iter, dtype=jnp.int32)

        def warmup_body(carry, xs):
            key, iteration = xs
            X, nuc_X, basis, log_step = carry
            step_size = jnp.exp(log_step)
            X_new, nuc_new, basis_new, accepted = _mala_step_lowrank(
                X, nuc_X, basis, key, Y, obs_mask, prec_noise, lambda_val_j,
                step_size, n_iter_arr,
            )
            log_step_new = _adapt_log_step(log_step, accepted, iteration, rm_step)
            logpost = _x_logpost(X_new, nuc_new, Y, obs_mask, prec_noise, lambda_val_j)
            trace = (lambda_val_j, accepted, jnp.nan, nuc_new, logpost, step_size)
            return (X_new, nuc_new, basis_new, log_step_new), trace

        (X_warmed, nuc_warmed, basis_final, log_step_final), warmup_trace = jax.lax.scan(
            warmup_body,
            (X0, nuc_X0, basis0, log_step_init),
            (jax.random.split(key_warmup, n_warmup), jnp.arange(n_warmup)),
        )

        step_size_final = jnp.exp(log_step_final)

        def sample_body(carry, key):
            X, nuc_X, sum_X, n_accept = carry
            X_new, nuc_new, _, accepted = _mala_step_lowrank(
                X, nuc_X, basis_final, key, Y, obs_mask, prec_noise,
                lambda_val_j, step_size_final, n_iter_arr,
            )
            logpost = _x_logpost(X_new, nuc_new, Y, obs_mask, prec_noise, lambda_val_j)
            trace = (lambda_val_j, accepted, jnp.nan, nuc_new, logpost, step_size_final)
            return (X_new, nuc_new, sum_X + X_new, n_accept + accepted), trace

        (_, _, sum_X, total_accepts), sample_trace = jax.lax.scan(
            sample_body,
            (X_warmed, nuc_warmed, jnp.zeros_like(X_warmed), jnp.array(0.0)),
            jax.random.split(key_sample, n_samples),
        )

    mu = sum_X / n_samples
    accept_rate = float(total_accepts) / n_samples

    trace_kwargs = _empty_trace()
    if return_trace:
        (
            lambda_trace, accept_trace, lambda_accept_trace, nuclear_trace,
            logpost_trace, step_size_trace,
        ) = sample_trace
        (
            warmup_lambda_trace, warmup_accept_trace, warmup_lambda_accept_trace,
            warmup_nuclear_trace, warmup_logpost_trace, warmup_step_size_trace,
        ) = warmup_trace
        trace_kwargs = {
            "lambda_trace": lambda_trace,
            "accept_trace": accept_trace,
            "lambda_accept_trace": lambda_accept_trace,
            "nuclear_trace": nuclear_trace,
            "logpost_trace": logpost_trace,
            "step_size_trace": step_size_trace,
            "warmup_lambda_trace": warmup_lambda_trace,
            "warmup_accept_trace": warmup_accept_trace,
            "warmup_lambda_accept_trace": warmup_lambda_accept_trace,
            "warmup_nuclear_trace": warmup_nuclear_trace,
            "warmup_logpost_trace": warmup_logpost_trace,
            "warmup_step_size_trace": warmup_step_size_trace,
        }

    return MCMCResult(
        mu=mu,
        lambda_bar=float(lambda_val),
        accept_rate=accept_rate,
        n_samples=n_samples,
        **trace_kwargs,
    )


# ---------------------------------------------------------------------------
# Gibbs helpers
# ---------------------------------------------------------------------------


_LAMBDA_LOG_STEP: float = 0.5  # fallback random-walk MH step size in log(lambda) space
_MALA_TARGET_ACCEPT: float = 0.574
_LOG_STEP_MIN: float = -20.0
_LOG_STEP_MAX: float = 5.0


def _default_step_size_init(
    prec_noise: jax.Array,
    obs_mask: jax.Array,
) -> float:
    """Dimension-scaled MALA step-size initializer.

    The old ``0.5 / max_precision`` default is reasonable for tiny matrices but
    much too aggressive for MALA in thousands of dimensions.  The standard
    product-target MALA scaling is O(d^-1/3); using the observed entry count as
    the effective dimension gives warmup a close starting point instead of
    spending most of burn-in rejecting oversized proposals.  We scale by
    sqrt(precision), not precision, because the stochastic proposal variance is
    on the X scale while the gradient drift already carries the precision.
    """
    max_prec = float(jnp.max(prec_noise))
    n_obs = float(jnp.sum(obs_mask))
    dim_scale = max(n_obs, 1.0) ** (1.0 / 3.0)
    return 0.7 / (max(max_prec, 1e-6) ** 0.5 * dim_scale)


def _adapt_log_step(
    log_step: jax.Array,
    accepted: jax.Array,
    iteration: jax.Array,
    rm_step: float,
) -> jax.Array:
    """Diminishing Robbins-Monro update for MALA step-size warmup."""
    i = jnp.asarray(iteration, dtype=log_step.dtype)
    gain = jnp.asarray(rm_step, dtype=log_step.dtype) / jnp.sqrt(i + 1.0)
    log_step_new = log_step + gain * (accepted - _MALA_TARGET_ACCEPT)
    return jnp.clip(log_step_new, _LOG_STEP_MIN, _LOG_STEP_MAX)


@jax.jit
def _gibbs_step(
    X: jax.Array,
    nuc_X: jax.Array,
    lambda_bar: jax.Array,
    zeta: jax.Array,
    log_step: jax.Array,
    key: jax.Array,
    Y: jax.Array,
    obs_mask: jax.Array,
    prec_noise: jax.Array,
    lambda_log_step: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """One joint (X, lambda) Gibbs step using MALA for X and MH for lambda.

    The nu auxiliary variables from the GSM are integrated out analytically:
    the marginal p(X | Y, lambda) ∝ p(Y|X) * exp(-lambda ||X||_*) is sampled
    exactly (with MH correction) by the proximal MALA kernel.  Lambda is then
    updated from its marginal conditional:

        p(lambda | X, zeta) ∝ lambda^{mn} exp(-lambda ||X||_* - lambda^2/(2*zeta))

    via a log-normal random-walk MH step.  The half-Cauchy auxiliary zeta is
    updated by its IG(1/2, (1+lambda^2)/2) conjugate.

    Args:
        X:          Current X, shape (m, n).
        nuc_X:      Nuclear norm of X (recomputed on proposal change by MALA).
        lambda_bar: Current lambda, scalar.
        zeta:       Current half-Cauchy auxiliary, scalar.
        log_step:   Log of the current MALA step size (adapted during warmup).
        key:        JAX random key.
        Y, obs_mask, prec_noise: Data (fixed throughout the chain).

    Returns:
        X_new, nuc_new, lambda_new, zeta_new, mala_accepted, lam_accepted
    """
    m, n = Y.shape
    k1, k2, k3, k4 = jax.random.split(key, 4)

    # --- Step A: proximal MALA step for X | lambda (exact via MH) ---
    step_size = jnp.exp(log_step)
    X_new, nuc_new, mala_accepted = _mala_step(
        X, nuc_X, k1, Y, obs_mask, prec_noise, lambda_bar, step_size
    )

    # --- Step B: IG update for zeta | lambda ---
    # IG(1/2) has infinite mean; clamp to prevent occasional spikes from
    # causing a runaway lambda → zeta → lambda feedback loop.
    zeta_new = (1.0 + lambda_bar ** 2) / (
        2.0 * jax.random.gamma(k2, jnp.array(0.5, X.dtype))
    )
    zeta_new = jnp.minimum(zeta_new, jnp.array(1e6, X.dtype))

    # --- Step C: log-normal MH for lambda | X_new, zeta_new ---
    def log_p_lam(log_lam: jax.Array) -> jax.Array:
        lam = jnp.exp(log_lam)
        return m * n * log_lam - lam * nuc_new - lam ** 2 / (2.0 * zeta_new)

    log_lam_old = jnp.log(jnp.maximum(lambda_bar, 1e-30))
    log_lam_prop = log_lam_old + lambda_log_step * jax.random.normal(k3)
    # Jacobian from log(lambda) → lambda parameterisation: +log_lam_prop - log_lam_old
    log_alpha = (
        log_p_lam(log_lam_prop) + log_lam_prop
        - log_p_lam(log_lam_old) - log_lam_old
    )
    lam_accepted = (jax.random.uniform(k4) < jnp.exp(jnp.minimum(log_alpha, 0.0))).astype(X.dtype)
    lambda_new = jnp.where(lam_accepted.astype(bool), jnp.exp(log_lam_prop), lambda_bar)
    # Safety clamp: when nuc_new→0 (X→0) the posterior on lambda diverges;
    # clip to prevent numerical overflow.  The clamp only activates in
    # degenerate cases where the posterior is effectively improper.
    lambda_new = jnp.minimum(lambda_new, jnp.array(1e4, X.dtype))

    return X_new, nuc_new, lambda_new, zeta_new, mala_accepted, lam_accepted


@jax.jit
def _gibbs_step_lowrank(
    X: jax.Array,
    nuc_X: jax.Array,
    basis: jax.Array,
    lambda_bar: jax.Array,
    zeta: jax.Array,
    log_step: jax.Array,
    key: jax.Array,
    Y: jax.Array,
    obs_mask: jax.Array,
    prec_noise: jax.Array,
    proposal_svd_n_iter: jax.Array,
    lambda_log_step: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """One GSM Gibbs step using a low-rank approximate SVT MALA proposal."""
    m, n = Y.shape
    k1, k2, k3, k4 = jax.random.split(key, 4)

    step_size = jnp.exp(log_step)
    X_new, nuc_new, basis_new, mala_accepted = _mala_step_lowrank(
        X, nuc_X, basis, k1, Y, obs_mask, prec_noise, lambda_bar,
        step_size, proposal_svd_n_iter,
    )

    zeta_new = (1.0 + lambda_bar ** 2) / (
        2.0 * jax.random.gamma(k2, jnp.array(0.5, X.dtype))
    )
    zeta_new = jnp.minimum(zeta_new, jnp.array(1e6, X.dtype))

    def log_p_lam(log_lam: jax.Array) -> jax.Array:
        lam = jnp.exp(log_lam)
        return m * n * log_lam - lam * nuc_new - lam ** 2 / (2.0 * zeta_new)

    log_lam_old = jnp.log(jnp.maximum(lambda_bar, 1e-30))
    log_lam_prop = log_lam_old + lambda_log_step * jax.random.normal(k3)
    log_alpha = (
        log_p_lam(log_lam_prop) + log_lam_prop
        - log_p_lam(log_lam_old) - log_lam_old
    )
    lam_accepted = (jax.random.uniform(k4) < jnp.exp(jnp.minimum(log_alpha, 0.0))).astype(X.dtype)
    lambda_new = jnp.where(lam_accepted.astype(bool), jnp.exp(log_lam_prop), lambda_bar)
    lambda_new = jnp.minimum(lambda_new, jnp.array(1e4, X.dtype))

    return (
        X_new, nuc_new, basis_new, lambda_new, zeta_new, mala_accepted,
        lam_accepted,
    )


# ---------------------------------------------------------------------------
# Public: GSM Gibbs
# ---------------------------------------------------------------------------


def mcmc_gsm_gibbs(
    Y: jax.Array,
    S: jax.Array,
    *,
    lambda_init: float | None = None,
    x_init: jax.Array | None = None,
    n_warmup: int = 100,
    n_samples: int = 300,
    step_size_init: float | None = None,
    rm_step: float = 0.5,
    proposal_svd_rank: int | None = None,
    proposal_svd_n_iter: int = 1,
    proposal_svd_oversample: int = 5,
    return_trace: bool = False,
    lambda_log_step: float | None = None,
    key: jax.Array | None = None,
) -> MCMCResult:
    """Joint (X, lambda) Gibbs sampler for the NND posterior.

    Targets p(X, lambda | Y, S) using:

    (a) Proximal MALA step for X | lambda — exact via Metropolis–Hastings
        correction.  Uses singular-value soft-thresholding as the proximal
        proposal mean.  Step size is adapted via Robbins–Monro during warmup.
    (b) IG(1/2, (1+lambda^2)/2) update for the half-Cauchy auxiliary zeta.
    (c) Log-normal random-walk MH step for lambda | X, zeta from

            p(lambda | X, zeta) ∝ lambda^{mn} exp(-lambda ||X||_*
                                                    - lambda^2 / (2*zeta)).

    The nu auxiliary variables from the GSM representation are integrated
    out analytically: no GIG sampling is needed.  This avoids the
    shrinkage-amplification divergence of the fully-augmented chain and
    ensures the sampler targets the correct posterior.

    Args:
        Y:              Observations, shape (m, n); ignored where S = inf.
        S:              Standard errors, shape (m, n); inf = missing.
        lambda_init:    Initial lambda; defaults to ``m*n / ||X0||_*``.
        x_init:         Initial X matrix; defaults to Y with missing=0.
        n_warmup:       Burn-in steps (step size adapted via RM).
        n_samples:      Post-warmup samples averaged for posterior mean.
        step_size_init: Initial MALA step size.  Defaults to an observed
                        dimension-scaled precision-adjusted value.
        rm_step:        Robbins–Monro learning-rate scale for step adaptation.
        proposal_svd_rank:
                        Optional rank for an approximate low-rank SVT proposal
                        in the MALA ``X | lambda`` move.  Uses full SVD when
                        ``None``.
        proposal_svd_n_iter:
                        Power iterations for the approximate proposal SVT.
        proposal_svd_oversample:
                        Extra right-basis vectors for the approximate proposal.
        return_trace:   Return scalar warmup and sampling traces for diagnostics.
        lambda_log_step:
                        Log-scale random-walk proposal SD for lambda.  Defaults
                        to ``1 / sqrt(m*n)`` because the conditional
                        log-density curvature scales with the number of matrix
                        entries.
        key:            JAX random key; fixed seed if None.

    Returns:
        MCMCResult with ``mu`` = posterior mean of X, ``lambda_bar`` = mean
        sampled lambda over post-warmup samples, ``accept_rate`` = mean
        MALA acceptance rate.
    """
    if key is None:
        key = jax.random.PRNGKey(99)
    if proposal_svd_n_iter < 0:
        raise ValueError("proposal_svd_n_iter must be non-negative.")

    obs_mask = jnp.isfinite(S)
    prec_noise = jnp.where(obs_mask, 1.0 / jnp.where(obs_mask, S, 1.0) ** 2, 0.0)
    m, n = Y.shape
    if lambda_log_step is None:
        lambda_log_step = min(_LAMBDA_LOG_STEP, 1.0 / float(jnp.sqrt(jnp.asarray(m * n, dtype=Y.dtype))))
    if lambda_log_step <= 0:
        raise ValueError("lambda_log_step must be positive.")
    lambda_log_step_j = jnp.asarray(lambda_log_step, dtype=Y.dtype)

    if step_size_init is None:
        step_size_init = _default_step_size_init(prec_noise, obs_mask)

    X0 = x_init if x_init is not None else jnp.where(obs_mask, Y, 0.0)
    nuc_X0 = jnp.linalg.svd(X0, compute_uv=False).sum()
    if lambda_init is None:
        lambda_init = float((m * n) / jnp.maximum(nuc_X0, 1e-10))
    lambda0 = jnp.array(lambda_init, dtype=Y.dtype)
    zeta0 = jnp.ones((), dtype=Y.dtype)
    log_step0 = jnp.log(jnp.array(step_size_init, dtype=Y.dtype))
    basis0 = _initial_right_basis(X0, proposal_svd_rank, proposal_svd_oversample)

    # --- Warmup: adapt MALA step size via Robbins–Monro ---
    key, key_warmup, key_sample = jax.random.split(key, 3)
    if basis0 is None:
        def warmup_body(carry, xs):
            key, iteration = xs
            X, nuc_X, lam, zeta, log_step = carry
            step_size = jnp.exp(log_step)
            X_new, nuc_new, lam_new, zeta_new, mala_acc, lam_acc = _gibbs_step(
                X, nuc_X, lam, zeta, log_step, key, Y, obs_mask, prec_noise,
                lambda_log_step_j,
            )
            logpost = _x_logpost(X_new, nuc_new, Y, obs_mask, prec_noise, lam_new)
            log_step_new = _adapt_log_step(log_step, mala_acc, iteration, rm_step)
            trace = (lam_new, mala_acc, lam_acc, nuc_new, logpost, step_size)
            return (X_new, nuc_new, lam_new, zeta_new, log_step_new), trace

        (X_warmed, nuc_warmed, lam_warmed, zeta_warmed, log_step_final), warmup_trace = jax.lax.scan(
            warmup_body,
            (X0, nuc_X0, lambda0, zeta0, log_step0),
            (jax.random.split(key_warmup, n_warmup), jnp.arange(n_warmup)),
        )

        def sample_body(carry, key):
            X, nuc_X, lam, zeta, sum_X, sum_lam, n_acc = carry
            step_size = jnp.exp(log_step_final)
            X_new, nuc_new, lam_new, zeta_new, mala_acc, lam_acc = _gibbs_step(
                X, nuc_X, lam, zeta, log_step_final, key, Y, obs_mask,
                prec_noise, lambda_log_step_j,
            )
            logpost = _x_logpost(X_new, nuc_new, Y, obs_mask, prec_noise, lam_new)
            trace = (lam_new, mala_acc, lam_acc, nuc_new, logpost, step_size)
            return (X_new, nuc_new, lam_new, zeta_new,
                    sum_X + X_new, sum_lam + lam_new, n_acc + mala_acc), trace

        (_, _, _, _, sum_X, sum_lam, total_acc), sample_trace = jax.lax.scan(
            sample_body,
            (X_warmed, nuc_warmed, lam_warmed, zeta_warmed,
             jnp.zeros_like(X_warmed), jnp.zeros((), dtype=Y.dtype), jnp.array(0.0)),
            jax.random.split(key_sample, n_samples),
        )
    else:
        n_iter_arr = jnp.asarray(proposal_svd_n_iter, dtype=jnp.int32)

        def warmup_body(carry, xs):
            key, iteration = xs
            X, nuc_X, basis, lam, zeta, log_step = carry
            step_size = jnp.exp(log_step)
            (
                X_new, nuc_new, basis_new, lam_new, zeta_new, mala_acc, lam_acc,
            ) = _gibbs_step_lowrank(
                X, nuc_X, basis, lam, zeta, log_step, key, Y, obs_mask,
                prec_noise, n_iter_arr, lambda_log_step_j,
            )
            log_step_new = _adapt_log_step(log_step, mala_acc, iteration, rm_step)
            logpost = _x_logpost(X_new, nuc_new, Y, obs_mask, prec_noise, lam_new)
            trace = (lam_new, mala_acc, lam_acc, nuc_new, logpost, step_size)
            return (X_new, nuc_new, basis_new, lam_new, zeta_new, log_step_new), trace

        (
            X_warmed, nuc_warmed, basis_final, lam_warmed, zeta_warmed,
            log_step_final,
        ), warmup_trace = jax.lax.scan(
            warmup_body,
            (X0, nuc_X0, basis0, lambda0, zeta0, log_step0),
            (jax.random.split(key_warmup, n_warmup), jnp.arange(n_warmup)),
        )

        def sample_body(carry, key):
            X, nuc_X, lam, zeta, sum_X, sum_lam, n_acc = carry
            step_size = jnp.exp(log_step_final)
            (
                X_new, nuc_new, _, lam_new, zeta_new, mala_acc, lam_acc,
            ) = _gibbs_step_lowrank(
                X, nuc_X, basis_final, lam, zeta, log_step_final, key, Y,
                obs_mask, prec_noise, n_iter_arr, lambda_log_step_j,
            )
            logpost = _x_logpost(X_new, nuc_new, Y, obs_mask, prec_noise, lam_new)
            trace = (lam_new, mala_acc, lam_acc, nuc_new, logpost, step_size)
            return (X_new, nuc_new, lam_new, zeta_new,
                    sum_X + X_new, sum_lam + lam_new, n_acc + mala_acc), trace

        (_, _, _, _, sum_X, sum_lam, total_acc), sample_trace = jax.lax.scan(
            sample_body,
            (X_warmed, nuc_warmed, lam_warmed, zeta_warmed,
             jnp.zeros_like(X_warmed), jnp.zeros((), dtype=Y.dtype), jnp.array(0.0)),
            jax.random.split(key_sample, n_samples),
        )

    mu = sum_X / n_samples
    lambda_bar = float(sum_lam / n_samples)
    accept_rate = float(total_acc) / n_samples

    trace_kwargs = _empty_trace()
    if return_trace:
        (
            lambda_trace, accept_trace, lambda_accept_trace, nuclear_trace,
            logpost_trace, step_size_trace,
        ) = sample_trace
        (
            warmup_lambda_trace, warmup_accept_trace, warmup_lambda_accept_trace,
            warmup_nuclear_trace, warmup_logpost_trace, warmup_step_size_trace,
        ) = warmup_trace
        trace_kwargs = {
            "lambda_trace": lambda_trace,
            "accept_trace": accept_trace,
            "lambda_accept_trace": lambda_accept_trace,
            "nuclear_trace": nuclear_trace,
            "logpost_trace": logpost_trace,
            "step_size_trace": step_size_trace,
            "warmup_lambda_trace": warmup_lambda_trace,
            "warmup_accept_trace": warmup_accept_trace,
            "warmup_lambda_accept_trace": warmup_lambda_accept_trace,
            "warmup_nuclear_trace": warmup_nuclear_trace,
            "warmup_logpost_trace": warmup_logpost_trace,
            "warmup_step_size_trace": warmup_step_size_trace,
        }

    return MCMCResult(
        mu=mu,
        lambda_bar=lambda_bar,
        accept_rate=accept_rate,
        n_samples=n_samples,
        **trace_kwargs,
    )
