"""
MCMC samplers for the matrix Laplace (Nuclear-Norm Distribution) posterior.

Two samplers adapted from Segert & Wycoff (2025), arXiv:2510.05447:

1. ``mcmc_proximal_mala`` – Proximal Metropolis-Adjusted Langevin (MALA)
   targeting p(X | Y, S, lambda).  Uses singular-value soft-thresholding as a
   proximal gradient step; accepts/rejects with the full MH ratio.  Step size
   is adapted via Robbins–Monro to target the optimal acceptance rate 0.574.

2. ``mcmc_gsm_gibbs`` – Gibbs sampler exploiting the Gaussian Scale Mixture
   (GSM) auxiliary variable Q.  Cycles through:
   (a) exact row-wise Gaussian sampling of X | Q, lambda,
   (b) Metropolis-within-Gibbs updates of Q's eigenvalues,
   (c) conjugate Gamma sampling of lambda | X, Q.

Both samplers run warmup and sampling phases inside ``jax.lax.scan`` for
efficient JIT compilation.  The returned ``mu`` is the posterior mean over
the post-warmup samples.

See ``refs/nnd_mcmc.tex`` for full algorithmic derivations.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla


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
    """

    mu: jax.Array
    lambda_bar: float
    accept_rate: float
    n_samples: int


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
    rm_step: float = 0.05,
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
        step_size_init: Initial MALA step size.  Defaults to
                        ``0.5 / max_observed_precision``.
        rm_step:        Robbins–Monro learning rate (default 0.05).
        key:            JAX random key.  A fixed seed is used when None.

    Returns:
        MCMCResult with ``mu`` = posterior mean of X, ``lambda_bar`` =
        input lambda_val (fixed), ``accept_rate``, ``n_samples``.
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    obs_mask = jnp.isfinite(S)
    prec_noise = jnp.where(obs_mask, 1.0 / jnp.where(obs_mask, S, 1.0) ** 2, 0.0)
    m, n = Y.shape

    if lambda_val is None:
        max_prec = jnp.max(prec_noise)
        lambda_val = float(jnp.sqrt(max(m, n)) / jnp.sqrt(jnp.maximum(max_prec, 1e-10)))

    lambda_val_j = jnp.array(lambda_val, dtype=Y.dtype)

    if step_size_init is None:
        max_prec = float(jnp.max(prec_noise))
        step_size_init = 0.5 / max(max_prec, 1e-6)

    # Initialize X: use provided x_init, else fall back to observed values
    if x_init is not None:
        X0 = x_init.astype(Y.dtype)
    else:
        X0 = jnp.where(obs_mask, Y, 0.0)
    nuc_X0 = jnp.linalg.svd(X0, compute_uv=False).sum()
    log_step_init = jnp.log(jnp.array(step_size_init, dtype=Y.dtype))

    # --- Warmup phase with Robbins–Monro step size adaptation ---
    def warmup_body(carry, key):
        X, nuc_X, log_step = carry
        step_size = jnp.exp(log_step)
        X_new, nuc_new, accepted = _mala_step(
            X, nuc_X, key, Y, obs_mask, prec_noise, lambda_val_j, step_size
        )
        log_step_new = log_step + rm_step * (accepted - 0.574)
        return (X_new, nuc_new, log_step_new), accepted

    key, key_warmup, key_sample = jax.random.split(key, 3)
    (X_warmed, nuc_warmed, log_step_final), warmup_accepts = jax.lax.scan(
        warmup_body,
        (X0, nuc_X0, log_step_init),
        jax.random.split(key_warmup, n_warmup),
    )

    # --- Sampling phase (fixed step size from end of warmup) ---
    step_size_final = jnp.exp(log_step_final)

    def sample_body(carry, key):
        X, nuc_X, sum_X, n_accept = carry
        X_new, nuc_new, accepted = _mala_step(
            X, nuc_X, key, Y, obs_mask, prec_noise, lambda_val_j, step_size_final
        )
        return (X_new, nuc_new, sum_X + X_new, n_accept + accepted), None

    (X_final, _, sum_X, total_accepts), _ = jax.lax.scan(
        sample_body,
        (X_warmed, nuc_warmed, jnp.zeros_like(X_warmed), jnp.array(0.0)),
        jax.random.split(key_sample, n_samples),
    )

    mu = sum_X / n_samples
    accept_rate = float(total_accepts) / n_samples

    return MCMCResult(
        mu=mu,
        lambda_bar=float(lambda_val),
        accept_rate=accept_rate,
        n_samples=n_samples,
    )


# ---------------------------------------------------------------------------
# GSM Gibbs helpers
# ---------------------------------------------------------------------------


@jax.jit
def _sample_x_rows(
    Y: jax.Array,
    obs_mask: jax.Array,
    prec_noise: jax.Array,
    Q_inv: jax.Array,
    lambda_bar: jax.Array,
    key: jax.Array,
) -> jax.Array:
    """Sample all rows x_i ~ N(mu_i, Sigma_i) in parallel.

    Sigma_i^{-1} = diag(prec_noise_i) + lambda_bar * Q^{-1}
    mu_i         = Sigma_i * (prec_noise_i * y_i)
    """
    lambda_Q_inv = lambda_bar * Q_inv  # (n, n)

    def sample_one_row(y_i, obs_i, prec_i, key_i):
        prec_obs = jnp.where(obs_i, prec_i, 0.0)
        A_i = jnp.diag(prec_obs) + lambda_Q_inv
        cho = jsla.cho_factor(A_i)
        rhs = prec_obs * jnp.where(jnp.isfinite(y_i), y_i, 0.0)
        mu_i = jsla.cho_solve(cho, rhs)
        # Sample: x_i = mu_i + L^{-T} eps  (L is Cholesky of A_i, A_i = L L^T)
        eps = jax.random.normal(key_i, rhs.shape)
        x_i = mu_i + jsla.solve_triangular(cho[0], eps, trans=True)
        return x_i

    row_keys = jax.random.split(key, Y.shape[0])
    return jax.vmap(sample_one_row)(Y, obs_mask, prec_noise, row_keys)


@jax.jit
def _gsm_gibbs_step(
    X: jax.Array,
    lambda_bar: jax.Array,
    key: jax.Array,
    Y: jax.Array,
    obs_mask: jax.Array,
    prec_noise: jax.Array,
    a0: jax.Array,
    b0: jax.Array,
    sigma_Q: jax.Array,
    lambda_max: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """One complete GSM Gibbs step.

    State: (X, lambda).  Q is re-derived from X at every step.

    Step A: Compute Q = Psi^{1/2} (MAP) from current X, then sample
            x_i | Q, lambda from the Gaussian conditional.

    Step B: From the new X, compute Psi_new = X_new^T X_new.  Initialise
            d = sqrt(psi_new) (MAP) and do one Metropolis step in log space
            to sample approximately from p(d | psi_new, lambda).  When
            sigma_Q = 0, d stays at the MAP (equivalent to CAVI Q update).

    Step C: Sample lambda | X_new, Q_new from the conjugate Gamma.

    Args:
        X:          Current X, shape (m, n).
        lambda_bar: Current lambda, scalar.
        key:        JAX random key.
        Y, obs_mask, prec_noise: Data arrays (fixed throughout the chain).
        a0, b0:     Gamma hyperprior shape/rate for lambda.
        sigma_Q:    Std of log-normal random walk for d proposals.
        lambda_max: Upper cap for sampled lambda (prevents divergence with
                    sparse data where X→0 drives tr(Q)→0).

    Returns:
        X_new, lambda_new, q_accept_rate (mean acceptance rate for d).
    """
    m, n = Y.shape
    key, k1, k2, k3, k4 = jax.random.split(key, 5)

    # --- Step A: compute MAP Q from current X, sample X_new | Q, lambda ---
    Psi = X.T @ X  # (n, n)
    psi_vals, V = jnp.linalg.eigh(Psi)
    psi_vals = jnp.maximum(psi_vals, 1e-10)
    d_map = jnp.sqrt(psi_vals)  # MAP eigenvalues: Q_opt = Psi^{1/2}
    Q_inv = V @ jnp.diag(1.0 / jnp.maximum(d_map, 1e-30)) @ V.T
    X_new = _sample_x_rows(Y, obs_mask, prec_noise, Q_inv, lambda_bar, k1)

    # --- Step B: single-step MH for d starting at MAP of Psi_new ---
    Psi_new = X_new.T @ X_new
    psi_new, V_new = jnp.linalg.eigh(Psi_new)
    psi_new = jnp.maximum(psi_new, 1e-10)
    d_init = jnp.sqrt(psi_new)  # MAP re-initialisation in new eigenbasis

    eps_d = jax.random.normal(k2, (n,))
    log_d_init = jnp.log(jnp.maximum(d_init, 1e-30))
    log_d_prop = log_d_init + sigma_Q * eps_d
    d_prop = jnp.exp(log_d_prop)

    # Target: p(d_k) ∝ exp(-lambda/2 * (d_k + psi_k/d_k))  (GIG(1, lambda, lambda*psi))
    # MH acceptance (random walk in log space, Jacobian = log d' - log d):
    log_alpha_k = (
        -0.5 * lambda_bar * (d_prop + psi_new / d_prop - d_init - psi_new / d_init)
        + (log_d_prop - log_d_init)
    )
    u_k = jax.random.uniform(k3, (n,))
    accept_k = jnp.log(u_k) < log_alpha_k
    d_new = jnp.where(accept_k, d_prop, d_init)

    # --- Step C: sample lambda | X_new, Q_new ~ Gamma(a0+mn, b0+tr(Q)) ---
    # Note: with very sparse data, X_new can shrink toward zero, making tr(Q)≈0
    # and b_N≈b0≈1e-3, driving lambda→large. We cap lambda_new at lambda_max
    # (default ≈ 10√max(m,n)) to prevent this runaway with sparse observations.
    a_N = a0 + jnp.array(m * n, dtype=a0.dtype)
    b_N = b0 + jnp.sum(d_new)
    lambda_new = jnp.minimum(jax.random.gamma(k4, a_N) / b_N, lambda_max)

    q_accept_rate = accept_k.astype(jnp.float32).mean()
    return X_new, lambda_new, q_accept_rate


# ---------------------------------------------------------------------------
# Public: GSM Gibbs
# ---------------------------------------------------------------------------


def mcmc_gsm_gibbs(
    Y: jax.Array,
    S: jax.Array,
    *,
    a0: float = 1e-3,
    b0: float = 1e-3,
    lambda_init: float | None = None,
    lambda_max: float | None = None,
    n_warmup: int = 100,
    n_samples: int = 300,
    sigma_Q: float = 0.3,
    key: jax.Array | None = None,
) -> MCMCResult:
    """GSM Gibbs sampler for the NND posterior.

    Each iteration cycles through:
    (a) Sample X row-wise from the Gaussian conditional given Q (MAP) and lambda.
    (b) Single-step Metropolis update for Q's eigenvalues starting at the MAP
        (GIG(1, lambda, lambda*psi) target; log-normal random walk proposal).
    (c) Conjugate Gamma sample for lambda given X and Q.

    Args:
        Y:            Observations, shape (m, n); ignored where S = inf.
        S:            Standard errors, shape (m, n); inf = missing.
        a0, b0:       Gamma hyperprior shape and rate for lambda.
        lambda_init:  Initial lambda; heuristic if None.
        lambda_max:   Cap for sampled lambda; prevents divergence when X
                      collapses toward zero with sparse data.  Defaults to
                      ``10 * sqrt(max(m, n))``.
        n_warmup:     Burn-in steps (samples discarded).
        n_samples:    Post-warmup samples averaged for posterior mean.
        sigma_Q:      Std of log-normal random walk for Q eigenvalue proposals.
                      Set to 0 to use pure MAP Q update (equivalent to CAVI).
        key:          JAX random key; fixed seed if None.

    Returns:
        MCMCResult with ``mu`` = posterior mean of X, ``lambda_bar`` = mean
        sampled lambda over post-warmup samples, ``accept_rate`` = mean Q
        eigenvalue acceptance rate.
    """
    if key is None:
        key = jax.random.PRNGKey(99)

    obs_mask = jnp.isfinite(S)
    prec_noise = jnp.where(obs_mask, 1.0 / jnp.where(obs_mask, S, 1.0) ** 2, 0.0)
    m, n = Y.shape

    if lambda_init is None:
        max_prec = float(jnp.max(prec_noise))
        lambda_init = float(jnp.sqrt(max(m, n)) / jnp.sqrt(max(max_prec, 1e-10)))

    if lambda_max is None:
        lambda_max = 10.0 * float(jnp.sqrt(max(m, n)))

    X0 = jnp.where(obs_mask, Y, 0.0)
    lambda0 = jnp.array(lambda_init, dtype=Y.dtype)

    a0_j = jnp.array(a0, dtype=Y.dtype)
    b0_j = jnp.array(b0, dtype=Y.dtype)
    sigma_Q_j = jnp.array(sigma_Q, dtype=Y.dtype)
    lambda_max_j = jnp.array(lambda_max, dtype=Y.dtype)

    def warmup_body(carry, key):
        X, lam = carry
        X_new, lam_new, _ = _gsm_gibbs_step(
            X, lam, key, Y, obs_mask, prec_noise, a0_j, b0_j, sigma_Q_j, lambda_max_j
        )
        return (X_new, lam_new), None

    def sample_body(carry, key):
        X, lam = carry
        X_new, lam_new, q_acc = _gsm_gibbs_step(
            X, lam, key, Y, obs_mask, prec_noise, a0_j, b0_j, sigma_Q_j, lambda_max_j
        )
        return (X_new, lam_new), (X_new, lam_new, q_acc)

    key, key_warmup, key_sample = jax.random.split(key, 3)

    (X_warmed, lam_warmed), _ = jax.lax.scan(
        warmup_body,
        (X0, lambda0),
        jax.random.split(key_warmup, n_warmup),
    )

    (_, _), (X_samples, lam_samples, q_acc_samples) = jax.lax.scan(
        sample_body,
        (X_warmed, lam_warmed),
        jax.random.split(key_sample, n_samples),
    )

    mu = X_samples.mean(axis=0)
    lambda_bar = float(lam_samples.mean())
    accept_rate = float(q_acc_samples.mean())

    return MCMCResult(
        mu=mu,
        lambda_bar=lambda_bar,
        accept_rate=accept_rate,
        n_samples=n_samples,
    )
