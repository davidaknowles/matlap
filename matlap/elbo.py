"""
ELBO computation for matlap CAVI.

The Evidence Lower Bound is used both for convergence monitoring (must be
non-decreasing) and for comparing models with different fixed lambda values
in the grid-search mode.
"""

import jax
import jax.numpy as jnp
import jax.scipy.special as jss


@jax.jit
def compute_elbo(
    Y: jax.Array,
    S2: jax.Array,
    mus: jax.Array,
    sigmas: jax.Array,
    log_det_sigmas: jax.Array,
    q_sqrt_vals: jax.Array,
    lambda_bar: jax.Array,
    a_N: jax.Array,
    b_N: jax.Array,
    a0: float,
    b0: float,
) -> jax.Array:
    """Compute the ELBO for the CAVI model.

    ::

        ELBO = E_q[log p(Y|X)]           (log-likelihood)
             + E_q[log p(X|lambda)]       (prior; variational lower bound at optimal Q)
             + E_q[log p(lambda)]         (hyperprior)
             - E_q[log q(X)]              (entropy of q(X))
             - E_q[log q(lambda)]         (entropy of q(lambda))

    Args:
        Y:               observations, shape (m, n); NaN where missing
        S2:              observation variances, shape (m, n); inf where missing
        mus:             posterior means, shape (m, n)
        sigmas:          posterior covariances, shape (m, n, n)
        log_det_sigmas:  log abs(Sigma_i) for each row, shape (m,)
        q_sqrt_vals:     sqrt eigenvalues of Psi (= eigenvalues of Q), shape (n,)
        lambda_bar:      E_q[lambda] = a_N / b_N, scalar
        a_N:             Gamma posterior shape, scalar
        b_N:             Gamma posterior rate, scalar
        a0:              Gamma prior shape, scalar
        b0:              Gamma prior rate, scalar

    Returns:
        Scalar ELBO value.
    """
    m, n = Y.shape

    # ------------------------------------------------------------------ #
    # 1. Log-likelihood: E_q[log p(Y|X)]                                  #
    #    = -1/2 * sum_{i,j:obs} [(y_ij - mu_ij)^2 / s_ij^2               #
    #                             + Sigma_i[j,j] / s_ij^2                 #
    #                             + log(2*pi*s_ij^2)]                     #
    # ------------------------------------------------------------------ #
    obs_mask = jnp.isfinite(S2) & jnp.isfinite(Y)  # (m, n)
    prec = jnp.where(obs_mask, 1.0 / S2, 0.0)

    # Zero out residuals at missing entries to avoid 0 * NaN = NaN
    resid2 = jnp.where(obs_mask, (Y - mus) ** 2, 0.0)
    sigma_diag = jnp.diagonal(sigmas, axis1=1, axis2=2)  # (m, n)

    ll = -0.5 * jnp.sum(prec * (resid2 + sigma_diag))
    ll -= 0.5 * jnp.sum(jnp.where(obs_mask, jnp.log(2.0 * jnp.pi * S2), 0.0))

    # ------------------------------------------------------------------ #
    # 2. Prior on X (variational lower bound at optimal Q):               #
    #    E_q[log p(X|lambda)] >= -lambda_bar * Tr(Q) + mn * E_q[log lam] #
    #    Tr(Q) = sum(q_sqrt_vals)                                         #
    #    E_q[log lambda] = psi(a_N) - log(b_N)                           #
    # ------------------------------------------------------------------ #
    trace_Q = jnp.sum(q_sqrt_vals)
    e_log_lam = jss.digamma(a_N) - jnp.log(b_N)
    prior_X = -lambda_bar * trace_Q + m * n * e_log_lam

    # ------------------------------------------------------------------ #
    # 3. Entropy of q(X) = sum_i H[N(mu_i, Sigma_i)]                     #
    #    = 1/2 * sum_i [log abs(Sigma_i) + n*(1 + log(2*pi))]                #
    # ------------------------------------------------------------------ #
    entropy_X = 0.5 * (jnp.sum(log_det_sigmas) + m * n * (1.0 + jnp.log(2.0 * jnp.pi)))

    # ------------------------------------------------------------------ #
    # 4. Lambda terms: E_q[log p(lambda)] - E_q[log q(lambda)]           #
    #    = -KL(q(lambda) || p(lambda))                                   #
    #    KL(Gamma(a_N, b_N) || Gamma(a0, b0))                           #
    #      = a_N*log(b_N) - a0*log(b0) - lgamma(a_N) + lgamma(a0)      #
    #        + (a_N - a0)*(psi(a_N) - log(b_N))                         #
    #        + (b0 - b_N)*(a_N/b_N)                                     #
    # ------------------------------------------------------------------ #
    kl_lam = (
        a_N * jnp.log(b_N) - a0 * jnp.log(b0)
        - jss.gammaln(a_N) + jss.gammaln(a0)
        + (a_N - a0) * e_log_lam
        + (b0 - b_N) * lambda_bar
    )
    neg_kl_lam = -kl_lam

    return ll + prior_X + entropy_X + neg_kl_lam


@jax.jit
def compute_elbo_from_diag(
    Y: jax.Array,
    S2: jax.Array,
    mus: jax.Array,
    sigma_diag: jax.Array,
    log_det_sigmas: jax.Array,
    q_sqrt_vals: jax.Array,
    lambda_bar: jax.Array,
    a_N: jax.Array,
    b_N: jax.Array,
    a0: float,
    b0: float,
) -> jax.Array:
    """ELBO for batched full CAVI — identical to compute_elbo but accepts the
    diagonal of each Sigma_i rather than the full (m, n, n) covariance tensor.

    This allows matlap_batched to avoid storing O(m n²) covariances: only the
    diagonal (m, n) is retained after each batch.  The prior and entropy terms
    still use m*n (the full model dimension), unlike compute_elbo_lowrank.

    Args:
        Y:               observations, shape (m, n); NaN where missing
        S2:              observation variances, shape (m, n); inf where missing
        mus:             posterior means, shape (m, n)
        sigma_diag:      diagonal of per-row posterior covariances, shape (m, n)
        log_det_sigmas:  log abs(Sigma_i) for each row, shape (m,)
        q_sqrt_vals:     sqrt eigenvalues of Psi, shape (n,)
        lambda_bar:      E_q[lambda] = a_N / b_N, scalar
        a_N:             Gamma posterior shape, scalar
        b_N:             Gamma posterior rate, scalar
        a0:              Gamma prior shape
        b0:              Gamma prior rate

    Returns:
        Scalar ELBO value.
    """
    m, n = Y.shape
    obs_mask = jnp.isfinite(S2) & jnp.isfinite(Y)
    prec = jnp.where(obs_mask, 1.0 / S2, 0.0)

    resid2 = jnp.where(obs_mask, (Y - mus) ** 2, 0.0)
    ll = -0.5 * jnp.sum(prec * (resid2 + sigma_diag))
    ll -= 0.5 * jnp.sum(jnp.where(obs_mask, jnp.log(2.0 * jnp.pi * S2), 0.0))

    trace_Q = jnp.sum(q_sqrt_vals)
    e_log_lam = jss.digamma(a_N) - jnp.log(b_N)
    prior_X = -lambda_bar * trace_Q + m * n * e_log_lam

    entropy_X = 0.5 * (jnp.sum(log_det_sigmas) + m * n * (1.0 + jnp.log(2.0 * jnp.pi)))

    kl_lam = (
        a_N * jnp.log(b_N) - a0 * jnp.log(b0)
        - jss.gammaln(a_N) + jss.gammaln(a0)
        + (a_N - a0) * e_log_lam
        + (b0 - b_N) * lambda_bar
    )
    return ll + prior_X + entropy_X + (-kl_lam)


@jax.jit
def compute_elbo_lowrank(
    Y: jax.Array,
    S2: jax.Array,
    mus: jax.Array,
    sigma_diag: jax.Array,
    log_det_sigmas: jax.Array,
    q_sqrt_vals: jax.Array,
    lambda_bar: jax.Array,
    a_N: jax.Array,
    b_N: jax.Array,
    a0: float,
    b0: float,
) -> jax.Array:
    """Compute the ELBO for the low-rank CAVI model.

    The factor model lives in R^{m×r} space (X = Z V_r^T), so entropy and
    lambda terms use r = ``q_sqrt_vals.shape[0]``, not n.  This ELBO monitors
    convergence within the low-rank family and is NOT directly comparable to
    the full CAVI ELBO.

    Args:
        Y:               observations, shape (m, n); NaN where missing.
        S2:              observation variances, shape (m, n); inf where missing.
        mus:             posterior means in original space, shape (m, n).
        sigma_diag:      diagonal of per-row posterior covariances, shape (m, n).
        log_det_sigmas:  log|A_r^{-1}_i| for each row (r-dim), shape (m,).
        q_sqrt_vals:     sqrt eigenvalues of Ψ_r (r-dim), shape (r,).
        lambda_bar:      E_q[lambda] = a_N / b_N, scalar.
        a_N:             Gamma posterior shape, scalar.
        b_N:             Gamma posterior rate, scalar.
        a0:              Gamma prior shape, scalar.
        b0:              Gamma prior rate, scalar.

    Returns:
        Scalar ELBO value.
    """
    m, n = Y.shape
    r = q_sqrt_vals.shape[0]

    # 1. Log-likelihood (uses sigma_diag directly; same formula as full CAVI)
    obs_mask = jnp.isfinite(S2) & jnp.isfinite(Y)
    prec = jnp.where(obs_mask, 1.0 / S2, 0.0)
    resid2 = jnp.where(obs_mask, (Y - mus) ** 2, 0.0)
    ll = -0.5 * jnp.sum(prec * (resid2 + sigma_diag))
    ll -= 0.5 * jnp.sum(jnp.where(obs_mask, jnp.log(2.0 * jnp.pi * S2), 0.0))

    # 2. Prior on X (in factor space — uses r, not n)
    trace_Q = jnp.sum(q_sqrt_vals)
    e_log_lam = jss.digamma(a_N) - jnp.log(b_N)
    prior_X = -lambda_bar * trace_Q + m * r * e_log_lam

    # 3. Entropy of q(Z) = sum_i H[N(z_i, A_r^{-1}_i)]  (r-dim Gaussians)
    entropy_X = 0.5 * (jnp.sum(log_det_sigmas) + m * r * (1.0 + jnp.log(2.0 * jnp.pi)))

    # 4. Lambda KL (identical to full CAVI)
    kl_lam = (
        a_N * jnp.log(b_N) - a0 * jnp.log(b0)
        - jss.gammaln(a_N) + jss.gammaln(a0)
        + (a_N - a0) * e_log_lam
        + (b0 - b_N) * lambda_bar
    )
    neg_kl_lam = -kl_lam

    return ll + prior_X + entropy_X + neg_kl_lam


@jax.jit
def compute_elbo_lowrank_iso(
    Y: jax.Array,
    S2: jax.Array,
    mus: jax.Array,
    sigma_diag: jax.Array,
    log_det_sigmas: jax.Array,
    d_r: jax.Array,
    Psi_perp: jax.Array,
    lambda_bar: jax.Array,
    a_N: jax.Array,
    b_N: jax.Array,
    a0: float,
    b0: float,
) -> jax.Array:
    """ELBO for the low-rank-plus-isotropic CAVI model.

    Uses the nuclear-norm variational bound for **both** in-subspace and
    off-subspace dimensions, evaluated at the optimal Q:

    .. math::

        Q = V_r\\,\\mathrm{diag}(d_r)\\,V_r^\\top + \\delta^*(I - V_r V_r^\\top),
        \\quad \\delta^* = \\sqrt{\\mathrm{Tr}(\\Psi_\\perp)/(n-r)}

    which gives :math:`\\mathrm{Tr}(Q) = \\sum_k d_{r,k} + (n-r)\\delta^*` and
    the tightened nuclear-norm bound:

    .. math::

        -\\bar\\lambda\\,\\mathrm{Tr}(Q) + mn\\,\\mathbb{E}[\\log\\lambda]

    This is consistent with the λ update :math:`b_N = b_0 + \\mathrm{Tr}(Q)`
    and :math:`a_N = a_0 + mn`, ensuring ELBO monotonicity for the Q and λ
    sub-updates.  The row update (which uses :math:`\\gamma=\\bar\\lambda`
    rather than the exact :math:`\\gamma=\\bar\\lambda/\\delta`) is an
    approximation that does not affect the ELBO formula but may cause small
    non-monotonicities in the log-likelihood term in practice.

    Args:
        Y:               Observations, shape (m, n); NaN where missing.
        S2:              Observation variances, shape (m, n); inf where missing.
        mus:             Posterior means (full n-dim), shape (m, n).
        sigma_diag:      Diagonal of per-row posterior covariances, shape (m, n).
        log_det_sigmas:  Full n-dim log|Σ_i| for each row, shape (m,).
        d_r:             Sqrt-eigenvalues of Ψ_r (in-subspace Q diagonal), shape (r,).
        Psi_perp:        Off-subspace second moment Tr(Ψ_⊥) = (n−r)·δ*², scalar.
        lambda_bar:      E_q[λ] = a_N / b_N, scalar.
        a_N:             Gamma posterior shape = a_0 + m·n, scalar.
        b_N:             Gamma posterior rate = b_0 + Tr(Q), scalar.
        a0:              Gamma prior shape.
        b0:              Gamma prior rate.

    Returns:
        Scalar ELBO value.
    """
    m, n = Y.shape
    r = d_r.shape[0]

    obs_mask = jnp.isfinite(S2) & jnp.isfinite(Y)
    prec = jnp.where(obs_mask, 1.0 / S2, 0.0)
    resid2 = jnp.where(obs_mask, (Y - mus) ** 2, 0.0)
    ll = -0.5 * jnp.sum(prec * (resid2 + sigma_diag))
    ll -= 0.5 * jnp.sum(jnp.where(obs_mask, jnp.log(2.0 * jnp.pi * S2), 0.0))

    e_log_lam = jss.digamma(a_N) - jnp.log(b_N)
    trace_d_r = jnp.sum(d_r)

    # Nuclear-norm bound at optimal Q: −λ̄·Tr(Q) + m·n·E[log λ]
    # Tr(Q) = Σ_k d_{r,k}  +  (n−r)·δ* = trace_d_r + sqrt((n−r)·Tr(Ψ_⊥))
    # Psi_perp = (n−r)·δ*², so (n−r)·δ* = sqrt((n−r)·Psi_perp)
    trace_Q_perp = jnp.sqrt((n - r) * Psi_perp)   # = (n−r)·δ*
    prior_X = (
        -lambda_bar * (trace_d_r + trace_Q_perp)   # −λ̄·Tr(Q)
        + m * n * e_log_lam                         # m·n·E[log λ]
    )

    entropy_X = 0.5 * (jnp.sum(log_det_sigmas) + m * n * (1.0 + jnp.log(2.0 * jnp.pi)))

    kl_lam = (
        a_N * jnp.log(b_N) - a0 * jnp.log(b0)
        - jss.gammaln(a_N) + jss.gammaln(a0)
        + (a_N - a0) * e_log_lam
        + (b0 - b_N) * lambda_bar
    )
    return ll + prior_X + entropy_X + (-kl_lam)

