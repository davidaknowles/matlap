"""
Numpyro SVI for Bayesian matrix denoising with three variational guide families.

All three guides approximate the same generative model as the CAVI algorithm
in ``matlap.core``::

    lambda ~ Gamma(a0, b0)              (or fixed)
    X      ~ MatrixLaplace(lambda)      implemented as
             ImproperUniform + factor(-lambda*||X||_* + mn*log(lambda))
    Y[obs] ~ N(X[obs], S[obs]^2)

Three variational families for q(X) are available::

    'diagonal'      -- fully factorised: q(X_ij) = N(mu_ij, sigma_ij^2)
    'row_mvn'       -- row-factorised:   q(X) = prod_i MVN(mu_i, L_i L_i^T)
    'matrix_normal' -- matrix-normal:    q(X) = MN(M, L_U L_U^T, L_V L_V^T)

When ``lambda_val=None`` a LogNormal guide is placed on lambda; otherwise
lambda is treated as a fixed hyperparameter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
import optax

from .linalg import approx_nuclear_norm


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class VIResult:
    """Result of Numpyro SVI.

    Attributes:
        mu:           Posterior mean E_q[X], shape (m, n).
        lambda_bar:   E_q[lambda].
        elbo_trace:   ELBO at each recorded step (negated SVI loss).
        converged:    True if relative ELBO change < tol before n_steps.
        n_iter:       Total SVI steps executed.
    """

    mu: jax.Array
    lambda_bar: float
    elbo_trace: list[float] = field(default_factory=list)
    converged: bool = False
    n_iter: int = 0


# ---------------------------------------------------------------------------
# Model  (X has 2-D event shape (m, n) throughout)
# ---------------------------------------------------------------------------


def _model(Y_flat, obs_idx, S2_flat, m, n, lambda_val, a0, b0, approx_rank=0):
    """Numpyro generative model for matrix denoising.

    X is sampled as a (m, n) event so all three guides can match it directly.

    Args:
        approx_rank: If > 0, use randomized SVD with this rank to approximate
            the nuclear norm.  Speeds up large-scale runs; set to 0 for exact.
    """
    mn = m * n

    if lambda_val is None:
        lam = numpyro.sample("lambda", dist.Gamma(a0, b0))
    else:
        lam = jnp.asarray(lambda_val, dtype=jnp.float32)

    # Flat prior; each guide will constrain X via its own parameterisation
    X = numpyro.sample(
        "X",
        dist.ImproperUniform(dist.constraints.real, (), (m, n)),
    )

    # Matrix Laplace prior: log p(X|lambda) = -lambda*||X||_* + mn*log(lambda)
    if approx_rank > 0:
        nuclear = approx_nuclear_norm(X, approx_rank)
    else:
        sv = jnp.linalg.svd(X, compute_uv=False)
        nuclear = sv.sum()
    numpyro.factor("nuclear_prior", -lam * nuclear + mn * jnp.log(lam))

    # Likelihood on observed entries only
    X_flat = X.reshape(-1)
    numpyro.sample(
        "Y",
        dist.Normal(X_flat[obs_idx], jnp.sqrt(S2_flat[obs_idx])),
        obs=Y_flat[obs_idx],
    )


# ---------------------------------------------------------------------------
# Guide factories
# ---------------------------------------------------------------------------


def _lambda_params(lambda_val):
    """Sample (or observe) lambda in the guide."""
    if lambda_val is None:
        loc = numpyro.param("log_lam_loc", jnp.array(0.0))
        scale = numpyro.param(
            "log_lam_scale",
            jnp.array(0.5),
            constraint=dist.constraints.positive,
        )
        numpyro.sample("lambda", dist.LogNormal(loc, scale))


def _make_diagonal_guide():
    """Fully factorised q(X_ij) = N(mu_ij, exp(log_sigma_ij)^2)."""

    def guide(Y_flat, obs_idx, S2_flat, m, n, lambda_val, a0, b0, approx_rank=0):
        mu = numpyro.param("mu", jnp.zeros((m, n)))
        log_sigma = numpyro.param("log_sigma", jnp.full((m, n), -1.0))
        numpyro.sample(
            "X",
            dist.Normal(mu, jnp.exp(log_sigma)).to_event(2),
        )
        _lambda_params(lambda_val)

    return guide


def _make_row_mvn_guide():
    """Row-factorised q(X) = prod_i MVN(mu_i, L_i L_i^T).

    Batch of MVNs (one per row), reinterpreted so the row batch axis becomes
    an event axis to match the (m, n) event shape of X in the model.
    """

    def guide(Y_flat, obs_idx, S2_flat, m, n, lambda_val, a0, b0, approx_rank=0):
        mu = numpyro.param("mu", jnp.zeros((m, n)))
        L = numpyro.param(
            "L",
            0.1 * jnp.broadcast_to(jnp.eye(n), (m, n, n)),
            constraint=dist.constraints.lower_cholesky,
        )
        # MultivariateNormal: batch=(m,), event=(n,)
        # .to_event(1): batch=(), event=(m, n) — product of independent MVNs
        numpyro.sample(
            "X",
            dist.MultivariateNormal(mu, scale_tril=L).to_event(1),
        )
        _lambda_params(lambda_val)

    return guide


def _make_matrix_normal_guide():
    """Matrix Normal guide: q(X) = MN(M, L_U L_U^T, L_V L_V^T).

    Captures both row and column correlations.
    O(m^2 + n^2) covariance parameters.
    """

    def guide(Y_flat, obs_idx, S2_flat, m, n, lambda_val, a0, b0, approx_rank=0):
        M = numpyro.param("M", jnp.zeros((m, n)))
        L_U = numpyro.param(
            "L_U",
            jnp.eye(m),
            constraint=dist.constraints.lower_cholesky,
        )
        L_V = numpyro.param(
            "L_V",
            jnp.eye(n),
            constraint=dist.constraints.lower_cholesky,
        )
        # MatrixNormal: batch=(), event=(m, n) -- matches X in model
        numpyro.sample("X", dist.MatrixNormal(M, L_U, L_V))
        _lambda_params(lambda_val)

    return guide


def _make_row_lowrank_guide(rank: int):
    """Per-row low-rank Gaussian guide.

    q(X) = prod_i LowRankMVN(mu_i, F_i F_i^T + diag(d_i))
    where F_i has shape (n, rank).  Memory O(mnr) ≈ 600 MB at r=15 for 10k×1k.
    """

    def guide(Y_flat, obs_idx, S2_flat, m, n, lambda_val, a0, b0, approx_rank=0):
        mu = numpyro.param("mu", jnp.zeros((m, n)))
        cov_factor = numpyro.param(
            "cov_factor",
            jnp.zeros((m, n, rank)),
        )
        cov_diag = numpyro.param(
            "cov_diag",
            jnp.ones((m, n)),
            constraint=dist.constraints.positive,
        )
        numpyro.sample(
            "X",
            dist.LowRankMultivariateNormal(mu, cov_factor, cov_diag).to_event(1),
        )
        _lambda_params(lambda_val)

    return guide


def _make_matrix_factor_guide(rank: int):
    """Shared-column-factor Gaussian guide.

    q(X) = prod_i LowRankMVN(mu_i, F F^T + diag(d_i))
    where F (shape n×rank) is shared across all rows.
    Memory O(mn + nk) ≈ O(mn) — same as diagonal but captures column space.
    """

    def guide(Y_flat, obs_idx, S2_flat, m, n, lambda_val, a0, b0, approx_rank=0):
        mu = numpyro.param("mu", jnp.zeros((m, n)))
        cov_factor = numpyro.param(
            "cov_factor",
            jnp.zeros((n, rank)),
        )
        cov_diag = numpyro.param(
            "cov_diag",
            jnp.ones((m, n)),
            constraint=dist.constraints.positive,
        )
        # cov_factor[None] broadcasts (1, n, rank) → (m, n, rank)
        numpyro.sample(
            "X",
            dist.LowRankMultivariateNormal(mu, cov_factor[None], cov_diag).to_event(1),
        )
        _lambda_params(lambda_val)

    return guide


def _make_guide(guide_type: str, guide_rank: int):
    """Create a guide function by type and (optional) rank."""
    if guide_type == "diagonal":
        return _make_diagonal_guide()
    elif guide_type == "row_mvn":
        return _make_row_mvn_guide()
    elif guide_type == "matrix_normal":
        return _make_matrix_normal_guide()
    elif guide_type == "row_lowrank":
        return _make_row_lowrank_guide(guide_rank)
    elif guide_type == "matrix_factor":
        return _make_matrix_factor_guide(guide_rank)
    else:
        valid = ["diagonal", "row_mvn", "matrix_normal", "row_lowrank", "matrix_factor"]
        raise ValueError(f"guide_type must be one of {valid}; got {guide_type!r}")


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def fit_vi(
    Y: jax.Array,
    S: jax.Array,
    lambda_val: float | None = None,
    guide_type: str = "diagonal",
    *,
    a0: float = 1e-3,
    b0: float = 1e-3,
    n_steps: int = 5000,
    lr: float = 1e-3,
    tol: float = 1e-5,
    record_every: int = 50,
    verbose: bool = False,
    guide_rank: int = 15,
    approx_rank: int = 0,
) -> VIResult:
    """Fit Bayesian matrix denoising via Numpyro SVI.

    Uses the same Matrix Laplace model as the CAVI implementation in
    ``matlap.core``, but optimises the ELBO with Adam rather than
    coordinate ascent.

    Variational families for q(X):

    * ``'diagonal'``:       q(X_ij) = N(mu_ij, sigma_ij^2)                O(mn) params
    * ``'row_mvn'``:        q(X) = prod_i MVN(mu_i, L_i L_i^T)            O(mn²) params
    * ``'matrix_normal'``:  q(X) = MN(M, L_U L_U^T, L_V L_V^T)           O(m²+n²) params
    * ``'row_lowrank'``:    q(X) = prod_i LRMVN(mu_i, F_i F_i^T+diag(d)) O(mnr) params
    * ``'matrix_factor'``:  q(X) = prod_i LRMVN(mu_i, FF^T+diag(d_i))    O(mn+nr) params

    Args:
        Y:            Observed matrix, shape (m, n).
        S:            Known standard errors; ``jnp.inf`` where missing.
        lambda_val:   Fix lambda to this value.  ``None`` places a
                      Gamma(a0, b0) hyperprior and estimates lambda.
        guide_type:   One of ``'diagonal'``, ``'row_mvn'``, ``'matrix_normal'``,
                      ``'row_lowrank'``, ``'matrix_factor'``.
        a0, b0:       Gamma hyperprior parameters for lambda.
        n_steps:      Number of Adam SVI steps.
        lr:           Adam learning rate.
        tol:          Relative ELBO convergence tolerance.
        record_every: Record ELBO every this many steps.
        verbose:      Print ELBO periodically.
        guide_rank:   Rank for ``'row_lowrank'`` and ``'matrix_factor'`` guides.
        approx_rank:  If > 0, use randomized SVD with this rank to approximate
                      the nuclear norm in the model (speeds up large-scale runs).

    Returns:
        VIResult with posterior mean and diagnostics.
    """
    Y = jnp.asarray(Y, dtype=jnp.float32)
    S = jnp.asarray(S, dtype=jnp.float32)
    S2 = S ** 2
    m, n = Y.shape

    obs_mask = jnp.isfinite(S2) & jnp.isfinite(Y)
    obs_idx = jnp.flatnonzero(obs_mask.reshape(-1))
    Y_flat = Y.reshape(-1)
    S2_flat = S2.reshape(-1)

    guide = _make_guide(guide_type, guide_rank)
    model_args = (Y_flat, obs_idx, S2_flat, m, n, lambda_val, a0, b0, approx_rank)

    optimizer = numpyro.optim.optax_to_numpyro(optax.adam(lr))
    svi = SVI(_model, guide, optimizer, loss=Trace_ELBO())
    svi_state = svi.init(jax.random.PRNGKey(0), *model_args)

    elbo_trace: list[float] = []
    converged = False
    step = 0

    for step in range(1, n_steps + 1):
        svi_state, loss = svi.update(svi_state, *model_args)

        if step % record_every == 0:
            elbo_val = -float(loss)
            elbo_trace.append(elbo_val)

            if verbose:
                print(f"  step {step:5d}  ELBO={elbo_val:.4f}")

            if len(elbo_trace) >= 2:
                prev = elbo_trace[-2]
                if abs(elbo_val - prev) / max(abs(prev), 1e-10) < tol:
                    converged = True
                    break

    params = svi.get_params(svi_state)

    if guide_type == "matrix_normal":
        mu = params["M"]
    else:
        mu = params["mu"]

    if lambda_val is None:
        loc = float(params.get("log_lam_loc", jnp.array(0.0)))
        scale = float(params.get("log_lam_scale", jnp.array(0.5)))
        lambda_bar = float(jnp.exp(loc + 0.5 * scale ** 2))
    else:
        lambda_bar = float(lambda_val)

    return VIResult(
        mu=mu,
        lambda_bar=lambda_bar,
        elbo_trace=elbo_trace,
        converged=converged,
        n_iter=step,
    )
