"""
Coordinate Ascent Variational Inference (CAVI) for Bayesian Matrix Denoising.

Main public functions:
    matlap()      — full CAVI with automatic lambda estimation (empirical Bayes)
    matlap_grid() — CAVI with fixed lambda on a user-supplied grid
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from .elbo import compute_elbo, compute_elbo_lowrank
from .linalg import SqrtDecomp, matrix_sqrt_eigh, trace_sqrt, update_rows, update_rows_lowrank, rsvd


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class CAVIResult:
    """Result of matlap() CAVI optimisation.

    Attributes:
        mu:          Posterior mean of X, shape (m, n).
        sigma:       Posterior covariances, one per row, shape (m, n, n).
        lambda_bar:  E_q[lambda], scalar.
        a_N:         Gamma posterior shape, scalar.
        b_N:         Gamma posterior rate, scalar.
        elbo_trace:  ELBO at the end of each iteration, list of floats.
        converged:   True if |ΔELBO|/|ELBO| < tol before max_iter.
        n_iter:      Number of iterations executed.
    """

    mu: jax.Array
    sigma: jax.Array
    lambda_bar: float
    a_N: float
    b_N: float
    elbo_trace: list[float] = field(default_factory=list)
    converged: bool = False
    n_iter: int = 0


@dataclass
class GridResult:
    """Result of matlap_grid().

    Attributes:
        best_lambda:  Lambda with the highest ELBO.
        best_result:  CAVIResult for the best lambda.
        results:      List of (lambda, CAVIResult) for every grid point,
                      sorted by lambda ascending.
    """

    best_lambda: float
    best_result: CAVIResult
    results: list[tuple[float, CAVIResult]]


# ---------------------------------------------------------------------------
# Internal helpers (JIT-compiled)
# ---------------------------------------------------------------------------


@jax.jit
def _compute_psi(mus: jax.Array, sigmas: jax.Array) -> jax.Array:
    """Psi = sum_i (mu_i mu_i^T + Sigma_i), shape (n, n)."""
    return jnp.einsum("im,in->mn", mus, mus) + sigmas.sum(axis=0)


def _one_cavi_step(
    Y: jax.Array,
    S2: jax.Array,
    mus: jax.Array,
    sigmas: jax.Array,
    decomp: SqrtDecomp,
    a_N: jax.Array,
    b_N: jax.Array,
    a0: float,
    b0: float,
    update_lambda: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, SqrtDecomp, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Execute one CAVI iteration.

    Takes the current decomp (already computed from the current mus/sigmas), so
    Q is always in sync with Psi(mus, sigmas) on entry. After updating rows,
    Q and lambda are refreshed from the new Psi before computing the ELBO,
    guaranteeing that the ELBO trace is non-decreasing.

    Returns:
        mus, sigmas, log_dets, decomp_new, lambda_bar, a_N, b_N, elbo
    """
    m, n = Y.shape

    # decomp is already sqrt(Psi(mus, sigmas)) — update lambda from it
    if update_lambda:
        trace_Q = trace_sqrt(decomp)
        a_N = jnp.asarray(a0 + m * n, dtype=jnp.float32)
        b_N = jnp.asarray(b0, dtype=jnp.float32) + trace_Q
    lambda_bar = a_N / b_N

    # Update posterior q(X) row by row (vectorised)
    mus, sigmas, log_dets = update_rows(Y, S2, decomp.sqrt_vals, decomp.vecs, lambda_bar)

    # Refresh Q from the NEW Psi so ELBO is computed at a fully consistent state
    Psi_new = _compute_psi(mus, sigmas)
    decomp_new = matrix_sqrt_eigh(Psi_new)

    if update_lambda:
        b_N = jnp.asarray(b0, dtype=jnp.float32) + trace_sqrt(decomp_new)
        lambda_bar = a_N / b_N

    elbo = compute_elbo(
        Y, S2, mus, sigmas, log_dets,
        decomp_new.sqrt_vals, lambda_bar, a_N, b_N, a0, b0,
    )

    return mus, sigmas, log_dets, decomp_new, lambda_bar, a_N, b_N, elbo


# ---------------------------------------------------------------------------
# Main public functions
# ---------------------------------------------------------------------------


def matlap(
    Y: jax.Array,
    S: jax.Array,
    *,
    a0: float = 1e-3,
    b0: float = 1e-3,
    max_iter: int = 200,
    tol: float = 1e-6,
    verbose: bool = False,
) -> CAVIResult:
    """Bayesian matrix denoising via CAVI with automatic lambda estimation.

    Fits the model:
        Y_ij ~ N(X_ij, s_ij^2)        (elementwise Gaussian likelihood)
        p(X | lambda) ∝ exp(-lambda * ||X||_*)   (Matrix Laplace prior)
        lambda ~ Gamma(a0, b0)         (hyperprior; estimated from data)

    Missing observations are encoded as ``S[i, j] = jnp.inf`` (and the
    corresponding ``Y[i, j]`` value is ignored).

    Args:
        Y:         Observed matrix, shape (m, n). NaN/any value where missing.
        S:         Known standard errors, shape (m, n). ``jnp.inf`` where missing.
        a0:        Gamma prior shape for lambda (default: 1e-3, weakly informative).
        b0:        Gamma prior rate for lambda (default: 1e-3, weakly informative).
        max_iter:  Maximum CAVI iterations.
        tol:       Convergence tolerance on relative ELBO change.
        verbose:   Print ELBO at each iteration.

    Returns:
        CAVIResult with posterior mean, covariances, lambda, and diagnostics.
    """
    Y = jnp.asarray(Y, dtype=jnp.float32)
    S = jnp.asarray(S, dtype=jnp.float32)
    S2 = S ** 2
    m, n = Y.shape

    # Initialise: mu_i = y_i (or 0 where missing), Sigma_i = I
    obs_mask = jnp.isfinite(S) & jnp.isfinite(Y)
    mus = jnp.where(obs_mask, Y, 0.0)
    sigmas = jnp.broadcast_to(jnp.eye(n), (m, n, n)).copy()

    # Compute initial Q from the initial Psi (required by restructured _one_cavi_step)
    Psi_init = _compute_psi(mus, sigmas)
    decomp = matrix_sqrt_eigh(Psi_init)

    # Initialise lambda from the initial Q
    a_N = jnp.asarray(a0 + m * n, dtype=jnp.float32)
    b_N = jnp.asarray(b0, dtype=jnp.float32) + trace_sqrt(decomp)
    lambda_bar = a_N / b_N

    elbo_trace: list[float] = []
    converged = False

    for i in range(max_iter):
        mus, sigmas, log_dets, decomp, lambda_bar, a_N, b_N, elbo = _one_cavi_step(
            Y, S2, mus, sigmas, decomp, a_N, b_N, a0, b0, update_lambda=True,
        )
        elbo_val = float(elbo)
        elbo_trace.append(elbo_val)

        if verbose:
            print(f"  iter {i+1:4d}  ELBO={elbo_val:.4f}  lambda={float(lambda_bar):.4f}")

        if i > 0:
            prev = elbo_trace[-2]
            denom = max(abs(prev), 1e-10)
            if abs(elbo_val - prev) / denom < tol:
                converged = True
                break

    return CAVIResult(
        mu=mus,
        sigma=sigmas,
        lambda_bar=float(lambda_bar),
        a_N=float(a_N),
        b_N=float(b_N),
        elbo_trace=elbo_trace,
        converged=converged,
        n_iter=len(elbo_trace),
    )


def matlap_grid(
    Y: jax.Array,
    S: jax.Array,
    lambda_grid: jax.Array,
    *,
    a0: float = 1e-3,
    b0: float = 1e-3,
    max_iter: int = 200,
    tol: float = 1e-6,
    verbose: bool = False,
) -> GridResult:
    """Bayesian matrix denoising with lambda selected over a grid.

    For each value of ``lambda`` in ``lambda_grid``, runs CAVI with lambda
    held fixed (only the X posterior is updated).  Returns the solution with
    the highest ELBO, together with all individual results for inspection.

    Args:
        Y:            Observed matrix, shape (m, n).
        S:            Known standard errors, shape (m, n). ``jnp.inf`` where missing.
        lambda_grid:  1-D array of lambda values to evaluate.
        a0, b0:       Gamma prior parameters (used only in ELBO computation).
        max_iter:     Maximum CAVI iterations per grid point.
        tol:          Convergence tolerance.
        verbose:      Print progress.

    Returns:
        GridResult with best lambda and all per-lambda results.
    """
    Y = jnp.asarray(Y, dtype=jnp.float32)
    S = jnp.asarray(S, dtype=jnp.float32)
    S2 = S ** 2
    m, n = Y.shape

    obs_mask = jnp.isfinite(S) & jnp.isfinite(Y)

    results: list[tuple[float, CAVIResult]] = []
    best_elbo = -jnp.inf
    best_lam = float(lambda_grid[0])
    best_res: CAVIResult | None = None

    for lam_val in lambda_grid:
        lam = jnp.asarray(lam_val, dtype=jnp.float32)
        # a_N, b_N are fixed (lambda not updated); compute implied b_N from lam
        a_N = jnp.asarray(a0 + m * n, dtype=jnp.float32)
        b_N = jnp.asarray(a_N / lam, dtype=jnp.float32)

        # Initialise
        mus = jnp.where(obs_mask, Y, 0.0)
        sigmas = jnp.broadcast_to(jnp.eye(n), (m, n, n)).copy()
        lambda_bar = lam

        # Pre-compute initial Q
        Psi_init = _compute_psi(mus, sigmas)
        decomp = matrix_sqrt_eigh(Psi_init)

        elbo_trace: list[float] = []
        converged = False

        for i in range(max_iter):
            mus, sigmas, log_dets, decomp, lambda_bar, a_N, b_N, elbo = _one_cavi_step(
                Y, S2, mus, sigmas, decomp, a_N, b_N, a0, b0, update_lambda=False,
            )
            elbo_val = float(elbo)
            elbo_trace.append(elbo_val)

            if i > 0:
                prev = elbo_trace[-2]
                denom = max(abs(prev), 1e-10)
                if abs(elbo_val - prev) / denom < tol:
                    converged = True
                    break

        res = CAVIResult(
            mu=mus,
            sigma=sigmas,
            lambda_bar=float(lambda_bar),
            a_N=float(a_N),
            b_N=float(b_N),
            elbo_trace=elbo_trace,
            converged=converged,
            n_iter=len(elbo_trace),
        )
        results.append((float(lam_val), res))

        if verbose:
            print(f"  lambda={float(lam_val):.4f}  ELBO={elbo_trace[-1]:.4f}  converged={converged}")

        if elbo_trace[-1] > best_elbo:
            best_elbo = elbo_trace[-1]
            best_lam = float(lam_val)
            best_res = res

    results.sort(key=lambda x: x[0])

    return GridResult(
        best_lambda=best_lam,
        best_result=best_res,
        results=results,
    )


# ---------------------------------------------------------------------------
# Low-rank result type and main function
# ---------------------------------------------------------------------------


@dataclass
class LowRankCAVIResult:
    """Result of matlap_lowrank() low-rank CAVI optimisation.

    Attributes:
        mu:          Posterior mean of X = Z V_r^T, shape (m, n).
        z:           Factor-space posterior means, shape (m, r).
        V_r:         Factor loading matrix, shape (n, r); orthonormal columns.
        lambda_bar:  E_q[lambda], scalar.
        a_N:         Gamma posterior shape, scalar.
        b_N:         Gamma posterior rate, scalar.
        elbo_trace:  ELBO at the end of each iteration, list of floats.
        converged:   True if |ΔELBO|/|ELBO| < tol before max_iter.
        n_iter:      Number of iterations executed.
    """

    mu: jax.Array
    z: jax.Array
    V_r: jax.Array
    lambda_bar: float
    a_N: float
    b_N: float
    elbo_trace: list[float] = field(default_factory=list)
    converged: bool = False
    n_iter: int = 0


def matlap_lowrank(
    Y: jax.Array,
    S: jax.Array,
    *,
    rank: int = 50,
    a0: float = 1e-3,
    b0: float = 1e-3,
    max_iter: int = 200,
    tol: float = 1e-6,
    verbose: bool = False,
) -> LowRankCAVIResult:
    """Low-rank Bayesian matrix denoising via CAVI.

    Restricts the variational family to rank-``rank`` row covariances
    Σ_i = V_r A_r^{(i)−1} V_r^T where V_r ∈ R^{n×r} is a shared factor
    basis updated at each iteration.  Memory is O(mn + nr + r²) — suitable
    for large matrices where full CAVI would require O(mn²) storage.

    The ELBO is computed in the r-dimensional factor space and uses ``mr``
    (not ``mn``) as the normalizing dimension; lambda estimates are therefore
    approximately r/n × those of full CAVI.

    Args:
        Y:        Observed matrix, shape (m, n).  NaN/any value where missing.
        S:        Known standard errors, shape (m, n). ``jnp.inf`` where missing.
        rank:     Rank of the factor subspace (default 50).
        a0:       Gamma prior shape for lambda (default: 1e-3).
        b0:       Gamma prior rate for lambda (default: 1e-3).
        max_iter: Maximum CAVI iterations.
        tol:      Convergence tolerance on relative ELBO change.
        verbose:  Print ELBO at each iteration.

    Returns:
        LowRankCAVIResult with posterior mean, factor coordinates, and diagnostics.
    """
    Y = jnp.asarray(Y, dtype=jnp.float32)
    S = jnp.asarray(S, dtype=jnp.float32)
    S2 = S ** 2
    m, n = Y.shape
    r = min(rank, n, m)

    # Initialise V_r via randomized SVD of observed data (missing → 0)
    Y_obs = jnp.where(jnp.isfinite(Y) & jnp.isfinite(S), Y, 0.0)
    _, _, Vt_r = rsvd(Y_obs, r)
    V_r = Vt_r.T  # (n, r), orthonormal columns

    # Initial d_r = ones (arbitrary; will be updated on first iteration)
    d_r = jnp.ones(r, dtype=jnp.float32)

    # Initialise lambda
    a_N = jnp.asarray(a0 + m * r, dtype=jnp.float32)
    b_N = jnp.asarray(a_N, dtype=jnp.float32)  # lambda_bar ~ 1 initially
    lambda_bar = a_N / b_N

    elbo_trace: list[float] = []
    converged = False
    zs: jax.Array | None = None

    for i in range(max_iter):
        # Update lambda from current d_r
        trace_Q = d_r.sum()
        a_N = jnp.asarray(a0 + m * r, dtype=jnp.float32)
        b_N = jnp.asarray(b0, dtype=jnp.float32) + trace_Q
        lambda_bar = a_N / b_N

        # Update q(X) for all rows via Woodbury r×r solve
        zs, A_r_invs, log_dets, diag_sigs = update_rows_lowrank(Y, S2, V_r, d_r, lambda_bar)
        # zs: (m, r), A_r_invs: (m, r, r), log_dets: (m,), diag_sigs: (m, n)

        # Posterior means in original space (mu_i = V_r z_i)
        mus = zs @ V_r.T  # (m, n)

        # Accumulate Ψ_r = sum_i (z_i z_i^T + A_r^{-1}_i) in factor space
        Psi_r = zs.T @ zs + A_r_invs.sum(axis=0)  # (r, r)

        # Refresh d_r and rotate V_r via eigh(Ψ_r)
        vals_r, vecs_r = jnp.linalg.eigh(Psi_r)  # ascending order
        d_r = jnp.sqrt(jnp.maximum(vals_r, 0.0))  # (r,)
        V_r = V_r @ vecs_r  # (n, r)  — rotate columns

        # Update lambda with new d_r
        b_N = jnp.asarray(b0, dtype=jnp.float32) + d_r.sum()
        lambda_bar = a_N / b_N

        # ELBO in factor space (log_dets and diag_sigs are rotation-invariant)
        elbo = compute_elbo_lowrank(
            Y, S2, mus, diag_sigs, log_dets, d_r, lambda_bar, a_N, b_N, a0, b0,
        )
        elbo_val = float(elbo)
        elbo_trace.append(elbo_val)

        if verbose:
            print(f"  iter {i+1:4d}  ELBO={elbo_val:.4f}  lambda={float(lambda_bar):.6f}")

        if i > 0:
            prev = elbo_trace[-2]
            denom = max(abs(prev), 1e-10)
            if abs(elbo_val - prev) / denom < tol:
                converged = True
                break

    return LowRankCAVIResult(
        mu=mus,
        z=zs,
        V_r=V_r,
        lambda_bar=float(lambda_bar),
        a_N=float(a_N),
        b_N=float(b_N),
        elbo_trace=elbo_trace,
        converged=converged,
        n_iter=len(elbo_trace),
    )
