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

from .elbo import compute_elbo, compute_elbo_from_diag, compute_elbo_lowrank
from .linalg import (
    SqrtDecomp, matrix_sqrt_eigh, trace_sqrt,
    update_rows, update_rows_and_reduce,
    update_rows_lowrank, update_rows_lowrank_isotropic,
    rsvd,
)


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


@dataclass
class BatchedCAVIResult:
    """Result of matlap_batched() — full CAVI with O(batch*n²) peak memory.

    Attributes:
        mu:          Posterior mean of X, shape (m, n).
        sigma_diag:  Diagonal of per-row posterior covariances, shape (m, n).
                     Full Sigma_i (n×n per row) is never stored simultaneously.
        lambda_bar:  E_q[lambda], scalar.
        a_N:         Gamma posterior shape, scalar.
        b_N:         Gamma posterior rate, scalar.
        elbo_trace:  ELBO at the end of each iteration, list of floats.
        converged:   True if |ΔELBO|/|ELBO| < tol before max_iter.
        n_iter:      Number of iterations executed.
    """

    mu: jax.Array
    sigma_diag: jax.Array
    lambda_bar: float
    a_N: float
    b_N: float
    elbo_trace: list[float] = field(default_factory=list)
    converged: bool = False
    n_iter: int = 0


@dataclass
class LowRankGridResult:
    """Result of matlap_grid_lowrank().

    Attributes:
        best_lambda:  Lambda with the highest ELBO.
        best_result:  LowRankCAVIResult for the best lambda.
        results:      List of (lambda, LowRankCAVIResult) for every grid point,
                      sorted by lambda ascending.
    """

    best_lambda: float
    best_result: "LowRankCAVIResult"
    results: list[tuple[float, "LowRankCAVIResult"]]


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


@dataclass
class LowRankIsotropicResult:
    """Result of matlap_lowrank_isotropic() CAVI optimisation.

    Implements Q = V_r diag(d_r) V_r^T + δ(I − V_r V_r^T), where δ is a
    variational parameter optimised by CAVI (not a hyperparameter).  The
    off-subspace prior precision γ = λ̄/δ is derived each iteration.

    Attributes:
        mu:          Full n-dim posterior mean of X, shape (m, n).
        z:           In-subspace projection V_r^T μ_i, shape (m, r).
        V_r:         Factor loading matrix, shape (n, r); orthonormal columns.
        lambda_bar:  E_q[lambda], scalar.
        a_N:         Gamma posterior shape (= a0 + m*n), scalar.
        b_N:         Gamma posterior rate, scalar.
        delta:       Converged off-subspace Q eigenvalue δ* = sqrt(Tr(Ψ⊥)/(n-r)).
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
    delta: float
    elbo_trace: list[float] = field(default_factory=list)
    converged: bool = False
    n_iter: int = 0


def matlap_lowrank(
    Y: jax.Array,
    S: jax.Array,
    lambda_val: float | None = None,
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
        Y:          Observed matrix, shape (m, n).  NaN/any value where missing.
        S:          Known standard errors, shape (m, n). ``jnp.inf`` where missing.
        lambda_val: If provided, fix lambda to this value (skip empirical-Bayes
                    update).  Pass as a positional arg to use with
                    :func:`~matlap.cv.cv_lambda`.
        rank:       Rank of the factor subspace (default 50).
        a0:         Gamma prior shape for lambda (default: 1e-3).
        b0:         Gamma prior rate for lambda (default: 1e-3).
        max_iter:   Maximum CAVI iterations.
        tol:        Convergence tolerance on relative ELBO change.
        verbose:    Print ELBO at each iteration.

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
        # Update lambda from current d_r (skip if fixed)
        trace_Q = d_r.sum()
        a_N = jnp.asarray(a0 + m * r, dtype=jnp.float32)
        if lambda_val is None:
            b_N = jnp.asarray(b0, dtype=jnp.float32) + trace_Q
            lambda_bar = a_N / b_N
        else:
            lambda_bar = jnp.asarray(lambda_val, dtype=jnp.float32)
            b_N = a_N / lambda_bar

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

        # Update lambda with new d_r (skip if fixed)
        if lambda_val is None:
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


def matlap_lowrank_isotropic(
    Y: jax.Array,
    S: jax.Array,
    lambda_val: float | None = None,
    *,
    rank: int = 50,
    a0: float = 1e-3,
    b0: float = 1e-3,
    max_iter: int = 200,
    tol: float = 1e-6,
    verbose: bool = False,
) -> LowRankIsotropicResult:
    """Low-rank-plus-isotropic Bayesian matrix denoising via CAVI.

    Implements Q = V_r diag(d_r) V_r^T + δ(I − V_r V_r^T) where δ is a
    variational parameter (not a hyperparameter) optimised in closed form each
    iteration as δ* = sqrt(Tr(Ψ⊥) / (n − r)).  The off-subspace prior
    precision γ = λ̄/δ is derived from δ and λ̄, so no extra hyperparameters
    are introduced beyond those in :func:`matlap_lowrank`.

    Compared to :func:`matlap_lowrank`:

    * μ_i captures off-subspace signal (components outside col(V_r)).
    * Entropy uses all n dimensions, correcting the auto-λ over-shrinkage.
    * δ is learned from the data; no γ hyperparameter to tune.

    Args:
        Y:          Observed matrix, shape (m, n).  NaN/any value where missing.
        S:          Known standard errors, shape (m, n). ``jnp.inf`` where missing.
        lambda_val: If provided, fix lambda to this value (skip empirical-Bayes
                    update).  Pass as a positional arg to use with
                    :func:`~matlap.cv.cv_lambda`.
        rank:       Rank of the factor subspace (default 50).
        a0:         Gamma prior shape for lambda (default: 1e-3).
        b0:         Gamma prior rate for lambda (default: 1e-3).
        max_iter:   Maximum CAVI iterations.
        tol:        Convergence tolerance on relative ELBO change.
        verbose:    Print ELBO at each iteration.

    Returns:
        :class:`LowRankIsotropicResult` with posterior mean, factor coordinates,
        and diagnostics (including converged δ).
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

    # a_N = a0 + m*n (correct full-dimension prior)
    a_N = jnp.asarray(a0 + m * n, dtype=jnp.float32)

    d_r = jnp.ones(r, dtype=jnp.float32)

    # δ: off-subspace Q eigenvalue.  Start at 1 → γ = λ̄/δ = 1 initially.
    delta = 1.0

    # Initialise λ ≈ 1: trace_Q = d_r.sum() + (n-r)*delta = r + (n-r) = n
    b_N = jnp.asarray(a0 + float(a_N), dtype=jnp.float32)
    lambda_bar = a_N / b_N  # ≈ 1.0

    elbo_trace: list[float] = []
    converged = False
    mus: jax.Array | None = None
    z_tildes: jax.Array | None = None

    for i in range(max_iter):
        # Derive γ from current λ̄ and δ
        gamma_val = float(lambda_bar) / max(delta, 1e-8)
        gamma_arr = jnp.asarray(gamma_val, dtype=jnp.float32)

        # Pre-row lambda update using current trace_Q
        trace_Q = float(d_r.sum()) + (n - r) * delta
        a_N = jnp.asarray(a0 + m * n, dtype=jnp.float32)
        if lambda_val is None:
            b_N = jnp.asarray(b0 + trace_Q, dtype=jnp.float32)
            lambda_bar = a_N / b_N
            gamma_val = float(lambda_bar) / max(delta, 1e-8)
            gamma_arr = jnp.asarray(gamma_val, dtype=jnp.float32)
        else:
            lambda_bar = jnp.asarray(lambda_val, dtype=jnp.float32)
            b_N = a_N / lambda_bar

        # Full n-dim Woodbury row updates
        mus, z_tildes, VtSigmaVs, log_dets, diag_sigs = (
            update_rows_lowrank_isotropic(Y, S2, V_r, d_r, lambda_bar, gamma_arr)
        )
        # mus: (m,n), z_tildes: (m,r), VtSigmaVs: (m,r,r), log_dets: (m,), diag_sigs: (m,n)

        # Ψ_r = V_r^T Ψ V_r = Σ_i (z̃_i z̃_i^T + V_r^T Σ_i V_r)
        Psi_r = z_tildes.T @ z_tildes + VtSigmaVs.sum(axis=0)  # (r, r)

        # Update d_r and rotate V_r via eigh(Ψ_r)
        vals_r, vecs_r = jnp.linalg.eigh(Psi_r)
        d_r = jnp.sqrt(jnp.maximum(vals_r, 0.0))
        V_r = V_r @ vecs_r  # (n, r) — rotate columns

        # Update δ: δ* = sqrt(Tr(Ψ⊥) / (n − r))
        # Tr(Ψ) = Σ_i (‖μ_i‖² + Tr(Σ_i))   (full n-dim)
        # Tr(Ψ_r) = Σ_k d_{r,k}²            (d_r = sqrt eigenvals of Ψ_r)
        Psi_total = float((mus ** 2 + diag_sigs).sum())
        Psi_r_tr = float((d_r ** 2).sum())
        Psi_perp = max(Psi_total - Psi_r_tr, 0.0)
        delta = max(float(jnp.sqrt(Psi_perp / max(n - r, 1))), 1e-6)

        # Post-row trace_Q and lambda update with refreshed d_r and delta
        trace_Q = float(d_r.sum()) + (n - r) * delta
        if lambda_val is None:
            b_N = jnp.asarray(b0 + trace_Q, dtype=jnp.float32)
            lambda_bar = a_N / b_N

        # ELBO: q_sqrt_vals sums to Tr(Q) = Σ_k d_{r,k} + (n−r)δ
        q_sqrt_vals = jnp.concatenate(
            [d_r, jnp.full(n - r, delta, dtype=jnp.float32)]
        )
        elbo = compute_elbo_from_diag(
            Y, S2, mus, diag_sigs, log_dets, q_sqrt_vals, lambda_bar, a_N, b_N, a0, b0,
        )
        elbo_val = float(elbo)
        elbo_trace.append(elbo_val)

        if verbose:
            print(f"  iter {i+1:4d}  ELBO={elbo_val:.4f}  lambda={float(lambda_bar):.4f}"
                  f"  delta={delta:.4f}  gamma={gamma_val:.4f}")

        if i > 0:
            prev = elbo_trace[-2]
            denom = max(abs(prev), 1e-10)
            if abs(elbo_val - prev) / denom < tol:
                converged = True
                break

    assert mus is not None and z_tildes is not None
    return LowRankIsotropicResult(
        mu=mus,
        z=z_tildes,
        V_r=V_r,
        lambda_bar=float(lambda_bar),
        a_N=float(a_N),
        b_N=float(b_N),
        delta=float(delta),
        elbo_trace=elbo_trace,
        converged=converged,
        n_iter=len(elbo_trace),
    )


# ---------------------------------------------------------------------------
# Batched full CAVI (memory-efficient)
# ---------------------------------------------------------------------------


def matlap_batched(
    Y: jax.Array,
    S: jax.Array,
    *,
    a0: float = 1e-3,
    b0: float = 1e-3,
    max_iter: int = 200,
    tol: float = 1e-6,
    batch_size: int = 64,
    verbose: bool = False,
) -> BatchedCAVIResult:
    """Full CAVI with memory-efficient batched row processing.

    Runs the same algorithm as :func:`matlap` but avoids storing all m row
    covariances Sigma_i simultaneously.  Instead each batch of ``batch_size``
    rows is processed by a single JIT call that:

    1. Computes ``(mu_b, Sigma_b, log_det_b)`` via Cholesky.
    2. Extracts ``diag(Sigma_b)`` and the batch's Psi contribution
       ``sum_i (mu_i mu_i^T + Sigma_i)``.
    3. Discards the full ``Sigma_b`` before returning.

    Peak memory is O(batch_size * n²) instead of O(m * n²), making it
    feasible for large m at moderate n.  At n = 1 000, a single row requires
    an O(n³) Cholesky, so this is only practical for n ≲ 300; use
    :func:`matlap_lowrank` for larger n.

    Args:
        Y:          Observed matrix, shape (m, n). NaN/any value where missing.
        S:          Known standard errors, shape (m, n). ``jnp.inf`` where missing.
        a0:         Gamma prior shape for lambda (default: 1e-3).
        b0:         Gamma prior rate for lambda (default: 1e-3).
        max_iter:   Maximum CAVI iterations.
        tol:        Convergence tolerance on relative ELBO change.
        batch_size: Number of rows processed per JIT call (controls peak memory).
        verbose:    Print ELBO at each iteration.

    Returns:
        BatchedCAVIResult with posterior mean, diagonal covariances, and diagnostics.
    """
    Y = jnp.asarray(Y, dtype=jnp.float32)
    S = jnp.asarray(S, dtype=jnp.float32)
    S2 = S ** 2
    m, n = Y.shape

    obs_mask = jnp.isfinite(S) & jnp.isfinite(Y)
    mus = jnp.where(obs_mask, Y, 0.0)

    # Initial Psi from a single pass over rows in batches (no sigma storage)
    # Use identity covariances as starting point: Psi = mumuT + mI
    Psi = jnp.einsum("im,in->mn", mus, mus) + m * jnp.eye(n, dtype=jnp.float32)
    decomp = matrix_sqrt_eigh(Psi)

    a_N = jnp.asarray(a0 + m * n, dtype=jnp.float32)
    b_N = jnp.asarray(b0 + trace_sqrt(decomp), dtype=jnp.float32)
    lambda_bar = a_N / b_N

    elbo_trace: list[float] = []
    converged = False
    sigma_diag = jnp.zeros((m, n), dtype=jnp.float32)
    log_dets = jnp.zeros(m, dtype=jnp.float32)

    for i in range(max_iter):
        new_Psi = jnp.zeros((n, n), dtype=jnp.float32)
        mus_parts: list[jax.Array] = []
        sd_parts: list[jax.Array] = []
        ld_parts: list[jax.Array] = []

        for start in range(0, m, batch_size):
            end = min(start + batch_size, m)
            mu_b, sd_b, ld_b, Psi_b = update_rows_and_reduce(
                Y[start:end], S2[start:end],
                decomp.sqrt_vals, decomp.vecs, lambda_bar,
            )
            mus_parts.append(mu_b)
            sd_parts.append(sd_b)
            ld_parts.append(ld_b)
            new_Psi = new_Psi + Psi_b

        mus = jnp.concatenate(mus_parts, axis=0)
        sigma_diag = jnp.concatenate(sd_parts, axis=0)
        log_dets = jnp.concatenate(ld_parts, axis=0)

        decomp = matrix_sqrt_eigh(new_Psi)

        trace_Q = trace_sqrt(decomp)
        a_N = jnp.asarray(a0 + m * n, dtype=jnp.float32)
        b_N = jnp.asarray(b0, dtype=jnp.float32) + trace_Q
        lambda_bar = a_N / b_N

        elbo = compute_elbo_from_diag(
            Y, S2, mus, sigma_diag, log_dets,
            decomp.sqrt_vals, lambda_bar, a_N, b_N, a0, b0,
        )
        elbo_val = float(elbo)
        elbo_trace.append(elbo_val)

        if verbose:
            print(f"  iter {i+1:4d}  ELBO={elbo_val:.4f}  lambda={float(lambda_bar):.4f}")

        if i > 0:
            prev = elbo_trace[-2]
            if abs(elbo_val - prev) / max(abs(prev), 1e-10) < tol:
                converged = True
                break

    return BatchedCAVIResult(
        mu=mus,
        sigma_diag=sigma_diag,
        lambda_bar=float(lambda_bar),
        a_N=float(a_N),
        b_N=float(b_N),
        elbo_trace=elbo_trace,
        converged=converged,
        n_iter=len(elbo_trace),
    )


# ---------------------------------------------------------------------------
# Low-rank grid search (warm-started regularisation path)
# ---------------------------------------------------------------------------


def matlap_grid_lowrank(
    Y: jax.Array,
    S: jax.Array,
    lambda_grid: jax.Array,
    *,
    rank: int = 50,
    a0: float = 1e-3,
    b0: float = 1e-3,
    max_iter: int = 50,
    tol: float = 1e-6,
    verbose: bool = False,
) -> LowRankGridResult:
    """Grid search over lambda using low-rank CAVI with warm-started path.

    Evaluates each lambda in ``lambda_grid`` using the low-rank CAVI algorithm
    from :func:`matlap_lowrank`.  Lambda values are processed in **decreasing**
    order so that the sparse solution at large lambda warm-starts the denser
    solution at the next smaller lambda.  This regularisation-path strategy
    typically needs far fewer iterations per grid point than cold-starting.

    The best lambda is selected by maximising the ELBO (computed in factor
    space, consistent with how :func:`matlap_grid` selects lambda in the full
    CAVI).

    Args:
        Y:            Observed matrix, shape (m, n). NaN/any value where missing.
        S:            Known standard errors, shape (m, n). ``jnp.inf`` where missing.
        lambda_grid:  1-D array of lambda values to evaluate.
        rank:         Rank of the factor subspace (default 50).
        a0:           Gamma prior shape (used in ELBO; lambda is fixed per point).
        b0:           Gamma prior rate.
        max_iter:     Maximum CAVI iterations **per grid point** (default 50).
        tol:          Convergence tolerance on relative ELBO change.
        verbose:      Print progress.

    Returns:
        LowRankGridResult with best lambda and per-lambda LowRankCAVIResults.
    """
    Y = jnp.asarray(Y, dtype=jnp.float32)
    S = jnp.asarray(S, dtype=jnp.float32)
    S2 = S ** 2
    m, n = Y.shape
    r = min(rank, n, m)

    # Initialise shared state via rSVD of the observed data
    Y_obs = jnp.where(jnp.isfinite(Y) & jnp.isfinite(S), Y, 0.0)
    _, _, Vt_r = rsvd(Y_obs, r)
    V_r = Vt_r.T  # (n, r)
    d_r = jnp.ones(r, dtype=jnp.float32)
    zs: jax.Array | None = None

    # Process lambda in decreasing order — warm-start regularisation path
    lambda_vals = sorted([float(lv) for lv in lambda_grid], reverse=True)

    results: list[tuple[float, LowRankCAVIResult]] = []
    best_elbo = -jnp.inf
    best_lam = lambda_vals[0]
    best_res: LowRankCAVIResult | None = None

    for lam_val in lambda_vals:
        lam = jnp.asarray(lam_val, dtype=jnp.float32)
        # Fix lambda: set a_N/b_N so lambda_bar = lam (same as matlap_grid)
        a_N = jnp.asarray(a0 + m * r, dtype=jnp.float32)
        b_N = jnp.asarray(a_N / lam, dtype=jnp.float32)
        lambda_bar = lam

        elbo_trace: list[float] = []
        converged = False

        for i in range(max_iter):
            # Update q(X) rows (warm-started from previous lambda's V_r, d_r)
            zs, A_r_invs, log_dets, diag_sigs = update_rows_lowrank(
                Y, S2, V_r, d_r, lambda_bar,
            )
            mus = zs @ V_r.T

            # Refresh V_r and d_r via eigh of factor-space Psi_r
            Psi_r = zs.T @ zs + A_r_invs.sum(axis=0)
            vals_r, vecs_r = jnp.linalg.eigh(Psi_r)
            d_r = jnp.sqrt(jnp.maximum(vals_r, 0.0))
            V_r = V_r @ vecs_r

            # ELBO with fixed lambda (b_N fixed, lambda_bar = lam throughout)
            elbo = compute_elbo_lowrank(
                Y, S2, mus, diag_sigs, log_dets, d_r, lambda_bar, a_N, b_N, a0, b0,
            )
            elbo_val = float(elbo)
            elbo_trace.append(elbo_val)

            if i > 0:
                prev = elbo_trace[-2]
                if abs(elbo_val - prev) / max(abs(prev), 1e-10) < tol:
                    converged = True
                    break

        res = LowRankCAVIResult(
            mu=mus,
            z=zs,
            V_r=V_r,
            lambda_bar=lam_val,
            a_N=float(a_N),
            b_N=float(b_N),
            elbo_trace=elbo_trace,
            converged=converged,
            n_iter=len(elbo_trace),
        )
        results.append((lam_val, res))

        if verbose:
            print(f"  lambda={lam_val:.4f}  ELBO={elbo_trace[-1]:.4f}  "
                  f"iters={len(elbo_trace)}  converged={converged}")

        if elbo_trace[-1] > best_elbo:
            best_elbo = elbo_trace[-1]
            best_lam = lam_val
            best_res = res

    results.sort(key=lambda x: x[0])
    return LowRankGridResult(
        best_lambda=best_lam,
        best_result=best_res,
        results=results,
    )
