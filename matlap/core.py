"""
Coordinate Ascent Variational Inference (CAVI) for Bayesian Matrix Denoising.

Main public functions:
    matlap()                       — full CAVI with automatic lambda estimation
    matlap_grid()                  — full CAVI, lambda selected by ELBO over grid
    matlap_batched()               — full CAVI, memory-efficient batched rows
    matlap_batched_warmstart()     — matlap_batched with FA-EM warm-start
    matlap_iso_warmstart()         — matlap_lowrank_isotropic with FA-EM warm-start
    matlap_lowrank()               — low-rank CAVI with automatic lambda
    matlap_grid_lowrank()          — low-rank CAVI, lambda selected by ELBO over grid
    matlap_lowrank_isotropic()     — low-rank+isotropic CAVI with automatic lambda
    matlap_grid_lowrank_isotropic()— low-rank+isotropic CAVI, ELBO grid search

Grid functions warm-start in decreasing-lambda order, calling through to the
corresponding single-lambda function with the previous point's (V_r, d_r[, δ])
as initialisation.  This means every algorithm lives in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from .elbo import compute_elbo, compute_elbo_from_diag, compute_elbo_lowrank, compute_elbo_lowrank_iso
from .linalg import (
    SqrtDecomp, matrix_sqrt_eigh, trace_sqrt,
    update_rows, update_rows_and_reduce,
    update_rows_lowrank, update_rows_lowrank_isotropic,
    rsvd,
)
from .scoring import closed_form_loo, compute_iso_prior_var, renyi_elbo, renyi_lambda_opt


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
        mu:              Posterior mean of X, shape (m, n).
        sigma_diag:      Diagonal of per-row posterior covariances, shape (m, n).
                         Full Sigma_i (n×n per row) is never stored simultaneously.
        psi_sqrt_diag:   Diagonal of √Ψ (= Q), shape (n,), where Ψ = Σᵢ E[xᵢxᵢᵀ].
                         Used as prior-variance proxy: prior_var_j = psi_sqrt_diag_j / λ.
        lambda_bar:      E_q[lambda], scalar.
        a_N:             Gamma posterior shape, scalar.
        b_N:             Gamma posterior rate, scalar.
        elbo_trace:      ELBO at the end of each iteration, list of floats.
        converged:       True if |ΔELBO|/|ELBO| < tol before max_iter.
        n_iter:          Number of iterations executed.
    """

    mu: jax.Array
    sigma_diag: jax.Array
    psi_sqrt_diag: jax.Array
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
        d_r:         Q eigenvalues sqrt(eig(Ψ_r)), shape (r,).  Warm-start state.
        lambda_bar:  E_q[lambda], scalar.
        a_N:         Gamma posterior shape, scalar.
        b_N:         Gamma posterior rate, scalar.
        diag_sigma:  Diagonal posterior variances, shape (m, n).
        elbo_trace:  ELBO at the end of each iteration, list of floats.
        converged:   True if |ΔELBO|/|ELBO| < tol before max_iter.
        n_iter:      Number of iterations executed.
    """

    mu: jax.Array
    z: jax.Array
    V_r: jax.Array
    d_r: jax.Array
    lambda_bar: float
    a_N: float
    b_N: float
    diag_sigma: jax.Array | None = None
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
        d_r:         Q eigenvalues sqrt(eig(Ψ_r)), shape (r,).  Warm-start state.
        lambda_bar:  E_q[lambda], scalar.
        a_N:         Gamma posterior shape (= a0 + m*n), scalar.
        b_N:         Gamma posterior rate, scalar.
        delta:       Converged off-subspace Q eigenvalue δ* = sqrt(Tr(Ψ⊥)/(n-r)).
        diag_sigma:  Diagonal posterior variances, shape (m, n).
        elbo_trace:  ELBO at the end of each iteration, list of floats.
        converged:   True if |ΔELBO|/|ELBO| < tol before max_iter.
        n_iter:      Number of iterations executed.
    """

    mu: jax.Array
    z: jax.Array
    V_r: jax.Array
    d_r: jax.Array
    lambda_bar: float
    a_N: float
    b_N: float
    delta: float
    diag_sigma: jax.Array | None = None
    elbo_trace: list[float] = field(default_factory=list)
    converged: bool = False
    n_iter: int = 0


@dataclass
class LowRankIsotropicGridResult:
    """Result of matlap_grid_lowrank_isotropic().

    Attributes:
        best_lambda:  Lambda with the highest ELBO.
        best_result:  LowRankIsotropicResult for the best lambda.
        results:      List of (lambda, LowRankIsotropicResult) for every grid
                      point, sorted by lambda ascending.
    """

    best_lambda: float
    best_result: "LowRankIsotropicResult"
    results: list[tuple[float, "LowRankIsotropicResult"]]


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
    V_r_init: jax.Array | None = None,
    d_r_init: jax.Array | None = None,
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
        V_r_init:   Warm-start factor loadings, shape (n, r).  If None, initialised
                    from rSVD of the observed data.
        d_r_init:   Warm-start Q eigenvalues, shape (r,).  If None, initialised to
                    ones.

    Returns:
        LowRankCAVIResult with posterior mean, factor coordinates, and diagnostics.
        The ``V_r`` and ``d_r`` fields can be passed as warm-start to the next call.
    """
    Y = jnp.asarray(Y, dtype=jnp.float32)
    S = jnp.asarray(S, dtype=jnp.float32)
    S2 = S ** 2
    m, n = Y.shape
    r = min(rank, n, m)

    if V_r_init is not None:
        V_r = jnp.asarray(V_r_init, dtype=jnp.float32)
        d_r = jnp.asarray(d_r_init, dtype=jnp.float32)
    else:
        Y_obs = jnp.where(jnp.isfinite(Y) & jnp.isfinite(S), Y, 0.0)
        _, _, Vt_r = rsvd(Y_obs, r)
        V_r = Vt_r.T  # (n, r), orthonormal columns
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
        d_r=d_r,
        lambda_bar=float(lambda_bar),
        a_N=float(a_N),
        b_N=float(b_N),
        diag_sigma=diag_sigs,
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
    V_r_init: jax.Array | None = None,
    d_r_init: jax.Array | None = None,
    delta_init: float | None = None,
    use_ldlt: bool = False,
    use_xla_ldlt: bool = False,
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
        Y:           Observed matrix, shape (m, n).  NaN/any value where missing.
        S:           Known standard errors, shape (m, n). ``jnp.inf`` where missing.
        lambda_val:  If provided, fix lambda to this value (skip empirical-Bayes
                     update).  Pass as a positional arg to use with
                     :func:`~matlap.cv.cv_lambda`.
        rank:        Rank of the factor subspace (default 50).
        a0:          Gamma prior shape for lambda (default: 1e-3).
        b0:          Gamma prior rate for lambda (default: 1e-3).
        max_iter:    Maximum CAVI iterations.
        tol:         Convergence tolerance on relative ELBO change.
        verbose:     Print ELBO at each iteration.
        V_r_init:    Warm-start factor loadings, shape (n, r).  If None, initialised
                     from rSVD of the observed data.
        d_r_init:    Warm-start Q eigenvalues, shape (r,).  If None, initialised to
                     ones.
        delta_init:  Warm-start off-subspace scale δ.  If None, derived from a_N.
        use_ldlt:    If True, use the CuPy CUDA LDL^T kernel for the B̃
                     factorisation.  ~4× faster than ``eigh``; requires CuPy.
                     Must be called outside ``jax.jit``.
        use_xla_ldlt: If True, use the XLA-native CUDA LDL^T kernel.  Runs on
                     the JAX-managed stream with no sync barriers; the entire
                     update is compiled into a single XLA program.  ~6× faster
                     than ``eigh``; requires the compiled ``_ldlt_kernel.so``.

    Returns:
        :class:`LowRankIsotropicResult` with posterior mean, factor coordinates,
        and diagnostics (including converged δ).  The ``V_r``, ``d_r``, and
        ``delta`` fields can be passed as warm-start to the next call.
    """
    Y = jnp.asarray(Y, dtype=jnp.float32)
    S = jnp.asarray(S, dtype=jnp.float32)
    S2 = S ** 2
    m, n = Y.shape
    r = min(rank, n, m)

    # a_N = a0 + m*n (correct full-dimension prior)
    a_N = jnp.asarray(a0 + m * n, dtype=jnp.float32)

    if V_r_init is not None:
        V_r = jnp.asarray(V_r_init, dtype=jnp.float32)
        d_r = jnp.asarray(d_r_init, dtype=jnp.float32)
    else:
        Y_obs = jnp.where(jnp.isfinite(Y) & jnp.isfinite(S), Y, 0.0)
        _, _, Vt_r = rsvd(Y_obs, r)
        V_r = Vt_r.T  # (n, r), orthonormal columns
        d_r = jnp.ones(r, dtype=jnp.float32)

    if delta_init is not None:
        delta = float(delta_init)
    else:
        # Initialise δ so that trace_Q ≈ a_N → lambda_bar ≈ 1 on the first iteration.
        delta = max((float(a_N) - float(r)) / max(n - r, 1), 1.0)

    b_N = jnp.asarray(b0 + float(a_N), dtype=jnp.float32)  # trace_Q ≈ a_N → lambda ≈ 1
    lambda_bar = a_N / b_N

    if use_xla_ldlt:
        from .ldlt_cuda import update_rows_lowrank_isotropic_xla_ldlt as _row_update
    elif use_ldlt:
        from .ldlt_cuda import update_rows_lowrank_isotropic_ldlt as _row_update
    else:
        _row_update = update_rows_lowrank_isotropic

    elbo_trace: list[float] = []
    converged = False
    mus: jax.Array | None = None
    z_tildes: jax.Array | None = None

    for i in range(max_iter):
        # γ = λ̄ (off-subspace prior precision equals λ̄, not λ̄/δ)
        gamma_val = float(lambda_bar)
        gamma_arr = jnp.asarray(gamma_val, dtype=jnp.float32)

        # Pre-row lambda update using current trace_Q
        trace_Q = float(d_r.sum()) + (n - r) * delta
        if lambda_val is None:
            b_N = jnp.asarray(b0 + trace_Q, dtype=jnp.float32)
            lambda_bar = a_N / b_N
            gamma_val = float(lambda_bar)
            gamma_arr = jnp.asarray(gamma_val, dtype=jnp.float32)
        else:
            lambda_bar = jnp.asarray(lambda_val, dtype=jnp.float32)
            b_N = a_N / lambda_bar

        # Full n-dim Woodbury row updates
        mus, z_tildes, VtSigmaVs, log_dets, diag_sigs = (
            _row_update(Y, S2, V_r, d_r, lambda_bar, gamma_arr)
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

        # Correct hybrid ELBO: nuclear-norm normaliser for r in-subspace dims,
        # Gaussian normaliser for (n-r) off-subspace dims.
        # Psi_perp = (n-r)*delta² is the total off-subspace second moment.
        Psi_perp_val = jnp.asarray(float(Psi_perp), dtype=jnp.float32)
        elbo = compute_elbo_lowrank_iso(
            Y, S2, mus, diag_sigs, log_dets, d_r, Psi_perp_val,
            lambda_bar, a_N, b_N, a0, b0,
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
        d_r=d_r,
        lambda_bar=float(lambda_bar),
        a_N=float(a_N),
        b_N=float(b_N),
        delta=float(delta),
        diag_sigma=diag_sigs,
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
    lambda_val: float | None = None,
    lambda_init: float | None = None,
    mu_init: jax.Array | None = None,
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
        Y:            Observed matrix, shape (m, n). NaN/any value where missing.
        S:            Known standard errors, shape (m, n). ``jnp.inf`` where missing.
        a0:           Gamma prior shape for lambda (default: 1e-3).
        b0:           Gamma prior rate for lambda (default: 1e-3).
        max_iter:     Maximum CAVI iterations.
        tol:          Convergence tolerance on relative ELBO change.
        batch_size:   Number of rows processed per JIT call (controls peak memory).
        lambda_val:   If provided, fix lambda to this value (skip empirical-Bayes
                      update). Useful for grid search and comparison studies.
        lambda_init:  If provided, initialise lambda to this value but allow CAVI
                      to update it. Useful for warm-starting from FA-EM/GradML.
        mu_init:      If provided, use this (m, n) array as the initial posterior
                      means instead of the observed Y. Warm-starting from a fast
                      method (e.g. FA-EM) typically halves the iteration count.
        verbose:      Print ELBO at each iteration.

    Returns:
        BatchedCAVIResult with posterior mean, diagonal covariances, and diagnostics.
    """
    Y = jnp.asarray(Y, dtype=jnp.float32)
    S = jnp.asarray(S, dtype=jnp.float32)
    S2 = S ** 2
    m, n = Y.shape

    obs_mask = jnp.isfinite(S) & jnp.isfinite(Y)
    if mu_init is not None:
        mus = jnp.asarray(mu_init, dtype=jnp.float32)
    else:
        mus = jnp.where(obs_mask, Y, 0.0)

    # Initial Psi from a single pass over rows in batches (no sigma storage)
    # Use identity covariances as starting point: Psi = mumuT + mI
    Psi = jnp.einsum("im,in->mn", mus, mus) + m * jnp.eye(n, dtype=jnp.float32)
    decomp = matrix_sqrt_eigh(Psi)

    a_N = jnp.asarray(a0 + m * n, dtype=jnp.float32)
    if lambda_val is not None:
        lambda_bar = jnp.asarray(lambda_val, dtype=jnp.float32)
        b_N = a_N / lambda_bar
    elif lambda_init is not None:
        lambda_bar = jnp.asarray(lambda_init, dtype=jnp.float32)
        b_N = a_N / lambda_bar
    else:
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
        if lambda_val is None:
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

    # diag(√Ψ)_j = Σ_k sqrt_vals[k] * vecs[j,k]²  — used as prior-var proxy
    psi_sqrt_diag = (decomp.vecs ** 2) @ decomp.sqrt_vals

    return BatchedCAVIResult(
        mu=mus,
        sigma_diag=sigma_diag,
        psi_sqrt_diag=psi_sqrt_diag,
        lambda_bar=float(lambda_bar),
        a_N=float(a_N),
        b_N=float(b_N),
        elbo_trace=elbo_trace,
        converged=converged,
        n_iter=len(elbo_trace),
    )


def matlap_batched_warmstart(
    Y: jax.Array,
    S: jax.Array,
    *,
    faem_rank: int = 10,
    faem_iters: int = 50,
    a0: float = 1e-3,
    b0: float = 1e-3,
    max_iter: int = 200,
    tol: float = 1e-6,
    batch_size: int = 64,
    lambda_val: float | None = None,
    verbose: bool = False,
) -> BatchedCAVIResult:
    """FA-EM warm-started version of :func:`matlap_batched`.

    Runs FA-EM for a small number of iterations to obtain good initial
    posterior means and a lambda estimate, then passes them to
    :func:`matlap_batched`.  This typically halves the number of CAVI
    iterations needed because the initial Q encodes the dominant subspace.

    FA-EM is a Gaussian factor model, so its mu is a good proxy for the
    Matrix Laplace posterior means; the warm-start does not bias the final
    result since CAVI updates all quantities from the initialisation.

    Args:
        Y:          Observed matrix, shape (m, n).
        S:          Standard errors, shape (m, n). ``jnp.inf`` where missing.
        faem_rank:  Rank used for FA-EM warm-start (default: 10).
        faem_iters: FA-EM iterations for warm-start (default: 50; fast).
        a0, b0, max_iter, tol, batch_size, lambda_val, verbose:
                    Passed through to :func:`matlap_batched`.

    Returns:
        BatchedCAVIResult (same as :func:`matlap_batched`).
    """
    from .faem import matlap_faem  # local import to avoid circular dep at module level

    faem = matlap_faem(Y, S, rank=faem_rank, max_iter=faem_iters)
    return matlap_batched(
        Y, S,
        mu_init=faem.mu,
        lambda_init=faem.lambda_bar if lambda_val is None else None,
        a0=a0, b0=b0,
        max_iter=max_iter, tol=tol,
        batch_size=batch_size,
        lambda_val=lambda_val,
        verbose=verbose,
    )


def matlap_iso_warmstart(
    Y: jax.Array,
    S: jax.Array,
    *,
    faem_rank: int = 10,
    faem_iters: int = 50,
    rank: int | None = None,
    a0: float = 1e-3,
    b0: float = 1e-3,
    max_iter: int = 200,
    tol: float = 1e-6,
    lambda_val: float | None = None,
    verbose: bool = False,
) -> LowRankIsotropicResult:
    """FA-EM warm-started version of :func:`matlap_lowrank_isotropic`.

    Runs FA-EM for a small number of iterations to obtain a good initial
    subspace, then passes ``V_r`` and ``d_r`` to
    :func:`matlap_lowrank_isotropic`.

    The mapping from FA-EM to low-rank+isotropic warm-start is::

        FA-EM W_r  (n × r)
          → thin SVD: W_r = U Σ Vt
          → V_r_init = U   (n × r, orthonormal)
          → d_r_init = Σ   (eigenvalues of Q_r = (W_r W_r^T)^{1/2})

    Args:
        Y:          Observed matrix, shape (m, n).
        S:          Standard errors, shape (m, n). ``jnp.inf`` where missing.
        faem_rank:  Rank used for FA-EM warm-start (default: 10).
        faem_iters: FA-EM iterations for warm-start (default: 50; fast).
        rank:       Rank for matlap_lowrank_isotropic. Defaults to faem_rank.
        a0, b0, max_iter, tol, lambda_val, verbose:
                    Passed through to :func:`matlap_lowrank_isotropic`.

    Returns:
        LowRankIsotropicResult (same as :func:`matlap_lowrank_isotropic`).
    """
    from .faem import matlap_faem

    if rank is None:
        rank = faem_rank

    faem = matlap_faem(Y, S, rank=faem_rank, max_iter=faem_iters)

    # Thin SVD of W_r: (n,r) → U (n,r), s (r,), Vt (r,r)
    W_r = jnp.asarray(faem.W_r, dtype=jnp.float32)
    U, s, _ = jnp.linalg.svd(W_r, full_matrices=False)
    # Pad/trim if rank != faem_rank
    if rank > faem_rank:
        pad = jnp.zeros((U.shape[0], rank - faem_rank), dtype=jnp.float32)
        U = jnp.concatenate([U, pad], axis=1)
        s = jnp.concatenate([s, jnp.ones(rank - faem_rank, dtype=jnp.float32)])
        # Re-orthogonalise via QR in case padding introduced non-orthogonality
        U, _ = jnp.linalg.qr(U)
    else:
        U = U[:, :rank]
        s = s[:rank]

    return matlap_lowrank_isotropic(
        Y, S, lambda_val,
        rank=rank,
        a0=a0, b0=b0,
        max_iter=max_iter, tol=tol,
        verbose=verbose,
        V_r_init=U,
        d_r_init=s,
    )



# ---------------------------------------------------------------------------
# Low-rank+isotropic CAVI with Rényi α-ELBO λ learning
# ---------------------------------------------------------------------------


def matlap_iso_renyi_lambda(
    Y: jax.Array,
    S: jax.Array,
    *,
    rank: int = 50,
    alpha: float = 0.5,
    n_outer: int = 20,
    outer_tol: float = 5e-3,
    a0: float = 1e-3,
    b0: float = 1e-3,
    max_iter: int = 200,
    tol: float = 1e-6,
    verbose: bool = False,
) -> LowRankIsotropicResult:
    """Low-rank+isotropic CAVI with Rényi α-ELBO λ selection.

    Alternates between two steps until λ converges:

    1. **CAVI** (inner loop, fixed λ, warm-started from the previous outer step):
       update q(X) = N(μ, Σ) via :func:`matlap_lowrank_isotropic`.
    2. **Rényi λ update**: given the current q, find
       λ* = argmax_λ R_α(λ | q) via BFGS on log λ.

    The Rényi α-ELBO is a tighter bound on the marginal likelihood than the
    standard ELBO (for α < 1), so the resulting λ is closer to the marginal-
    likelihood estimate and better calibrated for prediction than the ELBO
    auto-λ (which over-regularises).

    Initialised with one auto-λ CAVI run, so the first Rényi update starts
    from a good posterior approximation.

    Args:
        Y:          Observed matrix, shape (m, n).  NaN/any value where missing.
        S:          Known standard errors, shape (m, n). ``jnp.inf`` where missing.
        rank:       Rank of the factor subspace (default 50).
        alpha:      Rényi order; 0 ≤ α < 1 (default 0.5).  Smaller α gives a
                    tighter bound and stronger correction toward MLE of λ.
        n_outer:    Maximum number of outer CAVI ↔ Rényi-λ iterations (default 20).
        outer_tol:  Convergence threshold on relative λ change (default 5e-3).
        a0, b0:     Gamma prior hyperparameters for λ (default: 1e-3).
        max_iter:   Maximum inner CAVI iterations per outer step (default 200).
        tol:        Inner CAVI convergence tolerance (default 1e-6).
        verbose:    Print λ at each outer iteration.

    Returns:
        :class:`LowRankIsotropicResult` with the Rényi-optimal λ stored in
        ``lambda_bar`` and the converged posterior in ``mu``, ``V_r``, etc.
    """
    # --- Initialise: one auto-λ CAVI run to get a warm posterior and λ ---
    result = matlap_lowrank_isotropic(
        Y, S,
        rank=rank, a0=a0, b0=b0, max_iter=max_iter, tol=tol, verbose=False,
    )
    current_lambda = result.lambda_bar

    if verbose:
        print(f"  init  lambda={current_lambda:.4f}")

    for outer in range(n_outer):
        # --- Rényi λ update (M-step): given q, find argmax R_α(λ | q) ---
        new_lambda = renyi_lambda_opt(
            result.V_r, result.d_r, result.delta,
            result.mu, result.diag_sigma,
            Y, S, alpha=alpha, lambda_init=current_lambda,
        )

        rel_change = abs(new_lambda - current_lambda) / max(current_lambda, 1e-10)
        if verbose:
            print(f"  outer {outer + 1:2d}  lambda: {current_lambda:.4f} → {new_lambda:.4f}"
                  f"  (rel Δ={rel_change:.2e})")
        current_lambda = new_lambda

        # --- CAVI E-step: update q with fixed λ, warm-started ---
        result = matlap_lowrank_isotropic(
            Y, S, current_lambda,
            rank=rank, a0=a0, b0=b0, max_iter=max_iter, tol=tol, verbose=False,
            V_r_init=result.V_r, d_r_init=result.d_r, delta_init=result.delta,
        )

        if rel_change < outer_tol:
            if verbose:
                print(f"  converged at outer iter {outer + 1}")
            break

    # Return result with the Rényi-optimal lambda_bar
    return LowRankIsotropicResult(
        mu=result.mu,
        z=result.z,
        V_r=result.V_r,
        d_r=result.d_r,
        lambda_bar=current_lambda,
        a_N=result.a_N,
        b_N=result.b_N,
        delta=result.delta,
        diag_sigma=result.diag_sigma,
        elbo_trace=result.elbo_trace,
        converged=result.converged,
        n_iter=result.n_iter,
    )




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
    score_fn: str = "elbo",
    alpha: float = 0.5,
    verbose: bool = False,
) -> LowRankGridResult:
    """Grid search over lambda using low-rank CAVI with warm-started path.

    Calls :func:`matlap_lowrank` for each lambda in **decreasing** order,
    passing the previous point's ``(V_r, d_r)`` as warm-start.

    Selection score:
      * ``score_fn="elbo"`` (default): final ELBO at each grid point.
      * ``score_fn="loo"``: analytical Gaussian leave-one-out score.
      * ``score_fn="renyi"``: analytical Rényi α-ELBO.

    Args:
        Y:            Observed matrix, shape (m, n). NaN/any value where missing.
        S:            Known standard errors, shape (m, n). ``jnp.inf`` where missing.
        lambda_grid:  1-D array of lambda values to evaluate.
        rank:         Rank of the factor subspace (default 50).
        a0:           Gamma prior shape (used in ELBO; lambda is fixed per point).
        b0:           Gamma prior rate.
        max_iter:     Maximum CAVI iterations **per grid point** (default 50).
        tol:          Convergence tolerance on relative ELBO change.
        score_fn:     One of ``{"elbo", "loo", "renyi"}``.
        alpha:        Rényi order for ``score_fn="renyi"``; must satisfy 0≤α<1.
        verbose:      Print progress.

    Returns:
        :class:`LowRankGridResult` with best lambda and per-lambda results.
    """
    score_mode = score_fn.lower()
    if score_mode not in {"elbo", "loo", "renyi"}:
        raise ValueError(f"Unknown score_fn={score_fn!r}; expected 'elbo', 'loo', or 'renyi'.")
    if score_mode == "renyi" and not (0.0 <= alpha < 1.0):
        raise ValueError(f"alpha must be in [0, 1) for Rényi scoring, got {alpha}")

    lambda_vals = sorted([float(lv) for lv in lambda_grid], reverse=True)

    V_r: jax.Array | None = None
    d_r: jax.Array | None = None

    results: list[tuple[float, LowRankCAVIResult]] = []
    best_score = -float("inf")
    best_lam = lambda_vals[0]
    best_res: LowRankCAVIResult | None = None

    for lam_val in lambda_vals:
        res = matlap_lowrank(
            Y, S, lam_val,
            rank=rank, a0=a0, b0=b0, max_iter=max_iter, tol=tol,
            V_r_init=V_r, d_r_init=d_r,
        )
        V_r = res.V_r
        d_r = res.d_r

        if score_mode == "elbo":
            score = float(res.elbo_trace[-1])
        else:
            if res.diag_sigma is None:
                raise ValueError("diag_sigma is required for score_fn != 'elbo'.")
            if score_mode == "loo":
                score = closed_form_loo(res.mu, res.diag_sigma, Y, S)
            else:
                prior_var = compute_iso_prior_var(
                    res.V_r,
                    res.d_r,
                    delta=1e-6,
                    lambda_bar=lam_val,
                )
                score = renyi_elbo(res.mu, res.diag_sigma, prior_var, Y, S, alpha=alpha)

        if score > best_score:
            best_score = score
            best_lam = lam_val
            best_res = res

        results.append((lam_val, res))
        if verbose:
            print(
                f"  lambda={lam_val:.4f}  {score_mode.upper()}={score:.4f}  "
                f"ELBO={res.elbo_trace[-1]:.4f}  iters={res.n_iter}  converged={res.converged}"
            )

    results.sort(key=lambda x: x[0])
    return LowRankGridResult(
        best_lambda=best_lam,
        best_result=best_res,
        results=results,
    )


def matlap_grid_lowrank_isotropic(
    Y: jax.Array,
    S: jax.Array,
    lambda_grid: jax.Array,
    *,
    rank: int = 50,
    a0: float = 1e-3,
    b0: float = 1e-3,
    max_iter: int = 50,
    tol: float = 1e-6,
    score_fn: str = "renyi",
    alpha: float = 0.5,
    use_ldlt: bool = False,
    use_xla_ldlt: bool = False,
    verbose: bool = False,
) -> LowRankIsotropicGridResult:
    """Grid search over lambda using low-rank+isotropic CAVI with warm-started path.

    Calls :func:`matlap_lowrank_isotropic` for each lambda in **decreasing**
    order, passing the previous point's ``(V_r, d_r, delta)`` as warm-start.

    Selection score:
      * ``score_fn="renyi"`` (default): analytical Rényi α-ELBO (recommended).
      * ``score_fn="loo"``: analytical Gaussian leave-one-out score.
      * ``score_fn="elbo"``: final ELBO — **not recommended** for the iso model.
        The nuclear-norm normaliser ``m·n·log λ`` grows faster than the prior
        penalty ``-λ·Tr(Q)`` shrinks at large λ, so the ELBO is monotonically
        increasing across the grid and always selects the largest λ value.

    Args:
        Y:            Observed matrix, shape (m, n). NaN/any value where missing.
        S:            Known standard errors, shape (m, n). ``jnp.inf`` where missing.
        lambda_grid:  1-D array of lambda values to evaluate.
        rank:         Rank of the factor subspace (default 50).
        a0:           Gamma prior shape (used in ELBO; lambda is fixed per point).
        b0:           Gamma prior rate.
        max_iter:     Maximum CAVI iterations **per grid point** (default 50).
        tol:          Convergence tolerance on relative ELBO change.
        score_fn:     One of ``{"renyi", "loo", "elbo"}``.  Default ``"renyi"``.
        alpha:        Rényi order for ``score_fn="renyi"``; must satisfy 0≤α<1.
        use_ldlt:     If True, use the CuPy CUDA LDL^T kernel (requires GPU + CuPy).
        use_xla_ldlt: If True, use the XLA-native CUDA LDL^T kernel (no sync barriers).
        verbose:      Print progress.

    Returns:
        :class:`LowRankIsotropicGridResult` with best lambda and per-lambda results.
    """
    score_mode = score_fn.lower()
    if score_mode not in {"elbo", "loo", "renyi"}:
        raise ValueError(f"Unknown score_fn={score_fn!r}; expected 'elbo', 'loo', or 'renyi'.")
    if score_mode == "renyi" and not (0.0 <= alpha < 1.0):
        raise ValueError(f"alpha must be in [0, 1) for Rényi scoring, got {alpha}")

    lambda_vals = sorted([float(lv) for lv in lambda_grid], reverse=True)

    V_r: jax.Array | None = None
    d_r: jax.Array | None = None
    delta: float | None = None

    results: list[tuple[float, LowRankIsotropicResult]] = []
    best_score = -float("inf")
    best_lam = lambda_vals[0]
    best_res: LowRankIsotropicResult | None = None

    for lam_val in lambda_vals:
        res = matlap_lowrank_isotropic(
            Y, S, lam_val,
            rank=rank, a0=a0, b0=b0, max_iter=max_iter, tol=tol,
            V_r_init=V_r, d_r_init=d_r, delta_init=delta,
            use_ldlt=use_ldlt, use_xla_ldlt=use_xla_ldlt,
        )
        V_r = res.V_r
        d_r = res.d_r
        delta = res.delta

        if score_mode == "elbo":
            score = float(res.elbo_trace[-1])
        else:
            if res.diag_sigma is None:
                raise ValueError("diag_sigma is required for score_fn != 'elbo'.")
            if score_mode == "loo":
                score = closed_form_loo(res.mu, res.diag_sigma, Y, S)
            else:
                prior_var = compute_iso_prior_var(
                    res.V_r,
                    res.d_r,
                    delta=res.delta,
                    lambda_bar=lam_val,
                )
                score = renyi_elbo(res.mu, res.diag_sigma, prior_var, Y, S, alpha=alpha)

        if score > best_score:
            best_score = score
            best_lam = lam_val
            best_res = res

        results.append((lam_val, res))
        if verbose:
            print(
                f"  lambda={lam_val:.4f}  {score_mode.upper()}={score:.4f}  "
                f"ELBO={res.elbo_trace[-1]:.4f}  iters={res.n_iter}  converged={res.converged}"
            )

    results.sort(key=lambda x: x[0])
    return LowRankIsotropicGridResult(
        best_lambda=best_lam,
        best_result=best_res,
        results=results,
    )


from .adaptive import adaptive_lambda_search, batched_warm_state, iso_warm_state, lowrank_warm_state
from .scoring import (
    _get_prior_var, closed_form_loo, make_elbo_scorer, make_loo_scorer,
    make_renyi_scorer, renyi_elbo,
)


@dataclass
class BatchedGridResult:
    """Result of :func:`matlap_grid_batched` or :func:`matlap_adaptive_batched`.

    Attributes:
        best_lambda:  λ with the highest score.
        best_result:  :class:`BatchedCAVIResult` for the best λ.
        results:      List of (lambda, BatchedCAVIResult) for every λ tried,
                      sorted ascending by λ.
    """

    best_lambda: float
    best_result: BatchedCAVIResult
    results: list[tuple[float, BatchedCAVIResult]]


def matlap_grid_batched(
    Y: jax.Array,
    S: jax.Array,
    lambda_grid: jax.Array,
    *,
    a0: float = 1e-3,
    b0: float = 1e-3,
    max_iter: int = 200,
    tol: float = 1e-6,
    batch_size: int = 64,
    score_fn: str = "loo",
    alpha: float = 0.5,
    verbose: bool = False,
) -> BatchedGridResult:
    """Grid search over lambda using the full CAVI (batched) model.

    Evaluates :func:`matlap_batched` for each lambda in **decreasing** order
    (warm-starting each run from the previous posterior mean) and returns the
    solution with the highest score.

    Selection score:

    * ``score_fn="loo"`` (default): analytical Gaussian leave-one-out score.
      Unimodal for the batched model; peaks at the best-RMSE lambda.
    * ``score_fn="renyi"``: Rényi α-ELBO using the model-derived prior variance
      ``psi_sqrt_diag_j/λ`` (diagonal of Q/λ where Q = Ψ^{1/2} is the CAVI
      shape matrix).  This is the correct prior covariance for the row-factorised
      Gaussian approximation and approximates the marginal likelihood p(Y|λ).
    * ``score_fn="elbo"``: final ELBO at each grid point.  The ELBO peaks at
      the empirical-Bayes λ (same as :func:`matlap_batched`); not recommended
      for prediction-optimal lambda selection.

    Args:
        Y:            Observed matrix, shape (m, n).
        S:            Noise std devs, shape (m, n). ``jnp.inf`` where missing.
        lambda_grid:  1-D array of lambda values to evaluate.
        a0:           Gamma prior shape (default: 1e-3).
        b0:           Gamma prior rate (default: 1e-3).
        max_iter:     Maximum CAVI iterations per grid point (default 200).
        tol:          Convergence tolerance on relative ELBO change.
        batch_size:   Row batch size (controls peak memory, default 64).
        score_fn:     One of ``{"loo", "renyi", "elbo"}`` (default ``"loo"``).
        alpha:        Rényi order for ``score_fn="renyi"``; must satisfy 0≤α<1.
        verbose:      Print progress per grid point.

    Returns:
        :class:`BatchedGridResult` with best lambda and per-lambda results.
    """
    score_mode = score_fn.lower()
    if score_mode not in {"elbo", "loo", "renyi"}:
        raise ValueError(f"Unknown score_fn={score_fn!r}; expected 'elbo', 'loo', or 'renyi'.")
    if score_mode == "renyi" and not (0.0 <= alpha < 1.0):
        raise ValueError(f"alpha must be in [0, 1) for Rényi scoring, got {alpha}")

    Y = jnp.asarray(Y, dtype=jnp.float32)
    S = jnp.asarray(S, dtype=jnp.float32)

    lambda_vals = sorted([float(lv) for lv in lambda_grid], reverse=True)

    mu_init: jax.Array | None = None

    results: list[tuple[float, BatchedCAVIResult]] = []
    best_score = -float("inf")
    best_lam = lambda_vals[0]
    best_res: BatchedCAVIResult | None = None

    for lam_val in lambda_vals:
        warm = {"mu_init": mu_init} if mu_init is not None else {}
        res = matlap_batched(
            Y, S, a0=a0, b0=b0, max_iter=max_iter, tol=tol,
            batch_size=batch_size, lambda_val=lam_val, **warm,
        )
        mu_init = res.mu

        if score_mode == "elbo":
            score = float(res.elbo_trace[-1])
        elif score_mode == "loo":
            score = float(closed_form_loo(res.mu, res.sigma_diag, Y, S))
        else:
            pv = _get_prior_var(res, lam_val, delta_fallback=1e-6)
            score = float(renyi_elbo(res.mu, res.sigma_diag, pv, Y, S, alpha=alpha))

        if score > best_score:
            best_score = score
            best_lam = lam_val
            best_res = res

        results.append((lam_val, res))
        if verbose:
            print(
                f"  lambda={lam_val:.4f}  {score_mode.upper()}={score:.4f}  "
                f"ELBO={res.elbo_trace[-1]:.4f}  iters={res.n_iter}  converged={res.converged}"
            )

    results.sort(key=lambda x: x[0])
    return BatchedGridResult(
        best_lambda=best_lam,
        best_result=best_res,
        results=results,
    )


def matlap_adaptive_lowrank_isotropic(
    Y: jax.Array,
    S: jax.Array,
    *,
    lambda_start: float | None = None,
    lambda_min: float = 1e-4,
    rank: int = 50,
    a0: float = 1e-3,
    b0: float = 1e-3,
    max_iter: int = 50,
    tol: float = 1e-6,
    score_fn: str = "renyi",
    alpha: float = 0.5,
    n_folds: int = 3,
    patience: int = 2,
    verbose: bool = False,
) -> LowRankIsotropicGridResult:
    """Adaptive golden-ratio λ search for the low-rank+isotropic CAVI model.

    Starts from a high ``lambda_start`` and repeatedly multiplies λ by
    1/φ ≈ 0.618, scoring each converged CAVI run and stopping when the score
    fails to improve for ``patience`` consecutive steps.  Each run is
    warm-started from the previous λ.

    Compared with :func:`matlap_grid_lowrank_isotropic`, this avoids the need
    to specify a grid and automatically adapts to the scale of the problem.

    The search strategy is fully modular: supply any
    :func:`~matlap.adaptive.adaptive_lambda_search`-compatible ``fit_fn`` and
    ``score_fn`` for custom use cases.

    Args:
        Y:             Observed matrix, shape (m, n). NaN/any value where missing.
        S:             Known standard errors, shape (m, n). ``jnp.inf`` where missing.
        lambda_start:  Starting λ. If None, set to 100 × data heuristic.
        lambda_min:    Hard lower bound on λ (safety stop).
        rank:          Rank of the factor subspace (default 50).
        a0:            Gamma prior shape.
        b0:            Gamma prior rate.
        max_iter:      Maximum CAVI iterations per λ value.
        tol:           Convergence tolerance on relative ELBO change.
        score_fn:      One of ``{"renyi", "elbo", "loo", "cv"}`` (default ``"renyi"``).
        alpha:         Rényi order for ``score_fn="renyi"``; must satisfy 0 ≤ α < 1.
        n_folds:       Number of CV folds for ``score_fn="cv"``.
        patience:      Stop after this many consecutive non-improving reductions.
        verbose:       Print progress per λ step.

    Returns:
        :class:`LowRankIsotropicGridResult` with best lambda and per-lambda results.
    """
    def fit_fn(Y_, S_, lam, **warm):
        return matlap_lowrank_isotropic(
            Y_, S_, lam, rank=rank, a0=a0, b0=b0, max_iter=max_iter, tol=tol, **warm
        )

    scorer = _make_scorer(score_fn, alpha=alpha, n_folds=n_folds, fit_fn=fit_fn)

    best_lam, best_res, results = adaptive_lambda_search(
        Y, S, fit_fn, scorer,
        extract_warm_state=iso_warm_state,
        lambda_start=lambda_start,
        lambda_min=lambda_min,
        patience=patience,
        verbose=verbose,
    )
    return LowRankIsotropicGridResult(
        best_lambda=best_lam,
        best_result=best_res,
        results=results,
    )


def matlap_adaptive_lowrank(
    Y: jax.Array,
    S: jax.Array,
    *,
    lambda_start: float | None = None,
    lambda_min: float = 1e-4,
    rank: int = 50,
    a0: float = 1e-3,
    b0: float = 1e-3,
    max_iter: int = 50,
    tol: float = 1e-6,
    score_fn: str = "renyi",
    alpha: float = 0.5,
    n_folds: int = 3,
    patience: int = 2,
    verbose: bool = False,
) -> LowRankGridResult:
    """Adaptive golden-ratio λ search for the low-rank CAVI model.

    Identical strategy to :func:`matlap_adaptive_lowrank_isotropic` but uses
    :func:`matlap_lowrank` as the fitting function (no isotropic component).
    When ``score_fn="renyi"`` the off-subspace prior variance uses
    ``delta=1e-6`` as an approximation.

    Args:
        Y:             Observed matrix, shape (m, n). NaN/any value where missing.
        S:             Known standard errors, shape (m, n). ``jnp.inf`` where missing.
        lambda_start:  Starting λ. If None, set to 100 × data heuristic.
        lambda_min:    Hard lower bound on λ (safety stop).
        rank:          Rank of the factor subspace (default 50).
        a0:            Gamma prior shape.
        b0:            Gamma prior rate.
        max_iter:      Maximum CAVI iterations per λ value.
        tol:           Convergence tolerance on relative ELBO change.
        score_fn:      One of ``{"renyi", "elbo", "loo", "cv"}`` (default ``"renyi"``).
        alpha:         Rényi order for ``score_fn="renyi"``; must satisfy 0 ≤ α < 1.
        n_folds:       Number of CV folds for ``score_fn="cv"``.
        patience:      Stop after this many consecutive non-improving reductions.
        verbose:       Print progress per λ step.

    Returns:
        :class:`LowRankGridResult` with best lambda and per-lambda results.
    """
    def fit_fn(Y_, S_, lam, **warm):
        return matlap_lowrank(
            Y_, S_, lam, rank=rank, a0=a0, b0=b0, max_iter=max_iter, tol=tol, **warm
        )

    scorer = _make_scorer(score_fn, alpha=alpha, n_folds=n_folds, fit_fn=fit_fn)

    best_lam, best_res, results = adaptive_lambda_search(
        Y, S, fit_fn, scorer,
        extract_warm_state=lowrank_warm_state,
        lambda_start=lambda_start,
        lambda_min=lambda_min,
        patience=patience,
        verbose=verbose,
    )
    return LowRankGridResult(
        best_lambda=best_lam,
        best_result=best_res,
        results=results,
    )


def matlap_adaptive_batched(
    Y: jax.Array,
    S: jax.Array,
    *,
    lambda_start: float | None = None,
    lambda_min: float = 1e-4,
    a0: float = 1e-3,
    b0: float = 1e-3,
    max_iter: int = 200,
    tol: float = 1e-6,
    batch_size: int = 64,
    score_fn: str = "loo",
    alpha: float = 0.5,
    n_folds: int = 3,
    patience: int = 2,
    verbose: bool = False,
) -> BatchedGridResult:
    """Adaptive golden-ratio λ search for the full CAVI (batched) model.

    Identical search strategy to :func:`matlap_adaptive_lowrank` but uses
    :func:`matlap_batched` as the fitting function.  Each CAVI run is fixed
    at the candidate λ (``lambda_val``) and warm-started from the previous
    posterior mean via ``mu_init``.

    The empirical-Bayes λ update inside :func:`matlap_batched` inflates λ on
    low-rank data (all n dimensions contribute to Tr(√Ψ), not just the r
    signal dims).  LOO scoring (default) bypasses this mis-calibration and
    recovers the correct λ.  The Rényi scorer is **not recommended** for the
    batched model: it requires a prior-variance proxy derived from the
    current run's posterior, which becomes self-referential as λ→∞ and
    produces a non-unimodal score surface.

    Args:
        Y:             Observed matrix, shape (m, n).
        S:             Noise std matrix, shape (m, n). ``jnp.inf`` where missing.
        lambda_start:  Starting λ. If None, set to 100 × data heuristic.
        lambda_min:    Hard lower bound on λ (safety stop).
        a0:            Gamma prior shape (default: 1e-3).
        b0:            Gamma prior rate (default: 1e-3).
        max_iter:      Maximum CAVI iterations per λ value.
        tol:           Convergence tolerance on relative ELBO change.
        batch_size:    Row batch size (controls peak memory, default 64).
        score_fn:      One of ``{"loo", "renyi", "elbo", "cv"}`` (default ``"loo"``).
                       ``"loo"`` is recommended; ``"renyi"`` is not recommended for
                       the batched model (non-unimodal score surface).
        alpha:         Rényi order for ``score_fn="renyi"``; 0 ≤ α < 1.
        n_folds:       Number of CV folds for ``score_fn="cv"``.
        patience:      Stop after this many consecutive non-improving reductions.
        verbose:       Print progress per λ step.

    Returns:
        :class:`BatchedGridResult` with best lambda and per-lambda results.
    """
    def fit_fn(Y_, S_, lam, **warm):
        return matlap_batched(
            Y_, S_,
            a0=a0, b0=b0, max_iter=max_iter, tol=tol,
            batch_size=batch_size, lambda_val=lam, **warm,
        )

    scorer = _make_scorer(score_fn, alpha=alpha, n_folds=n_folds, fit_fn=fit_fn)

    best_lam, best_res, results = adaptive_lambda_search(
        Y, S, fit_fn, scorer,
        extract_warm_state=batched_warm_state,
        lambda_start=lambda_start,
        lambda_min=lambda_min,
        patience=patience,
        verbose=verbose,
    )
    return BatchedGridResult(
        best_lambda=best_lam,
        best_result=best_res,
        results=results,
    )


def _make_scorer(score_fn: str, *, alpha: float, n_folds: int, fit_fn):
    """Resolve a score_fn string to a callable scorer."""
    mode = score_fn.lower()
    if mode == "renyi":
        return make_renyi_scorer(alpha=alpha)
    if mode == "elbo":
        return make_elbo_scorer()
    if mode == "loo":
        return make_loo_scorer()
    if mode == "cv":
        from .cv import make_cv_scorer
        return make_cv_scorer(fit_fn, n_folds=n_folds)
    raise ValueError(
        f"Unknown score_fn={score_fn!r}; expected one of 'renyi', 'elbo', 'loo', 'cv'."
    )
