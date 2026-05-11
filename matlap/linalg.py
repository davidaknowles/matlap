"""
Linear algebra utilities for matlap.

All operations avoid jnp.linalg.inv; use eigh for symmetric matrix square roots
and Cholesky decomposition for solving linear systems.

Low-rank utilities
------------------
``rsvd``                 -- randomized truncated SVD via power iteration
``approx_nuclear_norm``  -- differentiable approximate nuclear norm (rSVD)
``update_row_lowrank``   -- O(nr) Woodbury row update for low-rank CAVI
``update_rows_lowrank``  -- vmapped + JIT version of ``update_row_lowrank``
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla


class SqrtDecomp(NamedTuple):
    """Eigendecomposition of a symmetric PSD matrix A = vecs @ diag(vals) @ vecs.T.

    Attributes:
        vals: eigenvalues, shape (n,), non-negative
        vecs: eigenvectors (columns), shape (n, n), orthonormal
        sqrt_vals: sqrt of eigenvalues, shape (n,)
    """
    vals: jax.Array
    vecs: jax.Array
    sqrt_vals: jax.Array


def matrix_sqrt_eigh(A: jax.Array) -> SqrtDecomp:
    """Compute the symmetric matrix square root A^{1/2} via eigendecomposition.

    A^{1/2} = vecs @ diag(sqrt_vals) @ vecs.T

    Args:
        A: symmetric PSD matrix, shape (n, n)

    Returns:
        SqrtDecomp with (vals, vecs, sqrt_vals)
    """
    vals, vecs = jnp.linalg.eigh(A)
    vals = jnp.maximum(vals, 0.0)  # numerical safety
    sqrt_vals = jnp.sqrt(vals)
    return SqrtDecomp(vals=vals, vecs=vecs, sqrt_vals=sqrt_vals)


def trace_sqrt(decomp: SqrtDecomp) -> jax.Array:
    """Tr(A^{1/2}) = sum of sqrt eigenvalues."""
    return decomp.sqrt_vals.sum()


@jax.jit
def update_row(
    y_i: jax.Array,
    s2_i: jax.Array,
    q_sqrt_vals: jax.Array,
    q_vecs: jax.Array,
    lambda_bar: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute variational posterior for one row.

    Computes the optimal variational parameters (mu_i, Sigma_i) for row i:

        Sigma_i = (diag(1/s2_i) + lambda_bar * Q^{-1})^{-1}
        mu_i    = Sigma_i @ (y_i / s2_i)

    where Q^{-1} = vecs @ diag(1/q_sqrt_vals) @ vecs.T.

    Missing observations are handled by setting s2_i[j] = inf, which makes
    the corresponding precision 0. The RHS y_i/s2_i is safe because
    jnp.where masks out NaN y values before dividing.

    Args:
        y_i:           observations for row i, shape (n,)
        s2_i:          observation variances for row i, shape (n,); inf = missing
        q_sqrt_vals:   sqrt eigenvalues of Psi (= eigenvalues of Q), shape (n,)
        q_vecs:        eigenvectors of Psi, shape (n, n)
        lambda_bar:    E_q[lambda], scalar

    Returns:
        mu_i:       posterior mean, shape (n,)
        Sigma_i:    posterior covariance, shape (n, n)
        log_det_i:  log|Sigma_i|, scalar (used for ELBO)
    """
    n = y_i.shape[0]
    prec_noise = jnp.where(jnp.isfinite(s2_i), 1.0 / s2_i, 0.0)  # shape (n,)

    # Precision matrix A_i = diag(prec_noise) + lambda_bar * Q^{-1}
    # Q^{-1} = vecs @ diag(1/q_sqrt_vals) @ vecs.T
    # Numerically: A_i = diag(prec_noise) + vecs @ diag(lambda_bar/q_sqrt_vals) @ vecs.T
    inv_q_vals = lambda_bar / jnp.maximum(q_sqrt_vals, 1e-30)
    A_i = jnp.diag(prec_noise) + (q_vecs * inv_q_vals) @ q_vecs.T

    # Cholesky decomposition of A_i (PD because lambda_bar * Q^{-1} is PD)
    cho = jsla.cho_factor(A_i)

    # mu_i = A_i^{-1} @ (prec_noise * y_i)   (safe: prec_noise=0 where y_i may be NaN)
    rhs_mu = prec_noise * jnp.where(jnp.isfinite(y_i), y_i, 0.0)
    mu_i = jsla.cho_solve(cho, rhs_mu)

    # Sigma_i = A_i^{-1}  (solve against identity columns)
    Sigma_i = jsla.cho_solve(cho, jnp.eye(n))

    # log|Sigma_i| = -log|A_i| = -2 * sum(log diag(L))
    L = cho[0]
    log_det_i = -2.0 * jnp.sum(jnp.log(jnp.diag(L)))

    return mu_i, Sigma_i, log_det_i


# Vectorised version: apply update_row over all rows simultaneously
_update_rows_vmap = jax.vmap(update_row, in_axes=(0, 0, None, None, None))

update_rows = jax.jit(_update_rows_vmap)


# ============================================================================
# Low-rank utilities
# ============================================================================


def rsvd(
    A: jax.Array,
    rank: int,
    n_iter: int = 4,
    key: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Randomized truncated SVD via power iteration.

    Computes approximate top-``rank`` singular triplets of A using the
    randomized range-finder algorithm (Halko et al. 2011).

    Time O(mnr) vs O(mn²) for full SVD.

    Args:
        A:      Input matrix, shape (m, n).
        rank:   Number of singular components to retain.
        n_iter: Power-iteration steps (default 4; more = more accurate).
        key:    JAX random key.  A fixed seed is used when ``None``.

    Returns:
        U_r:   Left singular vectors, shape (m, rank).
        s_r:   Singular values, shape (rank,), descending.
        Vt_r:  Right singular vectors transposed, shape (rank, n).
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    m, n = A.shape
    oversample = min(10, min(m, n) - rank)
    k = rank + oversample

    # Random Gaussian test matrix
    Omega = jax.random.normal(key, (n, k))

    # Power iteration to improve accuracy
    Z = A @ Omega
    for _ in range(n_iter):
        Z = A @ (A.T @ Z)

    # Orthonormalize the range
    Q, _ = jnp.linalg.qr(Z)  # (m, k)

    # Small SVD of projected matrix
    B = Q.T @ A  # (k, n)
    Ub, s_r, Vt_r = jnp.linalg.svd(B, full_matrices=False)

    # Recover left vectors
    U_r = Q @ Ub
    return U_r[:, :rank], s_r[:rank], Vt_r[:rank, :]


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def approx_nuclear_norm(A: jax.Array, rank: int, n_iter: int = 4) -> jax.Array:
    """Differentiable approximate nuclear norm via rSVD.

    Returns the sum of the top-``rank`` singular values of A, which is a
    differentiable lower bound on the true nuclear norm ‖A‖_*.  The gradient
    w.r.t. A is U_r V_r^T (the standard nuclear-norm subgradient from the
    top-r components).

    Args:
        A:      Input matrix, shape (m, n).
        rank:   Number of singular components to use.
        n_iter: Power-iteration steps (default 4).

    Returns:
        Approximate nuclear norm (sum of top-r singular values), scalar.
    """
    _, s_r, _ = rsvd(A, rank, n_iter)
    return s_r.sum()


def _approx_nuclear_norm_fwd(A, rank, n_iter):
    U_r, s_r, Vt_r = rsvd(A, rank, n_iter)
    return s_r.sum(), (U_r, Vt_r)


def _approx_nuclear_norm_bwd(rank, n_iter, residuals, g):
    U_r, Vt_r = residuals
    # Gradient of ||A||_* w.r.t. A is U V^T (top-r subgradient)
    return (g * (U_r @ Vt_r),)


approx_nuclear_norm.defvjp(_approx_nuclear_norm_fwd, _approx_nuclear_norm_bwd)


def update_row_lowrank(
    y_i: jax.Array,
    s2_i: jax.Array,
    V_r: jax.Array,
    d_r: jax.Array,
    lambda_bar: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Woodbury row update for the low-rank CAVI model.

    Restricts the variational posterior to rank-r covariances
    Σ_i = V_r A_r^{(i)−1} V_r^T  where V_r ∈ R^{n×r}.

    Posterior in factor space (z_i = V_r^T x_i):

        A_r^{(i)} = diag(λ/d_r) + V_r^T diag(p_i) V_r   [r×r]
        z_i       = A_r^{-1} V_r^T (p_i ⊙ y_i)

    Args:
        y_i:        Observations for row i, shape (n,).
        s2_i:       Observation variances, shape (n,); ``inf`` = missing.
        V_r:        Factor loading matrix, shape (n, r); orthonormal columns.
        d_r:        Sqrt-eigenvalues of Ψ_r (> 0), shape (r,).
        lambda_bar: E_q[λ], scalar.

    Returns:
        z_i:        Posterior mean in factor space, shape (r,).
        A_r_inv:    Posterior covariance in factor space, shape (r, r).
        log_det_i:  log|A_r^{-1}|, scalar (for ELBO).
        diag_sig_i: Diagonal of Σ_i in original space, shape (n,).
    """
    r = V_r.shape[1]
    prec_noise = jnp.where(jnp.isfinite(s2_i), 1.0 / s2_i, 0.0)  # (n,)

    # A_r = diag(lambda_bar/d_r) + V_r^T diag(prec_noise) V_r   [r×r]
    prior_prec_r = lambda_bar / jnp.maximum(d_r, 1e-30)  # (r,)
    VtP = V_r.T * prec_noise  # (r, n)  == V_r.T @ diag(prec_noise)
    A_r = jnp.diag(prior_prec_r) + VtP @ V_r  # (r, r)

    cho = jsla.cho_factor(A_r)

    # rhs = V_r^T (prec_noise ⊙ y_i)
    rhs = VtP @ jnp.where(jnp.isfinite(y_i), y_i, 0.0)  # (r,)
    z_i = jsla.cho_solve(cho, rhs)  # (r,)

    # A_r_inv = A_r^{-1}
    A_r_inv = jsla.cho_solve(cho, jnp.eye(r))  # (r, r)

    # log|A_r^{-1}| = -log|A_r| = -2 sum log diag(L)
    L = cho[0]
    log_det_i = -2.0 * jnp.sum(jnp.log(jnp.diag(L)))

    # diag(Sigma_i) in original space: diag(V_r A_r^{-1} V_r^T)
    VA = V_r @ A_r_inv  # (n, r)
    diag_sig_i = jnp.sum(VA * V_r, axis=1)  # (n,)

    return z_i, A_r_inv, log_det_i, diag_sig_i


# Vmapped + JIT version for all rows simultaneously
update_rows_lowrank = jax.jit(
    jax.vmap(update_row_lowrank, in_axes=(0, 0, None, None, None))
)
