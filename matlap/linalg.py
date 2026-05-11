"""
Linear algebra utilities for matlap.

All operations avoid jnp.linalg.inv; use eigh for symmetric matrix square roots
and Cholesky decomposition for solving linear systems.
"""

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
