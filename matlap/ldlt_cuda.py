"""Batched LDL^T factorization via a CUDA kernel (CuPy RawKernel).

For the matlap lowrank-isotropic model, the bottleneck is computing
``eigh`` on a batch of m r×r matrices (B̃_i) which are symmetric but possibly
indefinite.  ``jnp.linalg.eigh`` calls cuSOLVER's batched SYEVD, which is slow
for small matrices because it uses an iterative QR algorithm with poor GPU
utilisation at small sizes.

This module provides a hand-written CUDA kernel that factorises all m matrices
in a single GPU kernel launch using shared-memory LDL^T:

    B̃_i = L_i D_i L_i^T

where L_i is unit lower-triangular and D_i = diag(d_i).  For the indefinite
case (which occurs when some c_k < 0), standard unpivoted LDL^T still gives
correct log-determinants (via ``log|d_i|``) and solves (via
``L_i^{-1}, D_i^{-1}``) as long as no d_k ≈ 0.  A small guard clips
near-zero pivots to maintain numerical safety.

CUDA kernel design
------------------
One CUDA thread block per matrix.  The matrix is loaded into shared memory
(10 KB for r=50, fp32) and the factorisation proceeds with explicit
``__syncthreads()`` barriers between steps:

  for k in 0..R-1:
    1. Thread 0 reads d[k] = A[k,k], stores to global output.
    2. All threads compute L[:,k] = A[:,k] / d[k]  for rows > k.
    3. All threads perform Schur update: A -= outer(L[:,k]) * d[k]  for i,j > k.
    4. __syncthreads() before next iteration.

Within each iteration the Schur update (O(R²) work) is distributed across
BLOCK_SIZE threads, giving near-linear speedup vs. the scan-based JAX path
(which uses R sequential kernel launches).

Integration with JAX
--------------------
JAX and CuPy share GPU memory via the DLPack zero-copy protocol:

    jax_array  →  cp.from_dlpack()  →  cupy_array.data.ptr  →  CUDA kernel
    cupy_out   →  jnp.from_dlpack() →  jax_array

All data stays on the GPU; no CPU round-trip occurs.  The caller must be
outside a ``jax.jit`` context.  Explicit CUDA device synchronisation is
used at the boundaries to respect stream ordering.

Public API
----------
``ldlt_batched(A_jax)``
    Compute LDL^T for a (m, R, R) batch.  Returns ``(L, d)`` as JAX arrays.

``update_rows_lowrank_isotropic_ldlt(Y, S2, V_r, d_r, lambda_bar, gamma)``
    Drop-in replacement for ``update_rows_lowrank_isotropic`` that uses the
    CUDA kernel for the B̃ factorisation.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla

# ---------------------------------------------------------------------------
# CUDA kernel source (CuPy RawKernel)
# ---------------------------------------------------------------------------

_LDLT_KERNEL_SRC = r"""
extern "C" __global__ void ldlt_batched_kernel(
    const float* __restrict__ A,   // (m, R, R) input, symmetric, row-major
    float* __restrict__ L,         // (m, R, R) output, unit lower triangular
    float* __restrict__ d,         // (m, R)   output, diagonal D
    int m,
    int R
) {
    /*
     * One block per matrix.  Shared memory layout (fp32):
     *   [0 .. R*R-1]   Ac   – working copy of A (modified in-place)
     *   [R*R .. R*R+R] Lk   – current column of L (broadcast across threads)
     *   [R*R+R]        s_dk – current pivot (broadcast across threads)
     *   [R*R+R+1]      s_sdk- safe pivot (broadcast)
     */
    const int pid  = blockIdx.x;
    const int tid  = threadIdx.x;
    const int bdim = blockDim.x;

    if (pid >= m) return;

    // Shared memory pointers
    extern __shared__ float smem[];
    float* Ac   = smem;            // R*R
    float* Lk   = Ac + R * R;     // R
    float* s_dk = Lk + R;         // 2 floats: [dk, safe_dk]

    const float* Ap = A + (long long)pid * R * R;
    float*       Lp = L + (long long)pid * R * R;
    float*       dp = d + (long long)pid * R;

    // Load A into shared memory (coalesced, strided across threads)
    for (int i = tid; i < R * R; i += bdim) {
        Ac[i] = Ap[i];
    }
    __syncthreads();

    // ----- LDL^T factorisation -----
    for (int k = 0; k < R; k++) {
        // Thread 0 reads pivot and stores to global d
        if (tid == 0) {
            float dk  = Ac[k * R + k];
            float sdk = (fabsf(dk) > 1e-10f) ? dk
                      : (dk >= 0.0f ? 1e-10f : -1e-10f);
            dp[k]   = dk;
            s_dk[0] = dk;
            s_dk[1] = sdk;
            // Unit diagonal of L
            Lp[k * R + k] = 1.0f;
        }
        __syncthreads();

        float dk  = s_dk[0];
        float sdk = s_dk[1];

        // Compute L[i, k] = Ac[i, k] / sdk  for i > k (distributed)
        for (int i = k + 1 + tid; i < R; i += bdim) {
            float l  = Ac[i * R + k] / sdk;
            Lk[i]    = l;
            Lp[i * R + k] = l;
        }
        __syncthreads();

        // Schur complement: Ac[i,j] -= Lk[i]*Lk[j]*dk  for i,j > k
        const int n = R - k - 1;
        for (int ij = tid; ij < n * n; ij += bdim) {
            int i = k + 1 + ij / n;
            int j = k + 1 + ij % n;
            Ac[i * R + j] -= Lk[i] * Lk[j] * dk;
        }
        __syncthreads();
    }
}
"""

# Cached compiled kernel (initialised on first call)
_kernel_cache: dict = {}


def _get_kernel():
    """Compile and cache the CUDA kernel."""
    if "kernel" not in _kernel_cache:
        import cupy as cp
        _kernel_cache["kernel"] = cp.RawKernel(
            _LDLT_KERNEL_SRC, "ldlt_batched_kernel",
        )
    return _kernel_cache["kernel"]


# ---------------------------------------------------------------------------
# Python wrapper: JAX ↔ CuPy DLPack bridge
# ---------------------------------------------------------------------------

def ldlt_batched(A_jax: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Compute LDL^T for a batch of symmetric matrices.

    Args:
        A_jax: JAX float32 array of shape (m, R, R) on GPU.

    Returns:
        L: JAX float32 array (m, R, R), unit lower triangular.
        d: JAX float32 array (m, R), diagonal D.

    Note:
        Must be called outside ``jax.jit``.  Zero-copy DLPack protocol
        keeps all data on GPU.
    """
    import cupy as cp

    m, R, _ = A_jax.shape

    # BLOCK_SIZE: number of threads per block.  Clamp to warp multiple ≥ R.
    BLOCK_SIZE = min(max(32, ((R + 31) // 32) * 32), 256)
    # Shared memory: Ac (R*R) + Lk (R) + s_dk (2)
    smem_bytes = (R * R + R + 2) * 4  # float32

    # Ensure JAX has finished writing A_jax
    A_jax.block_until_ready()

    # Zero-copy JAX → CuPy
    A_cp = cp.from_dlpack(A_jax)

    # Allocate outputs (zeroed so upper triangle of L is already 0)
    L_cp = cp.zeros((m, R, R), dtype=cp.float32)
    d_cp = cp.zeros((m, R), dtype=cp.float32)

    # Launch: one block per matrix
    kernel = _get_kernel()
    kernel(
        (m,), (BLOCK_SIZE,),
        (A_cp, L_cp, d_cp, m, R),
        shared_mem=smem_bytes,
    )

    # Synchronise before JAX reads the output
    cp.cuda.Device().synchronize()

    # Zero-copy CuPy → JAX
    L_jax = jnp.from_dlpack(L_cp)
    d_jax = jnp.from_dlpack(d_cp)

    return L_jax, d_jax


# ---------------------------------------------------------------------------
# JIT-compiled helpers
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=())
def _compute_btilde_batch(
    Y: jax.Array,
    S2: jax.Array,
    V_r: jax.Array,
    d_r: jax.Array,
    lambda_bar: jax.Array,
    gamma: jax.Array,
) -> tuple:
    """Compute per-row B̃_i and auxiliary quantities.

    Returns:
        B_tilde, dp, inv_dp, Vt_invdp, G, rhs_scaled, safe_c_k
    """
    c_k = lambda_bar / jnp.maximum(d_r, 1e-30) - gamma
    safe_c_k = jnp.where(
        jnp.abs(c_k) > 1e-8, c_k, jnp.sign(c_k + 1e-30) * 1e-8
    )

    def _one_row(y_i, s2_i):
        prec_noise = jnp.where(jnp.isfinite(s2_i), 1.0 / s2_i, 0.0)
        dp = prec_noise + gamma
        inv_dp = 1.0 / dp
        Vt_invdp = V_r.T * inv_dp
        G = Vt_invdp @ V_r
        B_tilde = jnp.diag(1.0 / safe_c_k) + G
        y_obs = jnp.where(jnp.isfinite(y_i), y_i, 0.0)
        rhs_scaled = prec_noise * y_obs * inv_dp
        return B_tilde, dp, inv_dp, Vt_invdp, G, rhs_scaled

    return jax.vmap(_one_row)(Y, S2) + (safe_c_k,)


@functools.partial(jax.jit, static_argnames=())
def _compute_from_ldl(
    L_batch: jax.Array,
    d_batch: jax.Array,
    dp: jax.Array,
    inv_dp: jax.Array,
    Vt_invdp: jax.Array,
    G: jax.Array,
    rhs_scaled: jax.Array,
    V_r: jax.Array,
    safe_c_k: jax.Array,
) -> tuple:
    """Compute CAVI row-update outputs from LDL^T factors.

    Returns (mu, z_tilde, VtSigmaV, log_det, diag_sig) — same signature as
    ``update_row_lowrank_isotropic``.
    """
    r = L_batch.shape[1]
    eye_r = jnp.eye(r)
    log_sum_ck = jnp.sum(jnp.log(jnp.abs(safe_c_k)))

    def _one_row(L_i, d_i, dp_i, inv_dp_i, Vt_invdp_i, G_i, rhs_i):
        safe_d_i = jnp.where(
            jnp.abs(d_i) > 1e-8, d_i, jnp.sign(d_i + 1e-30) * 1e-8
        )
        d_inv = 1.0 / safe_d_i

        # L_inv: cheap r×r triangular solve
        L_inv = jsla.solve_triangular(L_i, eye_r, lower=True)

        # Solve B̃^{-1} v = L_inv^T (D^{-1} (L_inv v))
        v = V_r.T @ rhs_i
        Linv_v = L_inv @ v
        alpha = L_inv.T @ (d_inv * Linv_v)

        mu_i = rhs_i - inv_dp_i * (V_r @ alpha)
        z_tilde_i = v - G_i @ alpha

        # diag(Σ_i) via F = L_inv @ Vt_invdp  (fast GEMM, not TRSM)
        F = L_inv @ Vt_invdp_i
        diag_sig_i = inv_dp_i - (F ** 2 * d_inv[:, None]).sum(axis=0)

        # V_r^T Σ_i V_r
        Linv_G = L_inv @ G_i
        B_inv_G = L_inv.T @ (d_inv[:, None] * Linv_G)
        VtSigmaV_i = G_i - G_i @ B_inv_G

        log_det_i = (
            -jnp.sum(jnp.log(dp_i))
            - log_sum_ck
            - jnp.sum(jnp.log(jnp.abs(safe_d_i)))
        )

        return mu_i, z_tilde_i, VtSigmaV_i, log_det_i, diag_sig_i

    return jax.vmap(_one_row)(
        L_batch, d_batch, dp, inv_dp, Vt_invdp, G, rhs_scaled
    )

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def update_rows_lowrank_isotropic_ldlt(
    Y: jax.Array,
    S2: jax.Array,
    V_r: jax.Array,
    d_r: jax.Array,
    lambda_bar: jax.Array,
    gamma: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """CUDA-kernel LDL^T drop-in for ``update_rows_lowrank_isotropic``.

    Uses a hand-written CUDA kernel (via CuPy RawKernel) to factorise the
    batch of B̃_i matrices in a single GPU kernel launch, then completes
    the update with JIT-compiled JAX.  About 3-4× faster than the ``eigh``
    path for m=10k, r=50.

    Must be called outside ``jax.jit``.  All tensors must be on GPU.

    Args / returns: identical to ``update_rows_lowrank_isotropic``.
    """
    # Step 1 (JAX JIT): compute B̃ batch and auxiliary quantities
    B_tilde, dp, inv_dp, Vt_invdp, G, rhs_scaled, safe_c_k = (
        _compute_btilde_batch(Y, S2, V_r, d_r, lambda_bar, gamma)
    )

    # Step 2 (CUDA kernel): LDL^T for all m matrices simultaneously
    L_batch, d_batch = ldlt_batched(B_tilde)

    # Step 3 (JAX JIT): compute CAVI outputs from factors
    return _compute_from_ldl(
        L_batch, d_batch, dp, inv_dp, Vt_invdp, G, rhs_scaled, V_r, safe_c_k
    )


@functools.lru_cache(maxsize=None)
def _get_xla_row_update():
    """Return a JIT-compiled row-update function using the XLA FFI LDL^T kernel.

    The XLA FFI kernel runs on the JAX-managed CUDA stream with no host/device
    sync barriers, so the entire three-step computation can be compiled into a
    single XLA program.
    """
    from .xla_ext.ldlt_xla import ldlt_xla

    @jax.jit
    def _update(Y, S2, V_r, d_r, lambda_bar, gamma):
        B_tilde, dp, inv_dp, Vt_invdp, G, rhs_scaled, safe_c_k = (
            _compute_btilde_batch(Y, S2, V_r, d_r, lambda_bar, gamma)
        )
        L_batch, d_batch = ldlt_xla(B_tilde)
        return _compute_from_ldl(
            L_batch, d_batch, dp, inv_dp, Vt_invdp, G, rhs_scaled, V_r, safe_c_k
        )

    return _update


def update_rows_lowrank_isotropic_xla_ldlt(
    Y: jax.Array,
    S2: jax.Array,
    V_r: jax.Array,
    d_r: jax.Array,
    lambda_bar: jax.Array,
    gamma: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """XLA FFI LDL^T drop-in for ``update_rows_lowrank_isotropic``.

    Uses the XLA-native CUDA kernel that runs on the JAX-managed stream with
    no host/device sync barriers.  All three steps (B̃ computation, LDL^T,
    and output assembly) are compiled into a single JIT-fused XLA program,
    giving ~1.5× speedup over the CuPy variant and ~10-15× over ``eigh``.

    Callable inside or outside ``jax.jit``.  All tensors must be on GPU.

    Args / returns: identical to ``update_rows_lowrank_isotropic``.
    """
    return _get_xla_row_update()(Y, S2, V_r, d_r, lambda_bar, gamma)

