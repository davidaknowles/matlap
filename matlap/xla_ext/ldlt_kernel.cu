/* Batched LDL^T factorisation — XLA FFI custom call.
 *
 * Each CUDA block handles one r×r matrix. Threads within the block
 * parallelise the Schur-complement rank-1 update (O(r²) work per step).
 * The r sequential column-elimination steps are serialised with
 * __syncthreads() barriers.
 *
 * Memory layout: row-major (C order), matrices packed as (m, r, r).
 */

#include <cuda_runtime.h>
#include <math.h>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

// ---------------------------------------------------------------------------
// CUDA kernel
// ---------------------------------------------------------------------------

__global__ void ldlt_batched_kernel(
    const float* __restrict__ A_in,
    float*       __restrict__ L_out,
    float*       __restrict__ d_out,
    int r)
{
    // One block per matrix.
    int mat = blockIdx.x;
    int tid = threadIdx.x;
    int T   = blockDim.x;

    // Shared working copy of the matrix (r×r floats).
    extern __shared__ float S[];

    const float* A = A_in + (long long)mat * r * r;
    float*       L = L_out + (long long)mat * r * r;
    float*       d = d_out + (long long)mat * r;

    // Load A → shared memory (coalesced strided load).
    for (int idx = tid; idx < r * r; idx += T)
        S[idx] = A[idx];
    __syncthreads();

    // LDL^T elimination: r sequential pivot steps.
    for (int k = 0; k < r; k++) {
        float dk = S[k * r + k];
        // Regularise tiny or zero pivots.
        float safe_dk = (fabsf(dk) < 1e-8f)
                        ? (dk >= 0.0f ? 1e-8f : -1e-8f)
                        : dk;

        // Thread 0 writes d[k].
        if (tid == 0)
            d[k] = safe_dk;

        // Write column k of L (L is unit lower-triangular).
        // Thread tid handles row i = tid, tid+T, tid+2T, ...
        for (int i = tid; i < r; i += T) {
            float val;
            if      (i <  k) val = 0.0f;
            else if (i == k) val = 1.0f;
            else             val = S[i * r + k] / safe_dk;
            L[i * r + k] = val;
        }
        __syncthreads();  // ensure all reads from S[:,k] finish before update

        // Schur complement: S[i,j] -= S[i,k] * S[j,k] / safe_dk
        // for i,j in {k+1, ..., r-1}.
        int n = r - k - 1;           // number of rows/cols to update
        int total = n * n;
        for (int idx = tid; idx < total; idx += T) {
            int ii = idx / n + k + 1;
            int jj = idx % n + k + 1;
            S[ii * r + jj] -= S[ii * r + k] * S[jj * r + k] / safe_dk;
        }
        __syncthreads();  // all updates done before next pivot
    }
}

// ---------------------------------------------------------------------------
// XLA FFI handler
// ---------------------------------------------------------------------------

// The XLA FFI decodes PlatformStream<cudaStream_t> as cudaStream_t (the raw
// pointer type T), so the function receives the raw handle directly.
static ffi::Error LdltBatched(
    cudaStream_t                       stream,
    ffi::Buffer<ffi::F32>              A,
    ffi::Result<ffi::Buffer<ffi::F32>> L,
    ffi::Result<ffi::Buffer<ffi::F32>> d)
{
    auto dims = A.dimensions();
    if (dims.size() != 3)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "ldlt_batched: A must be rank-3 (m, r, r)");

    int m = (int)dims[0];
    int r = (int)dims[1];

    if ((int)dims[2] != r)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "ldlt_batched: A must be square in last two dims");

    // 256 threads per block; shared memory = r*r floats.
    int threads    = 256;
    int smem_bytes = r * r * sizeof(float);

    ldlt_batched_kernel<<<m, threads, smem_bytes, stream>>>(
        A.typed_data(),
        L->typed_data(),
        d->typed_data(),
        r);

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    LdltBatchedFfi,
    LdltBatched,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>());
