# matlap

**Bayesian matrix completion via Coordinate Ascent Variational Inference (CAVI)** with a Matrix Laplace prior.

Given an observed matrix `Y` with per-entry heteroscedastic noise `S`, `matlap` recovers a low-rank posterior mean `μ` while automatically estimating the regularisation strength `λ` from the data — no cross-validation required. The model is:
```
Y_ij ~ N(X_ij, s_ij²)               heteroscedastic Gaussian likelihood
p(X | λ) ∝ exp(-λ ‖X‖_*)           Matrix Laplace prior = "Nuclear Norm distribution"
λ ~ Gamma(a0, b0)                    hyperprior on regularisation strength
```

Installation
```bash
pip install -e ".[dev]"   # editable install with test deps
```

Requires JAX ≥ 0.4 (CPU or GPU). 

## Implementation details

`jnp.linalg.inv` is never called; `Q⁻¹` is applied implicitly through its eigendecomposition, Cholesky or LDLT. 
The ELBO is guaranteed non-decreasing: Q and λ are refreshed from the updated Ψ before the ELBO is computed.

### GPU acceleration: CUDA LDL^T kernels

Each CAVI iteration requires factoring `m` independent `r×r` indefinite linear systems `B̃ᵢ = diag(1/cₖ) + GᵢᵀGᵢ`. By default these are factored via `jnp.linalg.eigh`. On GPU with `r=50, m=10000` this costs ~68 ms/iteration.

Two CUDA LDL^T alternatives are provided:

#### XLA-native kernel (recommended)

Implemented as an XLA FFI custom call (`matlap/xla_ext/ldlt_kernel.cu`), compiled to `matlap/xla_ext/_ldlt_kernel.so`. Runs on the JAX-managed CUDA stream with **no host/device sync barriers** — all three steps (B̃ computation, LDL^T, output assembly) are fused into a single XLA program:

- ~**9× faster** than `eigh` for the full per-row update (`m=2000, r=50`)
- ~**4× faster** than the CuPy variant (see below)
- No CuPy dependency; callable inside `jax.jit`

```python
result = matlap_lowrank_isotropic(Y, S, rank=50, use_xla_ldlt=True)
result = matlap_grid_lowrank_isotropic(Y, S, grid, rank=50, use_xla_ldlt=True)
```

**Requirements:** GPU + pre-compiled `_ldlt_kernel.so`. Set `LD_LIBRARY_PATH` to include NVIDIA CUDA libs:

```bash
NVIDIA_LIBS=/path/to/venv/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$(find $NVIDIA_LIBS -name "lib" -type d | tr '\n' ':')$LD_LIBRARY_PATH
```

To recompile after kernel changes:
```bash
JAX_INC=/path/to/venv/lib/python3.12/site-packages/jaxlib/include
nvcc -O3 -shared --compiler-options '-fPIC' --std=c++17 -arch=sm_86 \
  -I"$JAX_INC" -I/usr/local/cuda/include -diag-suppress 940,2473 \
  -o matlap/xla_ext/_ldlt_kernel.so matlap/xla_ext/ldlt_kernel.cu
```

#### CuPy kernel (kept for comparison)

Uses a CuPy `RawKernel` in `matlap/ldlt_cuda.py`. ~3–4× faster than `eigh`; requires CuPy (`pip install cupy-cuda12x`) and explicit `block_until_ready()` sync barriers.

```python
result = matlap_lowrank_isotropic(Y, S, rank=50, use_ldlt=True)
```

**Requirements:** GPU + CuPy + `LD_LIBRARY_PATH` as above.

On CPU-only machines both flags fall back gracefully.

#### Approximate nuclear norm via rSVD for numpyro SVI methods

When `approx_rank > 0`, the model replaces the exact SVD nuclear norm with a
randomized SVD approximation using `approx_rank` singular values. This reduces
the per-step cost from O(mn·min(m,n)) to O(mn·approx_rank), typically giving
a 20–50× speedup at 10k×1k with negligible accuracy loss for low-rank matrices.
The gradient is the standard nuclear norm subgradient restricted to the top-r
component, implemented via `jax.custom_vjp`.

## Benchmark

```bash
# Full-scale benchmark (10k×1k, all methods, CPU+GPU)
python scripts/benchmark.py

# Faster run for testing
python scripts/benchmark.py \
    -s 3 \
    --proximal-iters 50 \
    --vi-steps 100 \
    --lowrank-iters 30 \
    --lowrank-rank 50 \
    --guide-rank 15 \
    --approx-rank 30 \
    --output results/benchmark_10k
```

Results are written to `results/benchmark_10k.md` (human-readable report)
and `results/benchmark_10k.csv` (raw numbers).

The benchmark includes `matlap_grid_lowrank_iso_xla_ldlt`, which uses the
XLA FFI CUDA LDL^T kernel (`use_xla_ldlt=True`). This requires the
pre-compiled `matlap/xla_ext/_ldlt_kernel.so` and the NVIDIA CUDA libs on
`LD_LIBRARY_PATH` (see [GPU acceleration](#gpu-acceleration-cuda-ldlt-kernels) above).
On CPU-only machines that method is skipped automatically.

### Results (10k×1k, rank-15, RTX 3090)

| Method | RMSE | GPU time (s) | Converged |
|---|---|---|---|
| **`matlap_faem`** | **0.081** | **2.4** | ✓ |
| `matlap_gradml` | 0.081 | 2.8 | ✓ |
| **`matlap_grid_lowrank`** | **0.098** | **2.1** | ✓ |
| `matlap_grid_lowrank_iso_ldlt` | 0.119 | 3.3 | ✓ (**3× faster than iso_renyi**) |
| `matlap_grid_lowrank_iso_xla_ldlt` | 0.122 | 3.4 | ✓ |
| `matlap_grid_lowrank_iso_renyi` | 0.120 | 10.2 | ✓ |
| `proximal_cv` | 0.105 | 122 | — |
| `proximal` | 0.123 | 22 | — |
| `matlap_batched` | 0.153 | 188 | ✓ |
| `vi_diagonal` | 0.242 | 41 | — (200 steps) |
| `matlap_lowrank` | 0.258 | 0.5 | ✓ |
| `vi_matrix_factor` | 0.270 | 19 | — (200 steps) |
| `vi_row_lowrank` | 0.270 | 25 | — (200 steps) |
| `vi_diagonal_approx` | 0.398 | 11 | — (200 steps) |
| `matlap_grid_lowrank_iso_elbo` | 0.254 | 11 | — (ELBO scoring broken, see note) |

**`matlap_faem` / `matlap_gradml` achieve the lowest RMSE (0.081) — ~23% better than proximal CV at 50× lower cost.**
**`matlap_grid_lowrank` is the best efficiency trade-off: RMSE 0.098 in 2.1 s, ~7% better than proximal CV at 58× lower cost.**
**`matlap_grid_lowrank_iso_ldlt` (Rényi scoring) matches iso_renyi in accuracy (RMSE 0.119) in 3.3 s — 3× faster, using the CuPy LDL^T kernel.**

- `matlap_lowrank` over-shrinks because the empirical-Bayes λ update in factor space is biased by a factor ~n/r. `matlap_grid_lowrank` fixes this by grid search.
- The iso CAVI variants (`matlap_grid_lowrank_iso_*`) converge in ~12 CAVI steps (well within the 50-iteration budget). They use a nuclear-norm + isotropic prior; the `_ldlt` and `_xla_ldlt` variants accelerate the per-step B̃ factorisation ~3–4× vs `eigh`, making them competitive with `matlap_grid_lowrank` in wall-clock time at better RMSE.
- **ELBO scoring does not work for the iso model** (`iso_elbo`, RMSE 0.254): the nuclear-norm normaliser `m·n·log λ` grows faster than the prior penalty `-λ·Tr(Q)` at large λ, so the ELBO is monotonically increasing across the grid and always selects the largest λ. Use `score_fn="renyi"` (default) instead.
- `matlap_batched` gives exact full-CAVI results but is slow at n=1000 (O(n³) per row); best used when n ≤ 300.
- rSVD nuclear-norm approximation (`vi_*_approx`) at rank 30 introduces gradient noise that prevents SVI convergence at this scale.

## NND benchmark (model-matched data)

The standard benchmark above uses synthetic low-rank data, which favours `matlap_grid_lowrank`.
To assess each method on data that matches its prior, `scripts/benchmark_nnd.py` simulates matrices
directly from the Nuclear Norm Distribution (NND) — the Matrix Laplace prior — using the SVD
representation from the paper:

```
σᵢ ~ Exp(λ_true)  (Gamma(1, λ) for square matrices), sorted descending
U ~ Stiefel(n, m) via QR decomposition (Haar measure)
V ~ O(n) via QR decomposition (Haar measure)
X = U diag(σ) Vᵀ,  Y = X + ε,  ε ~ N(0, σ²_noise I)
```

With `m=n=100, λ_true=0.05, σ_noise=1.0`:

```bash
python scripts/benchmark_nnd.py
```

### Results (m=n=100, λ_true=0.05, σ_noise=1.0, 10 seeds)

| Method | RMSE | λ selected |
|---|---|---|
| **`batched_loo`** | **0.904 ± 0.012** | 5.0 |
| `batched_eb` | 0.905 ± 0.012 | 4.56 (median) |
| `iso_r10` | 0.927 ± 0.014 | 0.20 |
| `iso_r20` | 0.942 ± 0.018 | 0.50 |
| `batched_renyi` | 0.928 ± 0.020 | 2.0 |
| `iso_r30` | 0.976 ± 0.015 | 1.00 |
| `lowrank_r50` | 1.035 ± 0.041 | 5.0 |
| `lowrank_r5` | 2.178 ± 0.198 | 5.0 |
| Noise floor (Y as estimate) | 1.000 | — |

**Key observations:**

- `batched_loo` (the correctly-specified model) achieves the lowest RMSE and is the only method
  that beats the noise floor on *all* seeds. This validates the CAVI derivation for NND data.
- Lowrank methods `(r ≤ 50)` perform **worse than no denoising** because NND matrices are full-rank;
  zeroing out the bottom `n−r` singular-value directions introduces more error than the noise.
- `iso_r10` (isotropic floor + rank-10 subspace) benefits from the isotropic floor covering 90
  dimensions, partially compensating for the full-rank structure.
- The CAVI-optimal λ (~5) is much larger than the true NND λ (0.05): within the Gaussian
  row-factorised approximation, stronger regularisation is needed to correctly shrink noisy singular values.
- `batched_renyi` now correctly uses model-derived prior variance `psi_sqrt_diag_j/λ` (the
  diagonal of Q/λ, the actual CAVI row prior covariance), selecting λ=2 and scoring RMSE=0.928.
  Previously a constant data-derived proxy was used; that was an unjustified approximation.

## λ selection strategy comparison

```bash
python scripts/lambda_study.py
```

Compares 12 λ-selection strategies on a 500×100 matrix with heteroscedastic noise
across true ranks 1–40 and SNR values 0.1–10 (3 seeds each). Results are written to
`results/lambda_study.csv`, `results/lambda_study.md`, `results/snr_study.csv`, and
`results/snr_study.md`.

### Mean test RMSE across sweeps

| Method | Rank sweep | SNR sweep |
|---|---:|---:|
| `proximal_cv` | 0.467 | 0.314 |
| `matlap_grid` | 0.615 | 0.450 |
| `lowrank_cv` | 0.600 | 0.485 |
| `lowrank_grid` | 0.637 | 0.634 |
| `iso_cv` | 0.585 | 0.445 |
| `iso_grid_renyi` | 0.585 | 0.446 |
| `iso_grid_is` | 0.587 | 0.464 |
| `iso_grid_loo` | 0.631 | 0.476 |

`iso_*` = `matlap_lowrank_isotropic`; `lowrank_*` = `matlap_lowrank`; `*_grid`
means warm-started λ search. `iso_grid_renyi` is the best non-CV selector overall;
`iso_grid_is` (α=0) is close but degrades at higher SNR, and `iso_grid_loo` is
consistently too conservative.

## Scalability

| Method | Memory | Per-iter compute | 10k×1k feasible? |
|---|---|---|---|
| `matlap` | O(mn²) ≈ **40 GB** | O(mn³) | ✗ OOM |
| `matlap_batched` | O(Bn²) ≈ 256 MB (B=64) | O(mn³) total | ✓ (slow at n=1000) |
| `matlap_lowrank` | O(mn + mr²) ≈ 44 MB | O(mnr) | ✓ |
| `matlap_grid_lowrank` | O(mn + mr²) ≈ 44 MB | O(G·mnr) | ✓ **recommended** |
| `matlap_lowrank_isotropic` | O(mn + mr²) ≈ 44 MB | O(mnr) | ✓ **unbiased λ** |
| `proximal` | O(mn) ≈ 40 MB | O(mn·min(m,n)) full SVD | ✓ |
| `vi_diagonal` | O(mn) ≈ 40 MB | O(mn·min(m,n)) full SVD | ✓ (slow) |
| `vi_diagonal` + rSVD | O(mn) ≈ 40 MB | O(mn·r) | ✓ |
| `vi_matrix_factor` | O(mn) ≈ 40 MB | O(mn·r) | ✓ |
| `vi_row_lowrank` | O(mnk) ≈ 600 MB | O(mn·r) | ✓ |
| `vi_row_mvn` | O(mn²) ≈ **40 GB** | O(mn³) | ✗ OOM |
| `vi_matrix_normal` | O(m²+n²) ≈ 400 MB | O(m²n) ≈ 10¹¹ | ✗ too slow |

## Tests

```bash
pytest tests/ -v
```

90+ tests covering: matrix-sqrt correctness, ELBO monotonicity, low-rank recovery,
missing-data handling, convergence flags, grid-search, proximal gradient optimality,
CV correctness, all SVI guide types, rSVD accuracy, low-rank CAVI vs full CAVI
agreement, and memory-feasibility at 10k×1k scale.
