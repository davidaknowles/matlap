# matlap

**Bayesian matrix denoising via Coordinate Ascent Variational Inference (CAVI)** with a Matrix Laplace prior.

Given an observed matrix `Y` with per-entry heteroscedastic noise `S`, `matlap` recovers a low-rank posterior mean `μ` while automatically estimating the regularisation strength `λ` from the data — no cross-validation required.

Scalable variants handle matrices up to **10 000 × 1 000** and beyond using low-rank factor subspaces, the Woodbury identity, and randomized SVD.

## Installation

```bash
pip install -e ".[dev]"   # editable install with test deps
```

Requires JAX ≥ 0.4 (CPU or GPU). The package is tested on the JAX venv at `~/venvs/jax`.

## Quick start

```python
import jax
import jax.numpy as jnp
from matlap import matlap, matlap_grid, matlap_lowrank, matlap_lowrank_isotropic

# --- synthetic data ---
key = jax.random.PRNGKey(0)
U = jax.random.normal(key, (50, 3))
V = jax.random.normal(jax.random.fold_in(key, 1), (20, 3))
X_true = U @ V.T                          # 50×20, rank-3
noise = 0.5 * jax.random.normal(jax.random.fold_in(key, 2), X_true.shape)
Y = X_true + noise
S = jnp.full_like(Y, 0.5)               # known noise level

# --- automatic lambda (empirical Bayes) ---
result = matlap(Y, S, verbose=True)
print(result.mu)           # posterior mean, shape (50, 20)
print(result.lambda_bar)   # estimated regularisation strength
print(result.converged)

# --- grid search over lambda ---
grid = matlap_grid(Y, S, lambda_grid=jnp.logspace(-2, 2, 20))
print(grid.best_lambda)
print(grid.best_result.mu)

# --- scalable low-rank CAVI (10k×1k scale) ---
result_lr = matlap_lowrank(Y_large, S_large, rank=50)
print(result_lr.mu)        # posterior mean (m, n)
print(result_lr.lambda_bar)

# --- low-rank-plus-isotropic CAVI (full n-dim posteriors, unbiased λ) ---
result_iso = matlap_lowrank_isotropic(Y_large, S_large, rank=50)
print(result_iso.mu)        # off-subspace mass included
print(result_iso.lambda_bar)  # unbiased estimate
```

## Missing data

Encode missing entries by setting the corresponding standard error to `jnp.inf`
(the `Y` value is ignored):

```python
S = S.at[3, 7].set(jnp.inf)   # entry (3,7) is missing
result = matlap(Y, S)
```

Internally, noise precision `1/s²` is zero for missing entries, so they
contribute nothing to the likelihood or the posterior mean.

## API

### `matlap(Y, S, *, a0, b0, max_iter, tol, verbose) → CAVIResult`

Full CAVI with automatic `λ` estimation via an empirical-Bayes Gamma
hyperprior `λ ~ Gamma(a0, b0)`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Y` | — | Observed matrix `(m, n)` |
| `S` | — | Known standard errors `(m, n)`; `jnp.inf` for missing |
| `a0` | `1e-3` | Gamma prior shape (weakly informative) |
| `b0` | `1e-3` | Gamma prior rate (weakly informative) |
| `max_iter` | `200` | Maximum CAVI iterations |
| `tol` | `1e-6` | Relative ELBO convergence tolerance |
| `verbose` | `False` | Print ELBO and λ each iteration |

**Returns** `CAVIResult`:

| Field | Description |
|-------|-------------|
| `mu` | Posterior mean `(m, n)` |
| `sigma` | Posterior row covariances `(m, n, n)` |
| `lambda_bar` | `E_q[λ]` |
| `a_N`, `b_N` | Gamma posterior parameters |
| `elbo_trace` | ELBO at the end of each iteration |
| `converged` | Whether tolerance was reached |
| `n_iter` | Number of iterations executed |

> **Memory note:** stores `sigma` of shape `(m, n, n)` — infeasible above ~200 columns on 16 GB RAM. Use `matlap_lowrank` for larger matrices.

---

### `matlap_lowrank_isotropic(Y, S, lambda_val=None, *, rank, gamma, a0, b0, max_iter, tol, verbose) → LowRankIsotropicResult`

Low-rank-plus-isotropic CAVI using prior precision `V_r diag(λ̄/d_r) V_rᵀ + γI`.
Computes **full n-dimensional** per-row posteriors via the Woodbury identity — a
strict improvement over `matlap_lowrank` at the same O(mnr + r³) asymptotic cost:

- **Correct n-dim entropy** in the ELBO (vs r-dim in `matlap_lowrank`) → unbiased λ
- **Off-subspace posterior mass** → lower reconstruction error
- `lambda_val` can be passed as a positional arg for use with `cv_lambda`

```python
from matlap import matlap_lowrank_isotropic
from matlap.cv import cv_lambda
import jax.numpy as jnp

# Auto lambda (n-dim empirical Bayes — unbiased)
result = matlap_lowrank_isotropic(Y, S, rank=50)
print(result.mu)          # posterior mean (m, n) — includes off-subspace
print(result.lambda_bar)  # unbiased E_q[λ]

# Lambda selection by CV
grid = jnp.logspace(-1, 2, 12)
best_lam, result = cv_lambda(
    Y, S, grid,
    lambda Y_, S_, lam: matlap_lowrank_isotropic(Y_, S_, lam, rank=50),
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Y` | — | Observed matrix `(m, n)` |
| `S` | — | Standard errors `(m, n)`; `jnp.inf` for missing |
| `lambda_val` | `None` | Fix λ (skip empirical-Bayes update); pass positionally for `cv_lambda` |
| `rank` | `50` | Rank of the factor subspace `r` |
| `gamma` | `1e-3` | Isotropic prior precision `γ` for off-subspace directions |
| `a0` | `1e-3` | Gamma prior shape |
| `b0` | `1e-3` | Gamma prior rate |
| `max_iter` | `200` | Maximum CAVI iterations |
| `tol` | `1e-6` | Relative ELBO convergence tolerance |
| `use_ldlt` | `False` | Use CuPy CUDA LDL^T kernel (requires CuPy; ~4× vs eigh) |
| `use_xla_ldlt` | `False` | Use XLA-native CUDA LDL^T kernel (no sync barriers; ~9× vs eigh) |

**Returns** `LowRankIsotropicResult`:

| Field | Description |
|-------|-------------|
| `mu` | Full n-dim posterior mean `(m, n)` (includes off-subspace directions) |
| `z` | In-subspace projection `Vᵣᵀ μᵢ`, shape `(m, r)` |
| `V_r` | Loading matrix `(n, r)`; orthonormal columns |
| `gamma` | Isotropic prior precision used |
| `lambda_bar` | `E_q[λ]` (unbiased: uses n-dim entropy) |
| `a_N`, `b_N` | Gamma posterior parameters |
| `elbo_trace` | ELBO per iteration |
| `converged` | Whether tolerance was reached |
| `n_iter` | Number of iterations executed |

**Memory:** same as `matlap_lowrank` — ~44 MB at `m=10000, n=1000, rank=50`.

> **Key difference vs `matlap_lowrank`:** The ELBO uses n-dimensional entropy
> (`a_N = a0 + m·n`) so λ is estimated correctly. `matlap_lowrank` uses
> `a_N = a0 + m·r`, biasing λ low by a factor of ~r/n and causing over-shrinkage.

#### GPU acceleration: CUDA LDL^T kernels

Each CAVI iteration requires factoring `m` independent `r×r` indefinite linear systems `B̃ᵢ = diag(1/cₖ) + GᵢᵀGᵢ`. By default these are factored via `jnp.linalg.eigh`. On GPU with `r=50, m=10000` this costs ~68 ms/iteration.

Two CUDA LDL^T alternatives are provided:

##### `use_xla_ldlt=True` — XLA-native kernel (recommended)

Implemented as an XLA FFI custom call (`matlap/xla_ext/ldlt_kernel.cu`), compiled to `matlap/xla_ext/_ldlt_kernel.so`. Runs on the JAX-managed CUDA stream with **no host/device sync barriers** — all three steps (B̃ computation, LDL^T, output assembly) are fused into a single XLA program:

- ~**9× faster** than `eigh` for the full per-row update (`m=2000, r=50`)
- ~**4× faster** than the CuPy variant
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

##### `use_ldlt=True` — CuPy kernel (kept for comparison)

Uses a CuPy `RawKernel` in `matlap/ldlt_cuda.py`. ~3–4× faster than `eigh`; requires CuPy (`pip install cupy-cuda12x`) and explicit `block_until_ready()` sync barriers.

```python
result = matlap_lowrank_isotropic(Y, S, rank=50, use_ldlt=True)
```

**Requirements:** GPU + CuPy + `LD_LIBRARY_PATH` as above.

On CPU-only machines both flags fall back gracefully.

---

### `matlap_grid_lowrank_isotropic(Y, S, lambda_grid, *, rank, a0, b0, max_iter, tol, score_fn="elbo", alpha=0.5, use_ldlt=False, use_xla_ldlt=False, verbose=False) → LowRankIsotropicGridResult`

Warm-started λ grid search for `matlap_lowrank_isotropic`. The grid is traversed from
largest to smallest λ, reusing `(V_r, d_r, delta)` from the previous point. Selection
supports `score_fn="elbo"` (default), `score_fn="loo"` (closed-form Gaussian LOO), or
`score_fn="renyi"` (Rényi α-ELBO); `alpha=0` gives the importance-style objective.

```python
from matlap import matlap_grid_lowrank_isotropic
import jax.numpy as jnp

grid = jnp.logspace(-1, 2, 12)
result = matlap_grid_lowrank_isotropic(Y, S, grid, rank=50, score_fn="renyi", alpha=0.5)
print(result.best_lambda)
print(result.best_result.mu)
```

**Memory:** same as `matlap_lowrank_isotropic`.

---

### `matlap_lowrank(Y, S, lambda_val=None, *, rank, …) → LowRankCAVIResult`

Low-rank CAVI that restricts the variational family to a rank-`r` factor subspace,
reducing memory from O(mn²) to O(mn + mr²). Uses the **Woodbury identity** so
each per-row update requires only an `r×r` Cholesky solve.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Y` | — | Observed matrix `(m, n)` |
| `S` | — | Known standard errors `(m, n)`; `jnp.inf` for missing |
| `rank` | `50` | Rank of the factor subspace `r` |
| `a0` | `1e-3` | Gamma prior shape |
| `b0` | `1e-3` | Gamma prior rate |
| `max_iter` | `200` | Maximum CAVI iterations |
| `tol` | `1e-6` | Relative ELBO convergence tolerance |
| `verbose` | `False` | Print ELBO and λ each iteration |

**Returns** `LowRankCAVIResult`:

| Field | Description |
|-------|-------------|
| `mu` | Posterior mean `(m, n)` |
| `z` | Factor-space means `(m, r)` |
| `V_r` | Loading matrix `(n, r)`; orthonormal columns |
| `lambda_bar` | `E_q[λ]` |
| `a_N`, `b_N` | Gamma posterior parameters |
| `elbo_trace` | ELBO per iteration |
| `converged` | Whether tolerance was reached |
| `n_iter` | Number of iterations executed |

**Memory:** ~44 MB at `m=10000, n=1000, rank=50` (vs 40 GB for full CAVI).

---

### `matlap_grid(Y, S, lambda_grid, *, a0, b0, max_iter, tol, verbose) → GridResult`

Run CAVI with `λ` held fixed at each point in `lambda_grid`.  
Selects the best `λ` by final ELBO.

**Returns** `GridResult`:

| Field | Description |
|-------|-------------|
| `best_lambda` | λ with the highest final ELBO |
| `best_result` | `CAVIResult` for the best λ |
| `results` | List of `(lambda, CAVIResult)` for every grid point |

---

### `matlap_batched(Y, S, *, batch_size, a0, b0, max_iter, tol, verbose) → BatchedCAVIResult`

Exact full CAVI with memory-efficient batched row processing.  
Instead of materialising all `m` row covariances at once (O(mn²) = 40 GB at 10k×1k),
rows are processed in mini-batches of `batch_size`. Each batch computes `Σᵢ`, immediately
extracts `diag(Σᵢ)` and its contribution to `Ψ`, then discards the full covariance.
Peak memory is O(B·n²) — with B=64 and n=300 this is ~21 MB.

> **Note:** Each row still requires an O(n³) Cholesky solve, so this is practical
> when `n` is moderate (≤300) but `m` is large. At n=1000 it is slow (~185 s/seed).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Y` | — | Observed matrix `(m, n)` |
| `S` | — | Standard errors `(m, n)`; `jnp.inf` for missing |
| `batch_size` | `64` | Mini-batch size `B` |
| `a0` | `1e-3` | Gamma prior shape |
| `b0` | `1e-3` | Gamma prior rate |
| `max_iter` | `200` | Maximum CAVI iterations |
| `tol` | `1e-6` | Relative ELBO convergence tolerance |

**Returns** `BatchedCAVIResult`:

| Field | Description |
|-------|-------------|
| `mu` | Posterior mean `(m, n)` |
| `sigma_diag` | Diagonal of each row posterior covariance `(m, n)` |
| `lambda_bar` | `E_q[λ]` |
| `elbo_trace` | ELBO per iteration |
| `converged` | Whether tolerance was reached |

---

### `matlap_grid_lowrank(Y, S, lambda_grid, *, rank, a0, b0, max_iter, tol, score_fn, alpha, verbose) → LowRankGridResult`

**Recommended for large matrices.** Combines the low-rank CAVI (rank-`r` factor subspace)
with a warm-started regularisation path. The grid is traversed from largest to smallest
`λ`; each grid point is warm-started from the previous solution so convergence is fast.
Selection supports `score_fn="elbo"` (default), `score_fn="loo"` (closed-form Gaussian
LOO), or `score_fn="renyi"` (Rényi α-ELBO).

```python
from matlap import matlap_grid_lowrank
import jax.numpy as jnp

grid = jnp.logspace(0, 3, 7)
result = matlap_grid_lowrank(Y, S, lambda_grid=grid, rank=30)
print(result.best_lambda)
print(result.best_result.mu)   # shape (m, n)

# Alternative selection criteria
result_loo = matlap_grid_lowrank(Y, S, grid, rank=30, score_fn="loo")
result_renyi = matlap_grid_lowrank(Y, S, grid, rank=30, score_fn="renyi", alpha=0.5)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Y` | — | Observed matrix `(m, n)` |
| `S` | — | Standard errors `(m, n)`; `jnp.inf` for missing |
| `lambda_grid` | — | 1-D array of λ values to search |
| `rank` | `30` | Rank of the factor subspace `r` |
| `a0` | `1e-3` | Gamma prior shape (used for ELBO comparison only) |
| `b0` | `1e-3` | Gamma prior rate |
| `max_iter` | `200` | Maximum CAVI iterations per grid point |
| `tol` | `1e-6` | Relative ELBO convergence tolerance |
| `score_fn` | `"elbo"` | Selection score: `"elbo"`, `"loo"`, or `"renyi"` |
| `alpha` | `0.5` | Rényi order in `[0, 1)` used only when `score_fn="renyi"` (`0` = IS-style objective) |

**Returns** `LowRankGridResult`:

| Field | Description |
|-------|-------------|
| `best_lambda` | λ with the highest final ELBO |
| `best_result` | `LowRankCAVIResult` for the best λ |
| `results` | List of `(lambda, LowRankCAVIResult)` for every grid point |

**Memory:** ~44 MB at `m=10000, n=1000, rank=30`.  
**Speed:** ~0.6 s on RTX 3090 for 7 grid points.

---

## Algorithm

The model is:

```
Y_ij ~ N(X_ij, s_ij²)               heteroscedastic Gaussian likelihood
p(X | λ) ∝ exp(-λ ‖X‖_*)           Matrix Laplace prior (nuclear-norm regularisation)
λ ~ Gamma(a0, b0)                    hyperprior on regularisation strength
```

Each CAVI iteration performs three closed-form updates:

1. **Q update** — set `Q = Ψ^½` where `Ψ = Σ_i (μᵢμᵢᵀ + Σᵢ)`, computed via eigendecomposition (`jnp.linalg.eigh`).
2. **λ update** — `a_N = a0 + mn`, `b_N = b0 + Tr(Q)`, `λ̄ = a_N/b_N`.  
   *(Skipped in grid mode.)*
3. **Row updates** — for each row `i`, `Σᵢ = (diag(1/s²ᵢ) + λ̄ Q⁻¹)⁻¹` and `μᵢ = Σᵢ (yᵢ/s²ᵢ)`, solved via Cholesky (`jax.scipy.linalg.cho_factor`/`cho_solve`). Vectorised with `jax.vmap`.

`jnp.linalg.inv` is never called; `Q⁻¹` is applied implicitly through its eigendecomposition.  
The ELBO is guaranteed non-decreasing: Q and λ are refreshed from the updated Ψ before the ELBO is computed.

### Low-rank CAVI (`matlap_lowrank`)

Maintains a shared loading matrix `V_r ∈ R^{n×r}` and per-row factor-space means `z_i ∈ R^r`.
The per-row precision in factor space is an `r×r` matrix:

```
A_r^(i) = diag(λ̄/d_r) + V_rᵀ diag(pᵢ) V_r
z_i     = cho_solve(A_r^(i), V_rᵀ (pᵢ ⊙ yᵢ))
μᵢ      = V_r zᵢ
```

After all rows, `Ψ_r = Σ_i (zᵢzᵢᵀ + A_r^(i)⁻¹)` is accumulated in `R^{r×r}`, then `eigh(Ψ_r)` rotates `V_r` and gives `d_r = sqrt(eigenvalues)`. Memory is O(mn + mr² + nr) instead of O(mn²).

> **λ bias:** `a_N = a0 + m·r` so the automatic λ is biased by ~r/n relative to full CAVI. Use `matlap_lowrank_isotropic` for unbiased λ estimation.

### Low-rank-plus-isotropic CAVI (`matlap_lowrank_isotropic`)

Uses prior precision `V_r diag(λ̄/d_r) V_rᵀ + γI`. By the **Woodbury matrix identity**, each per-row posterior is fully n-dimensional:

```
B̃ᵢ     = diag(dᵣ/λ̄) + V_rᵀ D̃ᵢ⁻¹ V_r     [r×r; D̃ᵢ = diag(pᵢ + γ)]
Σᵢ      = D̃ᵢ⁻¹ − D̃ᵢ⁻¹ V_r B̃ᵢ⁻¹ V_rᵀ D̃ᵢ⁻¹   [diagonal-minus-lowrank, n-dim]
μᵢ      = Σᵢ (pᵢ ⊙ yᵢ)                       [n-dim posterior mean]
```

The Q update projects `Ψ` onto `V_r` (same O(mnr) cost), while the ELBO uses the full n-dimensional entropy via the matrix determinant lemma — correcting the systematic under-counting in `matlap_lowrank` and giving unbiased λ.

## Comparator methods

### Nuclear-norm proximal gradient (`proximal_gradient` / `proximal_cv`)

Solves the penalised problem min_X 0.5 Σ_{obs} (Y_ij − X_ij)² / s_ij² + λ ‖X‖_* via FISTA (accelerated proximal gradient).  The proximal operator of λ ‖·‖_* is singular value soft-thresholding.

```python
from matlap.proximal import proximal_gradient, proximal_cv
import jax.numpy as jnp

# Fixed lambda
r = proximal_gradient(Y, S, lambda_val=1.0)
print(r.X)           # denoised estimate (m, n)
print(r.converged)

# Lambda selected by 5-fold entry-wise CV
grid = jnp.logspace(-1, 2, 15)
best_lam, r = proximal_cv(Y, S, grid, n_folds=5)
print(best_lam, r.X)
```

### General cross-validation (`cv_lambda`)

`cv_lambda` works with **any** fitting function that accepts `(Y, S, lambda_val, **kwargs)`:

```python
from matlap.cv import cv_lambda
from matlap.vi import fit_vi

grid = [0.1, 0.5, 1.0, 5.0]

def my_fit(Y, S, lam):
    return fit_vi(Y, S, lambda_val=lam, guide_type="diagonal")

best_lam, result = cv_lambda(Y, S, grid, my_fit, n_folds=5)
```

`cv_lambda` splits observed entries (where `S < inf`) into K folds, evaluates held-out MSE for each (λ, fold), selects the best λ, and refits on all training entries.  The `get_mu` argument (optional) extracts the prediction from the result object; if omitted, `.mu` or `.X` is detected automatically.

### Numpyro SVI (`fit_vi`)

Fits the same Matrix Laplace model as CAVI via gradient-based SVI using Numpyro.

#### Guide types

| `guide_type` | Variational family | Memory | Notes |
|---|---|---|---|
| `'diagonal'` | Fully-factorised Gaussian | O(mn) | Baseline |
| `'row_mvn'` | Product of row MVNs | O(mn²) | Exact row covariance; OOM at n≥200 |
| `'matrix_normal'` | Matrix Normal | O(m²+n²) | O(m²n) per step; impractical at m≫1 |
| `'matrix_factor'` | Shared column-factor + diagonal | O(mn) | Scalable structured covariance |
| `'row_lowrank'` | Per-row low-rank + diagonal | O(mnk) | ~600 MB at 10k×1k, k=15 |

```python
from matlap.vi import fit_vi

# Auto lambda (LogNormal hyperprior)
r = fit_vi(Y, S, guide_type="diagonal", n_steps=5000, lr=1e-3)
print(r.mu)           # E_q[X], shape (m, n)
print(r.lambda_bar)   # E_q[lambda]

# Scalable: diagonal guide + rSVD nuclear norm (approx_rank speeds up model)
r = fit_vi(Y, S, guide_type="diagonal", approx_rank=30, n_steps=200)

# Shared column-factor guide (captures column correlations, O(mn) memory)
r = fit_vi(Y, S, guide_type="matrix_factor", guide_rank=15, approx_rank=30)

# Per-row low-rank guide
r = fit_vi(Y, S, guide_type="row_lowrank", guide_rank=5, approx_rank=30)

# Fixed lambda
r = fit_vi(Y, S, lambda_val=2.0, guide_type="matrix_factor")
```

#### `fit_vi` parameters

| Parameter | Default | Description |
|---|---|---|
| `guide_type` | `'diagonal'` | Variational family (see table above) |
| `approx_rank` | `0` | rSVD rank for approximate nuclear norm in model; `0` = exact SVD |
| `guide_rank` | `15` | Low-rank factor dimension for `matrix_factor` / `row_lowrank` |
| `n_steps` | `5000` | SVI gradient steps |
| `lr` | `1e-3` | Adam learning rate |
| `lambda_val` | `None` | Fix λ (skips hyperprior); `None` = estimate from data |

**Returns** `VIResult`:

| Field | Description |
|---|---|
| `mu` | Posterior mean E_q[X], shape (m, n) |
| `lambda_bar` | E_q[λ] |
| `elbo_trace` | ELBO per SVI step (negated loss) |
| `converged` | Whether ELBO plateau was reached |
| `n_iter` | Number of SVI steps executed |

#### Approximate nuclear norm via rSVD

When `approx_rank > 0`, the model replaces the exact SVD nuclear norm with a
randomized SVD approximation using `approx_rank` singular values. This reduces
the per-step cost from O(mn·min(m,n)) to O(mn·approx_rank), typically giving
a 20–50× speedup at 10k×1k with negligible accuracy loss for low-rank matrices.
The gradient is the standard nuclear norm subgradient restricted to the top-r
component, implemented via `jax.custom_vjp`.

### Benchmark

```bash
# Full-scale benchmark (10k×1k, all 9 methods, CPU+GPU)
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

### Results (10k×1k, rank-15, RTX 3090)

| Method | RMSE | GPU time (s) | Converged |
|---|---|---|---|
| **`matlap_faem`** | **0.081** | **3.6** | ✓ |
| `matlap_gradml` | 0.081 | 3.2 | ✓ |
| **`matlap_grid_lowrank`** | **0.099** | **2.4** | ✓ |
| `matlap_grid_lowrank_iso_elbo` | 0.114 | 27 | — (50-iter budget) |
| `matlap_grid_lowrank_iso_xla_ldlt` | 0.114 | ~3 | — (50-iter budget, **9× faster**) |
| `proximal_cv` | 0.105 | 121 | — |
| `proximal` | 0.123 | 23 | — |
| `matlap_grid_lowrank_iso_renyi` | 0.124 | 27 | — (50-iter budget) |
| `matlap_batched` | 0.153 | 184 | ✓ |
| `vi_diagonal` | 0.242 | 42 | — (200 steps) |
| `matlap_lowrank` | 0.257 | 1.0 | ✓ |
| `vi_matrix_factor` | 0.269 | 19 | — (200 steps) |
| `vi_row_lowrank` | 0.270 | 25 | — (200 steps) |
| `vi_diagonal_approx` | 0.397 | 11 | — (200 steps) |

**`matlap_faem` / `matlap_gradml` achieve the lowest RMSE (0.081) — ~23% better than proximal CV at 30× lower cost.**
**`matlap_grid_lowrank` is the best efficiency trade-off: RMSE 0.099 in 2.4 s, ~6% better than proximal CV at 50× lower cost.**

- `matlap_lowrank` over-shrinks because the empirical-Bayes λ update in factor space is biased by a factor ~n/r. `matlap_grid_lowrank` fixes this by grid search.
- The iso CAVI variants (`matlap_grid_lowrank_iso_*`) use a hybrid nuclear-norm + isotropic Gaussian prior. They are strictly more expressive than `matlap_grid_lowrank` but require more iterations to converge (11× higher per-iteration cost); within the 50-iteration benchmark budget they are competitive with `proximal_cv`. The `_xla_ldlt` variant uses the XLA-native CUDA LDL^T kernel (~9× faster per iteration than `eigh`), making iso CAVI competitive in wall-clock time.
- `matlap_batched` gives exact full-CAVI results but is slow at n=1000 (O(n³) per row); best used when n ≤ 300.
- rSVD nuclear-norm approximation (`vi_*_approx`) at rank 30 introduces gradient noise that prevents SVI convergence at this scale.

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
