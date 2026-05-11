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
from matlap import matlap, matlap_grid, matlap_lowrank

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

### `matlap_lowrank(Y, S, *, rank, a0, b0, max_iter, tol, verbose) → LowRankCAVIResult`

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
# Full-scale benchmark (10k×1k, all 7 methods, 10 seeds)
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

## Scalability

| Method | Memory | Per-iter compute | 10k×1k feasible? |
|---|---|---|---|
| `matlap` | O(mn²) ≈ **40 GB** | O(mn³) | ✗ OOM |
| `matlap_lowrank` | O(mn + mr²) ≈ 44 MB | O(mnr) | ✓ |
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
