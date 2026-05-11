# matlap

**Bayesian matrix denoising via Coordinate Ascent Variational Inference (CAVI)** with a Matrix Laplace prior.

Given an observed matrix `Y` with per-entry heteroscedastic noise `S`, `matlap` recovers a low-rank posterior mean `μ` while automatically estimating the regularisation strength `λ` from the data — no cross-validation required.

## Installation

```bash
pip install -e ".[dev]"   # editable install with test deps
```

Requires JAX ≥ 0.4 (CPU or GPU). The package is tested on the JAX venv at `~/venvs/jax`.

## Quick start

```python
import jax.numpy as jnp
from matlap import matlap, matlap_grid

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

Fits the same Matrix Laplace model as CAVI via gradient-based SVI using Numpyro.  Three variational families are available:

| `guide_type` | Variational family | Parameters |
|---|---|---|
| `'diagonal'` | Fully-factorised Gaussian | O(mn) |
| `'row_mvn'` | Product of row MVNs | O(mn + mn²/2) |
| `'matrix_normal'` | Matrix Normal | O(mn + m²/2 + n²/2) |

```python
from matlap.vi import fit_vi

# Auto lambda (LogNormal hyperprior)
r = fit_vi(Y, S, guide_type="diagonal", n_steps=5000, lr=1e-3)
print(r.mu)           # E_q[X], shape (m, n)
print(r.lambda_bar)   # E_q[lambda]

# Fixed lambda
r = fit_vi(Y, S, lambda_val=2.0, guide_type="matrix_normal")
```

**Returns** `VIResult`:

| Field | Description |
|---|---|
| `mu` | Posterior mean E_q[X], shape (m, n) |
| `lambda_bar` | E_q[λ] |
| `elbo_trace` | ELBO per SVI step (negated loss) |
| `converged` | Whether ELBO plateau was reached |
| `n_iter` | Number of SVI steps executed |

### Benchmark

```bash
python scripts/benchmark.py --seeds 5 --rows 50 --cols 20
```

Simulates a rank-3 matrix, masks ~20% of entries as a test set, fits all six methods, and prints a RMSE / runtime table averaged over multiple seeds.

## Tests

```bash
pytest tests/ -v
```

51 tests covering matrix-sqrt correctness, ELBO monotonicity, low-rank recovery, missing-data handling, convergence flags, grid-search, proximal gradient optimality, CV correctness, and all three SVI guide types.
