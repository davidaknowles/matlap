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

## Tests

```bash
pytest tests/ -v
```

21 tests covering matrix-sqrt correctness, ELBO monotonicity, low-rank recovery, missing-data handling, convergence flags, and grid-search logic.
