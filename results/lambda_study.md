# matlap Lambda Selection Study

Generated: 2026-05-12 23:30  |  Matrix: 500×100  |  Seeds: 3  |  Low-rank rank: 50  |  CV folds: 3  |  Grid points: 8

## Test RMSE (mean ± std over seeds)

| Method | rank=1 | rank=3 | rank=5 | rank=10 | rank=20 | rank=40 |
|---|---|---|---|---|---|---|
| `proximal_cv` | 0.230 ± 0.004 | 0.292 ± 0.007 | 0.316 ± 0.003 | 0.310 ± 0.018 | 0.228 ± 0.004 | 0.163 ± 0.002 |
| `matlap_auto` | 0.255 ± 0.005 | 0.392 ± 0.007 | 0.404 ± 0.004 | 0.319 ± 0.012 | 0.229 ± 0.004 | 0.162 ± 0.002 |
| `lowrank_auto` | 1.055 ± 0.028 | 0.595 ± 0.013 | 0.457 ± 0.005 | 0.323 ± 0.012 | 0.230 ± 0.004 | 0.162 ± 0.002 |
| `lowrank_grid` | 0.448 ± 0.006 | 0.383 ± 0.009 | 0.383 ± 0.004 | 0.334 ± 0.008 | 0.267 ± 0.024 | 0.173 ± 0.002 |
| `lowrank_cv` | 0.315 ± 0.006 | 0.383 ± 0.009 | 0.383 ± 0.004 | 0.315 ± 0.011 | 0.230 ± 0.004 | 0.162 ± 0.002 |
| `iso_auto` | 0.259 ± 0.005 | 0.393 ± 0.008 | 0.404 ± 0.004 | 0.319 ± 0.012 | 0.229 ± 0.004 | 0.162 ± 0.002 |
| `iso_cv` | 0.245 ± 0.004 | 0.354 ± 0.008 | 0.360 ± 0.003 | 0.305 ± 0.011 | 0.229 ± 0.004 | 0.166 ± 0.003 |

## Chosen λ (mean ± std over seeds)

| Method | rank=1 | rank=3 | rank=5 | rank=10 | rank=20 | rank=40 |
|---|---|---|---|---|---|---|
| `proximal_cv` | 16.088 ± 0.020 | 16.088 ± 0.020 | 16.088 ± 0.020 | 21.073 ± 7.035 | 31.062 ± 0.038 | 31.062 ± 0.038 |
| `matlap_auto` | 30.114 ± 0.527 | 36.097 ± 0.252 | 44.323 ± 0.574 | 56.527 ± 0.612 | 63.396 ± 0.352 | 67.066 ± 0.143 |
| `lowrank_auto` | 342.829 ± 24.351 | 461.782 ± 0.913 | 481.363 ± 4.977 | 499.407 ± 0.015 | 499.573 ± 0.008 | 499.649 ± 0.001 |
| `lowrank_grid` | 16.088 ± 0.020 | 31.062 ± 0.038 | 31.062 ± 0.038 | 31.062 ± 0.038 | 40.714 ± 13.687 | 59.971 ± 0.074 |
| `lowrank_cv` | 31.062 ± 0.038 | 31.062 ± 0.038 | 31.062 ± 0.038 | 59.971 ± 0.074 | 115.786 ± 0.143 | 223.548 ± 0.276 |
| `iso_auto` | 30.128 ± 0.526 | 36.065 ± 0.264 | 44.407 ± 0.514 | 56.529 ± 0.595 | 63.305 ± 0.338 | 66.899 ± 0.130 |
| `iso_cv` | 16.088 ± 0.020 | 16.088 ± 0.020 | 16.088 ± 0.020 | 16.088 ± 0.020 | 26.073 ± 7.064 | 13.246 ± 12.634 |

## Runtime in seconds (mean over seeds)

| Method | rank=1 | rank=3 | rank=5 | rank=10 | rank=20 | rank=40 |
|---|---|---|---|---|---|---|
| `proximal_cv` | 30.7s | 35.9s | 32.5s | 34.2s | 31.2s | 32.0s |
| `matlap_auto` | 1.7s | 2.4s | 3.4s | 3.3s | 3.5s | 3.9s |
| `lowrank_auto` | 2.9s | 1.8s | 1.7s | 0.1s | 0.1s | 0.1s |
| `lowrank_grid` | 5.3s | 2.7s | 2.3s | 2.2s | 2.3s | 2.1s |
| `lowrank_cv` | 13.5s | 9.9s | 10.2s | 10.5s | 9.4s | 9.6s |
| `iso_auto` | 2.3s | 3.0s | 4.7s | 4.7s | 4.7s | 4.7s |
| `iso_cv` | 33.7s | 26.9s | 26.0s | 26.1s | 26.0s | 25.7s |

## Notes

- Matrix: 500×100, heteroscedastic noise σ ~ Uniform(0.5, 1.5)
- Test fraction: 20% of observed entries
- Low-rank rank: 50
- Lambda grid: 8 log-spaced values, 0.1×–10× heuristic
- `lowrank_auto` λ is biased (diverges to ~m) because trace_Q uses only r dims;
  use `iso_auto`, `lowrank_grid`, or `lowrank_cv` instead.
- `lowrank_grid` = `matlap_grid_lowrank` (ELBO-based λ selection).
  ELBO prefers slightly lower λ than CV at low rank.
- `iso_auto`/`iso_cv` use `matlap_lowrank_isotropic` (low-rank+isotropic prior).
  δ is a variational parameter optimised each iteration (δ*=sqrt(Tr(Ψ⊥)/(n-r)));
  γ=λ̄/δ is derived, not a hyperparameter.  `iso_auto` gives the same λ as
  `matlap_auto` at O(mnr) cost (vs O(mn²) for full CAVI).
  `iso_cv` grid selects λ that is slightly lower than auto-λ.

- **`proximal_cv`**: proximal_cv    (FISTA + 3-fold CV)
- **`matlap_auto`**: matlap_auto    (full CAVI, auto-λ)
- **`lowrank_auto`**: lowrank_auto   (rank-r CAVI, auto-λ, biased)
- **`lowrank_grid`**: lowrank_grid   (matlap_grid_lowrank, best ELBO)
- **`lowrank_cv`**: lowrank_cv     (rank-r CAVI, grid+CV)
- **`iso_auto`**: iso_auto       (lowrank+iso CAVI, auto-λ, δ learned)
- **`iso_cv`**: iso_cv         (lowrank+iso CAVI, grid+CV, δ learned)
