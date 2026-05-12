# matlap Lambda Selection Study

Generated: 2026-05-12 14:49  |  Matrix: 500×100  |  Seeds: 3  |  Low-rank rank: 50  |  CV folds: 3  |  Grid points: 8

## Test RMSE (mean ± std over seeds)

| Method | rank=1 | rank=3 | rank=5 | rank=10 | rank=20 | rank=40 |
|---|---|---|---|---|---|---|
| `proximal_cv` | 0.230 ± 0.004 | 0.292 ± 0.007 | 0.316 ± 0.003 | 0.310 ± 0.018 | 0.228 ± 0.004 | 0.163 ± 0.002 |
| `matlap_auto` | 0.255 ± 0.005 | 0.392 ± 0.007 | 0.404 ± 0.004 | 0.319 ± 0.012 | 0.229 ± 0.004 | 0.162 ± 0.002 |
| `lowrank_auto` | 1.055 ± 0.028 | 0.595 ± 0.013 | 0.457 ± 0.005 | 0.323 ± 0.012 | 0.230 ± 0.004 | 0.162 ± 0.002 |
| `lowrank_grid` | 0.448 ± 0.006 | 0.383 ± 0.009 | 0.383 ± 0.004 | 0.334 ± 0.008 | 0.267 ± 0.024 | 0.173 ± 0.002 |
| `lowrank_cv` | 0.315 ± 0.006 | 0.383 ± 0.009 | 0.383 ± 0.004 | 0.315 ± 0.011 | 0.230 ± 0.004 | 0.162 ± 0.002 |
| `iso_auto` | 1.042 ± 0.028 | 0.606 ± 0.013 | 0.466 ± 0.005 | 0.330 ± 0.012 | 0.233 ± 0.004 | 0.164 ± 0.002 |
| `iso_cv` | 1.052 ± 0.030 | 0.595 ± 0.013 | 0.457 ± 0.005 | 0.324 ± 0.012 | 0.230 ± 0.004 | 0.162 ± 0.002 |

## Chosen λ (mean ± std over seeds)

| Method | rank=1 | rank=3 | rank=5 | rank=10 | rank=20 | rank=40 |
|---|---|---|---|---|---|---|
| `proximal_cv` | 16.088 ± 0.020 | 16.088 ± 0.020 | 16.088 ± 0.020 | 21.073 ± 7.035 | 31.062 ± 0.038 | 31.062 ± 0.038 |
| `matlap_auto` | 30.114 ± 0.527 | 36.097 ± 0.252 | 44.323 ± 0.574 | 56.527 ± 0.612 | 63.396 ± 0.352 | 67.066 ± 0.143 |
| `lowrank_auto` | 342.829 ± 24.351 | 461.782 ± 0.913 | 481.363 ± 4.977 | 499.407 ± 0.015 | 499.573 ± 0.008 | 499.649 ± 0.001 |
| `lowrank_grid` | 16.088 ± 0.020 | 31.062 ± 0.038 | 31.062 ± 0.038 | 31.062 ± 0.038 | 40.714 ± 13.687 | 59.971 ± 0.074 |
| `lowrank_cv` | 31.062 ± 0.038 | 31.062 ± 0.038 | 31.062 ± 0.038 | 59.971 ± 0.074 | 115.786 ± 0.143 | 223.548 ± 0.276 |
| `iso_auto` | 80.018 ± 1.179 | 90.737 ± 0.315 | 92.789 ± 0.105 | 94.200 ± 0.057 | 95.015 ± 0.122 | 95.448 ± 0.125 |
| `iso_cv` | 2.235 ± 0.003 | 2.235 ± 0.003 | 223.548 ± 0.276 | 223.548 ± 0.276 | 223.548 ± 0.276 | 223.548 ± 0.276 |

## Runtime in seconds (mean over seeds)

| Method | rank=1 | rank=3 | rank=5 | rank=10 | rank=20 | rank=40 |
|---|---|---|---|---|---|---|
| `proximal_cv` | 33.6s | 36.6s | 35.7s | 38.5s | 32.8s | 33.9s |
| `matlap_auto` | 2.0s | 2.4s | 3.6s | 3.8s | 3.8s | 3.4s |
| `lowrank_auto` | 2.3s | 1.8s | 1.4s | 0.1s | 0.1s | 0.1s |
| `lowrank_grid` | 4.8s | 2.6s | 2.6s | 2.8s | 2.3s | 2.2s |
| `lowrank_cv` | 15.3s | 10.2s | 9.7s | 10.3s | 9.6s | 9.9s |
| `iso_auto` | 1.7s | 1.4s | 1.4s | 1.7s | 1.3s | 1.4s |
| `iso_cv` | 7.2s | 6.9s | 7.2s | 6.8s | 6.9s | 6.9s |

## Notes

- Matrix: 500×100, heteroscedastic noise σ ~ Uniform(0.5, 1.5)
- Test fraction: 20% of observed entries
- Low-rank rank: 50
- Lambda grid: 8 log-spaced values, 0.1×–10× heuristic
- `lowrank_auto` λ is biased by factor ~n/r = 100/50 = 2.0×
  (r-dim trace vs full n-dim); use `lowrank_grid` or `lowrank_cv` instead.
- `lowrank_grid` = `matlap_grid_lowrank` (ELBO-based λ selection).
  ELBO prefers lower λ than CV → tends to under-regularize.
- `iso_auto`/`iso_cv` use `matlap_lowrank_isotropic` (lowrank+isotropic prior).
  γ is set to λ̄ (auto) or λ (CV), regularizing off-subspace at the
  same scale as in-subspace.  With γ=1e-3 (old default) off-subspace
  directions are unregularised and the RMSE degrades to noise level.

- **`proximal_cv`**: proximal_cv    (FISTA + 3-fold CV)
- **`matlap_auto`**: matlap_auto    (full CAVI, auto-λ)
- **`lowrank_auto`**: lowrank_auto   (rank-r CAVI, auto-λ, biased)
- **`lowrank_grid`**: lowrank_grid   (matlap_grid_lowrank, best ELBO)
- **`lowrank_cv`**: lowrank_cv     (rank-r CAVI, grid+CV)
- **`iso_auto`**: iso_auto       (lowrank+iso CAVI, auto-λ, γ=λ̄)
- **`iso_cv`**: iso_cv         (lowrank+iso CAVI, grid+CV, γ=λ)
