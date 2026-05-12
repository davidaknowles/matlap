# matlap Lambda Selection Study

Generated: 2026-05-12 14:15  |  Matrix: 500×100  |  Seeds: 3  |  Low-rank rank: 50  |  CV folds: 3  |  Grid points: 8

## Test RMSE (mean ± std over seeds)

| Method | rank=1 | rank=3 | rank=5 | rank=10 | rank=20 | rank=40 |
|---|---|---|---|---|---|---|
| `proximal_cv` | 0.230 ± 0.004 | 0.292 ± 0.007 | 0.316 ± 0.003 | 0.310 ± 0.018 | 0.228 ± 0.004 | 0.163 ± 0.002 |
| `matlap_auto` | 0.255 ± 0.005 | 0.392 ± 0.007 | 0.404 ± 0.004 | 0.319 ± 0.012 | 0.229 ± 0.004 | 0.162 ± 0.002 |
| `lowrank_auto` | 1.055 ± 0.028 | 0.595 ± 0.013 | 0.457 ± 0.005 | 0.323 ± 0.012 | 0.230 ± 0.004 | 0.162 ± 0.002 |
| `lowrank_elbo` | 0.448 ± 0.006 | 0.383 ± 0.009 | 0.383 ± 0.004 | 0.334 ± 0.008 | 0.267 ± 0.024 | 0.173 ± 0.002 |
| `lowrank_cv` | 0.315 ± 0.006 | 0.383 ± 0.009 | 0.383 ± 0.004 | 0.315 ± 0.011 | 0.230 ± 0.004 | 0.162 ± 0.002 |
| `iso_auto` | 1.581 ± 0.010 | 1.660 ± 0.009 | 1.618 ± 0.002 | 1.544 ± 0.013 | 1.475 ± 0.013 | 1.443 ± 0.017 |
| `iso_cv` | 1.475 ± 0.009 | 1.546 ± 0.009 | 1.436 ± 0.005 | 1.332 ± 0.013 | 1.262 ± 0.011 | 1.231 ± 0.013 |

## Chosen λ (mean ± std over seeds)

| Method | rank=1 | rank=3 | rank=5 | rank=10 | rank=20 | rank=40 |
|---|---|---|---|---|---|---|
| `proximal_cv` | 16.088 ± 0.020 | 16.088 ± 0.020 | 16.088 ± 0.020 | 21.073 ± 7.035 | 31.062 ± 0.038 | 31.062 ± 0.038 |
| `matlap_auto` | 30.114 ± 0.527 | 36.097 ± 0.252 | 44.323 ± 0.574 | 56.527 ± 0.612 | 63.396 ± 0.352 | 67.066 ± 0.143 |
| `lowrank_auto` | 342.829 ± 24.351 | 461.782 ± 0.913 | 481.363 ± 4.977 | 499.407 ± 0.015 | 499.573 ± 0.008 | 499.649 ± 0.001 |
| `lowrank_elbo` | 16.088 ± 0.020 | 31.062 ± 0.038 | 31.062 ± 0.038 | 31.062 ± 0.038 | 40.714 ± 13.687 | 59.971 ± 0.074 |
| `lowrank_cv` | 31.062 ± 0.038 | 31.062 ± 0.038 | 31.062 ± 0.038 | 59.971 ± 0.074 | 115.786 ± 0.143 | 223.548 ± 0.276 |
| `iso_auto` | 9.338 ± 0.130 | 10.816 ± 0.072 | 11.178 ± 0.034 | 11.437 ± 0.017 | 11.592 ± 0.031 | 11.672 ± 0.031 |
| `iso_cv` | 2.235 ± 0.003 | 2.235 ± 0.003 | 2.235 ± 0.003 | 2.235 ± 0.003 | 2.235 ± 0.003 | 2.235 ± 0.003 |

## Runtime in seconds (mean over seeds)

| Method | rank=1 | rank=3 | rank=5 | rank=10 | rank=20 | rank=40 |
|---|---|---|---|---|---|---|
| `proximal_cv` | 32.8s | 37.8s | 34.5s | 36.4s | 34.7s | 33.6s |
| `matlap_auto` | 1.8s | 2.2s | 4.2s | 3.8s | 3.7s | 3.6s |
| `lowrank_auto` | 2.4s | 1.9s | 1.4s | 0.1s | 0.1s | 0.1s |
| `lowrank_elbo` | 5.7s | 2.6s | 2.7s | 2.2s | 2.5s | 2.2s |
| `lowrank_cv` | 14.0s | 10.0s | 9.9s | 10.7s | 9.6s | 9.5s |
| `iso_auto` | 1.0s | 1.1s | 0.9s | 0.8s | 0.9s | 0.8s |
| `iso_cv` | 23.3s | 17.0s | 17.6s | 16.8s | 16.5s | 17.5s |

## Notes

- Matrix: 500×100, heteroscedastic noise σ ~ Uniform(0.5, 1.5)
- Test fraction: 20% of observed entries
- Low-rank rank: 50
- Lambda grid: 8 log-spaced values, 0.1×–10× heuristic
- `lowrank_auto` λ is biased by factor ~n/r = 100/50 = 2.0×
  vs the unbiased `iso_auto` which uses the full n-dim ELBO.

- **`proximal_cv`**: proximal_cv    (FISTA + 3-fold CV)
- **`matlap_auto`**: matlap_auto    (full CAVI, auto-λ)
- **`lowrank_auto`**: lowrank_auto   (rank-r CAVI, auto-λ, biased)
- **`lowrank_elbo`**: lowrank_elbo   (rank-r CAVI, grid+ELBO)
- **`lowrank_cv`**: lowrank_cv     (rank-r CAVI, grid+CV)
- **`iso_auto`**: iso_auto       (isotropic CAVI, auto-λ, unbiased)
- **`iso_cv`**: iso_cv         (isotropic CAVI, grid+CV)
