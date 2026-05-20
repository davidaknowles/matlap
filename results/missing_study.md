# matlap Lambda Selection Study — Missing-Fraction Sweep

Generated: 2026-05-19 11:33  |  Matrix: 500×100  |  SNR=1.0  |  rank=5  |  Seeds: 3  |  Low-rank rank: 50  |  CV folds: 3  |  Grid: 12 pts, 10^-2×–10^2× heuristic

The noise heuristic λ̄ = √max(m,n) / √median_precision depends only on the *noise level* of observed entries, not on how many are observed.  The optimal λ/λ̄ ratio therefore rises with missing fraction, providing genuine variation for comparing λ-selection strategies.

## Test RMSE (mean ± std over seeds)

| Method | miss=0.02 | miss=0.1 | miss=0.3 | miss=0.6 | miss=0.9 |
|---|---|---|---|---|---|
| `proximal_cv` | 0.359 ± 0.006 | 0.363 ± 0.001 | 0.392 ± 0.006 | 0.494 ± 0.004 | 0.883 ± 0.010 |
| `matlap_auto` | 0.439 ± 0.016 | 0.468 ± 0.001 | 0.576 ± 0.006 | 0.835 ± 0.003 | 1.021 ± 0.013 |
| `matlap_grid` | 0.466 ± 0.018 | 0.498 ± 0.001 | 0.613 ± 0.007 | 0.871 ± 0.004 | 1.022 ± 0.012 |
| `lowrank_auto` | 1.027 ± 0.036 | 1.018 ± 0.009 | 1.018 ± 0.008 | 1.020 ± 0.012 | 1.024 ± 0.013 |
| `lowrank_grid` | 0.561 ± 0.005 | 0.571 ± 0.001 | 0.623 ± 0.008 | 0.795 ± 0.005 | 1.016 ± 0.012 |
| `lowrank_cv` | 0.465 ± 0.012 | 0.490 ± 0.003 | 0.623 ± 0.008 | 0.796 ± 0.005 | 1.016 ± 0.013 |
| `iso_auto` | 0.444 ± 0.016 | 0.472 ± 0.001 | 0.580 ± 0.006 | 0.836 ± 0.003 | 1.021 ± 0.013 |
| `iso_grid` | 0.469 ± 0.017 | 0.501 ± 0.001 | 0.615 ± 0.007 | 0.871 ± 0.004 | 1.022 ± 0.012 |
| `iso_cv` | 0.420 ± 0.015 | 0.451 ± 0.001 | 0.567 ± 0.006 | 0.839 ± 0.004 | 1.021 ± 0.013 |
| `iso_grid_loo` | 0.469 ± 0.017 | 0.501 ± 0.001 | 0.615 ± 0.007 | 0.871 ± 0.004 | 1.022 ± 0.013 |
| `iso_grid_renyi` | 0.421 ± 0.015 | 0.451 ± 0.001 | 0.567 ± 0.006 | 0.838 ± 0.004 | 1.021 ± 0.013 |
| `iso_grid_is` | 0.469 ± 0.017 | 0.484 ± 0.024 | 0.567 ± 0.006 | 0.838 ± 0.004 | 1.021 ± 0.013 |
| `iso_adaptive` | 0.425 ± 0.016 | 0.455 ± 0.001 | 0.567 ± 0.006 | 0.834 ± 0.003 | 1.021 ± 0.013 |

## Chosen λ (mean ± std over seeds)

| Method | miss=0.02 | miss=0.1 | miss=0.3 | miss=0.6 | miss=0.9 |
|---|---|---|---|---|---|
| `proximal_cv` | 14.716 ± 0.028 | 14.713 ± 0.023 | 14.721 ± 0.037 | 14.720 ± 0.027 | 6.345 ± 0.007 |
| `matlap_auto` | 26.169 ± 0.123 | 25.696 ± 0.124 | 24.548 ± 0.183 | 22.645 ± 0.185 | 21.333 ± 0.373 |
| `matlap_grid` | 33.995 ± 0.064 | 33.990 ± 0.053 | 34.007 ± 0.085 | 34.005 ± 0.061 | 27.459 ± 9.048 |
| `lowrank_auto` | 353.957 ± 3.521 | 371.892 ± 2.486 | 409.403 ± 2.726 | 452.878 ± 0.657 | 499.540 ± 0.033 |
| `lowrank_grid` | 14.716 ± 0.028 | 14.713 ± 0.023 | 14.721 ± 0.037 | 14.720 ± 0.027 | 14.658 ± 0.017 |
| `lowrank_cv` | 33.995 ± 0.064 | 33.990 ± 0.053 | 14.721 ± 0.037 | 14.720 ± 0.027 | 27.453 ± 9.035 |
| `iso_auto` | 26.146 ± 0.132 | 25.677 ± 0.115 | 24.530 ± 0.181 | 22.626 ± 0.171 | 13.656 ± 0.103 |
| `iso_grid` | 33.995 ± 0.064 | 33.990 ± 0.053 | 34.007 ± 0.085 | 34.005 ± 0.061 | 27.459 ± 9.048 |
| `iso_cv` | 14.716 ± 0.028 | 14.713 ± 0.023 | 14.721 ± 0.037 | 14.720 ± 0.027 | 14.658 ± 0.017 |
| `iso_grid_loo` | 33.995 ± 0.064 | 33.990 ± 0.053 | 34.007 ± 0.085 | 34.005 ± 0.061 | 33.863 ± 0.039 |
| `iso_grid_renyi` | 14.716 ± 0.028 | 14.713 ± 0.023 | 14.721 ± 0.037 | 14.720 ± 0.027 | 14.658 ± 0.017 |
| `iso_grid_is` | 33.995 ± 0.064 | 27.550 ± 9.055 | 14.721 ± 0.037 | 14.720 ± 0.027 | 14.658 ± 0.017 |
| `iso_adaptive` | 18.185 ± 0.034 | 18.182 ± 0.029 | 18.192 ± 0.045 | 18.191 ± 0.033 | 18.115 ± 0.021 |

## Runtime in seconds (mean over seeds)

| Method | miss=0.02 | miss=0.1 | miss=0.3 | miss=0.6 | miss=0.9 |
|---|---|---|---|---|---|
| `proximal_cv` | 40.7s | 41.4s | 43.3s | 43.3s | 39.7s |
| `matlap_auto` | 0.9s | 1.0s | 1.1s | 1.5s | 1.4s |
| `matlap_grid` | 6.8s | 7.2s | 8.3s | 7.8s | 7.9s |
| `lowrank_auto` | 1.7s | 2.2s | 2.0s | 1.7s | 0.1s |
| `lowrank_grid` | 3.9s | 4.5s | 3.4s | 4.1s | 4.4s |
| `lowrank_cv` | 12.7s | 13.2s | 13.3s | 14.1s | 14.0s |
| `iso_auto` | 1.5s | 1.6s | 2.1s | 3.6s | 4.5s |
| `iso_grid` | 10.5s | 11.4s | 12.0s | 11.8s | 11.4s |
| `iso_cv` | 42.4s | 42.4s | 43.0s | 42.2s | 41.9s |
| `iso_grid_loo` | 10.6s | 11.5s | 11.9s | 11.8s | 11.4s |
| `iso_grid_renyi` | 10.6s | 11.5s | 12.0s | 11.9s | 11.5s |
| `iso_grid_is` | 10.6s | 11.5s | 12.0s | 11.9s | 11.4s |
| `iso_adaptive` | 14.1s | 14.5s | 13.5s | 13.7s | 13.2s |

## Notes

- Matrix: 500×100, heteroscedastic noise σ ~ Uniform(0.5, 1.5)
- Signal scale = rank^0.25 / SNR → per-entry signal std = SNR, noise std ≈ 1
- Lambda grid: 12 log-spaced values, 10^-2×–10^2× heuristic
- Low-rank rank: 50
- `lowrank_auto` λ diverges (~m) due to rank-r trace; use iso_auto instead.
- `iso_auto`/`iso_cv`: δ* = sqrt(Tr(Ψ⊥)/(n−r)) each iteration; γ = λ̄/δ.

- **`proximal_cv`**: proximal_cv    (FISTA + 3-fold CV)
- **`matlap_auto`**: matlap_auto    (full CAVI, auto-λ)
- **`matlap_grid`**: matlap_grid    (full CAVI, best ELBO over grid)
- **`lowrank_auto`**: lowrank_auto   (rank-r CAVI, auto-λ, biased)
- **`lowrank_grid`**: lowrank_grid   (matlap_grid_lowrank, best ELBO)
- **`lowrank_cv`**: lowrank_cv     (rank-r CAVI, grid+CV)
- **`iso_auto`**: iso_auto       (lowrank+iso CAVI, auto-λ, δ learned)
- **`iso_grid`**: iso_grid       (lowrank+iso CAVI, best ELBO over grid)
- **`iso_cv`**: iso_cv         (lowrank+iso CAVI, grid+CV, δ learned)
- **`iso_grid_loo`**: iso_grid_loo   (lowrank+iso CAVI, best closed-form LOO over grid)
- **`iso_grid_renyi`**: iso_grid_renyi (lowrank+iso CAVI, best Rényi α=0.5 over grid)
- **`iso_grid_is`**: iso_grid_is    (lowrank+iso CAVI, best α=0 importance objective)
- **`iso_adaptive`**: iso_adaptive   (lowrank+iso CAVI, adaptive golden-ratio λ, Rényi α=0.5)
