# matlap Lambda Selection Study — Rank Sweep

Generated: 2026-05-19 10:49  |  Matrix: 500×100  |  SNR=1  |  missing=20%  |  Seeds: 3  |  Low-rank rank: 50  |  CV folds: 3  |  Grid: 12 pts, 10^-2×–10^2× heuristic

## Test RMSE (mean ± std over seeds)

| Method | rank=1 | rank=3 | rank=5 | rank=10 | rank=20 | rank=40 |
|---|---|---|---|---|---|---|
| `proximal_cv` | 0.244 ± 0.004 | 0.320 ± 0.007 | 0.376 ± 0.003 | 0.487 ± 0.006 | 0.630 ± 0.008 | 0.787 ± 0.009 |
| `matlap_auto` | 0.255 ± 0.005 | 0.419 ± 0.008 | 0.518 ± 0.006 | 0.669 ± 0.015 | 0.807 ± 0.012 | 0.908 ± 0.010 |
| `matlap_grid` | 0.263 ± 0.005 | 0.443 ± 0.009 | 0.551 ± 0.006 | 0.711 ± 0.018 | 0.847 ± 0.013 | 0.934 ± 0.011 |
| `lowrank_auto` | 1.055 ± 0.028 | 1.031 ± 0.023 | 1.020 ± 0.012 | 1.022 ± 0.038 | 1.027 ± 0.017 | 1.025 ± 0.014 |
| `lowrank_grid` | 0.474 ± 0.006 | 0.548 ± 0.009 | 0.595 ± 0.007 | 0.672 ± 0.009 | 0.761 ± 0.003 | 0.858 ± 0.012 |
| `lowrank_cv` | 0.308 ± 0.006 | 0.448 ± 0.011 | 0.595 ± 0.007 | 0.672 ± 0.009 | 0.761 ± 0.003 | 0.869 ± 0.012 |
| `iso_auto` | 0.259 ± 0.005 | 0.423 ± 0.009 | 0.523 ± 0.006 | 0.672 ± 0.015 | 0.810 ± 0.011 | 0.910 ± 0.011 |
| `iso_grid` | 0.266 ± 0.005 | 0.446 ± 0.010 | 0.555 ± 0.006 | 0.713 ± 0.018 | 0.849 ± 0.013 | 0.936 ± 0.011 |
| `iso_cv` | 0.245 ± 0.004 | 0.407 ± 0.009 | 0.504 ± 0.005 | 0.653 ± 0.016 | 0.794 ± 0.011 | 0.899 ± 0.011 |
| `iso_grid_loo` | 0.266 ± 0.005 | 0.446 ± 0.010 | 0.555 ± 0.006 | 0.713 ± 0.018 | 0.849 ± 0.013 | 0.936 ± 0.011 |
| `iso_grid_renyi` | 0.245 ± 0.004 | 0.407 ± 0.009 | 0.504 ± 0.005 | 0.653 ± 0.015 | 0.794 ± 0.011 | 0.900 ± 0.011 |
| `iso_grid_is` | 0.266 ± 0.005 | 0.407 ± 0.009 | 0.504 ± 0.005 | 0.653 ± 0.015 | 0.794 ± 0.011 | 0.900 ± 0.011 |
| `iso_adaptive` | 0.245 ± 0.004 | 0.409 ± 0.009 | 0.507 ± 0.005 | 0.658 ± 0.016 | 0.799 ± 0.012 | 0.903 ± 0.011 |

## Chosen λ (mean ± std over seeds)

| Method | rank=1 | rank=3 | rank=5 | rank=10 | rank=20 | rank=40 |
|---|---|---|---|---|---|---|
| `proximal_cv` | 14.708 ± 0.018 | 14.708 ± 0.018 | 14.708 ± 0.018 | 14.708 ± 0.018 | 14.708 ± 0.018 | 14.708 ± 0.018 |
| `matlap_auto` | 30.114 ± 0.527 | 26.382 ± 0.187 | 25.163 ± 0.173 | 23.743 ± 0.460 | 22.764 ± 0.190 | 22.486 ± 0.235 |
| `matlap_grid` | 33.977 ± 0.042 | 33.977 ± 0.042 | 33.977 ± 0.042 | 33.977 ± 0.042 | 33.977 ± 0.042 | 33.977 ± 0.042 |
| `lowrank_auto` | 342.829 ± 24.351 | 383.631 ± 4.369 | 392.119 ± 3.516 | 396.310 ± 6.186 | 398.045 ± 3.130 | 404.628 ± 3.108 |
| `lowrank_grid` | 14.708 ± 0.018 | 14.708 ± 0.018 | 14.708 ± 0.018 | 14.708 ± 0.018 | 14.708 ± 0.018 | 14.708 ± 0.018 |
| `lowrank_cv` | 33.977 ± 0.042 | 33.977 ± 0.042 | 14.708 ± 0.018 | 14.708 ± 0.018 | 14.708 ± 0.018 | 33.977 ± 0.042 |
| `iso_auto` | 30.128 ± 0.526 | 26.368 ± 0.181 | 25.145 ± 0.180 | 23.711 ± 0.459 | 22.716 ± 0.187 | 22.429 ± 0.233 |
| `iso_grid` | 33.977 ± 0.042 | 33.977 ± 0.042 | 33.977 ± 0.042 | 33.977 ± 0.042 | 33.977 ± 0.042 | 33.977 ± 0.042 |
| `iso_cv` | 14.708 ± 0.018 | 14.708 ± 0.018 | 14.708 ± 0.018 | 14.708 ± 0.018 | 14.708 ± 0.018 | 14.708 ± 0.018 |
| `iso_grid_loo` | 33.977 ± 0.042 | 33.977 ± 0.042 | 33.977 ± 0.042 | 33.977 ± 0.042 | 33.977 ± 0.042 | 33.977 ± 0.042 |
| `iso_grid_renyi` | 14.708 ± 0.018 | 14.708 ± 0.018 | 14.708 ± 0.018 | 14.708 ± 0.018 | 14.708 ± 0.018 | 14.708 ± 0.018 |
| `iso_grid_is` | 33.977 ± 0.042 | 14.708 ± 0.018 | 14.708 ± 0.018 | 14.708 ± 0.018 | 14.708 ± 0.018 | 14.708 ± 0.018 |
| `iso_adaptive` | 18.176 ± 0.022 | 18.176 ± 0.022 | 18.176 ± 0.022 | 18.176 ± 0.022 | 18.176 ± 0.022 | 18.176 ± 0.022 |

## Runtime in seconds (mean over seeds)

| Method | rank=1 | rank=3 | rank=5 | rank=10 | rank=20 | rank=40 |
|---|---|---|---|---|---|---|
| `proximal_cv` | 41.2s | 41.9s | 43.9s | 42.9s | 46.1s | 44.9s |
| `matlap_auto` | 1.6s | 1.2s | 1.4s | 1.3s | 1.1s | 1.0s |
| `matlap_grid` | 9.5s | 8.1s | 7.9s | 7.9s | 8.1s | 7.2s |
| `lowrank_auto` | 2.1s | 1.7s | 1.7s | 1.7s | 1.7s | 2.0s |
| `lowrank_grid` | 5.0s | 4.1s | 4.1s | 3.8s | 4.0s | 3.7s |
| `lowrank_cv` | 16.0s | 13.7s | 13.0s | 12.4s | 12.3s | 12.1s |
| `iso_auto` | 2.3s | 1.9s | 1.9s | 1.7s | 1.7s | 1.7s |
| `iso_grid` | 12.7s | 12.1s | 11.8s | 11.4s | 11.1s | 11.0s |
| `iso_cv` | 48.6s | 43.5s | 43.1s | 42.0s | 41.7s | 41.6s |
| `iso_grid_loo` | 12.8s | 12.2s | 11.8s | 11.4s | 11.1s | 11.1s |
| `iso_grid_renyi` | 12.8s | 12.2s | 11.9s | 11.4s | 11.1s | 11.0s |
| `iso_grid_is` | 12.7s | 12.2s | 11.8s | 11.4s | 11.1s | 11.1s |
| `iso_adaptive` | 14.7s | 13.9s | 13.8s | 13.4s | 13.1s | 12.9s |

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
