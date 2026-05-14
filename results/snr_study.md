# matlap Lambda Selection Study — SNR Sweep

Generated: 2026-05-14 14:15  |  Matrix: 500×100  |  rank=5  |  Seeds: 3  |  Low-rank rank: 50  |  CV folds: 3  |  Grid points: 8

SNR = signal per-entry std / mean noise std  ≈  signal_scale / 1

## Test RMSE (mean ± std over seeds)

| Method | SNR=0.1 | SNR=0.3 | SNR=1.0 | SNR=3.0 | SNR=10.0 |
|---|---|---|---|---|---|
| `proximal_cv` | 0.102 ± 0.001 | 0.282 ± 0.002 | 0.368 ± 0.003 | 0.402 ± 0.004 | 0.414 ± 0.004 |
| `matlap_auto` | 0.102 ± 0.001 | 0.298 ± 0.003 | 0.518 ± 0.006 | 0.582 ± 0.008 | 0.671 ± 0.012 |
| `matlap_grid` | 0.102 ± 0.001 | 0.299 ± 0.003 | 0.539 ± 0.006 | 0.581 ± 0.008 | 0.728 ± 0.013 |
| `lowrank_auto` | 0.102 ± 0.001 | 0.306 ± 0.004 | 1.020 ± 0.012 | 0.815 ± 0.012 | 1.380 ± 0.046 |
| `lowrank_grid` | 0.118 ± 0.001 | 0.310 ± 0.003 | 0.578 ± 0.007 | 0.846 ± 0.012 | 1.316 ± 0.041 |
| `lowrank_cv` | 0.102 ± 0.001 | 0.294 ± 0.004 | 0.563 ± 0.017 | 0.672 ± 0.010 | 0.793 ± 0.022 |
| `iso_auto` | 0.102 ± 0.001 | 0.298 ± 0.003 | 0.523 ± 0.006 | 0.602 ± 0.008 | 0.800 ± 0.018 |
| `iso_grid` | 0.102 ± 0.001 | 0.299 ± 0.003 | 0.543 ± 0.006 | 0.600 ± 0.008 | 0.850 ± 0.019 |
| `iso_cv` | 0.103 ± 0.003 | 0.274 ± 0.002 | 0.505 ± 0.005 | 0.600 ± 0.008 | 0.744 ± 0.016 |
| `iso_grid_loo` | 0.102 ± 0.001 | 0.299 ± 0.003 | 0.543 ± 0.006 | 0.645 ± 0.009 | 0.788 ± 0.017 |
| `iso_grid_renyi` | 0.108 ± 0.001 | 0.274 ± 0.002 | 0.505 ± 0.005 | 0.600 ± 0.008 | 0.744 ± 0.017 |
| `iso_grid_is` | 0.108 ± 0.001 | 0.274 ± 0.002 | 0.505 ± 0.005 | 0.645 ± 0.009 | 0.788 ± 0.017 |

## Chosen λ (mean ± std over seeds)

| Method | SNR=0.1 | SNR=0.3 | SNR=1.0 | SNR=3.0 | SNR=10.0 |
|---|---|---|---|---|---|
| `proximal_cv` | 31.062 ± 0.038 | 16.088 ± 0.020 | 16.088 ± 0.020 | 16.088 ± 0.020 | 16.088 ± 0.020 |
| `matlap_auto` | 69.361 ± 0.094 | 57.275 ± 0.381 | 25.163 ± 0.173 | 13.308 ± 0.083 | 5.972 ± 0.046 |
| `matlap_grid` | 223.548 ± 0.276 | 59.971 ± 0.074 | 31.062 ± 0.038 | 16.088 ± 0.020 | 4.316 ± 0.005 |
| `lowrank_auto` | 499.688 ± 0.001 | 499.328 ± 0.126 | 392.119 ± 3.516 | 9.169 ± 0.063 | 3.749 ± 0.026 |
| `lowrank_grid` | 59.971 ± 0.074 | 31.062 ± 0.038 | 16.088 ± 0.020 | 8.333 ± 0.010 | 4.316 ± 0.005 |
| `lowrank_cv` | 223.548 ± 0.276 | 59.971 ± 0.074 | 21.088 ± 7.089 | 16.088 ± 0.020 | 31.062 ± 0.038 |
| `iso_auto` | 69.147 ± 0.087 | 57.268 ± 0.372 | 25.145 ± 0.180 | 13.279 ± 0.085 | 5.919 ± 0.046 |
| `iso_grid` | 223.548 ± 0.276 | 59.971 ± 0.074 | 31.062 ± 0.038 | 16.088 ± 0.020 | 4.316 ± 0.005 |
| `iso_cv` | 40.733 ± 27.223 | 16.088 ± 0.020 | 16.088 ± 0.020 | 16.088 ± 0.020 | 16.088 ± 0.020 |
| `iso_grid_loo` | 223.548 ± 0.276 | 59.971 ± 0.074 | 31.062 ± 0.038 | 31.062 ± 0.038 | 31.062 ± 0.038 |
| `iso_grid_renyi` | 16.088 ± 0.020 | 16.088 ± 0.020 | 16.088 ± 0.020 | 16.088 ± 0.020 | 16.088 ± 0.020 |
| `iso_grid_is` | 16.088 ± 0.020 | 16.088 ± 0.020 | 16.088 ± 0.020 | 31.062 ± 0.038 | 31.062 ± 0.038 |

## Runtime in seconds (mean over seeds)

| Method | SNR=0.1 | SNR=0.3 | SNR=1.0 | SNR=3.0 | SNR=10.0 |
|---|---|---|---|---|---|
| `proximal_cv` | 30.6s | 32.4s | 37.6s | 38.5s | 44.0s |
| `matlap_auto` | 3.7s | 3.2s | 1.2s | 0.7s | 0.6s |
| `matlap_grid` | 4.3s | 4.3s | 5.2s | 10.9s | 12.2s |
| `lowrank_auto` | 0.1s | 0.1s | 1.8s | 1.0s | 0.6s |
| `lowrank_grid` | 2.1s | 2.4s | 3.3s | 5.8s | 6.3s |
| `lowrank_cv` | 9.3s | 9.4s | 12.4s | 22.4s | 31.9s |
| `iso_auto` | 4.6s | 4.7s | 1.9s | 1.2s | 1.4s |
| `iso_grid` | 6.1s | 6.1s | 7.2s | 13.9s | 21.8s |
| `iso_cv` | 25.6s | 25.8s | 28.3s | 57.2s | 79.6s |
| `iso_grid_loo` | 6.1s | 6.2s | 7.3s | 14.0s | 21.8s |
| `iso_grid_renyi` | 6.1s | 6.2s | 7.2s | 13.9s | 21.9s |
| `iso_grid_is` | 6.1s | 6.2s | 7.2s | 13.9s | 21.8s |

## Notes

- Matrix: 500×100, heteroscedastic noise σ ~ Uniform(0.5, 1.5)
- Signal scale = rank^0.25 / snr → Var(X_entry) = snr² (std = snr, noise std ≈ 1)
- Test fraction: 20% of observed entries
- Low-rank rank: 50
- Lambda grid: 8 log-spaced values, 0.1×–10× heuristic
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
