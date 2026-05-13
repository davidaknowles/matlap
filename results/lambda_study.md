# matlap Lambda Selection Study — Rank Sweep

Generated: 2026-05-13 12:39  |  Matrix: 500×100  |  SNR=1  |  Seeds: 3  |  Low-rank rank: 50  |  CV folds: 3  |  Grid points: 8

## Test RMSE (mean ± std over seeds)

| Method | rank=1 | rank=3 | rank=5 | rank=10 | rank=20 | rank=40 |
|---|---|---|---|---|---|---|
| `proximal_cv` | 0.230 ± 0.004 | 0.310 ± 0.007 | 0.368 ± 0.003 | 0.481 ± 0.007 | 0.626 ± 0.008 | 0.786 ± 0.009 |
| `matlap_auto` | 0.255 ± 0.005 | 0.419 ± 0.008 | 0.518 ± 0.006 | 0.669 ± 0.015 | 0.807 ± 0.012 | 0.908 ± 0.010 |
| `lowrank_auto` | 1.055 ± 0.028 | 1.031 ± 0.023 | 1.020 ± 0.012 | 1.022 ± 0.038 | 1.027 ± 0.017 | 1.025 ± 0.014 |
| `lowrank_grid` | 0.448 ± 0.006 | 0.528 ± 0.009 | 0.578 ± 0.007 | 0.661 ± 0.009 | 0.754 ± 0.003 | 0.851 ± 0.012 |
| `lowrank_cv` | 0.315 ± 0.006 | 0.447 ± 0.011 | 0.563 ± 0.017 | 0.661 ± 0.009 | 0.754 ± 0.003 | 0.860 ± 0.012 |
| `iso_auto` | 0.259 ± 0.005 | 0.423 ± 0.009 | 0.523 ± 0.006 | 0.672 ± 0.015 | 0.810 ± 0.011 | 0.910 ± 0.011 |
| `iso_cv` | 0.245 ± 0.004 | 0.407 ± 0.009 | 0.505 ± 0.005 | 0.654 ± 0.016 | 0.796 ± 0.012 | 0.901 ± 0.011 |

## Chosen λ (mean ± std over seeds)

| Method | rank=1 | rank=3 | rank=5 | rank=10 | rank=20 | rank=40 |
|---|---|---|---|---|---|---|
| `proximal_cv` | 16.088 ± 0.020 | 16.088 ± 0.020 | 16.088 ± 0.020 | 16.088 ± 0.020 | 16.088 ± 0.020 | 16.088 ± 0.020 |
| `matlap_auto` | 30.114 ± 0.527 | 26.382 ± 0.187 | 25.163 ± 0.173 | 23.743 ± 0.460 | 22.764 ± 0.190 | 22.486 ± 0.235 |
| `lowrank_auto` | 342.829 ± 24.351 | 383.631 ± 4.369 | 392.119 ± 3.516 | 396.310 ± 6.186 | 398.045 ± 3.130 | 404.628 ± 3.108 |
| `lowrank_grid` | 16.088 ± 0.020 | 16.088 ± 0.020 | 16.088 ± 0.020 | 16.088 ± 0.020 | 16.088 ± 0.020 | 16.088 ± 0.020 |
| `lowrank_cv` | 31.062 ± 0.038 | 31.062 ± 0.038 | 21.088 ± 7.089 | 16.088 ± 0.020 | 16.088 ± 0.020 | 31.062 ± 0.038 |
| `iso_auto` | 30.128 ± 0.526 | 26.368 ± 0.181 | 25.145 ± 0.180 | 23.711 ± 0.459 | 22.716 ± 0.187 | 22.429 ± 0.233 |
| `iso_cv` | 16.088 ± 0.020 | 16.088 ± 0.020 | 16.088 ± 0.020 | 16.088 ± 0.020 | 16.088 ± 0.020 | 16.088 ± 0.020 |

## Runtime in seconds (mean over seeds)

| Method | rank=1 | rank=3 | rank=5 | rank=10 | rank=20 | rank=40 |
|---|---|---|---|---|---|---|
| `proximal_cv` | 32.4s | 31.7s | 34.6s | 41.5s | 44.0s | 41.2s |
| `matlap_auto` | 1.7s | 1.3s | 1.2s | 1.1s | 1.1s | 1.1s |
| `lowrank_auto` | 2.2s | 1.8s | 2.3s | 2.0s | 1.8s | 2.1s |
| `lowrank_grid` | 4.9s | 3.5s | 3.1s | 3.0s | 3.2s | 2.8s |
| `lowrank_cv` | 14.4s | 11.8s | 11.1s | 11.0s | 10.7s | 10.0s |
| `iso_auto` | 2.3s | 1.9s | 1.9s | 1.8s | 1.7s | 1.7s |
| `iso_cv` | 33.7s | 29.8s | 28.6s | 27.5s | 26.9s | 27.0s |

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
- **`lowrank_auto`**: lowrank_auto   (rank-r CAVI, auto-λ, biased)
- **`lowrank_grid`**: lowrank_grid   (matlap_grid_lowrank, best ELBO)
- **`lowrank_cv`**: lowrank_cv     (rank-r CAVI, grid+CV)
- **`iso_auto`**: iso_auto       (lowrank+iso CAVI, auto-λ, δ learned)
- **`iso_cv`**: iso_cv         (lowrank+iso CAVI, grid+CV, δ learned)
