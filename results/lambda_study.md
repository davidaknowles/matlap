# matlap Lambda Selection Study — Rank Sweep

Generated: 2026-05-14 13:36  |  Matrix: 500×100  |  SNR=1  |  Seeds: 3  |  Low-rank rank: 50  |  CV folds: 3  |  Grid points: 8

## Test RMSE (mean ± std over seeds)

| Method | rank=1 | rank=3 | rank=5 | rank=10 | rank=20 | rank=40 |
|---|---|---|---|---|---|---|
| `proximal_cv` | 0.230 ± 0.004 | 0.310 ± 0.007 | 0.368 ± 0.003 | 0.481 ± 0.007 | 0.626 ± 0.008 | 0.786 ± 0.009 |
| `matlap_auto` | 0.255 ± 0.005 | 0.419 ± 0.008 | 0.518 ± 0.006 | 0.669 ± 0.015 | 0.807 ± 0.012 | 0.908 ± 0.010 |
| `matlap_grid` | 0.257 ± 0.004 | 0.433 ± 0.009 | 0.539 ± 0.006 | 0.698 ± 0.017 | 0.836 ± 0.013 | 0.927 ± 0.011 |
| `lowrank_auto` | 1.055 ± 0.028 | 1.031 ± 0.023 | 1.020 ± 0.012 | 1.022 ± 0.038 | 1.027 ± 0.017 | 1.025 ± 0.014 |
| `lowrank_grid` | 0.448 ± 0.006 | 0.528 ± 0.009 | 0.578 ± 0.007 | 0.661 ± 0.009 | 0.754 ± 0.003 | 0.851 ± 0.012 |
| `lowrank_cv` | 0.315 ± 0.006 | 0.447 ± 0.011 | 0.563 ± 0.017 | 0.661 ± 0.009 | 0.754 ± 0.003 | 0.860 ± 0.012 |
| `iso_auto` | 0.259 ± 0.005 | 0.423 ± 0.009 | 0.523 ± 0.006 | 0.672 ± 0.015 | 0.810 ± 0.011 | 0.910 ± 0.011 |
| `iso_grid` | 0.260 ± 0.004 | 0.436 ± 0.010 | 0.543 ± 0.006 | 0.700 ± 0.017 | 0.838 ± 0.013 | 0.929 ± 0.011 |
| `iso_cv` | 0.245 ± 0.004 | 0.407 ± 0.009 | 0.505 ± 0.005 | 0.654 ± 0.016 | 0.796 ± 0.012 | 0.901 ± 0.011 |
| `iso_grid_loo` | 0.337 ± 0.005 | 0.436 ± 0.010 | 0.543 ± 0.006 | 0.700 ± 0.017 | 0.838 ± 0.013 | 0.929 ± 0.011 |
| `iso_grid_renyi` | 0.245 ± 0.004 | 0.407 ± 0.009 | 0.505 ± 0.005 | 0.654 ± 0.015 | 0.796 ± 0.012 | 0.901 ± 0.011 |
| `iso_grid_is` | 0.260 ± 0.004 | 0.407 ± 0.009 | 0.505 ± 0.005 | 0.654 ± 0.015 | 0.796 ± 0.012 | 0.901 ± 0.011 |

## Chosen λ (mean ± std over seeds)

| Method | rank=1 | rank=3 | rank=5 | rank=10 | rank=20 | rank=40 |
|---|---|---|---|---|---|---|
| `proximal_cv` | 16.088 ± 0.020 | 16.088 ± 0.020 | 16.088 ± 0.020 | 16.088 ± 0.020 | 16.088 ± 0.020 | 16.088 ± 0.020 |
| `matlap_auto` | 30.114 ± 0.527 | 26.382 ± 0.187 | 25.163 ± 0.173 | 23.743 ± 0.460 | 22.764 ± 0.190 | 22.486 ± 0.235 |
| `matlap_grid` | 31.062 ± 0.038 | 31.062 ± 0.038 | 31.062 ± 0.038 | 31.062 ± 0.038 | 31.062 ± 0.038 | 31.062 ± 0.038 |
| `lowrank_auto` | 342.829 ± 24.351 | 383.631 ± 4.369 | 392.119 ± 3.516 | 396.310 ± 6.186 | 398.045 ± 3.130 | 404.628 ± 3.108 |
| `lowrank_grid` | 16.088 ± 0.020 | 16.088 ± 0.020 | 16.088 ± 0.020 | 16.088 ± 0.020 | 16.088 ± 0.020 | 16.088 ± 0.020 |
| `lowrank_cv` | 31.062 ± 0.038 | 31.062 ± 0.038 | 21.088 ± 7.089 | 16.088 ± 0.020 | 16.088 ± 0.020 | 31.062 ± 0.038 |
| `iso_auto` | 30.128 ± 0.526 | 26.368 ± 0.181 | 25.145 ± 0.180 | 23.711 ± 0.459 | 22.716 ± 0.187 | 22.429 ± 0.233 |
| `iso_grid` | 31.062 ± 0.038 | 31.062 ± 0.038 | 31.062 ± 0.038 | 31.062 ± 0.038 | 31.062 ± 0.038 | 31.062 ± 0.038 |
| `iso_cv` | 16.088 ± 0.020 | 16.088 ± 0.020 | 16.088 ± 0.020 | 16.088 ± 0.020 | 16.088 ± 0.020 | 16.088 ± 0.020 |
| `iso_grid_loo` | 59.971 ± 0.074 | 31.062 ± 0.038 | 31.062 ± 0.038 | 31.062 ± 0.038 | 31.062 ± 0.038 | 31.062 ± 0.038 |
| `iso_grid_renyi` | 16.088 ± 0.020 | 16.088 ± 0.020 | 16.088 ± 0.020 | 16.088 ± 0.020 | 16.088 ± 0.020 | 16.088 ± 0.020 |
| `iso_grid_is` | 31.062 ± 0.038 | 16.088 ± 0.020 | 16.088 ± 0.020 | 16.088 ± 0.020 | 16.088 ± 0.020 | 16.088 ± 0.020 |

## Runtime in seconds (mean over seeds)

| Method | rank=1 | rank=3 | rank=5 | rank=10 | rank=20 | rank=40 |
|---|---|---|---|---|---|---|
| `proximal_cv` | 31.2s | 30.2s | 32.0s | 36.9s | 42.3s | 38.5s |
| `matlap_auto` | 1.7s | 1.3s | 1.2s | 1.1s | 1.1s | 1.1s |
| `matlap_grid` | 6.3s | 5.7s | 5.3s | 4.8s | 4.9s | 4.5s |
| `lowrank_auto` | 2.2s | 2.1s | 2.1s | 1.8s | 2.1s | 2.1s |
| `lowrank_grid` | 5.0s | 3.1s | 3.1s | 3.2s | 2.7s | 2.5s |
| `lowrank_cv` | 13.5s | 11.2s | 10.9s | 10.0s | 9.9s | 10.0s |
| `iso_auto` | 2.3s | 1.9s | 1.9s | 1.8s | 1.7s | 1.7s |
| `iso_grid` | 8.7s | 7.3s | 7.3s | 7.0s | 6.9s | 6.8s |
| `iso_cv` | 33.4s | 29.6s | 28.3s | 27.3s | 26.7s | 26.8s |
| `iso_grid_loo` | 8.8s | 7.4s | 7.3s | 7.0s | 6.9s | 6.8s |
| `iso_grid_renyi` | 8.8s | 7.4s | 7.3s | 7.0s | 6.9s | 6.8s |
| `iso_grid_is` | 8.8s | 7.4s | 7.3s | 7.0s | 6.9s | 6.8s |

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
