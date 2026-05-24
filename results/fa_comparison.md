# All-methods comparison across missing fractions

Generated 2026-05-24 15:30

Settings: m=200, n=50, true_rank=5, model_rank=10, SNR=1.0, seeds=[0, 1, 2]

## RMSE on held-out entries (lower is better)

| Missing | proximal_cv      (FISTA + 3-fold CV) | batched_auto     (full CAVI, auto-λ) | batched_grid     (full CAVI, best ELBO over grid) | batched_warmstart(full CAVI, FA-EM warm-start) | iso_auto         (lowrank+iso CAVI, auto-λ) | iso_grid         (lowrank+iso CAVI, best ELBO over grid) | iso_cv           (lowrank+iso CAVI, grid+CV) | iso_warmstart    (lowrank+iso CAVI, FA-EM warm-start) | iso_then_proximal(iso λ → proximal_gradient) | iso_renyi        (lowrank+iso CAVI, Rényi α=0.5 λ) | faem             (FA EM, free subspace, Gaussian factor model) | gradml           (gradient marginal LL, free subspace) | mcmc_mala        (Proximal MALA, proximal-cv λ warm-start, 400 samples) | mcmc_gibbs       (GSM Gibbs, half-Cauchy λ, 400 samples) |
|--------:|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|
| 2%    | 0.5131 | 0.6796 | 0.7302 | 0.6795 | 0.6814 | 0.7312 | 0.6617 | 0.6619 | 0.4913 | 0.6667 | 0.4254 | 0.4264 | 0.5131 | 0.6921 |
| 10%    | 0.5433 | 0.7388 | 0.7876 | 0.7388 | 0.7408 | 0.7892 | 0.7266 | 0.7203 | 0.5325 | 0.7281 | 0.4588 | 0.4577 | 0.5433 | 0.7552 |
| 30%    | 0.5938 | 0.8397 | 0.8766 | 0.8397 | 0.8393 | 0.8764 | 0.8368 | 0.8238 | 0.6168 | 0.8332 | 0.5413 | 0.5403 | 0.5938 | 0.8668 |
| 60%    | 0.8800 | 0.9988 | 1.0025 | 0.9988 | 0.9984 | 1.0024 | 0.9985 | 0.9959 | 0.9457 | 0.9979 | 0.8525 | 0.8521 | 0.8800 | 1.0075 |
| 90%    | 0.9542 | 0.9529 | 0.9530 | 0.9529 | 0.9530 | 0.9530 | 0.9530 | 0.9530 | 0.9530 | 0.9530 | 1.0702 | 1.0789 | 0.9542 | 0.9530 |

## Auto/CV-selected λ

| Missing | proximal_cv      (FISTA + 3-fold CV) | batched_auto     (full CAVI, auto-λ) | batched_grid     (full CAVI, best ELBO over grid) | batched_warmstart(full CAVI, FA-EM warm-start) | iso_auto         (lowrank+iso CAVI, auto-λ) | iso_grid         (lowrank+iso CAVI, best ELBO over grid) | iso_cv           (lowrank+iso CAVI, grid+CV) | iso_warmstart    (lowrank+iso CAVI, FA-EM warm-start) | iso_then_proximal(iso λ → proximal_gradient) | iso_renyi        (lowrank+iso CAVI, Rényi α=0.5 λ) | faem             (FA EM, free subspace, Gaussian factor model) | gradml           (gradient marginal LL, free subspace) | mcmc_mala        (Proximal MALA, proximal-cv λ warm-start, 400 samples) | mcmc_gibbs       (GSM Gibbs, half-Cauchy λ, 400 samples) |
|--------:|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|
| 2%    | 8.404 | 15.410 | 23.385 | 15.408 | 15.382 | 23.385 | 8.404 | 15.399 | 15.382 | 11.724 | 0.331 | 0.716 | 8.404 | 18.437 |
| 10%    | 8.403 | 15.244 | 23.381 | 15.250 | 15.219 | 23.381 | 8.403 | 15.231 | 15.219 | 11.472 | 0.356 | 0.711 | 8.403 | 18.809 |
| 30%    | 8.431 | 14.785 | 23.459 | 14.785 | 14.744 | 23.459 | 8.431 | 14.751 | 14.744 | 10.876 | 0.429 | 0.713 | 8.431 | 21.125 |
| 60%    | 8.502 | 14.533 | 23.656 | 14.544 | 14.493 | 23.656 | 8.502 | 14.488 | 14.493 | 10.189 | 0.646 | 0.762 | 8.502 | 871.258 |
| 90%    | 18.311 | 13.593 | 23.441 | 13.300 | 12.829 | 23.441 | 13.554 | 12.826 | 12.829 | 26889.684 | 0.854 | 0.102 | 18.311 | 899.435 |

## Wall-clock time (seconds)

| Missing | proximal_cv      (FISTA + 3-fold CV) | batched_auto     (full CAVI, auto-λ) | batched_grid     (full CAVI, best ELBO over grid) | batched_warmstart(full CAVI, FA-EM warm-start) | iso_auto         (lowrank+iso CAVI, auto-λ) | iso_grid         (lowrank+iso CAVI, best ELBO over grid) | iso_cv           (lowrank+iso CAVI, grid+CV) | iso_warmstart    (lowrank+iso CAVI, FA-EM warm-start) | iso_then_proximal(iso λ → proximal_gradient) | iso_renyi        (lowrank+iso CAVI, Rényi α=0.5 λ) | faem             (FA EM, free subspace, Gaussian factor model) | gradml           (gradient marginal LL, free subspace) | mcmc_mala        (Proximal MALA, proximal-cv λ warm-start, 400 samples) | mcmc_gibbs       (GSM Gibbs, half-Cauchy λ, 400 samples) |
|--------:|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|
| 2%    | 9.7 | 0.5 | 0.8 | 0.7 | 0.3 | 0.4 | 1.2 | 0.2 | 0.5 | 2.4 | 0.1 | 0.3 | 3.3 | 2.9 |
| 10%    | 9.0 | 0.2 | 0.8 | 0.3 | 0.1 | 0.4 | 1.2 | 0.2 | 0.2 | 2.3 | 0.1 | 0.1 | 3.2 | 2.7 |
| 30%    | 10.1 | 0.2 | 1.0 | 0.4 | 0.1 | 0.4 | 1.2 | 0.2 | 0.4 | 1.7 | 0.1 | 0.1 | 3.2 | 2.9 |
| 60%    | 8.3 | 0.4 | 0.8 | 0.7 | 0.3 | 0.5 | 1.2 | 0.4 | 0.9 | 1.4 | 0.1 | 0.2 | 3.1 | 2.9 |
| 90%    | 8.3 | 0.0 | 0.7 | 1.7 | 0.7 | 0.4 | 1.3 | 0.7 | 0.8 | 2.0 | 0.3 | 1.6 | 3.2 | 2.9 |

## Method descriptions

- **`proximal_cv`**: proximal_cv      (FISTA + 3-fold CV)
- **`batched_auto`**: batched_auto     (full CAVI, auto-λ)
- **`batched_grid`**: batched_grid     (full CAVI, best ELBO over grid)
- **`batched_warmstart`**: batched_warmstart(full CAVI, FA-EM warm-start)
- **`iso_auto`**: iso_auto         (lowrank+iso CAVI, auto-λ)
- **`iso_grid`**: iso_grid         (lowrank+iso CAVI, best ELBO over grid)
- **`iso_cv`**: iso_cv           (lowrank+iso CAVI, grid+CV)
- **`iso_warmstart`**: iso_warmstart    (lowrank+iso CAVI, FA-EM warm-start)
- **`iso_then_proximal`**: iso_then_proximal(iso λ → proximal_gradient)
- **`iso_renyi`**: iso_renyi        (lowrank+iso CAVI, Rényi α=0.5 λ)
- **`faem`**: faem             (FA EM, free subspace, Gaussian factor model)
- **`gradml`**: gradml           (gradient marginal LL, free subspace)
- **`mcmc_mala`**: mcmc_mala        (Proximal MALA, proximal-cv λ warm-start, 400 samples)
- **`mcmc_gibbs`**: mcmc_gibbs       (GSM Gibbs, half-Cauchy λ, 400 samples)
