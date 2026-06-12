# Proximal Noise EB Benchmark

- device: `cuda:0`
- shapes: `[(30, 30)]`
- n_seeds: `1`
- lambda_true: `0.05`
- g_true: `0.25`
- train missing fraction: `0.3`
- lambda_init: `2.0`
- lambda_grid: `[1.0, 4.0, 16.0]`
- parameterizations: `['effective', 'base']`
- selectors: `['loo', 'taylor_elbo']`
- cv_frac: `0.2`
- max_outer: `5`
- prox_max_iter: `25`
- raw CSV: `results/benchmark_prox_noise_effective_vs_base_30x30_raw.csv`

| shape | method | param | X RMSE | test Y RMSE | g hat | lambda grid/base | lambda eff | median time s | median outer iters | converged |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 30x30 | grid_taylor_elbo_effective | effective | 2.710647 | 4.904734 | 0.001473 | 1.000000 | 1.000000 | 6.144 | 5.0 | 0/1 |
| 30x30 | grid_loo_effective | effective | 2.675180 | 4.810536 | 0.049570 | 4.000000 | 4.000000 | 2.474 | 5.0 | 0/1 |
| 30x30 | grid_taylor_elbo_base | base | 2.676665 | 4.830501 | 0.588999 | 1.000000 | 1.302995 | 4.815 | 5.0 | 0/1 |
| 30x30 | grid_loo_base | base | 2.676665 | 4.830501 | 0.588999 | 1.000000 | 1.302995 | 5.472 | 5.0 | 0/1 |
