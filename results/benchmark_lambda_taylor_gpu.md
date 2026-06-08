# Lambda Benchmark: Taylor GPU

- GPU: `cuda:0`
- N_SEEDS: `1`
- MAX_ITER: `8`
- TAYLOR_ITER: `8`
- PROX_INIT_ITER: `12`
- RANK: `30`
- SIGMA_NOISE: `1.0`
- LAM_GRID_BATCHED: `[0.1, 0.3, 0.7, 1.5, 3.0, 7.0, 15.0, 30.0, 70.0]`
- LAM_GRID_LR: `[0.003, 0.007, 0.015, 0.03, 0.07, 0.15, 0.3, 0.7, 1.5, 3.0]`
- CONDITION_LIMIT: `1`

## SNR Sweep

### RMSE Mean

| Method | SNR=high |
|---|---:|
| noisy_Y | 0.9998 |
| proximal_cv | 1.0158 |
| batched_eb | 0.9956 |
| batched_elbo | 0.9959 |
| batched_loo | 1.0130 |
| batched_renyi | 0.9976 |
| lowrank_elbo | 5.9024 |
| lowrank_loo | 5.9030 |
| lowrank_renyi | 5.9030 |
| iso_elbo | 1.2203 |
| iso_loo | 0.9980 |
| iso_renyi | 1.6330 |
| taylor_elbo | 0.9972 |
| taylor_loo | 1.0144 |
| taylor_renyi | 0.9984 |
| taylor_prox_elbo | 0.9973 |
| taylor_prox_loo | 1.0146 |
| taylor_prox_renyi | 0.9983 |
| prox_taylor_elbo | 0.9957 |
| prox_taylor_loo | 1.0158 |
| prox_taylor_renyi | 0.9975 |

### Lambda Median

| Method | SNR=high |
|---|---:|
| noisy_Y | nan |
| proximal_cv | 3.0000 |
| batched_eb | 0.8388 |
| batched_elbo | 0.7000 |
| batched_loo | 3.0000 |
| batched_renyi | 0.3000 |
| lowrank_elbo | 0.3000 |
| lowrank_loo | 3.0000 |
| lowrank_renyi | 3.0000 |
| iso_elbo | 0.1500 |
| iso_loo | 0.0150 |
| iso_renyi | 0.3000 |
| taylor_elbo | 0.7000 |
| taylor_loo | 3.0000 |
| taylor_renyi | 0.3000 |
| taylor_prox_elbo | 0.7000 |
| taylor_prox_loo | 3.0000 |
| taylor_prox_renyi | 0.3000 |
| prox_taylor_elbo | 0.7000 |
| prox_taylor_loo | 3.0000 |
| prox_taylor_renyi | 0.3000 |

### Log Lambda Ratio

| Method | SNR=high |
|---|---:|
| noisy_Y | nan |
| proximal_cv | 5.7038 |
| batched_eb | 4.4294 |
| batched_elbo | 4.2485 |
| batched_loo | 5.7038 |
| batched_renyi | 3.4012 |
| lowrank_elbo | 3.4012 |
| lowrank_loo | 5.7038 |
| lowrank_renyi | 5.7038 |
| iso_elbo | 2.7081 |
| iso_loo | 0.4055 |
| iso_renyi | 3.4012 |
| taylor_elbo | 4.2485 |
| taylor_loo | 5.7038 |
| taylor_renyi | 3.4012 |
| taylor_prox_elbo | 4.2485 |
| taylor_prox_loo | 5.7038 |
| taylor_prox_renyi | 3.4012 |
| prox_taylor_elbo | 4.2485 |
| prox_taylor_loo | 5.7038 |
| prox_taylor_renyi | 3.4012 |

### Median Time (s)

| Method | SNR=high |
|---|---:|
| noisy_Y | 0.0000 |
| proximal_cv | 4.2690 |
| batched_eb | 3.3181 |
| batched_elbo | 0.8700 |
| batched_loo | 0.4111 |
| batched_renyi | 0.4117 |
| lowrank_elbo | 4.4700 |
| lowrank_loo | 0.0571 |
| lowrank_renyi | 0.2743 |
| iso_elbo | 2.1287 |
| iso_loo | 0.1334 |
| iso_renyi | 0.1604 |
| taylor_elbo | 15.5713 |
| taylor_loo | 2.3767 |
| taylor_renyi | 2.3528 |
| taylor_prox_elbo | 2.2035 |
| taylor_prox_loo | 2.2024 |
| taylor_prox_renyi | 2.1926 |
| prox_taylor_elbo | 0.3284 |
| prox_taylor_loo | 0.3445 |
| prox_taylor_renyi | 0.3286 |

## Dimension Sweep

### RMSE Mean

| Method | 40×40 |
|---|---:|
| noisy_Y | 0.9870 |
| proximal_cv | 0.9794 |
| batched_eb | 0.9570 |
| batched_elbo | 0.9576 |
| batched_loo | 0.9695 |
| batched_renyi | 0.9682 |
| lowrank_elbo | 0.9886 |
| lowrank_loo | 0.9886 |
| lowrank_renyi | 0.9996 |
| iso_elbo | 0.9657 |
| iso_loo | 0.9657 |
| iso_renyi | 0.9839 |
| taylor_elbo | 0.9623 |
| taylor_loo | 0.9622 |
| taylor_renyi | 0.9717 |
| taylor_prox_elbo | 0.9918 |
| taylor_prox_loo | 0.9828 |
| taylor_prox_renyi | 0.9778 |
| prox_taylor_elbo | 0.9586 |
| prox_taylor_loo | 0.9794 |
| prox_taylor_renyi | 0.9671 |

### Lambda Median

| Method | 40×40 |
|---|---:|
| noisy_Y | nan |
| proximal_cv | 3.0000 |
| batched_eb | 1.6474 |
| batched_elbo | 1.5000 |
| batched_loo | 3.0000 |
| batched_renyi | 0.7000 |
| lowrank_elbo | 1.5000 |
| lowrank_loo | 1.5000 |
| lowrank_renyi | 3.0000 |
| iso_elbo | 1.5000 |
| iso_loo | 1.5000 |
| iso_renyi | 3.0000 |
| taylor_elbo | 1.5000 |
| taylor_loo | 3.0000 |
| taylor_renyi | 0.7000 |
| taylor_prox_elbo | 1.5000 |
| taylor_prox_loo | 3.0000 |
| taylor_prox_renyi | 0.7000 |
| prox_taylor_elbo | 1.5000 |
| prox_taylor_loo | 3.0000 |
| prox_taylor_renyi | 0.7000 |

### Log Lambda Ratio

| Method | 40×40 |
|---|---:|
| noisy_Y | nan |
| proximal_cv | 4.0943 |
| batched_eb | 3.4949 |
| batched_elbo | 3.4012 |
| batched_loo | 4.0943 |
| batched_renyi | 2.6391 |
| lowrank_elbo | 3.4012 |
| lowrank_loo | 3.4012 |
| lowrank_renyi | 4.0943 |
| iso_elbo | 3.4012 |
| iso_loo | 3.4012 |
| iso_renyi | 4.0943 |
| taylor_elbo | 3.4012 |
| taylor_loo | 4.0943 |
| taylor_renyi | 2.6391 |
| taylor_prox_elbo | 3.4012 |
| taylor_prox_loo | 4.0943 |
| taylor_prox_renyi | 2.6391 |
| prox_taylor_elbo | 3.4012 |
| prox_taylor_loo | 4.0943 |
| prox_taylor_renyi | 2.6391 |

### Median Time (s)

| Method | 40×40 |
|---|---:|
| noisy_Y | 0.0000 |
| proximal_cv | 2.3270 |
| batched_eb | 1.3668 |
| batched_elbo | 0.4266 |
| batched_loo | 0.3701 |
| batched_renyi | 0.2779 |
| lowrank_elbo | 2.6775 |
| lowrank_loo | 0.1029 |
| lowrank_renyi | 0.2533 |
| iso_elbo | 1.6333 |
| iso_loo | 0.1483 |
| iso_renyi | 0.1759 |
| taylor_elbo | 4.9249 |
| taylor_loo | 0.9189 |
| taylor_renyi | 0.9105 |
| taylor_prox_elbo | 0.7895 |
| taylor_prox_loo | 0.8237 |
| taylor_prox_renyi | 0.8023 |
| prox_taylor_elbo | 0.2074 |
| prox_taylor_loo | 0.2100 |
| prox_taylor_renyi | 0.2014 |
