# Proximal Lambda CV Benchmark

- device: `cuda:0`
- shapes: `[(100, 1000)]`
- n_seeds: `1`
- lambda_grid: `[0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]`
- lambda_true_for_simulation: `5.0`
- gamma2_true_for_simulation: `0.25`
- missing_frac: `0.3`
- max_iter: `80`
- fixed_iter: `True`
- cv_folds: `3`
- svd_rank: `exact`
- taylor_scores: `True`
- raw CSV: `results/benchmark_proximal_lambda_cv_taylor_100x1000_raw.csv`

| shape | selector | lambda | CV MSE | X RMSE | train Y RMSE | test Y RMSE | median time s |
|---|---|---:|---:|---:|---:|---:|---:|
| 100x1000 | fixed_lambda | 0.5 | nan | 3.265453 | 0.018844 | 5.935087 | 0.314 |
| 100x1000 | fixed_lambda | 1 | nan | 3.264779 | 0.037713 | 5.934018 | 0.313 |
| 100x1000 | fixed_lambda | 2 | nan | 3.264750 | 0.075539 | 5.933696 | 0.313 |
| 100x1000 | fixed_lambda | 4 | nan | 3.265801 | 0.151141 | 5.933547 | 0.313 |
| 100x1000 | fixed_lambda | 8 | nan | 3.271469 | 0.302381 | 5.933080 | 0.314 |
| 100x1000 | fixed_lambda | 16 | nan | 3.297941 | 0.604885 | 5.933196 | 0.316 |
| 100x1000 | fixed_lambda | 32 | nan | 3.406917 | 1.209827 | 5.933102 | 0.323 |
| 100x1000 | fixed_lambda | 64 | nan | 3.821661 | 2.419733 | 5.933105 | 0.336 |
| 100x1000 | fixed_lambda | 128 | nan | 5.103055 | 4.764963 | 5.875129 | 0.349 |
| 100x1000 | oracle_x_grid | 2 | nan | 3.264750 | 0.075539 | 5.933696 | 2.892 |
| 100x1000 | oracle_test_y_grid | 128 | nan | 5.103055 | 4.764963 | 5.875129 | 2.892 |
| 100x1000 | taylor_elbo_lambda | 4 | nan | 3.265801 | 0.151141 | 5.933547 | 7.828 |
| 100x1000 | taylor_loo_lambda | 4 | nan | 3.265801 | 0.151141 | 5.933547 | 7.828 |
| 100x1000 | taylor_elbo_score | 0.5 | 35927.679688 | nan | nan | nan | 4.936 |
| 100x1000 | taylor_loo_score | 0.5 | -266650.906250 | nan | nan | nan | 4.936 |
| 100x1000 | taylor_elbo_score | 1 | 87200.562500 | nan | nan | nan | 4.936 |
| 100x1000 | taylor_loo_score | 1 | -246070.468750 | nan | nan | nan | 4.936 |
| 100x1000 | taylor_elbo_score | 2 | 130913.476562 | nan | nan | nan | 4.936 |
| 100x1000 | taylor_loo_score | 2 | -229175.000000 | nan | nan | nan | 4.936 |
| 100x1000 | taylor_elbo_score | 4 | 159730.234375 | nan | nan | nan | 4.936 |
| 100x1000 | taylor_loo_score | 4 | -219606.125000 | nan | nan | nan | 4.936 |
| 100x1000 | taylor_elbo_score | 8 | 159631.343750 | nan | nan | nan | 4.936 |
| 100x1000 | taylor_loo_score | 8 | -224724.718750 | nan | nan | nan | 4.936 |
| 100x1000 | taylor_elbo_score | 16 | 105196.265625 | nan | nan | nan | 4.936 |
| 100x1000 | taylor_loo_score | 16 | -259209.687500 | nan | nan | nan | 4.936 |
| 100x1000 | taylor_elbo_score | 32 | -44009.093750 | nan | nan | nan | 4.936 |
| 100x1000 | taylor_loo_score | 32 | -352402.125000 | nan | nan | nan | 4.936 |
| 100x1000 | taylor_elbo_score | 64 | -328269.750000 | nan | nan | nan | 4.936 |
| 100x1000 | taylor_loo_score | 64 | -563033.250000 | nan | nan | nan | 4.936 |
| 100x1000 | taylor_elbo_score | 128 | -745506.375000 | nan | nan | nan | 4.936 |
| 100x1000 | taylor_loo_score | 128 | -1000787.125000 | nan | nan | nan | 4.936 |
| 100x1000 | cv_lambda | 128 | 33.303207 | 5.103055 | 4.764963 | 5.875129 | 9.057 |

## Selection Scores

| shape | score | lambda | value |
|---|---|---:|---:|
| 100x1000 | taylor_elbo_score | 0.5 | 35927.679688 |
| 100x1000 | taylor_loo_score | 0.5 | -266650.906250 |
| 100x1000 | taylor_elbo_score | 1 | 87200.562500 |
| 100x1000 | taylor_loo_score | 1 | -246070.468750 |
| 100x1000 | taylor_elbo_score | 2 | 130913.476562 |
| 100x1000 | taylor_loo_score | 2 | -229175.000000 |
| 100x1000 | taylor_elbo_score | 4 | 159730.234375 |
| 100x1000 | taylor_loo_score | 4 | -219606.125000 |
| 100x1000 | taylor_elbo_score | 8 | 159631.343750 |
| 100x1000 | taylor_loo_score | 8 | -224724.718750 |
| 100x1000 | taylor_elbo_score | 16 | 105196.265625 |
| 100x1000 | taylor_loo_score | 16 | -259209.687500 |
| 100x1000 | taylor_elbo_score | 32 | -44009.093750 |
| 100x1000 | taylor_loo_score | 32 | -352402.125000 |
| 100x1000 | taylor_elbo_score | 64 | -328269.750000 |
| 100x1000 | taylor_loo_score | 64 | -563033.250000 |
| 100x1000 | taylor_elbo_score | 128 | -745506.375000 |
| 100x1000 | taylor_loo_score | 128 | -1000787.125000 |
| 100x1000 | cv_score | 0.5 | 35.807596 |
| 100x1000 | cv_score | 1 | 35.792643 |
| 100x1000 | cv_score | 2 | 35.774438 |
| 100x1000 | cv_score | 4 | 35.772040 |
| 100x1000 | cv_score | 8 | 35.769145 |
| 100x1000 | cv_score | 16 | 35.767893 |
| 100x1000 | cv_score | 32 | 35.766773 |
| 100x1000 | cv_score | 64 | 35.628654 |
| 100x1000 | cv_score | 128 | 33.303207 |
