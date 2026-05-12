# matlap Benchmark Report

Generated: 2026-05-12 07:23  |  Matrix: 10000×1000, rank 15  |  Missing: 20%  |  Seeds: 3

## Configuration

| Parameter | Value |
|---|---|
| Rows (m) | 10,000 |
| Columns (n) | 1,000 |
| True rank | 15 |
| Missing fraction | 20% |
| Seeds | 3 |
| FISTA iterations | 300 |
| SVI steps | 1000 |
| matlap_lowrank iters | 80 |
| matlap_lowrank rank | 30 |
| VI guide rank | 20 |
| rSVD approx rank | 30 |
| Devices | CPU, GPU |

## Methods Included

| Method | Description |
|---|---|
| `proximal` | Nuclear-norm FISTA, λ set by heuristic |
| `proximal_cv` | Nuclear-norm FISTA, λ by entry-wise CV |
| `matlap_lowrank` | Low-rank CAVI (Woodbury, rank-r factor subspace), auto-λ |
| `matlap_grid_lowrank` | Low-rank CAVI on λ-grid, warm-started path, best ELBO |
| `matlap_batched` | Full CAVI, batched rows (O(batch·n²) peak mem), auto-λ |
| `vi_diagonal` | SVI, fully-factorised Gaussian guide, auto-λ |
| `vi_diagonal_approx` | SVI, fully-factorised Gaussian + rSVD nuclear norm, auto-λ |
| `vi_matrix_factor` | SVI, shared column-factor guide + rSVD, auto-λ; O(mn) memory |
| `vi_row_lowrank` | SVI, per-row low-rank guide + rSVD, auto-λ; O(mnr) memory |

## Methods Excluded

The following methods were excluded due to memory or compute constraints:

| Method | Reason |
|---|---|
| `matlap` | O(m·n³) compute even with batching — at 10k×1k each row needs an n=1000 Cholesky (~10⁹ FLOPs); 10k rows per iter ≈ 10¹³ FLOPs/iter (infeasible). Use `matlap_batched` at n ≲ 300. |
| `matlap_grid` | Same O(m·n³) compute limit as matlap; replaced by matlap_grid_lowrank. |
| `vi_row_mvn` | Guide stores m row-MVN covariances of size n×n ≈ 40 GB for 10k×1k (OOM). |
| `vi_matrix_normal` | Guide scale_tril_row is m×m (400 MB for m=10k); each SVI step costs O(m²·n) ≈ 10¹¹ FLOPs — impractical on CPU. |

## Results — CPU

### Test-Set RMSE

RMSE on held-out entries (lower is better).

| Method | Mean RMSE | Std RMSE | Converged (%) |
|---|---|---|---|
| proximal | 0.1228 | 0.0002 | 0% |
| proximal_cv | 0.1051 | 0.0002 | 0% |
| matlap_lowrank | 0.2574 | 0.0008 | 100% |
| matlap_grid_lowrank | 0.0924 | 0.0001 | 100% |
| matlap_batched | 0.1526 | 0.0001 | 100% |
| vi_diagonal | 0.2069 | 0.0003 | 0% |
| vi_diagonal_approx | 0.6537 | 0.0034 | 0% |
| vi_matrix_factor | 0.5304 | 0.0031 | 0% |
| vi_row_lowrank | 0.5287 | 0.0031 | 0% |

### Lambda Agreement

Estimated regularisation strength λ per method across seeds.

| Method | Mean λ | Std λ | Min λ | Max λ |
|---|---|---|---|---|
| proximal | 50.002 | 0.008 | 49.992 | 50.009 |
| proximal_cv | 158.121 | 0.024 | 158.088 | 158.144 |
| matlap_lowrank | 9965.745 | 0.240 | 9965.412 | 9965.970 |
| matlap_grid_lowrank | 50.002 | 0.008 | 49.992 | 50.009 |
| matlap_batched | 350.855 | 0.550 | 350.140 | 351.479 |
| vi_diagonal | 19.139 | 0.000 | 19.139 | 19.140 |
| vi_diagonal_approx | 22.976 | 0.000 | 22.976 | 22.976 |
| vi_matrix_factor | 23.035 | 0.000 | 23.035 | 23.035 |
| vi_row_lowrank | 23.035 | 0.000 | 23.035 | 23.035 |

<details><summary>Per-seed lambda values</summary>

| Seed | proximal | proximal_cv | matlap_lowrank | matlap_grid_lowrank | matlap_batched | vi_diagonal | vi_diagonal_approx | vi_matrix_factor | vi_row_lowrank |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 50.009 | 158.144 | 9965.412 | 50.009 | 350.140 | 19.140 | 22.976 | 23.035 | 23.035 |
| 1 | 49.992 | 158.088 | 9965.970 | 49.992 | 351.479 | 19.139 | 22.976 | 23.035 | 23.035 |
| 2 | 50.006 | 158.132 | 9965.853 | 50.006 | 350.945 | 19.139 | 22.976 | 23.035 | 23.035 |

</details>

### Runtimes

Wall-clock time per seed (seconds). Seed 0 may include JAX JIT compilation overhead.

| Method | Mean (s) | Std (s) | Min (s) | Max (s) |
|---|---|---|---|---|
| proximal | 286.5 | 2.4 | 283.1 | 288.2 |
| proximal_cv | 1604.2 | 11.7 | 1589.9 | 1618.7 |
| matlap_lowrank | 2.1 | 0.4 | 1.8 | 2.6 |
| matlap_grid_lowrank | 36.0 | 0.4 | 35.4 | 36.4 |
| matlap_batched | 3608.3 | 33.2 | 3584.8 | 3655.3 |
| vi_diagonal | 1152.4 | 26.2 | 1115.5 | 1174.3 |
| vi_diagonal_approx | 655.4 | 16.8 | 631.7 | 669.1 |
| vi_matrix_factor | 1989.0 | 16.3 | 1966.8 | 2005.4 |
| vi_row_lowrank | 5285.0 | 43.1 | 5225.1 | 5324.6 |

<details><summary>Per-seed runtimes (s)</summary>

| Seed | proximal | proximal_cv | matlap_lowrank | matlap_grid_lowrank | matlap_batched | vi_diagonal | vi_diagonal_approx | vi_matrix_factor | vi_row_lowrank |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 283.1 | 1589.9 | 2.6 | 35.4 | 3655.3 | 1174.3 | 665.5 | 2005.4 | 5324.6 |
| 1 | 288.2 | 1604.0 | 1.8 | 36.2 | 3584.8 | 1167.4 | 669.1 | 1994.8 | 5305.4 |
| 2 | 288.1 | 1618.7 | 1.9 | 36.4 | 3584.8 | 1115.5 | 631.7 | 1966.8 | 5225.1 |

</details>

## Results — GPU

### Test-Set RMSE

RMSE on held-out entries (lower is better).

| Method | Mean RMSE | Std RMSE | Converged (%) |
|---|---|---|---|
| proximal | 0.1228 | 0.0002 | 0% |
| proximal_cv | 0.1051 | 0.0002 | 0% |
| matlap_lowrank | 0.2574 | 0.0008 | 100% |
| matlap_grid_lowrank | 0.0920 | 0.0001 | 100% |
| matlap_batched | 0.1526 | 0.0001 | 100% |
| vi_diagonal | 0.2069 | 0.0003 | 0% |
| vi_diagonal_approx | 0.6537 | 0.0034 | 0% |
| vi_matrix_factor | 0.5304 | 0.0031 | 0% |
| vi_row_lowrank | 0.5287 | 0.0031 | 0% |

### Lambda Agreement

Estimated regularisation strength λ per method across seeds.

| Method | Mean λ | Std λ | Min λ | Max λ |
|---|---|---|---|---|
| proximal | 50.002 | 0.008 | 49.992 | 50.009 |
| proximal_cv | 158.121 | 0.024 | 158.088 | 158.144 |
| matlap_lowrank | 9965.768 | 0.234 | 9965.443 | 9965.985 |
| matlap_grid_lowrank | 50.002 | 0.008 | 49.992 | 50.009 |
| matlap_batched | 350.837 | 0.819 | 349.680 | 351.456 |
| vi_diagonal | 19.139 | 0.000 | 19.139 | 19.140 |
| vi_diagonal_approx | 22.976 | 0.000 | 22.976 | 22.976 |
| vi_matrix_factor | 23.035 | 0.000 | 23.035 | 23.035 |
| vi_row_lowrank | 23.035 | 0.000 | 23.035 | 23.035 |

<details><summary>Per-seed lambda values</summary>

| Seed | proximal | proximal_cv | matlap_lowrank | matlap_grid_lowrank | matlap_batched | vi_diagonal | vi_diagonal_approx | vi_matrix_factor | vi_row_lowrank |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 50.009 | 158.144 | 9965.443 | 50.009 | 349.680 | 19.140 | 22.976 | 23.035 | 23.035 |
| 1 | 49.992 | 158.088 | 9965.985 | 49.992 | 351.456 | 19.139 | 22.976 | 23.035 | 23.035 |
| 2 | 50.006 | 158.132 | 9965.876 | 50.006 | 351.375 | 19.139 | 22.976 | 23.035 | 23.035 |

</details>

### Runtimes

Wall-clock time per seed (seconds). Seed 0 may include JAX JIT compilation overhead.

| Method | Mean (s) | Std (s) | Min (s) | Max (s) |
|---|---|---|---|---|
| proximal | 66.7 | 0.7 | 65.9 | 67.6 |
| proximal_cv | 345.4 | 0.6 | 344.5 | 346.0 |
| matlap_lowrank | 1.7 | 2.3 | 0.0 | 4.9 |
| matlap_grid_lowrank | 0.6 | 0.0 | 0.6 | 0.6 |
| matlap_batched | 184.9 | 2.0 | 182.2 | 187.1 |
| vi_diagonal | 202.8 | 1.0 | 202.0 | 204.2 |
| vi_diagonal_approx | 51.6 | 0.3 | 51.3 | 51.9 |
| vi_matrix_factor | 93.4 | 2.8 | 91.0 | 97.3 |
| vi_row_lowrank | 141.4 | 0.7 | 140.4 | 142.2 |

<details><summary>Per-seed runtimes (s)</summary>

| Seed | proximal | proximal_cv | matlap_lowrank | matlap_grid_lowrank | matlap_batched | vi_diagonal | vi_diagonal_approx | vi_matrix_factor | vi_row_lowrank |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 67.6 | 344.5 | 4.9 | 0.6 | 185.4 | 204.2 | 51.6 | 97.3 | 142.2 |
| 1 | 66.6 | 346.0 | 0.0 | 0.6 | 182.2 | 202.1 | 51.3 | 91.8 | 140.4 |
| 2 | 65.9 | 345.6 | 0.0 | 0.6 | 187.1 | 202.0 | 51.9 | 91.0 | 141.5 |

</details>

## Scalability Notes

Memory and compute scaling at 10k×1k (m=10000, n=1000).

| Method | Memory | Per-iter compute | Notes |
|---|---|---|---|
| matlap | O(m·n²) — **40 GB OOM** | O(m·n³) | Exact but infeasible at n=1k |
| matlap_batched | O(B·n²), B=64 — 64×4MB=256 MB | O(m·n³) — **slow** at n=1k | Feasible memory; use at n≲300 |
| matlap_lowrank | O(mn + nr²) at r=30 — ~44 MB | O(mn·r) Woodbury | Exact in rank-r subspace |
| matlap_grid_lowrank | O(mn + nr²) at r=30 | O(G·mn·r) warm path | G=7 grid pts, warm-started |
| proximal | O(mn) — 40 MB | O(mn·min(m,n)) full SVD | ~1s/iter on CPU |
| vi_diagonal_approx | O(mn) — 40 MB | O(mn·r) rSVD, r=30 | ~30× faster per step vs full SVD |
| vi_matrix_factor | O(mn) — 40 MB | O(mn·r) rSVD, r=30 | Shared column-factor guide |
| vi_row_lowrank | O(mn·r) at r=20 — ~600 MB | O(mn·r) rSVD | Per-row low-rank covariance |
| vi_row_mvn | O(mn²) — **40 GB OOM** | O(mn³) | Infeasible |
| vi_matrix_normal | O(m²+n²) — 400 MB | O(m²n) — **impractical** | 10¹¹ FLOPs/step |
