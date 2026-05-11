# matlap Benchmark Report

Generated: 2026-05-11 17:35  |  Matrix: 10000×1000, rank 15  |  Missing: 20%  |  Seeds: 3

## Configuration

| Parameter | Value |
|---|---|
| Rows (m) | 10,000 |
| Columns (n) | 1,000 |
| True rank | 15 |
| Missing fraction | 20% |
| Seeds | 3 |
| FISTA iterations | 50 |
| SVI steps | 100 |
| matlap_lowrank iters | 30 |
| matlap_lowrank rank | 30 |
| VI guide rank | 5 |
| rSVD approx rank | 20 |
| Devices | CPU |

> **Note:** No GPU detected. All results are CPU-only. Re-run with a CUDA-enabled JAX install to include GPU timings.

## Methods Included

| Method | Description |
|---|---|
| `proximal` | Nuclear-norm FISTA, λ set by heuristic |
| `proximal_cv` | Nuclear-norm FISTA, λ by 2-fold entry-wise CV |
| `vi_diagonal` | SVI, fully-factorised Gaussian guide, auto-λ |
| `vi_diagonal_approx` | SVI, fully-factorised Gaussian + rSVD nuclear norm, auto-λ |
| `matlap_lowrank` | Low-rank CAVI (Woodbury, rank-r factor subspace), auto-λ |
| `vi_matrix_factor` | SVI, shared column-factor guide + rSVD, auto-λ; O(mn) memory |
| `vi_row_lowrank` | SVI, per-row low-rank guide + rSVD, auto-λ; O(mnr) memory |

## Methods Excluded

The following methods were excluded due to memory or compute constraints:

| Method | Reason |
|---|---|
| `matlap` | Stores O(m·n²) posterior covariances. At 10k×1k that is 10 000 × 1 000 × 1 000 × 4 bytes ≈ **40 GB** (OOM). |
| `matlap_grid` | Same O(m·n²) memory requirement as matlap. |
| `vi_row_mvn` | Guide stores m row-MVN covariances of size n×n ≈ 40 GB for 10k×1k (OOM). |
| `vi_matrix_normal` | Guide scale_tril_row is m×m (400 MB for m=10k); each SVI step costs O(m²·n) ≈ 10¹¹ FLOPs — impractical on CPU. |

## Results — CPU

### Test-Set RMSE

RMSE on held-out entries (lower is better).

| Method | Mean RMSE | Std RMSE | Converged (%) |
|---|---|---|---|
| proximal | 0.1228 | 0.0002 | 0% |
| proximal_cv | 0.1232 | 0.0002 | 0% |
| vi_diagonal | 0.2555 | 0.0007 | 0% |
| vi_diagonal_approx | 0.2849 | 0.0008 | 0% |
| matlap_lowrank | 0.2574 | 0.0008 | 100% |
| vi_matrix_factor | 0.2595 | 0.0007 | 0% |
| vi_row_lowrank | 0.2595 | 0.0007 | 0% |

### Lambda Agreement

Estimated regularisation strength λ per method across seeds.

| Method | Mean λ | Std λ | Min λ | Max λ |
|---|---|---|---|---|
| proximal | 50.002 | 0.008 | 49.992 | 50.009 |
| proximal_cv | 50.002 | 0.008 | 49.992 | 50.009 |
| vi_diagonal | 1.534 | 0.000 | 1.534 | 1.534 |
| vi_diagonal_approx | 1.534 | 0.000 | 1.534 | 1.534 |
| matlap_lowrank | 9965.745 | 0.240 | 9965.412 | 9965.970 |
| vi_matrix_factor | 1.534 | 0.000 | 1.534 | 1.534 |
| vi_row_lowrank | 1.534 | 0.000 | 1.534 | 1.534 |

<details><summary>Per-seed lambda values</summary>

| Seed | proximal | proximal_cv | vi_diagonal | vi_diagonal_approx | matlap_lowrank | vi_matrix_factor | vi_row_lowrank |
|---|---|---|---|---|---|---|---|
| 0 | 50.009 | 50.009 | 1.534 | 1.534 | 9965.412 | 1.534 | 1.534 |
| 1 | 49.992 | 49.992 | 1.534 | 1.534 | 9965.970 | 1.534 | 1.534 |
| 2 | 50.006 | 50.006 | 1.534 | 1.534 | 9965.853 | 1.534 | 1.534 |

</details>

### Runtimes

Wall-clock time per seed (seconds). Seed 0 may include JAX JIT compilation overhead.

| Method | Mean (s) | Std (s) | Min (s) | Max (s) |
|---|---|---|---|---|
| proximal | 49.9 | 1.8 | 48.0 | 52.3 |
| proximal_cv | 231.5 | 1.4 | 229.6 | 232.9 |
| vi_diagonal | 120.2 | 2.4 | 118.3 | 123.5 |
| vi_diagonal_approx | 64.6 | 1.6 | 62.4 | 66.1 |
| matlap_lowrank | 2.1 | 0.4 | 1.8 | 2.6 |
| vi_matrix_factor | 109.2 | 3.3 | 104.5 | 111.5 |
| vi_row_lowrank | 205.9 | 0.7 | 205.1 | 206.8 |

<details><summary>Per-seed runtimes (s)</summary>

| Seed | proximal | proximal_cv | vi_diagonal | vi_diagonal_approx | matlap_lowrank | vi_matrix_factor | vi_row_lowrank |
|---|---|---|---|---|---|---|---|
| 0 | 52.3 | 229.6 | 123.5 | 66.1 | 2.6 | 111.5 | 206.8 |
| 1 | 49.3 | 232.9 | 118.3 | 65.4 | 1.8 | 111.5 | 205.6 |
| 2 | 48.0 | 232.1 | 118.8 | 62.4 | 1.9 | 104.5 | 205.1 |

</details>

## Scalability Notes

Memory and compute scaling at 10k×1k (m=10000, n=1000).

| Method | Memory | Per-iter compute | Notes |
|---|---|---|---|
| matlap (CAVI) | O(m·n²) — **40 GB OOM** | O(m·n³) | Exact but infeasible |
| proximal | O(mn) — 40 MB | O(mn·min(m,n)) full SVD | ~1s/iter on CPU |
| vi_diagonal | O(mn) — 40 MB | O(mn·min(m,n)) full SVD | ~1s/step on CPU |
| vi_diagonal_approx | O(mn) — 40 MB | O(mn·r) rSVD, r=20 | ~30× faster per step |
| matlap_lowrank | O(mn + nr²) — ~44 MB at r=30 | O(mn·r) Woodbury | Exact in rank-r subspace |
| vi_matrix_factor | O(mn) — 40 MB | O(mn·r) rSVD, r=20 | Shared column-factor guide |
| vi_row_lowrank | O(mn·r) — ~600 MB at r=5 | O(mn·r) rSVD | Per-row low-rank covariance |
| vi_row_mvn | O(mn²) — **40 GB OOM** | O(mn³) | Infeasible |
| vi_matrix_normal | O(m²+n²) — 400 MB | O(m²n) — **impractical** | 10¹¹ FLOPs/step |
