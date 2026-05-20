# matlap Benchmark Report

Generated: 2026-05-19 22:48  |  Matrix: 500×100, rank 10  |  Missing: 20%  |  Seeds: 3

## Configuration

| Parameter | Value |
|---|---|
| Rows (m) | 500 |
| Columns (n) | 100 |
| True rank | 10 |
| Missing fraction | 20% |
| Seeds | 3 |
| FISTA iterations | 50 |
| SVI steps | 100 |
| matlap_lowrank iters | 30 |
| matlap_lowrank rank | 15 |
| VI guide rank | 15 |
| rSVD approx rank | 30 |
| Devices | CPU |

> **Note:** No GPU detected. All results are CPU-only. Re-run with a CUDA-enabled JAX install to include GPU timings.

## Methods Included

| Method | Description |
|---|---|
| `proximal` | Nuclear-norm FISTA, λ set by heuristic |
| `proximal_cv` | Nuclear-norm FISTA, λ by entry-wise CV |
| `matlap_faem` | Factor Analysis EM, free subspace W_r, auto-λ (EB) |
| `matlap_gradml` | Gradient marginal LL (Adam), free subspace W_r, auto-λ |
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
| proximal | 0.2346 | 0.0032 | 0% |
| proximal_cv | 0.2120 | 0.0043 | 0% |
| matlap_faem | 0.1950 | 0.0014 | 100% |
| matlap_gradml | 0.1952 | 0.0015 | 100% |
| matlap_lowrank | 0.3178 | 0.0115 | 0% |
| matlap_grid_lowrank | 0.2103 | 0.0036 | 100% |
| matlap_batched | 0.2759 | 0.0081 | 33% |
| vi_diagonal | 0.3234 | 0.0120 | 0% |
| vi_diagonal_approx | 0.3274 | 0.0120 | 0% |
| vi_matrix_factor | 0.3250 | 0.0123 | 0% |
| vi_row_lowrank | 0.3250 | 0.0122 | 0% |

### Lambda Agreement

Estimated regularisation strength λ per method across seeds.

| Method | Mean λ | Std λ | Min λ | Max λ |
|---|---|---|---|---|
| proximal | 11.178 | 0.011 | 11.166 | 11.193 |
| proximal_cv | 35.348 | 0.035 | 35.311 | 35.395 |
| matlap_faem | 1.082 | 0.030 | 1.040 | 1.108 |
| matlap_gradml | 1.224 | 0.016 | 1.206 | 1.246 |
| matlap_lowrank | 274.870 | 12.546 | 257.127 | 283.766 |
| matlap_grid_lowrank | 35.348 | 0.035 | 35.311 | 35.395 |
| matlap_batched | 65.972 | 1.285 | 64.156 | 66.956 |
| vi_diagonal | 1.533 | 0.000 | 1.533 | 1.533 |
| vi_diagonal_approx | 1.534 | 0.000 | 1.534 | 1.534 |
| vi_matrix_factor | 1.533 | 0.000 | 1.533 | 1.533 |
| vi_row_lowrank | 1.533 | 0.000 | 1.533 | 1.533 |

<details><summary>Per-seed lambda values</summary>

| Seed | proximal | proximal_cv | matlap_faem | matlap_gradml | matlap_lowrank | matlap_grid_lowrank | matlap_batched | vi_diagonal | vi_diagonal_approx | vi_matrix_factor | vi_row_lowrank |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 11.175 | 35.337 | 1.040 | 1.246 | 257.127 | 35.337 | 64.156 | 1.533 | 1.534 | 1.533 | 1.533 |
| 1 | 11.166 | 35.311 | 1.099 | 1.222 | 283.717 | 35.311 | 66.803 | 1.533 | 1.534 | 1.533 | 1.533 |
| 2 | 11.193 | 35.395 | 1.108 | 1.206 | 283.766 | 35.395 | 66.956 | 1.533 | 1.534 | 1.533 | 1.533 |

</details>

### Runtimes

Wall-clock time per seed (seconds). Seed 0 may include JAX JIT compilation overhead.

| Method | Mean (s) | Std (s) | Min (s) | Max (s) |
|---|---|---|---|---|
| proximal | 85.1 | 25.9 | 50.1 | 111.9 |
| proximal_cv | 479.9 | 83.7 | 411.6 | 597.7 |
| matlap_faem | 1.3 | 1.0 | 0.5 | 2.7 |
| matlap_gradml | 1.2 | 0.5 | 0.7 | 1.9 |
| matlap_lowrank | 0.6 | 0.4 | 0.3 | 1.1 |
| matlap_grid_lowrank | 0.5 | 0.0 | 0.5 | 0.5 |
| matlap_batched | 212.7 | 62.4 | 125.5 | 268.1 |
| vi_diagonal | 157.6 | 15.6 | 141.5 | 178.8 |
| vi_diagonal_approx | 27.1 | 3.0 | 23.0 | 30.2 |
| vi_matrix_factor | 28.8 | 1.9 | 26.2 | 30.7 |
| vi_row_lowrank | 28.7 | 3.7 | 23.7 | 32.6 |

<details><summary>Per-seed runtimes (s)</summary>

| Seed | proximal | proximal_cv | matlap_faem | matlap_gradml | matlap_lowrank | matlap_grid_lowrank | matlap_batched | vi_diagonal | vi_diagonal_approx | vi_matrix_factor | vi_row_lowrank |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 50.1 | 597.7 | 2.7 | 1.9 | 1.1 | 0.5 | 268.1 | 178.8 | 30.2 | 30.7 | 32.6 |
| 1 | 111.9 | 430.3 | 0.6 | 0.8 | 0.3 | 0.5 | 125.5 | 152.6 | 23.0 | 26.2 | 23.7 |
| 2 | 93.1 | 411.6 | 0.5 | 0.7 | 0.3 | 0.5 | 244.5 | 141.5 | 28.2 | 29.4 | 29.9 |

</details>

## Scalability Notes

Memory and compute scaling at 10k×1k (m=10000, n=1000).

| Method | Memory | Per-iter compute | Notes |
|---|---|---|---|
| matlap | O(m·n²) — **40 GB OOM** | O(m·n³) | Exact but infeasible at n=1k |
| matlap_batched | O(B·n²), B=64 — 64×4MB=256 MB | O(m·n³) — **slow** at n=1k | Feasible memory; use at n≲300 |
| matlap_faem | O(mr² + mn) at r=15 — ~44 MB | O(mnr + mr³) | Free subspace; FA EM M-step |
| matlap_gradml | O(mr² + mn) at r=15 — ~44 MB | O(mnr + mr³) per step | Free subspace; Adam on marginal LL |
| matlap_lowrank | O(mn + nr²) at r=15 — ~44 MB | O(mn·r) Woodbury | Exact in rank-r subspace |
| matlap_grid_lowrank | O(mn + nr²) at r=15 | O(G·mn·r) warm path | G=7 grid pts, warm-started |
| proximal | O(mn) — 40 MB | O(mn·min(m,n)) full SVD | ~1s/iter on CPU |
| vi_diagonal_approx | O(mn) — 40 MB | O(mn·r) rSVD, r=30 | ~30× faster per step vs full SVD |
| vi_matrix_factor | O(mn) — 40 MB | O(mn·r) rSVD, r=30 | Shared column-factor guide |
| vi_row_lowrank | O(mn·r) at r=15 — ~600 MB | O(mn·r) rSVD | Per-row low-rank covariance |
| vi_row_mvn | O(mn²) — **40 GB OOM** | O(mn³) | Infeasible |
| vi_matrix_normal | O(m²+n²) — 400 MB | O(m²n) — **impractical** | 10¹¹ FLOPs/step |
