# matlap Benchmark Report

Generated: 2026-05-25 10:16  |  Matrix: 10000×1000, rank 15  |  Missing: 20%  |  Seeds: 3

## Configuration

| Parameter | Value |
|---|---|
| Rows (m) | 10,000 |
| Columns (n) | 1,000 |
| True rank | 15 |
| Missing fraction | 20% |
| Seeds | 3 |
| FISTA iterations | 100 |
| SVI steps | 200 |
| matlap_lowrank iters | 50 |
| matlap_lowrank rank | 50 |
| VI guide rank | 15 |
| rSVD approx rank | 30 |
| Devices | GPU |

## Methods Included

| Method | Description |
|---|---|
| `proximal` | Nuclear-norm FISTA, λ set by heuristic |
| `proximal_cv` | Nuclear-norm FISTA, λ by entry-wise CV |
| `matlap_faem` | Factor Analysis EM, free subspace W_r, auto-λ (EB) |
| `matlap_gradml` | Gradient marginal LL (Adam), free subspace W_r, auto-λ |
| `matlap_lowrank` | Low-rank CAVI (Woodbury, rank-r factor subspace), auto-λ |
| `matlap_grid_lowrank` | Low-rank CAVI on λ-grid, warm-started path, best ELBO |
| `matlap_grid_lowrank_iso_elbo` | Low-rank+iso CAVI on λ-grid, warm-started path, best ELBO |
| `matlap_grid_lowrank_iso_renyi` | Low-rank+iso CAVI on λ-grid, warm-started path, best Rényi α=0.5 |
| `matlap_batched` | Full CAVI, batched rows (O(batch·n²) peak mem), auto-λ |
| `vi_diagonal` | SVI, fully-factorised Gaussian guide, auto-λ |
| `vi_diagonal_approx` | SVI, fully-factorised Gaussian + rSVD nuclear norm, auto-λ |
| `vi_matrix_factor` | SVI, shared column-factor guide + rSVD, auto-λ; O(mn) memory |
| `vi_row_lowrank` | SVI, per-row low-rank guide + rSVD, auto-λ; O(mnr) memory |
| `mcmc_mala` | Proximal MALA, cold start, heuristic λ (fixed) |
| `mcmc_gibbs` | MALA+MH Gibbs, cold start, heuristic λ init, λ sampled |

## Methods Excluded

The following methods were excluded due to memory or compute constraints:

| Method | Reason |
|---|---|
| `matlap` | O(m·n³) compute even with batching — at 10k×1k each row needs an n=1000 Cholesky (~10⁹ FLOPs); 10k rows per iter ≈ 10¹³ FLOPs/iter (infeasible). Use `matlap_batched` at n ≲ 300. |
| `matlap_grid` | Same O(m·n³) compute limit as matlap; replaced by matlap_grid_lowrank. |
| `vi_row_mvn` | Guide stores m row-MVN covariances of size n×n ≈ 40 GB for 10k×1k (OOM). |
| `vi_matrix_normal` | Guide scale_tril_row is m×m (400 MB for m=10k); each SVI step costs O(m²·n) ≈ 10¹¹ FLOPs — impractical on CPU. |

## Results — GPU

### Test-Set RMSE

RMSE on held-out entries (lower is better).

| Method | Mean RMSE | Std RMSE | Converged (%) |
|---|---|---|---|
| proximal | 0.1228 | 0.0002 | 0% |
| proximal_cv | 0.1051 | 0.0002 | 0% |
| matlap_faem | 0.0809 | 0.0002 | 100% |
| matlap_gradml | 0.0811 | 0.0002 | 100% |
| matlap_lowrank | 0.2574 | 0.0008 | 100% |
| matlap_grid_lowrank | 0.0985 | 0.0001 | 100% |
| matlap_grid_lowrank_iso_elbo | 0.1799 | 0.0002 | 100% |
| matlap_grid_lowrank_iso_renyi | 0.1341 | 0.0001 | 100% |
| matlap_batched | 0.1526 | 0.0001 | 100% |
| vi_diagonal | 0.2415 | 0.0007 | 0% |
| vi_diagonal_approx | 0.3965 | 0.0019 | 0% |
| vi_matrix_factor | 0.2688 | 0.0009 | 0% |
| vi_row_lowrank | 0.2696 | 0.0009 | 0% |

### Lambda Agreement

Estimated regularisation strength λ per method across seeds.

| Method | Mean λ | Std λ | Min λ | Max λ |
|---|---|---|---|---|
| proximal | 50.002 | 0.008 | 49.992 | 50.009 |
| proximal_cv | 158.121 | 0.024 | 158.088 | 158.144 |
| matlap_faem | 0.496 | 0.003 | 0.492 | 0.499 |
| matlap_gradml | 1.131 | 0.004 | 1.126 | 1.136 |
| matlap_lowrank | 9978.610 | 0.145 | 9978.409 | 9978.749 |
| matlap_grid_lowrank | 158.121 | 0.024 | 158.088 | 158.144 |
| matlap_grid_lowrank_iso_elbo | 500.023 | 0.076 | 499.918 | 500.095 |
| matlap_grid_lowrank_iso_renyi | 50.002 | 0.008 | 49.992 | 50.009 |
| matlap_batched | 350.985 | 0.735 | 350.117 | 351.914 |
| vi_diagonal | 2.051 | 0.000 | 2.051 | 2.051 |
| vi_diagonal_approx | 2.054 | 0.000 | 2.054 | 2.054 |
| vi_matrix_factor | 2.054 | 0.000 | 2.054 | 2.054 |
| vi_row_lowrank | 2.054 | 0.000 | 2.054 | 2.054 |

<details><summary>Per-seed lambda values</summary>

| Seed | proximal | proximal_cv | matlap_faem | matlap_gradml | matlap_lowrank | matlap_grid_lowrank | matlap_grid_lowrank_iso_elbo | matlap_grid_lowrank_iso_renyi | matlap_batched | vi_diagonal | vi_diagonal_approx | vi_matrix_factor | vi_row_lowrank |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 50.009 | 158.144 | 0.492 | 1.126 | 9978.409 | 158.144 | 500.095 | 50.009 | 350.117 | 2.051 | 2.054 | 2.054 | 2.054 |
| 1 | 49.992 | 158.088 | 0.499 | 1.136 | 9978.749 | 158.088 | 499.918 | 49.992 | 351.914 | 2.051 | 2.054 | 2.054 | 2.054 |
| 2 | 50.006 | 158.132 | 0.498 | 1.131 | 9978.672 | 158.132 | 500.058 | 50.006 | 350.923 | 2.051 | 2.054 | 2.054 | 2.054 |

</details>

### Runtimes

Wall-clock time per seed (seconds). Seed 0 may include JAX JIT compilation overhead.

| Method | Mean (s) | Std (s) | Min (s) | Max (s) |
|---|---|---|---|---|
| proximal | 22.6 | 0.7 | 22.0 | 23.5 |
| proximal_cv | 121.4 | 0.4 | 120.9 | 121.8 |
| matlap_faem | 3.7 | 3.3 | 1.3 | 8.3 |
| matlap_gradml | 3.2 | 1.4 | 2.2 | 5.2 |
| matlap_lowrank | 1.0 | 1.3 | 0.1 | 2.9 |
| matlap_grid_lowrank | 1.5 | 0.1 | 1.4 | 1.6 |
| matlap_grid_lowrank_iso_elbo | 7.1 | 0.8 | 6.5 | 8.3 |
| matlap_grid_lowrank_iso_renyi | 6.7 | 0.1 | 6.5 | 6.8 |
| matlap_batched | 186.7 | 3.2 | 182.4 | 190.0 |
| vi_diagonal | 41.9 | 1.3 | 41.0 | 43.7 |
| vi_diagonal_approx | 10.7 | 0.9 | 10.0 | 12.0 |
| vi_matrix_factor | 19.3 | 3.0 | 17.2 | 23.5 |
| vi_row_lowrank | 24.5 | 1.1 | 23.8 | 26.1 |

<details><summary>Per-seed runtimes (s)</summary>

| Seed | proximal | proximal_cv | matlap_faem | matlap_gradml | matlap_lowrank | matlap_grid_lowrank | matlap_grid_lowrank_iso_elbo | matlap_grid_lowrank_iso_renyi | matlap_batched | vi_diagonal | vi_diagonal_approx | vi_matrix_factor | vi_row_lowrank |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 23.5 | 120.9 | 8.3 | 5.2 | 2.9 | 1.6 | 8.3 | 6.8 | 190.0 | 43.7 | 12.0 | 23.5 | 26.1 |
| 1 | 22.2 | 121.8 | 1.4 | 2.2 | 0.1 | 1.5 | 6.6 | 6.6 | 187.7 | 41.0 | 10.3 | 17.3 | 23.8 |
| 2 | 22.0 | 121.5 | 1.3 | 2.3 | 0.1 | 1.4 | 6.5 | 6.5 | 182.4 | 41.0 | 10.0 | 17.2 | 23.8 |

</details>

## Scalability Notes

Memory and compute scaling at 10k×1k (m=10000, n=1000).

| Method | Memory | Per-iter compute | Notes |
|---|---|---|---|
| matlap | O(m·n²) — **40 GB OOM** | O(m·n³) | Exact but infeasible at n=1k |
| matlap_batched | O(B·n²), B=64 — 64×4MB=256 MB | O(m·n³) — **slow** at n=1k | Feasible memory; use at n≲300 |
| matlap_faem | O(mr² + mn) at r=50 — ~44 MB | O(mnr + mr³) | Free subspace; FA EM M-step |
| matlap_gradml | O(mr² + mn) at r=50 — ~44 MB | O(mnr + mr³) per step | Free subspace; Adam on marginal LL |
| matlap_lowrank | O(mn + nr²) at r=50 — ~44 MB | O(mn·r) Woodbury | Exact in rank-r subspace |
| matlap_grid_lowrank | O(mn + nr²) at r=50 | O(G·mn·r) warm path | G=7 grid pts, warm-started |
| matlap_grid_lowrank_iso_elbo | O(mn + nr²) at r=50 | O(G·mn·r) warm path | iso; G=7 grid pts, ELBO scoring |
| matlap_grid_lowrank_iso_renyi | O(mn + nr²) at r=50 | O(G·mn·r) warm path | iso; G=7 grid pts, Rényi α=0.5 |
| proximal | O(mn) — 40 MB | O(mn·min(m,n)) full SVD | ~1s/iter on CPU |
| vi_diagonal_approx | O(mn) — 40 MB | O(mn·r) rSVD, r=30 | ~30× faster per step vs full SVD |
| vi_matrix_factor | O(mn) — 40 MB | O(mn·r) rSVD, r=30 | Shared column-factor guide |
| vi_row_lowrank | O(mn·r) at r=15 — ~600 MB | O(mn·r) rSVD | Per-row low-rank covariance |
| vi_row_mvn | O(mn²) — **40 GB OOM** | O(mn³) | Infeasible |
| vi_matrix_normal | O(m²+n²) — 400 MB | O(m²n) — **impractical** | 10¹¹ FLOPs/step |
| mcmc_mala | O(mn) — 40 MB | O(mn·min(m,n)) × T steps full SVD | Gold standard; slow at large mn |
| mcmc_gibbs | O(mn) — 40 MB | O(mn·min(m,n)) × T steps full SVD | Gold standard; also samples λ |
