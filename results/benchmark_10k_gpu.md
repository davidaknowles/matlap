# matlap Benchmark Report

Generated: 2026-05-25 19:58  |  Matrix: 10000×1000, rank 15  |  Missing: 20%  |  Seeds: 10

## Configuration

| Parameter | Value |
|---|---|
| Rows (m) | 10,000 |
| Columns (n) | 1,000 |
| True rank | 15 |
| Missing fraction | 20% |
| Seeds | 10 |
| FISTA iterations | 100 |
| SVI steps | 200 |
| matlap_lowrank iters | 50 |
| matlap_lowrank rank | 50 |
| VI guide rank | 15 |
| rSVD approx rank | 30 |
| MCMC warmup steps | 300 |
| MCMC sample steps | 400 |
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
| `matlap_grid_lowrank_iso_ldlt` | Low-rank+iso CAVI on λ-grid, CUDA LDL^T kernel, best ELBO |
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
| proximal | 0.1227 | 0.0002 | 0% |
| proximal_cv | 0.1050 | 0.0001 | 0% |
| matlap_faem | 0.0808 | 0.0002 | 100% |
| matlap_gradml | 0.0810 | 0.0002 | 100% |
| matlap_lowrank | 0.2579 | 0.0017 | 100% |
| matlap_grid_lowrank | 0.0984 | 0.0001 | 100% |
| matlap_grid_lowrank_iso_elbo | 0.1131 | 0.0079 | 10% |
| matlap_grid_lowrank_iso_renyi | 0.1253 | 0.0019 | 0% |
| matlap_grid_lowrank_iso_ldlt | 0.1326 | 0.0149 | 0% |
| matlap_batched | 0.1525 | 0.0001 | 100% |
| vi_diagonal | 0.2419 | 0.0015 | 0% |
| vi_diagonal_approx | 0.3977 | 0.0027 | 0% |
| vi_matrix_factor | 0.2695 | 0.0017 | 0% |
| vi_row_lowrank | 0.2703 | 0.0017 | 0% |
| mcmc_mala | 0.2580 | 0.0017 | 100% |
| mcmc_gibbs | 0.2580 | 0.0017 | 100% |

### Lambda Agreement

Estimated regularisation strength λ per method across seeds.

| Method | Mean λ | Std λ | Min λ | Max λ |
|---|---|---|---|---|
| proximal | 50.000 | 0.007 | 49.988 | 50.009 |
| proximal_cv | 158.112 | 0.022 | 158.074 | 158.144 |
| matlap_faem | 0.494 | 0.006 | 0.484 | 0.502 |
| matlap_gradml | 1.130 | 0.005 | 1.124 | 1.135 |
| matlap_lowrank | 9978.540 | 0.254 | 9978.125 | 9978.894 |
| matlap_grid_lowrank | 158.112 | 0.022 | 158.074 | 158.144 |
| matlap_grid_lowrank_iso_elbo | 158.112 | 0.022 | 158.074 | 158.144 |
| matlap_grid_lowrank_iso_renyi | 50.000 | 0.007 | 49.988 | 50.009 |
| matlap_grid_lowrank_iso_ldlt | 158.112 | 0.022 | 158.074 | 158.144 |
| matlap_batched | 350.575 | 1.133 | 348.848 | 352.431 |
| vi_diagonal | 2.051 | 0.000 | 2.051 | 2.051 |
| vi_diagonal_approx | 2.054 | 0.000 | 2.054 | 2.054 |
| vi_matrix_factor | 2.054 | 0.000 | 2.054 | 2.054 |
| vi_row_lowrank | 2.054 | 0.000 | 2.054 | 2.054 |
| mcmc_mala | 50.000 | 0.007 | 49.988 | 50.009 |
| mcmc_gibbs | 206.823 | 0.275 | 206.484 | 207.132 |

<details><summary>Per-seed lambda values</summary>

| Seed | proximal | proximal_cv | matlap_faem | matlap_gradml | matlap_lowrank | matlap_grid_lowrank | matlap_grid_lowrank_iso_elbo | matlap_grid_lowrank_iso_renyi | matlap_grid_lowrank_iso_ldlt | matlap_batched | vi_diagonal | vi_diagonal_approx | vi_matrix_factor | vi_row_lowrank | mcmc_mala | mcmc_gibbs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 50.009 | 158.144 | 0.492 | 1.127 | 9978.412 | 158.144 | 158.144 | 50.009 | 158.144 | 349.681 | 2.051 | 2.054 | 2.054 | 2.054 | 50.009 | 206.528 |
| 1 | 49.992 | 158.088 | 0.499 | 1.134 | 9978.750 | 158.088 | 158.088 | 49.992 | 158.088 | 351.457 | 2.051 | 2.054 | 2.054 | 2.054 | 49.992 | 207.083 |
| 2 | 50.006 | 158.132 | 0.498 | 1.133 | 9978.672 | 158.132 | 158.132 | 50.006 | 158.132 | 350.923 | 2.051 | 2.054 | 2.054 | 2.054 | 50.006 | 206.675 |
| 3 | 49.999 | 158.111 | 0.485 | 1.124 | 9978.173 | 158.111 | 158.111 | 49.999 | 158.111 | 348.848 | 2.051 | 2.054 | 2.054 | 2.054 | 49.999 | 206.484 |
| 4 | 49.997 | 158.103 | 0.496 | 1.135 | 9978.597 | 158.103 | 158.103 | 49.997 | 158.103 | 350.768 | 2.051 | 2.054 | 2.054 | 2.054 | 49.997 | 207.102 |
| 5 | 50.003 | 158.125 | 0.502 | 1.135 | 9978.894 | 158.125 | 158.125 | 50.003 | 158.125 | 352.007 | 2.051 | 2.054 | 2.054 | 2.054 | 50.003 | 207.132 |
| 6 | 50.001 | 158.117 | 0.484 | 1.124 | 9978.125 | 158.117 | 158.117 | 50.001 | 158.117 | 349.065 | 2.051 | 2.054 | 2.054 | 2.054 | 50.001 | 206.549 |
| 7 | 49.988 | 158.074 | 0.493 | 1.129 | 9978.503 | 158.074 | 158.074 | 49.988 | 158.074 | 350.581 | 2.051 | 2.054 | 2.054 | 2.054 | 49.988 | 207.065 |
| 8 | 50.008 | 158.140 | 0.490 | 1.126 | 9978.402 | 158.140 | 158.140 | 50.008 | 158.140 | 349.989 | 2.051 | 2.054 | 2.054 | 2.054 | 50.008 | 206.523 |
| 9 | 49.992 | 158.090 | 0.502 | 1.135 | 9978.876 | 158.090 | 158.090 | 49.992 | 158.090 | 352.431 | 2.051 | 2.054 | 2.054 | 2.054 | 49.992 | 207.086 |

</details>

### Runtimes

Wall-clock time per seed (seconds). Seed 0 may include JAX JIT compilation overhead.

| Method | Mean (s) | Std (s) | Min (s) | Max (s) |
|---|---|---|---|---|
| proximal | 22.3 | 0.5 | 21.8 | 23.6 |
| proximal_cv | 122.0 | 2.1 | 120.6 | 128.2 |
| matlap_faem | 2.4 | 2.2 | 1.1 | 8.2 |
| matlap_gradml | 2.8 | 1.1 | 2.2 | 5.0 |
| matlap_lowrank | 0.5 | 0.9 | 0.1 | 2.9 |
| matlap_grid_lowrank | 2.1 | 0.7 | 1.1 | 3.4 |
| matlap_grid_lowrank_iso_elbo | 27.7 | 4.0 | 22.0 | 34.6 |
| matlap_grid_lowrank_iso_renyi | 27.7 | 4.0 | 22.2 | 35.8 |
| matlap_grid_lowrank_iso_ldlt | 8.6 | 2.5 | 6.5 | 15.6 |
| matlap_batched | 187.7 | 8.3 | 182.5 | 211.8 |
| vi_diagonal | 41.4 | 0.9 | 40.6 | 43.3 |
| vi_diagonal_approx | 10.5 | 0.8 | 9.8 | 12.2 |
| vi_matrix_factor | 18.7 | 2.4 | 17.1 | 23.6 |
| vi_row_lowrank | 24.6 | 1.0 | 23.5 | 26.3 |
| mcmc_mala | 252.1 | 6.7 | 248.2 | 271.9 |
| mcmc_gibbs | 252.8 | 1.8 | 250.2 | 256.1 |

<details><summary>Per-seed runtimes (s)</summary>

| Seed | proximal | proximal_cv | matlap_faem | matlap_gradml | matlap_lowrank | matlap_grid_lowrank | matlap_grid_lowrank_iso_elbo | matlap_grid_lowrank_iso_renyi | matlap_grid_lowrank_iso_ldlt | matlap_batched | vi_diagonal | vi_diagonal_approx | vi_matrix_factor | vi_row_lowrank | mcmc_mala | mcmc_gibbs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 23.6 | 120.6 | 8.2 | 5.0 | 2.9 | 2.0 | 31.5 | 30.2 | 15.6 | 185.9 | 43.3 | 12.2 | 23.6 | 26.0 | 249.2 | 250.2 |
| 1 | 22.2 | 121.5 | 1.3 | 2.4 | 0.1 | 3.4 | 23.4 | 23.7 | 8.4 | 182.5 | 40.7 | 9.9 | 17.2 | 23.5 | 248.2 | 250.7 |
| 2 | 22.0 | 121.1 | 1.4 | 2.2 | 0.1 | 1.6 | 26.3 | 26.3 | 7.5 | 182.5 | 41.6 | 10.6 | 17.7 | 23.8 | 251.0 | 253.3 |
| 3 | 22.5 | 121.6 | 4.9 | 5.0 | 1.8 | 1.9 | 32.0 | 31.0 | 7.3 | 185.6 | 42.9 | 11.8 | 23.4 | 26.0 | 249.3 | 253.7 |
| 4 | 22.1 | 121.1 | 1.3 | 2.4 | 0.1 | 3.3 | 28.2 | 28.4 | 7.9 | 182.6 | 40.6 | 10.1 | 17.1 | 23.6 | 249.7 | 252.1 |
| 5 | 22.1 | 121.6 | 1.2 | 2.3 | 0.1 | 1.7 | 22.0 | 22.2 | 9.1 | 182.6 | 40.7 | 9.8 | 17.1 | 24.1 | 251.3 | 253.3 |
| 6 | 21.8 | 121.9 | 1.4 | 2.2 | 0.1 | 1.1 | 22.6 | 22.7 | 8.2 | 187.6 | 41.2 | 10.5 | 17.6 | 24.3 | 251.2 | 256.1 |
| 7 | 23.1 | 128.2 | 1.4 | 2.4 | 0.1 | 1.5 | 34.6 | 35.8 | 6.5 | 211.8 | 41.2 | 10.5 | 18.7 | 26.3 | 271.9 | 254.7 |
| 8 | 21.9 | 121.6 | 1.4 | 2.3 | 0.1 | 1.9 | 27.5 | 27.6 | 7.2 | 187.7 | 40.7 | 9.8 | 17.2 | 23.9 | 251.0 | 253.5 |
| 9 | 22.0 | 121.1 | 1.1 | 2.4 | 0.1 | 2.7 | 29.1 | 29.4 | 8.3 | 187.7 | 41.0 | 10.0 | 17.5 | 24.2 | 248.3 | 250.5 |

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
| matlap_grid_lowrank_iso_ldlt | O(mn + nr²) at r=50 | O(G·mn·r) warm path | iso+LDL^T kernel; G=7 grid pts, ELBO scoring |
| proximal | O(mn) — 40 MB | O(mn·min(m,n)) full SVD | ~1s/iter on CPU |
| vi_diagonal_approx | O(mn) — 40 MB | O(mn·r) rSVD, r=30 | ~30× faster per step vs full SVD |
| vi_matrix_factor | O(mn) — 40 MB | O(mn·r) rSVD, r=30 | Shared column-factor guide |
| vi_row_lowrank | O(mn·r) at r=15 — ~600 MB | O(mn·r) rSVD | Per-row low-rank covariance |
| vi_row_mvn | O(mn²) — **40 GB OOM** | O(mn³) | Infeasible |
| vi_matrix_normal | O(m²+n²) — 400 MB | O(m²n) — **impractical** | 10¹¹ FLOPs/step |
| mcmc_mala | O(mn) — 40 MB | O(mn·min(m,n)) × T steps full SVD | Gold standard; slow at large mn |
| mcmc_gibbs | O(mn) — 40 MB | O(mn·min(m,n)) × T steps full SVD | Gold standard; also samples λ |
