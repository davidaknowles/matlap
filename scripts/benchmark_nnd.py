#!/usr/bin/env python
"""Benchmark on data generated from the Nuclear Norm Distribution (NND).

Unlike the standard low-rank Gaussian benchmark, here X is sampled from
the NND p(X) ∝ exp(-λ‖X‖_*) via the SVD representation:

    σᵢ ~ i.i.d. Gamma(m-n+1, rate=λ_true)   [= Exp(λ) for m=n]
    U  ~ Stiefel St(n, m),  V ~ O(n)  (Haar measure)
    X  = U diag(σ) V^T

All n singular values are nonzero, so the data is full rank.  This
eliminates model-data mismatch for the batched CAVI (which places a full-rank
prior on each row) while the low-rank models (which project onto a rank-r
subspace) are structurally misspecified.

Square case m=n=100 with λ_true=0.05 gives singular values ~ Exp(0.05),
mean = 20.  Observation noise σ=1 → noise SV threshold ≈ 2σ√n = 20, so
~37 % of singular values are detectable.

Methods compared
----------------
batched_eb        : matlap_batched with empirical-Bayes λ
batched_loo       : matlap_grid_batched, LOO scoring
batched_renyi     : matlap_grid_batched, Rényi scoring
lowrank_r{k}      : matlap_grid_lowrank, ranks k ∈ {5,10,20,30,50}, LOO
iso_r{k}          : matlap_grid_lowrank_isotropic, ranks k ∈ {10,20,30}, LOO
"""

from __future__ import annotations

import os
import sys
import time

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matlap import (
    matlap_batched,
    matlap_grid_batched,
    matlap_grid_lowrank,
    matlap_grid_lowrank_isotropic,
    sample_nnd,
)

# ── Experiment parameters ──────────────────────────────────────────────────────
M = 100            # rows
N = 100            # columns  (square → SVs ~ Exp(λ_true))
LAM_TRUE = 0.05   # NND λ; mean SV = 1/λ = 20
SIGMA_NOISE = 1.0  # observation noise std (homoscedastic)
N_SEEDS = 10
MAX_ITER = 100

# Lambda grid: spans 3 orders of magnitude around the true λ
LAM_GRID = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

LOWRANK_RANKS = [5, 10, 20, 30, 50]
ISO_RANKS = [10, 20, 30]


# ── Benchmark per seed ─────────────────────────────────────────────────────────
def run_seed(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    X_true, sigma_true = sample_nnd(rng, M, N, LAM_TRUE)
    Y = jnp.array(X_true + rng.standard_normal((M, N)) * SIGMA_NOISE)
    S = SIGMA_NOISE * jnp.ones((M, N))

    def rmse(mu) -> float:
        return float(jnp.sqrt(jnp.mean((jnp.array(mu) - X_true) ** 2)))

    results: dict[str, dict] = {}

    # batched: empirical-Bayes λ
    t0 = time.time()
    res = matlap_batched(Y, S, max_iter=MAX_ITER)
    results["batched_eb"] = dict(
        rmse=rmse(res.mu), lam=float(res.lambda_bar), t=time.time() - t0
    )

    # batched: grid + LOO
    t0 = time.time()
    res = matlap_grid_batched(Y, S, LAM_GRID, max_iter=MAX_ITER, score_fn="loo")
    results["batched_loo"] = dict(
        rmse=rmse(res.best_result.mu), lam=float(res.best_lambda), t=time.time() - t0
    )

    # batched: grid + Rényi
    t0 = time.time()
    res = matlap_grid_batched(Y, S, LAM_GRID, max_iter=MAX_ITER, score_fn="renyi")
    results["batched_renyi"] = dict(
        rmse=rmse(res.best_result.mu), lam=float(res.best_lambda), t=time.time() - t0
    )

    # lowrank with different ranks, LOO scoring
    for rank in LOWRANK_RANKS:
        t0 = time.time()
        res = matlap_grid_lowrank(
            Y, S, LAM_GRID, rank=rank, max_iter=MAX_ITER, score_fn="loo"
        )
        results[f"lowrank_r{rank}"] = dict(
            rmse=rmse(res.best_result.mu), lam=float(res.best_lambda), t=time.time() - t0
        )

    # iso+lowrank with different ranks, LOO scoring
    for rank in ISO_RANKS:
        t0 = time.time()
        res = matlap_grid_lowrank_isotropic(
            Y, S, LAM_GRID, rank=rank, max_iter=MAX_ITER, score_fn="loo"
        )
        results[f"iso_r{rank}"] = dict(
            rmse=rmse(res.best_result.mu), lam=float(res.best_lambda), t=time.time() - t0
        )

    return results, sigma_true


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    noise_sv_threshold = 2 * SIGMA_NOISE * N**0.5
    frac_detectable = np.exp(-LAM_TRUE * noise_sv_threshold)

    print(f"NND Benchmark: m={M}, n={N} (square), λ_true={LAM_TRUE}, σ_noise={SIGMA_NOISE}")
    print(f"  SVs ~ Exp({LAM_TRUE}): mean={1/LAM_TRUE:.1f}, noise SV threshold≈{noise_sv_threshold:.1f}")
    print(f"  Expected detectable SVs: ~{frac_detectable*N:.0f}/{N} ({frac_detectable:.0%})")
    print()

    all_rmse: dict[str, list[float]] = {}
    all_lam: dict[str, list[float]] = {}

    for seed in range(N_SEEDS):
        t_seed = time.time()
        res, sigma = run_seed(seed)
        elapsed = time.time() - t_seed

        n_above = int((sigma > noise_sv_threshold).sum())
        print(
            f"  seed {seed}: batched_loo={res['batched_loo']['rmse']:.4f}  "
            f"lowrank_r50={res['lowrank_r50']['rmse']:.4f}  "
            f"iso_r30={res['iso_r30']['rmse']:.4f}  "
            f"(SVs above threshold: {n_above}/{N})  [{elapsed:.1f}s]"
        )
        for method, d in res.items():
            all_rmse.setdefault(method, []).append(d["rmse"])
            all_lam.setdefault(method, []).append(d["lam"])

    print("\n─── RMSE (mean ± std over {} seeds) ─────────────────────────────".format(N_SEEDS))
    methods_sorted = sorted(all_rmse, key=lambda k: np.mean(all_rmse[k]))
    for method in methods_sorted:
        vals = all_rmse[method]
        lams = all_lam[method]
        print(
            f"  {method:20s}: RMSE={np.mean(vals):.4f}±{np.std(vals):.4f}  "
            f"λ={np.median(lams):.4f} (median)"
        )
