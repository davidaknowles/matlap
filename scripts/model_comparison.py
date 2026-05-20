#!/usr/bin/env python
"""
Compare matlap_batched (full-rank mean) vs matlap_lowrank_isotropic
across missing fractions to test whether rank-constraint explains
lambda invariance.

For each missing fraction:
  1. Simulate data (m=200, n=50, rank=5, SNR=1).
  2. Grid-search lambda for each model, measuring:
       - oracle RMSE on held-out entries (to find oracle-optimal λ)
       - ELBO-selected λ
       - auto-λ (empirical Bayes, batched only)
  3. Report optimal λ/λ̄ ratios across conditions.

If the rank-50 constraint is the reason for lambda invariance, we expect
matlap_batched to show varying optimal λ with missing fraction, while
matlap_lowrank_isotropic does not.

Output: results/model_comparison.csv and results/model_comparison.md
"""

from __future__ import annotations

import csv
import os
import sys
from datetime import datetime

import time

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matlap
from matlap.proximal import proximal_gradient, proximal_cv

# ── Experiment parameters ───────────────────────────────────────────────────
M, N    = 200, 50
R_TRUE  = 5
SNR     = 1.0
SEEDS   = [0, 1, 2]
MISSING_FRACS = [0.02, 0.10, 0.30, 0.60, 0.90]
RANK    = min(30, N - 1)          # rank for lowrank_isotropic model
GRID_POINTS  = 16
GRID_LOG_LO  = -2.0               # 0.01× heuristic
GRID_LOG_HI  = 2.0                # 100× heuristic
MAX_ITER_FAEM  = 300
MAX_ITER_GRAD  = 3000

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUT_DIR, exist_ok=True)


def heuristic_lambda(S: np.ndarray) -> float:
    obs = np.isfinite(S)
    med_prec = np.median(1.0 / S[obs] ** 2)
    return float(np.sqrt(max(M, N)) / np.sqrt(med_prec))


def simulate(seed: int, missing_frac: float):
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((M, R_TRUE))
    V = rng.standard_normal((N, R_TRUE))
    X_true = U @ V.T / np.sqrt(R_TRUE)
    signal_std = np.std(X_true)
    noise_std   = signal_std / SNR
    sigma = rng.uniform(0.5 * noise_std, 1.5 * noise_std, size=(M, N))
    Y_full = X_true + rng.standard_normal((M, N)) * sigma

    # Held-out mask (separate from missing)
    obs_frac = 1.0 - missing_frac
    test_frac = min(0.2, obs_frac / 2)
    mask = rng.uniform(size=(M, N))
    train_mask = mask > (missing_frac + test_frac)
    test_mask  = (mask > missing_frac) & ~train_mask

    Y_train = np.where(train_mask, Y_full, np.nan)
    S_train = np.where(train_mask, sigma, np.inf)
    return Y_train, S_train, Y_full, X_true, test_mask


def rmse(mu: np.ndarray, X_true: np.ndarray, mask: np.ndarray) -> float:
    diff = (mu - X_true)[mask]
    return float(np.sqrt(np.mean(diff ** 2)))


def run_batched_grid(Y, S, lam_grid):
    """Grid search for matlap_batched, return (lambdas, rmsees, elbos)."""
    results = []
    for lam in lam_grid:
        r = matlap.matlap_batched(Y, S, lambda_val=float(lam), max_iter=100)
        results.append((lam, r))
    return results


def run_iso_grid(Y, S, lam_grid):
    """Grid search for matlap_lowrank_isotropic, return list of (lam, result)."""
    results = []
    prev = None
    for lam in sorted(lam_grid, reverse=True):  # warm-start from high lambda
        kw = {}
        if prev is not None:
            kw = matlap.iso_warm_state(prev)
        r = matlap.matlap_lowrank_isotropic(
            Y, S, rank=RANK, lambda_val=float(lam), max_iter=100, **kw
        )
        results.append((lam, r))
        prev = r
    return results


def run_proximal_grid(Y, S, lam_grid):
    """Grid search for proximal_gradient, return list of (lam, result)."""
    results = []
    for lam in lam_grid:
        r = proximal_gradient(Y, S, float(lam), max_iter=300)
        results.append((lam, r))
    return results


def run_one_seed(seed: int, missing_frac: float) -> dict:
    Y_train, S_train, Y_full, X_true, test_mask = simulate(seed, missing_frac)
    Y_j  = jnp.array(Y_train, dtype=jnp.float32)
    S_j  = jnp.array(S_train, dtype=jnp.float32)

    lam_heur = heuristic_lambda(S_train)
    lam_grid = list(lam_heur * np.logspace(GRID_LOG_LO, GRID_LOG_HI, GRID_POINTS))

    print(f"  seed={seed} missing={missing_frac:.0%}  λ̄={lam_heur:.3f}", flush=True)

    # ── matlap_batched (full-rank) auto-λ ────────────────────────────────
    r_auto = matlap.matlap_batched(Y_j, S_j, max_iter=100)
    lam_batched_auto = r_auto.lambda_bar
    rmse_batched_auto = rmse(np.array(r_auto.mu), X_true, test_mask)

    # ── matlap_batched grid ──────────────────────────────────────────────
    batched_grid_results = run_batched_grid(Y_j, S_j, lam_grid)
    batched_rmses  = [rmse(np.array(r.mu), X_true, test_mask) for _, r in batched_grid_results]
    batched_elbos  = [r.elbo_trace[-1] for _, r in batched_grid_results]
    batched_oracle_idx = int(np.argmin(batched_rmses))
    batched_elbo_idx   = int(np.argmax(batched_elbos))
    lam_batched_oracle = lam_grid[batched_oracle_idx]
    lam_batched_elbo   = lam_grid[batched_elbo_idx]
    rmse_batched_oracle = batched_rmses[batched_oracle_idx]
    rmse_batched_elbo   = batched_rmses[batched_elbo_idx]

    # ── matlap_lowrank_isotropic grid ────────────────────────────────────
    iso_grid_results = run_iso_grid(Y_j, S_j, lam_grid)
    iso_rmses  = [rmse(np.array(r.mu), X_true, test_mask) for _, r in iso_grid_results]
    iso_elbos  = [r.elbo_trace[-1] for _, r in iso_grid_results]
    iso_oracle_idx = int(np.argmin(iso_rmses))
    iso_elbo_idx   = int(np.argmax(iso_elbos))
    lam_iso_oracle = iso_grid_results[iso_oracle_idx][0]
    lam_iso_elbo   = iso_grid_results[iso_elbo_idx][0]
    rmse_iso_oracle = iso_rmses[iso_oracle_idx]
    rmse_iso_elbo   = iso_rmses[iso_elbo_idx]

    # ── matlap_lowrank_isotropic auto-λ ──────────────────────────────────
    r_iso_auto = matlap.matlap_lowrank_isotropic(Y_j, S_j, rank=RANK, max_iter=100)
    lam_iso_auto = r_iso_auto.lambda_bar
    rmse_iso_auto = rmse(np.array(r_iso_auto.mu), X_true, test_mask)

    # ── proximal_gradient oracle grid ────────────────────────────────────
    prox_grid_results = run_proximal_grid(Y_j, S_j, lam_grid)
    prox_rmses = [rmse(np.array(r.X), X_true, test_mask) for _, r in prox_grid_results]
    prox_oracle_idx = int(np.argmin(prox_rmses))
    lam_prox_oracle = lam_grid[prox_oracle_idx]
    rmse_prox_oracle = prox_rmses[prox_oracle_idx]

    # ── proximal_cv (3-fold CV selected) ─────────────────────────────────
    lam_prox_cv, r_prox_cv = proximal_cv(Y_j, S_j, jnp.array(lam_grid), n_folds=3, max_iter=300)
    rmse_prox_cv = rmse(np.array(r_prox_cv.X), X_true, test_mask)

    # ── matlap_faem (auto-λ) ─────────────────────────────────────────────
    t0 = time.perf_counter()
    r_faem = matlap.matlap_faem(Y_j, S_j, rank=RANK, max_iter=MAX_ITER_FAEM)
    t_faem = time.perf_counter() - t0
    lam_faem = float(r_faem.lambda_bar)
    rmse_faem = rmse(np.array(r_faem.mu), X_true, test_mask)

    # ── matlap_gradml (auto-λ) ───────────────────────────────────────────
    t0 = time.perf_counter()
    r_grad = matlap.matlap_gradml(Y_j, S_j, rank=RANK, max_iter=MAX_ITER_GRAD)
    t_grad = time.perf_counter() - t0
    lam_gradml = float(r_grad.lambda_bar)
    rmse_gradml = rmse(np.array(r_grad.mu), X_true, test_mask)

    return dict(
        seed=seed,
        missing_frac=missing_frac,
        lam_heur=lam_heur,
        # batched
        lam_batched_auto=lam_batched_auto,
        lam_batched_oracle=lam_batched_oracle,
        lam_batched_elbo=lam_batched_elbo,
        rmse_batched_auto=rmse_batched_auto,
        rmse_batched_oracle=rmse_batched_oracle,
        rmse_batched_elbo=rmse_batched_elbo,
        # iso
        lam_iso_auto=lam_iso_auto,
        lam_iso_oracle=lam_iso_oracle,
        lam_iso_elbo=lam_iso_elbo,
        rmse_iso_auto=rmse_iso_auto,
        rmse_iso_oracle=rmse_iso_oracle,
        rmse_iso_elbo=rmse_iso_elbo,
        # proximal
        lam_prox_oracle=lam_prox_oracle,
        lam_prox_cv=lam_prox_cv,
        rmse_prox_oracle=rmse_prox_oracle,
        rmse_prox_cv=rmse_prox_cv,
        # faem / gradml
        lam_faem=lam_faem,
        rmse_faem=rmse_faem,
        t_faem=t_faem,
        lam_gradml=lam_gradml,
        rmse_gradml=rmse_gradml,
        t_gradml=t_grad,
    )


def main():
    rows = []
    for mfrac in MISSING_FRACS:
        for seed in SEEDS:
            r = run_one_seed(seed, mfrac)
            rows.append(r)
            ratio_b = r["lam_batched_oracle"] / r["lam_heur"]
            ratio_i = r["lam_iso_oracle"]    / r["lam_heur"]
            ratio_p = r["lam_prox_oracle"]   / r["lam_heur"]
            print(
                f"    oracle λ/λ̄ — batched={ratio_b:.2f}  iso={ratio_i:.2f}  prox={ratio_p:.2f} "
                f"| RMSE batched={r['rmse_batched_oracle']:.4f}  iso={r['rmse_iso_oracle']:.4f}"
                f"  prox={r['rmse_prox_oracle']:.4f}"
                f"  faem={r['rmse_faem']:.4f}  gradml={r['rmse_gradml']:.4f}",
                flush=True,
            )

    # ── Write CSV ────────────────────────────────────────────────────────────
    csv_path = os.path.join(OUT_DIR, "model_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {csv_path}")

    # ── Build summary table ──────────────────────────────────────────────────
    from collections import defaultdict
    by_frac: dict[float, list[dict]] = defaultdict(list)
    for r in rows:
        by_frac[r["missing_frac"]].append(r)

    lines = [
        "# Model comparison: batched vs lowrank+iso vs proximal vs FA-EM vs GradML",
        f"\nGenerated {datetime.now():%Y-%m-%d %H:%M}",
        f"\nSettings: m={M}, n={N}, rank={R_TRUE}, SNR={SNR}, rank_model={RANK}, seeds={SEEDS}",
        f"\nGrid: {GRID_POINTS} log-spaced λ from 10^{GRID_LOG_LO:.0f}×–10^{GRID_LOG_HI:.0f}× heuristic",
        "\n## Oracle-optimal λ / λ̄  (grid search, lower RMSE = better λ choice)",
        "",
        "| Missing | batched oracle λ/λ̄ | iso oracle λ/λ̄ | prox oracle λ/λ̄ | batched RMSE | iso RMSE | prox RMSE |",
        "|--------:|-------------------:|----------------:|----------------:|-------------:|---------:|----------:|",
    ]
    for frac in MISSING_FRACS:
        rs = by_frac[frac]
        b_ratio = np.mean([r["lam_batched_oracle"] / r["lam_heur"] for r in rs])
        i_ratio = np.mean([r["lam_iso_oracle"]     / r["lam_heur"] for r in rs])
        p_ratio = np.mean([r["lam_prox_oracle"]    / r["lam_heur"] for r in rs])
        b_rmse  = np.mean([r["rmse_batched_oracle"] for r in rs])
        i_rmse  = np.mean([r["rmse_iso_oracle"]     for r in rs])
        p_rmse  = np.mean([r["rmse_prox_oracle"]    for r in rs])
        lines.append(
            f"| {frac:.0%}    | {b_ratio:>19.2f} | {i_ratio:>15.2f} | {p_ratio:>15.2f} | {b_rmse:>12.4f} | {i_rmse:>8.4f} | {p_rmse:>9.4f} |"
        )

    lines += [
        "",
        "## Auto/CV-selected λ — all methods",
        "",
        "| Missing | batched auto | iso auto | prox CV | FA-EM | GradML |",
        "|--------:|-------------:|---------:|--------:|------:|-------:|",
    ]
    for frac in MISSING_FRACS:
        rs = by_frac[frac]
        b = np.mean([r["rmse_batched_auto"] for r in rs])
        i = np.mean([r["rmse_iso_auto"]     for r in rs])
        p = np.mean([r["rmse_prox_cv"]      for r in rs])
        f = np.mean([r["rmse_faem"]         for r in rs])
        g = np.mean([r["rmse_gradml"]       for r in rs])
        lines.append(f"| {frac:.0%}    | {b:.4f} | {i:.4f} | {p:.4f} | {f:.4f} | {g:.4f} |")

    lines += [
        "",
        "## Auto/CV-selected λ values",
        "",
        "| Missing | batched auto λ/λ̄ | iso auto λ/λ̄ | prox CV λ/λ̄ | FA-EM λ | GradML λ |",
        "|--------:|------------------:|-------------:|------------:|--------:|---------:|",
    ]
    for frac in MISSING_FRACS:
        rs = by_frac[frac]
        b_ratio = np.mean([r["lam_batched_auto"] / r["lam_heur"] for r in rs])
        i_ratio = np.mean([r["lam_iso_auto"]     / r["lam_heur"] for r in rs])
        p_ratio = np.mean([r["lam_prox_cv"]      / r["lam_heur"] for r in rs])
        f_lam   = np.mean([r["lam_faem"]         for r in rs])
        g_lam   = np.mean([r["lam_gradml"]       for r in rs])
        lines.append(
            f"| {frac:.0%}    | {b_ratio:>18.2f} | {i_ratio:>12.2f} | {p_ratio:>11.2f} | {f_lam:>7.3f} | {g_lam:>8.3f} |"
        )

    md_path = os.path.join(OUT_DIR, "model_comparison.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {md_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
