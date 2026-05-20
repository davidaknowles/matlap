#!/usr/bin/env python
"""
Benchmark all matlap methods across missing fractions.

Compares all denoising approaches on a small matrix (200×50, true rank 5,
model rank 10) at varying missing fractions to examine how free-subspace
methods (FA-EM, GradML) compare to frozen-subspace CAVI and proximal methods
as data becomes sparser.

Output: results/fa_comparison.csv and results/fa_comparison.md
"""

from __future__ import annotations

import csv
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matlap

# ── Experiment parameters ────────────────────────────────────────────────────
M, N         = 200, 50
R_TRUE       = 5
SNR          = 1.0
SEEDS        = [0, 1, 2]
MISSING_FRACS = [0.02, 0.10, 0.30, 0.60, 0.90]
RANK         = 10          # model rank (slight over-specification)
GRID_POINTS  = 10
GRID_LOG_LO  = -2.0
GRID_LOG_HI  =  2.0
N_FOLDS      = 3
MAX_ITER_CAVI  = 200
MAX_ITER_FAEM  = 300
MAX_ITER_GRAD  = 3000

METHODS = [
    "proximal_cv",
    "batched_auto",
    "batched_grid",
    "batched_warmstart",
    "iso_auto",
    "iso_grid",
    "iso_cv",
    "iso_warmstart",
    "iso_then_proximal",
    "iso_renyi",
    "faem",
    "gradml",
]

METHOD_LABELS = {
    "proximal_cv":       "proximal_cv      (FISTA + 3-fold CV)",
    "batched_auto":      "batched_auto     (full CAVI, auto-λ)",
    "batched_grid":      "batched_grid     (full CAVI, best ELBO over grid)",
    "batched_warmstart": "batched_warmstart(full CAVI, FA-EM warm-start)",
    "iso_auto":          "iso_auto         (lowrank+iso CAVI, auto-λ)",
    "iso_grid":          "iso_grid         (lowrank+iso CAVI, best ELBO over grid)",
    "iso_cv":            "iso_cv           (lowrank+iso CAVI, grid+CV)",
    "iso_warmstart":     "iso_warmstart    (lowrank+iso CAVI, FA-EM warm-start)",
    "iso_then_proximal": "iso_then_proximal(iso λ → proximal_gradient)",
    "iso_renyi":         "iso_renyi        (lowrank+iso CAVI, Rényi α=0.5 λ)",
    "faem":              "faem             (FA EM, free subspace, Gaussian factor model)",
    "gradml":            "gradml           (gradient marginal LL, free subspace)",
}

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUT_DIR, exist_ok=True)


def simulate(seed: int, missing_frac: float):
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((M, R_TRUE))
    V = rng.standard_normal((N, R_TRUE))
    X_true = U @ V.T / np.sqrt(R_TRUE)
    signal_std = np.std(X_true)
    noise_std = signal_std / SNR
    sigma = rng.uniform(0.5 * noise_std, 1.5 * noise_std, size=(M, N))
    Y_full = X_true + rng.standard_normal((M, N)) * sigma

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


def heuristic_lambda(S: np.ndarray) -> float:
    obs = np.isfinite(S)
    med_prec = np.median(1.0 / S[obs] ** 2)
    return float(np.sqrt(max(M, N)) / np.sqrt(med_prec))


def run_one_seed(seed: int, missing_frac: float) -> list[dict]:
    Y_train, S_train, Y_full, X_true, test_mask = simulate(seed, missing_frac)
    Y_j = jnp.array(Y_train, dtype=jnp.float32)
    S_j = jnp.array(S_train, dtype=jnp.float32)

    lam_heur = heuristic_lambda(S_train)
    lam_grid = list(lam_heur * np.logspace(GRID_LOG_LO, GRID_LOG_HI, GRID_POINTS))

    rows = []

    def record(name, mu, lam, t):
        r = rmse(np.array(mu), X_true, test_mask)
        print(f"    {name:14s}  RMSE={r:.4f}  λ={lam:.4f}  t={t:.1f}s", flush=True)
        rows.append(dict(seed=seed, missing_frac=missing_frac, method=name,
                         rmse=r, lam=lam, time=t))

    # proximal_cv
    t0 = time.perf_counter()
    from matlap.proximal import proximal_cv
    best_lam, r_pcv = proximal_cv(Y_j, S_j, jnp.array(lam_grid), n_folds=N_FOLDS, max_iter=300)
    record("proximal_cv", np.array(r_pcv.X), float(best_lam), time.perf_counter() - t0)

    # batched_auto
    t0 = time.perf_counter()
    r = matlap.matlap_batched(Y_j, S_j, max_iter=MAX_ITER_CAVI)
    record("batched_auto", r.mu, r.lambda_bar, time.perf_counter() - t0)

    # batched_grid
    t0 = time.perf_counter()
    from matlap.core import matlap_grid
    r_bg = matlap_grid(Y_j, S_j, jnp.array(lam_grid), max_iter=MAX_ITER_CAVI)
    record("batched_grid", r_bg.best_result.mu, r_bg.best_lambda, time.perf_counter() - t0)

    # batched_warmstart
    t0 = time.perf_counter()
    r = matlap.matlap_batched_warmstart(Y_j, S_j, faem_rank=RANK, faem_iters=50,
                                        max_iter=MAX_ITER_CAVI)
    record("batched_warmstart", r.mu, r.lambda_bar, time.perf_counter() - t0)

    # iso_auto
    t0 = time.perf_counter()
    r = matlap.matlap_lowrank_isotropic(Y_j, S_j, rank=RANK, max_iter=MAX_ITER_CAVI)
    record("iso_auto", r.mu, r.lambda_bar, time.perf_counter() - t0)

    # iso_grid
    t0 = time.perf_counter()
    from matlap.core import matlap_grid_lowrank_isotropic
    r_ig = matlap_grid_lowrank_isotropic(Y_j, S_j, jnp.array(lam_grid), rank=RANK,
                                          max_iter=MAX_ITER_CAVI)
    record("iso_grid", r_ig.best_result.mu, r_ig.best_lambda, time.perf_counter() - t0)

    # iso_cv
    t0 = time.perf_counter()
    from matlap.cv import cv_lambda
    def iso_fit(Y_, S_, lam):
        return matlap.matlap_lowrank_isotropic(Y_, S_, lam, rank=RANK, max_iter=MAX_ITER_CAVI)
    best_lam_iso, r_ic = cv_lambda(Y_j, S_j, lam_grid, iso_fit, n_folds=N_FOLDS)
    record("iso_cv", r_ic.mu, float(best_lam_iso), time.perf_counter() - t0)

    # iso_warmstart
    t0 = time.perf_counter()
    r = matlap.matlap_iso_warmstart(Y_j, S_j, faem_rank=RANK, faem_iters=50,
                                    rank=RANK, max_iter=MAX_ITER_CAVI)
    record("iso_warmstart", r.mu, r.lambda_bar, time.perf_counter() - t0)

    # iso_then_proximal: use iso auto-λ as regulariser for proximal_gradient
    t0 = time.perf_counter()
    from matlap.proximal import proximal_gradient
    r_iso_lam = matlap.matlap_lowrank_isotropic(Y_j, S_j, rank=RANK, max_iter=MAX_ITER_CAVI)
    r_p = proximal_gradient(Y_j, S_j, r_iso_lam.lambda_bar, max_iter=300, tol=1e-6)
    record("iso_then_proximal", r_p.X, r_iso_lam.lambda_bar, time.perf_counter() - t0)

    # iso_renyi: Rényi α=0.5 λ learning (alternating CAVI + Rényi λ opt)
    t0 = time.perf_counter()
    r = matlap.matlap_iso_renyi_lambda(Y_j, S_j, rank=RANK, alpha=0.5,
                                       n_outer=20, max_iter=MAX_ITER_CAVI)
    record("iso_renyi", r.mu, r.lambda_bar, time.perf_counter() - t0)

    # faem
    t0 = time.perf_counter()
    r = matlap.matlap_faem(Y_j, S_j, rank=RANK, max_iter=MAX_ITER_FAEM)
    record("faem", r.mu, r.lambda_bar, time.perf_counter() - t0)

    # gradml
    t0 = time.perf_counter()
    r = matlap.matlap_gradml(Y_j, S_j, rank=RANK, max_iter=MAX_ITER_GRAD)
    record("gradml", r.mu, r.lambda_bar, time.perf_counter() - t0)

    return rows


def main():
    all_rows = []
    for mfrac in MISSING_FRACS:
        for seed in SEEDS:
            print(f"\nmissing={mfrac:.0%}  seed={seed}", flush=True)
            all_rows.extend(run_one_seed(seed, mfrac))

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_path = os.path.join(OUT_DIR, "fa_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "missing_frac", "method", "rmse", "lam", "time"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {csv_path}")

    # ── Summary markdown ─────────────────────────────────────────────────────
    by_frac: dict[float, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in all_rows:
        by_frac[r["missing_frac"]][r["method"]].append(r)

    def mean(rows, key):
        return float(np.mean([r[key] for r in rows]))

    lines = [
        "# All-methods comparison across missing fractions",
        f"\nGenerated {datetime.now():%Y-%m-%d %H:%M}",
        f"\nSettings: m={M}, n={N}, true_rank={R_TRUE}, model_rank={RANK}, SNR={SNR}, seeds={SEEDS}",
    ]

    for title, key, fmt in [
        ("RMSE on held-out entries (lower is better)", "rmse", ".4f"),
        ("Auto/CV-selected λ", "lam", ".3f"),
        ("Wall-clock time (seconds)", "time", ".1f"),
    ]:
        lines += ["", f"## {title}", "",
                  "| Missing | " + " | ".join(METHOD_LABELS[m] for m in METHODS) + " |",
                  "|--------:|" + "|".join(["-" * 30 for _ in METHODS]) + "|"]
        for frac in MISSING_FRACS:
            vals = [format(mean(by_frac[frac][m], key), fmt) for m in METHODS]
            lines.append(f"| {frac:.0%}    | " + " | ".join(vals) + " |")

    lines += ["", "## Method descriptions", ""]
    for m in METHODS:
        lines.append(f"- **`{m}`**: {METHOD_LABELS[m]}")

    md_path = os.path.join(OUT_DIR, "fa_comparison.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {md_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
