#!/usr/bin/env python
"""
Lambda selection strategy comparison across matrices of varying true rank.

Compares seven combinations of method × lambda-selection strategy:

  1. proximal_cv        Nuclear-norm FISTA with entry-wise 3-fold CV
  2. matlap_auto        Full CAVI, automatic empirical-Bayes λ  (n=100 → feasible)
  3. lowrank_auto       Low-rank CAVI (rank-r factor space), auto-λ  (biased by n/r)
  4. lowrank_grid       Low-rank CAVI on λ-grid, best ELBO (= matlap_grid_lowrank)
  5. lowrank_cv         Low-rank CAVI on λ-grid, best by 3-fold entry-wise CV
  6. iso_auto           Low-rank+isotropic CAVI, auto-λ; γ=λ̄ (off-subspace ∝ in-subspace)
  7. iso_cv             Low-rank+isotropic CAVI on λ-grid; γ=λ per grid point

Experiment design
-----------------
  Matrix size:    m=500, n=100
  True ranks:     1, 3, 5, 10, 20, 40
  Noise:          heteroscedastic σ_ij ~ Uniform(0.5, 1.5)
  Test fraction:  20% of observed entries held out
  Lambda grid:    12 log-spaced values from 0.1 × heuristic to 10 × heuristic
  Low-rank rank:  min(50, n-1) = 50
  Seeds:          3

Outputs
-------
  results/lambda_study.csv   — raw (method, r_true, seed, lambda_chosen, rmse_test)
  results/lambda_study.md    — formatted tables with mean ± std across seeds
"""

from __future__ import annotations

import csv
import os
import sys
import time
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Line-buffered stdout so progress is visible when redirected to a file
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------

M = 500
N = 100
TRUE_RANKS = [1, 3, 5, 10, 20, 40]
N_SEEDS = 3
MISSING_FRAC = 0.20
LOWRANK_RANK = 50
GRID_POINTS = 8
N_FOLDS = 3
MAX_ITER = 50   # full CAVI converges quickly at m=500, n=100
LOWRANK_ITERS = 50

METHODS = [
    "proximal_cv",
    "matlap_auto",
    "lowrank_auto",
    "lowrank_grid",
    "lowrank_cv",
    "iso_auto",
    "iso_cv",
]

METHOD_LABELS = {
    "proximal_cv":  "proximal_cv    (FISTA + 3-fold CV)",
    "matlap_auto":  "matlap_auto    (full CAVI, auto-λ)",
    "lowrank_auto": "lowrank_auto   (rank-r CAVI, auto-λ, biased)",
    "lowrank_grid": "lowrank_grid   (matlap_grid_lowrank, best ELBO)",
    "lowrank_cv":   "lowrank_cv     (rank-r CAVI, grid+CV)",
    "iso_auto":     "iso_auto       (lowrank+iso CAVI, auto-λ, δ learned)",
    "iso_cv":       "iso_cv         (lowrank+iso CAVI, grid+CV, δ learned)",
}


# ---------------------------------------------------------------------------
# Data simulation
# ---------------------------------------------------------------------------

def simulate(seed: int, m: int, n: int, rank: int, missing_frac: float):
    key = jax.random.PRNGKey(seed)
    scale = float(rank) ** 0.5
    U = jax.random.normal(key, (m, rank)) / scale
    V = jax.random.normal(jax.random.fold_in(key, 1), (n, rank)) / scale
    X_true = U @ V.T

    s_vals = 0.5 + 1.0 * jax.random.uniform(jax.random.fold_in(key, 2), (m, n))
    noise = s_vals * jax.random.normal(jax.random.fold_in(key, 3), (m, n))
    Y_full = X_true + noise

    test_mask = jax.random.uniform(jax.random.fold_in(key, 4), (m, n)) < missing_frac
    S_train = jnp.where(test_mask, jnp.inf, s_vals)
    return X_true, Y_full, S_train, test_mask, s_vals


def test_rmse(pred, truth, mask):
    diff = jnp.where(mask, pred - truth, 0.0)
    return float(jnp.sqrt(jnp.sum(diff ** 2) / jnp.sum(mask)))


def heuristic_lambda(S_train: jnp.Array) -> float:
    prec = jnp.where(jnp.isfinite(S_train), 1.0 / S_train ** 2, 0.0)
    med_prec = float(jnp.median(prec[prec > 0]))
    return float(jnp.sqrt(max(S_train.shape)) / (med_prec ** 0.5))


# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------

def run_proximal_cv(Y, S, lam_grid, n_folds=N_FOLDS):
    from matlap.proximal import proximal_cv
    best_lam, r = proximal_cv(Y, S, jnp.array(lam_grid), n_folds=n_folds, max_iter=300)
    return r.X, float(best_lam)


def run_matlap_auto(Y, S):
    from matlap.core import matlap
    r = matlap(Y, S, max_iter=MAX_ITER)
    return r.mu, float(r.lambda_bar)


def run_lowrank_auto(Y, S, rank=LOWRANK_RANK):
    from matlap.core import matlap_lowrank
    r = matlap_lowrank(Y, S, rank=rank, max_iter=LOWRANK_ITERS)
    return r.mu, float(r.lambda_bar)


def run_lowrank_grid(Y, S, lam_grid, rank=LOWRANK_RANK):
    from matlap.core import matlap_grid_lowrank
    r = matlap_grid_lowrank(Y, S, jnp.array(lam_grid), rank=rank, max_iter=LOWRANK_ITERS)
    return r.best_result.mu, float(r.best_lambda)


def run_lowrank_cv(Y, S, lam_grid, rank=LOWRANK_RANK, n_folds=N_FOLDS):
    from matlap.core import matlap_lowrank
    from matlap.cv import cv_lambda

    def fit_fn(Y_, S_, lam):
        return matlap_lowrank(Y_, S_, lam, rank=rank, max_iter=LOWRANK_ITERS)

    best_lam, r = cv_lambda(Y, S, lam_grid, fit_fn, n_folds=n_folds)
    return r.mu, float(best_lam)


def run_iso_auto(Y, S, rank=LOWRANK_RANK):
    """Single-pass: δ is learned as a variational parameter each iteration."""
    from matlap.core import matlap_lowrank_isotropic
    r = matlap_lowrank_isotropic(Y, S, rank=rank, max_iter=LOWRANK_ITERS)
    return r.mu, float(r.lambda_bar)


def run_iso_cv(Y, S, lam_grid, rank=LOWRANK_RANK, n_folds=N_FOLDS):
    """Grid+CV: δ is learned as a variational parameter for each fixed λ."""
    from matlap.core import matlap_lowrank_isotropic
    from matlap.cv import cv_lambda

    def fit_fn(Y_, S_, lam):
        return matlap_lowrank_isotropic(Y_, S_, lam, rank=rank, max_iter=LOWRANK_ITERS)

    best_lam, r = cv_lambda(Y, S, lam_grid, fit_fn, n_folds=n_folds)
    return r.mu, float(best_lam)


# ---------------------------------------------------------------------------
# Single-seed benchmark
# ---------------------------------------------------------------------------

def run_one_seed(seed: int, r_true: int) -> dict:
    X_true, Y, S, mask, _ = simulate(seed, M, N, r_true, MISSING_FRAC)

    lam_heuristic = heuristic_lambda(S)
    lam_grid = list(float(lam_heuristic) * np.logspace(-1.0, 1.0, GRID_POINTS))

    results = {}

    runners = [
        ("proximal_cv",  lambda: run_proximal_cv(Y, S, lam_grid)),
        ("matlap_auto",  lambda: run_matlap_auto(Y, S)),
        ("lowrank_auto", lambda: run_lowrank_auto(Y, S)),
        ("lowrank_grid", lambda: run_lowrank_grid(Y, S, lam_grid)),
        ("lowrank_cv",   lambda: run_lowrank_cv(Y, S, lam_grid)),
        ("iso_auto",     lambda: run_iso_auto(Y, S)),
        ("iso_cv",       lambda: run_iso_cv(Y, S, lam_grid)),
    ]

    for name, fn in runners:
        t0 = time.perf_counter()
        try:
            mu_hat, lam_est = fn()
            _ = mu_hat.block_until_ready()
            elapsed = time.perf_counter() - t0
            rmse = test_rmse(mu_hat, X_true, mask)
            results[name] = {
                "rmse": rmse,
                "lambda": lam_est,
                "time": elapsed,
                "error": None,
            }
            print(f"  r_true={r_true:2d}  seed={seed}  {name:<20}  "
                  f"RMSE={rmse:.4f}  λ={lam_est:.3f}  t={elapsed:.1f}s")
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - t0
            results[name] = {
                "rmse": float("nan"),
                "lambda": float("nan"),
                "time": elapsed,
                "error": str(exc),
            }
            print(f"  r_true={r_true:2d}  seed={seed}  {name:<20}  ERROR: {exc}")

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _fmt(mean, std):
    """Format mean ± std compactly."""
    return f"{mean:.3f} ± {std:.3f}"


def build_report(all_data: list[dict]) -> str:
    """all_data: list of {r_true, seed, method, rmse, lambda, time}."""
    lines = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines.append("# matlap Lambda Selection Study")
    lines.append("")
    lines.append(
        f"Generated: {ts}  |  Matrix: {M}×{N}  |  "
        f"Seeds: {N_SEEDS}  |  Low-rank rank: {LOWRANK_RANK}  |  "
        f"CV folds: {N_FOLDS}  |  Grid points: {GRID_POINTS}"
    )
    lines.append("")

    def collect(method, rank, field):
        vals = [d[field] for d in all_data
                if d["method"] == method and d["r_true"] == rank
                and not (isinstance(d[field], float) and np.isnan(d[field]))]
        return vals

    # ── Table 1: Test RMSE
    lines.append("## Test RMSE (mean ± std over seeds)")
    lines.append("")
    header = "| Method | " + " | ".join(f"rank={r}" for r in TRUE_RANKS) + " |"
    sep = "|---|" + "---|" * len(TRUE_RANKS)
    lines.append(header)
    lines.append(sep)
    for m in METHODS:
        row = f"| `{m}` |"
        for r in TRUE_RANKS:
            vals = collect(m, r, "rmse")
            if vals:
                row += f" {_fmt(float(np.mean(vals)), float(np.std(vals)))} |"
            else:
                row += " — |"
        lines.append(row)
    lines.append("")

    # ── Table 2: Chosen lambda
    lines.append("## Chosen λ (mean ± std over seeds)")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for m in METHODS:
        row = f"| `{m}` |"
        for r in TRUE_RANKS:
            vals = collect(m, r, "lambda")
            if vals:
                row += f" {_fmt(float(np.mean(vals)), float(np.std(vals)))} |"
            else:
                row += " — |"
        lines.append(row)
    lines.append("")

    # ── Table 3: Runtime
    lines.append("## Runtime in seconds (mean over seeds)")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for m in METHODS:
        row = f"| `{m}` |"
        for r in TRUE_RANKS:
            vals = collect(m, r, "time")
            if vals:
                row += f" {float(np.mean(vals)):.1f}s |"
            else:
                row += " — |"
        lines.append(row)
    lines.append("")

    # ── Notes
    lines.append("## Notes")
    lines.append("")
    lines.append(f"- Matrix: {M}×{N}, heteroscedastic noise σ ~ Uniform(0.5, 1.5)")
    lines.append(f"- Test fraction: {MISSING_FRAC:.0%} of observed entries")
    lines.append(f"- Low-rank rank: {LOWRANK_RANK}")
    lines.append(f"- Lambda grid: {GRID_POINTS} log-spaced values, 0.1×–10× heuristic")
    lines.append(f"- `lowrank_auto` λ is biased by factor ~n/r = {N}/{LOWRANK_RANK} = {N/LOWRANK_RANK:.1f}×")
    lines.append(f"  (r-dim trace vs full n-dim); use `lowrank_grid` or `lowrank_cv` instead.")
    lines.append(f"- `lowrank_grid` = `matlap_grid_lowrank` (ELBO-based λ selection).")
    lines.append(f"  ELBO prefers lower λ than CV → tends to under-regularize.")
    lines.append(f"- `iso_auto`/`iso_cv` use `matlap_lowrank_isotropic` (lowrank+isotropic prior).")
    lines.append(f"  γ is set to λ̄ (auto) or λ (CV), regularizing off-subspace at the")
    lines.append(f"  same scale as in-subspace.  With γ=1e-3 (old default) off-subspace")
    lines.append(f"  directions are unregularised and the RMSE degrades to noise level.")
    lines.append("")

    for m in METHODS:
        lines.append(f"- **`{m}`**: {METHOD_LABELS[m]}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs("results", exist_ok=True)

    csv_path = "results/lambda_study.csv"
    fieldnames = ["r_true", "seed", "method", "rmse", "lambda", "time", "error"]

    # Open CSV for incremental writing so results are saved as they arrive
    csv_file = open(csv_path, "w", newline="")  # noqa: SIM115
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    csv_file.flush()

    all_flat: list[dict] = []

    for r_true in TRUE_RANKS:
        print(f"\n{'='*60}")
        print(f"True rank = {r_true}")
        print(f"{'='*60}")
        for seed in range(N_SEEDS):
            res = run_one_seed(seed, r_true)
            for method, info in res.items():
                row = {
                    "r_true": r_true,
                    "seed": seed,
                    "method": method,
                    "rmse": info["rmse"],
                    "lambda": info["lambda"],
                    "time": info["time"],
                    "error": info.get("error") or "",
                }
                writer.writerow(row)
                csv_file.flush()
                all_flat.append(row)

    csv_file.close()
    print(f"\nSaved raw results to {csv_path}")

    # Write Markdown report
    md_path = "results/lambda_study.md"
    report = build_report(all_flat)
    with open(md_path, "w") as f:
        f.write(report)
    print(f"Saved report to {md_path}")
    print()
    print(report)


if __name__ == "__main__":
    main()
