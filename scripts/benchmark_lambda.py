#!/usr/bin/env python
"""Lambda estimation benchmark on NND data.

Compares EB vs grid (ELBO/LOO/Rényi) × batched/lowrank/lowrank+iso/Taylor
across SNR conditions (varying λ_true) and matrix dimensions.

Metrics:
  RMSE              -- primary quality metric
  λ selected        -- median selected lambda
  log(λ/λ_true)     -- log-ratio to true lambda (0 = perfect, < 0 = over-shrink)
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
    proximal_gradient,
    sample_nnd,
    taylor_gradient,
)
from matlap.scoring import closed_form_loo

# ── Shared config ──────────────────────────────────────────────────────────────
SIGMA_NOISE = 1.0
N_SEEDS     = int(os.environ.get("MATLAP_BENCH_N_SEEDS", "5"))
MAX_ITER    = int(os.environ.get("MATLAP_BENCH_MAX_ITER", "80"))
TAYLOR_ITER = int(os.environ.get("MATLAP_BENCH_TAYLOR_ITER", str(MAX_ITER)))
PROX_INIT_ITER = int(os.environ.get("MATLAP_BENCH_PROX_INIT_ITER", "120"))
METHOD_FILTER = {
    s.strip() for s in os.environ.get("MATLAP_BENCH_METHODS", "").split(",") if s.strip()
}
CONDITION_LIMIT = int(os.environ.get("MATLAP_BENCH_CONDITION_LIMIT", "0"))

# The batched model's λ is a per-element precision; its optimal value scales
# with element variance (~mean_SV/sqrt(m)), which is much larger than λ_true.
# The lowrank/iso models' λ is a per-SV penalty, comparable to λ_true.
# Use separate grids so each family can find its optimum.
LAM_GRID_BATCHED = [0.1, 0.3, 0.7, 1.5, 3.0, 7.0, 15.0, 30.0, 70.0]
LAM_GRID_LR      = [0.003, 0.007, 0.015, 0.03, 0.07, 0.15, 0.3, 0.7, 1.5, 3.0]
PROX_CV_FOLDS    = 3

RANK = 30  # fixed rank for lowrank / iso models

# ── Experimental conditions ────────────────────────────────────────────────────
# SNR sweep: m=n=100, noise SV threshold ≈ 2×1×√100 = 20
#   fraction detectable ≈ exp(-λ_true × 20)
SNR_CONDITIONS = [
    {"label": "SNR=high",   "m": 100, "n": 100, "lam_true": 0.010},
    {"label": "SNR=med-hi", "m": 100, "n": 100, "lam_true": 0.025},
    {"label": "SNR=medium", "m": 100, "n": 100, "lam_true": 0.050},
    {"label": "SNR=med-lo", "m": 100, "n": 100, "lam_true": 0.100},
    {"label": "SNR=low",    "m": 100, "n": 100, "lam_true": 0.200},
]

# Dimension sweep: fix λ_true=0.05, square matrices only.
# NND with m=n gives SVs ~ Exp(λ) (mode=0, sparsity-inducing).
# For m > n, SVs ~ Gamma(m-n+1, λ) concentrate far from 0, making
# the benchmark trivial (optimal estimator ≈ Y), so we avoid that case.
DIM_CONDITIONS = [
    {"label": "40×40",    "m":  40, "n":  40, "lam_true": 0.05},
    {"label": "70×70",    "m":  70, "n":  70, "lam_true": 0.05},
    {"label": "100×100",  "m": 100, "n": 100, "lam_true": 0.05},
    {"label": "150×150",  "m": 150, "n": 150, "lam_true": 0.05},
    {"label": "200×200",  "m": 200, "n": 200, "lam_true": 0.05},
]

# Method definitions: (display_name, fn, kwargs)
# All grid methods use the same LAM_GRID; EB = matlap_batched (no grid)
def _make_methods():
    methods = [
        ("proximal_cv",    "proximal_cv",   {}),
        ("batched_eb",     "batched", {}),
        ("batched_elbo",   "batched_grid", {"score_fn": "elbo"}),
        ("batched_loo",    "batched_grid", {"score_fn": "loo"}),
        ("batched_renyi",  "batched_grid", {"score_fn": "renyi"}),
        ("lowrank_elbo",   "lowrank_grid", {"score_fn": "elbo"}),
        ("lowrank_loo",    "lowrank_grid", {"score_fn": "loo"}),
        ("lowrank_renyi",  "lowrank_grid", {"score_fn": "renyi"}),
        ("iso_elbo",       "iso_grid",     {"score_fn": "elbo"}),
        ("iso_loo",        "iso_grid",     {"score_fn": "loo"}),
        ("iso_renyi",      "iso_grid",     {"score_fn": "renyi"}),
        ("taylor_elbo",            "taylor_grid", {"score_fn": "elbo",  "prox_init": False}),
        ("taylor_loo",             "taylor_grid", {"score_fn": "loo",   "prox_init": False}),
        ("taylor_renyi",           "taylor_grid", {"score_fn": "renyi", "prox_init": False}),
        ("taylor_prox_elbo",       "taylor_grid", {"score_fn": "elbo",  "prox_init": True}),
        ("taylor_prox_loo",        "taylor_grid", {"score_fn": "loo",   "prox_init": True}),
        ("taylor_prox_renyi",      "taylor_grid", {"score_fn": "renyi", "prox_init": True}),
    ]
    if METHOD_FILTER:
        methods = [m for m in methods if m[0] in METHOD_FILTER or m[1] in METHOD_FILTER]
    return methods


def _score_taylor_result(res, Y, S, score_fn: str) -> float:
    if score_fn == "elbo":
        return float(res.elbo)
    if score_fn == "loo":
        return float(closed_form_loo(res.mu, res.sigma_diag, Y, S))
    if score_fn == "renyi":
        return float(res.renyi_elbo)
    raise ValueError(f"Unknown Taylor score_fn={score_fn!r}")


def run_taylor_grid(Y, S, lambda_grid, *, score_fn: str, prox_init: bool):
    """Warm-started Taylor lambda grid with optional same-lambda proximal init."""
    lambda_vals = sorted([float(lv) for lv in lambda_grid], reverse=True)
    mu_init = None
    svd_basis = None
    best_score = -float("inf")
    best_lam = lambda_vals[0]
    best_res = None

    for lam in lambda_vals:
        if prox_init:
            prox = proximal_gradient(Y, S, lam, max_iter=PROX_INIT_ITER, tol=1e-6)
            init_mu = prox.X
            init_basis = None
        else:
            init_mu = mu_init
            init_basis = svd_basis

        res = taylor_gradient(
            Y, S, lam, max_iter=TAYLOR_ITER, tol=1e-6,
            init_mu=init_mu, init_svd_basis=init_basis,
            recover_sigma=False,
        )
        if not prox_init:
            mu_init = res.mu
            svd_basis = res.svd_basis

        score = _score_taylor_result(res, Y, S, score_fn)
        if score > best_score:
            best_score = score
            best_lam = lam
            best_res = res

    return best_lam, best_res, best_score


def run_proximal_cv(Y, S):
    """Entry-wise CV comparator for nuclear-norm proximal gradient."""
    from matlap.proximal import proximal_cv

    best_lam, res = proximal_cv(
        Y, S, jnp.array(LAM_GRID_LR), n_folds=PROX_CV_FOLDS, max_iter=PROX_INIT_ITER, tol=1e-6
    )
    return best_lam, res


# ── Per-seed runner ────────────────────────────────────────────────────────────
def run_one_seed(seed: int, m: int, n: int, lam_true: float) -> dict[str, dict]:
    rng = np.random.default_rng(seed)
    X_true, _ = sample_nnd(rng, m, n, lam_true)
    Y = jnp.array(X_true + rng.standard_normal((m, n)) * SIGMA_NOISE)
    S = SIGMA_NOISE * jnp.ones((m, n))

    def rmse(mu):
        return float(jnp.sqrt(jnp.mean((jnp.array(mu) - X_true) ** 2)))

    results: dict[str, dict] = {}

    # baseline: noisy observation (no denoising)
    results["noisy_Y"] = {"rmse": rmse(Y), "lam": float("nan"), "t": 0.0}

    for name, kind, kw in _make_methods():
        t0 = time.time()
        try:
            if kind == "batched":
                res = matlap_batched(Y, S, max_iter=MAX_ITER)
                mu  = res.mu
                lam = float(res.lambda_bar)
            elif kind == "batched_grid":
                res = matlap_grid_batched(Y, S, LAM_GRID_BATCHED, max_iter=MAX_ITER, **kw)
                mu  = res.best_result.mu
                lam = float(res.best_lambda)
            elif kind == "lowrank_grid":
                res = matlap_grid_lowrank(Y, S, LAM_GRID_LR, rank=RANK, max_iter=MAX_ITER, **kw)
                mu  = res.best_result.mu
                lam = float(res.best_lambda)
            elif kind == "iso_grid":
                res = matlap_grid_lowrank_isotropic(Y, S, LAM_GRID_LR, rank=RANK, max_iter=MAX_ITER, **kw)
                mu  = res.best_result.mu
                lam = float(res.best_lambda)
            elif kind == "taylor_grid":
                lam, res, _ = run_taylor_grid(Y, S, LAM_GRID_LR, **kw)
                mu = res.mu
            elif kind == "proximal_cv":
                lam, res = run_proximal_cv(Y, S)
                mu = res.X
            results[name] = {"rmse": rmse(mu), "lam": lam, "t": time.time() - t0}
        except Exception as e:
            results[name] = {"rmse": float("nan"), "lam": float("nan"), "t": time.time() - t0,
                             "error": str(e)}
    return results


# ── Condition runner ───────────────────────────────────────────────────────────
def run_condition(cond: dict) -> dict[str, dict[str, list]]:
    m, n, lam_true = cond["m"], cond["n"], cond["lam_true"]
    accum: dict[str, dict[str, list]] = {}
    for seed in range(N_SEEDS):
        res = run_one_seed(seed, m, n, lam_true)
        for method, d in res.items():
            if method not in accum:
                accum[method] = {"rmse": [], "lam": [], "t": []}
            accum[method]["rmse"].append(d["rmse"])
            accum[method]["lam"].append(d["lam"])
            accum[method]["t"].append(d["t"])
    return accum


# ── Printing ───────────────────────────────────────────────────────────────────
def _nanmedian_or_nan(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return float("nan")
    return float(np.nanmedian(arr))


def print_condition(cond: dict, accum: dict[str, dict[str, list]]):
    lam_true = cond["lam_true"]
    noise_thr = 2.0 * SIGMA_NOISE * cond["n"] ** 0.5
    frac_det  = np.exp(-lam_true * noise_thr)

    print(f"\n{'='*76}")
    print(f"  {cond['label']:12s}  m={cond['m']}, n={cond['n']},  "
          f"λ_true={lam_true:.3f}  ({frac_det:.0%} of SVs detectable)")
    print(f"{'='*76}")

    # Group by model family for readability
    families = [
        ("baseline", ["noisy_Y"]),
        ("proximal", ["proximal_cv"]),
        ("batched", ["batched_eb", "batched_elbo", "batched_loo", "batched_renyi"]),
        ("lowrank", ["lowrank_elbo", "lowrank_loo", "lowrank_renyi"]),
        ("iso",     ["iso_elbo",    "iso_loo",     "iso_renyi"]),
        ("taylor",  ["taylor_elbo", "taylor_loo", "taylor_renyi",
                      "taylor_prox_elbo", "taylor_prox_loo", "taylor_prox_renyi"]),
    ]

    hdr = f"  {'Method':22s}  {'RMSE mean±std':>18s}  {'λ median':>10s}  {'log(λ/λt)':>10s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for fam_name, methods in families:
        print(f"  -- {fam_name} --")
        # Sort within family by mean RMSE
        methods_present = [m for m in methods if m in accum]
        methods_sorted  = sorted(methods_present, key=lambda k: np.nanmean(accum[k]["rmse"]))
        for method in methods_sorted:
            d        = accum[method]
            rmse_m   = np.nanmean(d["rmse"])
            rmse_s   = np.nanstd(d["rmse"])
            lam_med  = _nanmedian_or_nan(d["lam"])
            log_r    = np.log(lam_med / lam_true) if (lam_med > 0 and np.isfinite(lam_med)) else float("nan")
            log_str  = f"{log_r:+.2f}" if np.isfinite(log_r) else "  N/A"
            print(f"  {method:22s}  {rmse_m:.4f} ± {rmse_s:.4f}   {lam_med:>8.4f}     {log_str}")


# ── Summary table across conditions ───────────────────────────────────────────
def print_summary(conditions: list[dict], all_accum: list[dict], title: str):
    method_names = list(all_accum[0].keys())
    print(f"\n{'#'*76}")
    print(f"# {title}")
    print(f"{'#'*76}")
    print(f"\n  RMSE — rows=methods, cols=conditions")

    col_labels = [c["label"] for c in conditions]
    col_w = 13
    header = f"  {'Method':22s}" + "".join(f"  {lbl:>{col_w}s}" for lbl in col_labels)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for method in method_names:
        row = f"  {method:22s}"
        for accum in all_accum:
            if method in accum:
                v = np.nanmean(accum[method]["rmse"])
                row += f"  {v:>{col_w}.4f}"
            else:
                row += f"  {'N/A':>{col_w}s}"
        print(row)

    print(f"\n  log(λ_sel/λ_true) — rows=methods, cols=conditions")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for method in method_names:
        row = f"  {method:22s}"
        for cond, accum in zip(conditions, all_accum):
            if method in accum:
                lam_med = _nanmedian_or_nan(accum[method]["lam"])
                lr = np.log(lam_med / cond["lam_true"]) if (lam_med > 0 and np.isfinite(lam_med)) else float("nan")
                lr_str = f"{lr:+.2f}" if np.isfinite(lr) else "N/A"
                row += f"  {lr_str:>{col_w}s}"
            else:
                row += f"  {'N/A':>{col_w}s}"
        print(row)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Lambda estimation benchmark (NND data)")
    print(f"  N_SEEDS={N_SEEDS}, MAX_ITER={MAX_ITER}, TAYLOR_ITER={TAYLOR_ITER}, "
          f"PROX_INIT_ITER={PROX_INIT_ITER}, RANK={RANK}, σ_noise={SIGMA_NOISE}")
    if METHOD_FILTER:
        print(f"  METHOD_FILTER={sorted(METHOD_FILTER)}")
    if CONDITION_LIMIT:
        print(f"  CONDITION_LIMIT={CONDITION_LIMIT}")
    print(f"  LAM_GRID_BATCHED = {LAM_GRID_BATCHED}")
    print(f"  LAM_GRID_LR      = {LAM_GRID_LR}")

    # ── SNR sweep ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("  SWEEP 1: SNR (m=n=100, varying λ_true)")
    print("=" * 76)
    snr_accums = []
    snr_conditions = SNR_CONDITIONS[:CONDITION_LIMIT] if CONDITION_LIMIT else SNR_CONDITIONS
    for cond in snr_conditions:
        print(f"\n  Running {cond['label']} (λ_true={cond['lam_true']:.3f}) ...")
        t0 = time.time()
        accum = run_condition(cond)
        snr_accums.append(accum)
        print(f"  ... done in {time.time()-t0:.1f}s")
        print_condition(cond, accum)

    print_summary(snr_conditions, snr_accums, "SNR SWEEP — RMSE & λ selection summary")

    # ── Dimension sweep ────────────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("  SWEEP 2: Dimensions (λ_true=0.05, varying m×n)")
    print("=" * 76)
    dim_accums = []
    dim_conditions = DIM_CONDITIONS[:CONDITION_LIMIT] if CONDITION_LIMIT else DIM_CONDITIONS
    for cond in dim_conditions:
        print(f"\n  Running {cond['label']} ...")
        t0 = time.time()
        accum = run_condition(cond)
        dim_accums.append(accum)
        print(f"  ... done in {time.time()-t0:.1f}s")
        print_condition(cond, accum)

    print_summary(dim_conditions, dim_accums, "DIMENSION SWEEP — RMSE & λ selection summary")
