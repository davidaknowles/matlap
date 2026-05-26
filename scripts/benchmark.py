#!/usr/bin/env python
"""
Large-scale benchmark: matrix imputation on a 10 000 × 1 000 rank-15 matrix.

Tests all methods that scale to this size, repeated over multiple seeds,
then writes a Markdown report with:

  1. Test-set RMSE (mean ± std across seeds)
  2. Lambda agreement across methods (mean ± std across seeds)
  3. Wall-clock runtimes on CPU (and GPU if available)

Methods included
----------------
proximal            Nuclear-norm FISTA with a heuristic lambda
proximal_cv         Same with lambda selected by 2-fold entry-wise CV
vi_diagonal         Numpyro SVI, fully-factorised Gaussian guide, auto-λ
vi_diagonal_approx  Same guide, rSVD nuclear norm (approx_rank), much faster
fa_em               Factor Analysis EM, free subspace loading matrix, auto-λ
gradml              Gradient ascent on marginal LL (Adam), free subspace, auto-λ
matlap_lowrank      Low-rank CAVI (Woodbury rank-r updates), auto-λ
vi_matrix_factor    SVI with shared column-factor guide + rSVD; O(mn) memory
vi_row_lowrank      SVI with per-row low-rank guide + rSVD; O(mnr) memory
mcmc_mala           Proximal MALA (opt-in, --mcmc): gold-standard MCMC, slow
mcmc_gibbs          MALA+MH Gibbs (opt-in, --mcmc): full posterior with λ sampling

Methods excluded (with reason documented in report)
---------------------------------------------------
matlap          Stores O(m·n²) posterior covariances → 40 GB for 10k×1k (OOM)
matlap_grid     Same O(m·n²) memory requirement as matlap
vi_row_mvn      Guide stores m row-MVN covariances of size n×n ≈ 40 GB (OOM)
vi_matrix_normal  O(m²) parameters (400 MB) + O(m²·n) per-step cost, impractical

Usage
-----
    python scripts/benchmark.py [options]

    -m, --rows           Matrix rows m           (default 10000)
    -k, --cols           Matrix columns n        (default 1000)
    -r, --rank           True rank               (default 15)
    -s, --seeds          Number of random seeds  (default 10)
    --missing            Fraction missing        (default 0.20)
    --proximal-iters     FISTA iterations        (default 100)
    --vi-steps           SVI gradient steps      (default 200)
    --faem-iters         FA EM iterations        (default 100)
    --gradml-steps       GradML Adam steps       (default 500)
    --lowrank-iters      matlap_lowrank iters    (default 50)
    --lowrank-rank       Rank for matlap_lowrank (default 50)
    --guide-rank         Rank for low-rank VI guides (default 15)
    --approx-rank        Rank for rSVD nuclear norm  (default 30)
    --mcmc               Include MCMC methods (slow; warm-started from proximal_cv)
    --mcmc-warmup        MCMC warmup steps       (default 100)
    --mcmc-samples       MCMC sample steps       (default 200)
    --no-gpu             Force CPU even if GPU available
    --output             Path prefix for .md and .csv outputs
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def get_devices(force_cpu: bool, gpu_only: bool = False) -> list[tuple[str, object]]:
    """Return list of (name, jax_device) to benchmark."""
    devices = [] if gpu_only else [("CPU", jax.devices("cpu")[0])]
    if not force_cpu:
        try:
            gpu = jax.devices("gpu")[0]
            devices.append(("GPU", gpu))
        except RuntimeError:
            pass
    if not devices:
        devices = [("CPU", jax.devices("cpu")[0])]
    return devices


# ---------------------------------------------------------------------------
# Data simulation
# ---------------------------------------------------------------------------

def simulate(seed: int, m: int, n: int, rank: int,
             missing_frac: float) -> tuple[jnp.Array, jnp.Array, jnp.Array, jnp.Array]:
    """
    Returns (X_true, Y_full, S_train, test_mask).

    Y_full    : noisy observations everywhere, shape (m, n)
    S_train   : noise std; set to jnp.inf on ~missing_frac entries (test set)
    test_mask : bool array True at held-out entries
    """
    key = jax.random.PRNGKey(seed)
    scale = float(rank) ** 0.5
    U = jax.random.normal(key, (m, rank)) / scale
    V = jax.random.normal(jax.random.fold_in(key, 1), (n, rank)) / scale
    X_true = U @ V.T

    s_vals = 0.3 + 0.4 * jax.random.uniform(jax.random.fold_in(key, 2), (m, n))
    noise = s_vals * jax.random.normal(jax.random.fold_in(key, 3), (m, n))
    Y_full = X_true + noise

    test_mask = jax.random.uniform(jax.random.fold_in(key, 4), (m, n)) < missing_frac
    S_train = jnp.where(test_mask, jnp.inf, s_vals)
    return X_true, Y_full, S_train, test_mask


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_rmse(pred: jnp.Array, truth: jnp.Array, mask: jnp.Array) -> float:
    diff = jnp.where(mask, pred - truth, 0.0)
    return float(jnp.sqrt(jnp.sum(diff ** 2) / jnp.sum(mask)))


def heuristic_lambda(S_train: jnp.Array) -> float:
    """Median(1/s²)^{-0.5} * sqrt(max(m,n)) heuristic."""
    m, n = S_train.shape
    prec = jnp.where(jnp.isfinite(S_train), 1.0 / S_train ** 2, 0.0)
    med_prec = float(jnp.median(prec[prec > 0]))
    return float(jnp.sqrt(max(m, n)) / (med_prec ** 0.5))


# ---------------------------------------------------------------------------
# Method runners — each returns (mu_hat, lambda_estimate, converged)
# ---------------------------------------------------------------------------

def run_proximal(Y, S, lam, max_iter):
    from matlap.proximal import proximal_gradient
    r = proximal_gradient(Y, S, lam, max_iter=max_iter, tol=1e-6)
    return r.X, lam, r.converged


def run_proximal_cv(Y, S, lam_grid, max_iter, n_folds=2):
    from matlap.proximal import proximal_cv
    best_lam, r = proximal_cv(Y, S, jnp.array(lam_grid),
                               n_folds=n_folds, max_iter=max_iter, tol=1e-6)
    return r.X, float(best_lam), r.converged


def run_vi_diagonal(Y, S, n_steps, lr=3e-3):
    from matlap.vi import fit_vi
    r = fit_vi(Y, S, guide_type="diagonal", n_steps=n_steps, lr=lr)
    return r.mu, float(r.lambda_bar), r.converged


def run_vi_diagonal_approx(Y, S, n_steps, approx_rank, lr=3e-3):
    from matlap.vi import fit_vi
    r = fit_vi(Y, S, guide_type="diagonal", n_steps=n_steps, lr=lr,
               approx_rank=approx_rank)
    return r.mu, float(r.lambda_bar), r.converged


def run_matlap_faem(Y, S, rank, max_iter):
    from matlap.faem import matlap_faem
    r = matlap_faem(Y, S, rank=rank, max_iter=max_iter)
    return r.mu, float(r.lambda_bar), r.converged


def run_matlap_gradml(Y, S, rank, max_iter):
    from matlap.faem import matlap_gradml
    r = matlap_gradml(Y, S, rank=rank, max_iter=max_iter)
    return r.mu, float(r.lambda_bar), r.converged


def run_matlap_lowrank(Y, S, rank, max_iter):
    from matlap.core import matlap_lowrank
    r = matlap_lowrank(Y, S, rank=rank, max_iter=max_iter)
    return r.mu, float(r.lambda_bar), r.converged


def run_vi_matrix_factor(Y, S, n_steps, guide_rank, approx_rank, lr=3e-3):
    from matlap.vi import fit_vi
    r = fit_vi(Y, S, guide_type="matrix_factor", n_steps=n_steps, lr=lr,
               guide_rank=guide_rank, approx_rank=approx_rank)
    return r.mu, float(r.lambda_bar), r.converged


def run_vi_row_lowrank(Y, S, n_steps, guide_rank, approx_rank, lr=3e-3):
    from matlap.vi import fit_vi
    r = fit_vi(Y, S, guide_type="row_lowrank", n_steps=n_steps, lr=lr,
               guide_rank=guide_rank, approx_rank=approx_rank)
    return r.mu, float(r.lambda_bar), r.converged


def run_matlap_batched(Y, S, max_iter, batch_size=64):
    from matlap.core import matlap_batched
    r = matlap_batched(Y, S, max_iter=max_iter, batch_size=batch_size)
    return r.mu, float(r.lambda_bar), r.converged


def run_matlap_grid_lowrank(Y, S, lam_grid, rank, max_iter):
    from matlap.core import matlap_grid_lowrank
    r = matlap_grid_lowrank(Y, S, lam_grid, rank=rank, max_iter=max_iter)
    return r.best_result.mu, float(r.best_lambda), r.best_result.converged


def run_matlap_grid_lowrank_iso(Y, S, lam_grid, rank, max_iter, score_fn="elbo"):
    from matlap.core import matlap_grid_lowrank_isotropic
    r = matlap_grid_lowrank_isotropic(
        Y, S, lam_grid, rank=rank, max_iter=max_iter, score_fn=score_fn
    )
    return r.best_result.mu, float(r.best_lambda), r.best_result.converged


def run_matlap_grid_lowrank_iso_ldlt(Y, S, lam_grid, rank, max_iter, score_fn="elbo"):
    from matlap.core import matlap_grid_lowrank_isotropic
    r = matlap_grid_lowrank_isotropic(
        Y, S, lam_grid, rank=rank, max_iter=max_iter, score_fn=score_fn, use_ldlt=True
    )
    return r.best_result.mu, float(r.best_lambda), r.best_result.converged


def run_matlap_grid_lowrank_iso_xla_ldlt(Y, S, lam_grid, rank, max_iter, score_fn="elbo"):
    from matlap.core import matlap_grid_lowrank_isotropic
    r = matlap_grid_lowrank_isotropic(
        Y, S, lam_grid, rank=rank, max_iter=max_iter, score_fn=score_fn, use_xla_ldlt=True
    )
    return r.best_result.mu, float(r.best_lambda), r.best_result.converged


def run_mcmc_mala(Y, S, lam, n_warmup, n_samples):
    from matlap.mcmc import mcmc_proximal_mala
    r = mcmc_proximal_mala(Y, S, lambda_val=lam, n_warmup=n_warmup, n_samples=n_samples)
    return r.mu, float(lam), True


def run_mcmc_gibbs(Y, S, lam, n_warmup, n_samples):
    from matlap.mcmc import mcmc_gsm_gibbs
    r = mcmc_gsm_gibbs(Y, S, lambda_init=lam, n_warmup=n_warmup, n_samples=n_samples)
    return r.mu, float(r.lambda_bar), True





# ---------------------------------------------------------------------------
# Single-seed benchmark
# ---------------------------------------------------------------------------

def benchmark_seed(
    seed: int,
    m: int,
    n: int,
    rank: int,
    missing_frac: float,
    proximal_iters: int,
    vi_steps: int,
    faem_iters: int,
    gradml_steps: int,
    lowrank_iters: int,
    lowrank_rank: int,
    guide_rank: int,
    approx_rank: int,
    grid_points: int,
    batch_size: int,
    device,
    run_mcmc: bool = False,
    mcmc_warmup: int = 300,
    mcmc_samples: int = 400,
    verbose: bool = True,
    cached: dict | None = None,
    on_result=None,
) -> dict:
    """Run all methods on one seed and one device; return metrics dict."""
    with jax.default_device(device):
        X_true, Y, S, mask = simulate(seed, m, n, rank, missing_frac)
        lam_heuristic = heuristic_lambda(S)

        # Lambda grid: grid_points log-spaced values ±1.5 decades from heuristic
        lam_grid = float(lam_heuristic) * np.logspace(-1.5, 1.5, grid_points)

        # Pre-populate from cache (results already computed in a previous run)
        results = dict(cached) if cached else {}

        methods = [
            ("proximal",
             run_proximal,
             (Y, S, lam_heuristic, proximal_iters)),
            ("proximal_cv",
             run_proximal_cv,
             (Y, S, lam_grid, proximal_iters // 2)),
            ("matlap_faem",
             run_matlap_faem,
             (Y, S, lowrank_rank, faem_iters)),
            ("matlap_gradml",
             run_matlap_gradml,
             (Y, S, lowrank_rank, gradml_steps)),
            ("matlap_lowrank",
             run_matlap_lowrank,
             (Y, S, lowrank_rank, lowrank_iters)),
            ("matlap_grid_lowrank",
             run_matlap_grid_lowrank,
             (Y, S, lam_grid, lowrank_rank, lowrank_iters)),
            ("matlap_grid_lowrank_iso_elbo",
             run_matlap_grid_lowrank_iso,
             (Y, S, lam_grid, lowrank_rank, lowrank_iters, "elbo")),
            ("matlap_grid_lowrank_iso_renyi",
             run_matlap_grid_lowrank_iso,
             (Y, S, lam_grid, lowrank_rank, lowrank_iters, "renyi")),
            ("matlap_grid_lowrank_iso_ldlt",
             run_matlap_grid_lowrank_iso_ldlt,
             (Y, S, lam_grid, lowrank_rank, lowrank_iters, "renyi")),
            ("matlap_grid_lowrank_iso_xla_ldlt",
             run_matlap_grid_lowrank_iso_xla_ldlt,
             (Y, S, lam_grid, lowrank_rank, lowrank_iters, "renyi")),
            ("matlap_batched",
             run_matlap_batched,
             (Y, S, lowrank_iters, batch_size)),
            ("vi_diagonal",
             run_vi_diagonal,
             (Y, S, vi_steps)),
            ("vi_diagonal_approx",
             run_vi_diagonal_approx,
             (Y, S, vi_steps, approx_rank)),
            ("vi_matrix_factor",
             run_vi_matrix_factor,
             (Y, S, vi_steps, guide_rank, approx_rank)),
            ("vi_row_lowrank",
             run_vi_row_lowrank,
             (Y, S, vi_steps, guide_rank, approx_rank)),
        ]

        all_run_methods = list(methods)
        if run_mcmc:
            all_run_methods += [
                ("mcmc_mala",
                 run_mcmc_mala,
                 (Y, S, lam_heuristic, mcmc_warmup, mcmc_samples)),
                ("mcmc_gibbs",
                 run_mcmc_gibbs,
                 (Y, S, lam_heuristic, mcmc_warmup, mcmc_samples)),
            ]

        for name, fn, args in all_run_methods:
            if name in results:
                if verbose:
                    r = results[name]
                    print(f"  seed={seed:2d} {name:<22}  [cached]  "
                          f"RMSE={r['rmse']:.4f}  λ={r['lambda']:.3f}  "
                          f"t={r['time']:.1f}s")
                continue
            t0 = time.perf_counter()
            try:
                mu_hat, lam_est, converged = fn(*args)
                _ = mu_hat.block_until_ready()
                elapsed = time.perf_counter() - t0
                rmse = test_rmse(mu_hat, X_true, mask)
                results[name] = {
                    "rmse": rmse,
                    "lambda": lam_est,
                    "time": elapsed,
                    "converged": converged,
                    "error": None,
                }
                if verbose:
                    print(f"  seed={seed:2d} {name:<22}  "
                          f"RMSE={rmse:.4f}  λ={lam_est:.3f}  "
                          f"t={elapsed:.1f}s  conv={converged}")
            except Exception as exc:  # noqa: BLE001
                elapsed = time.perf_counter() - t0
                results[name] = {
                    "rmse": float("nan"),
                    "lambda": float("nan"),
                    "time": elapsed,
                    "converged": False,
                    "error": str(exc),
                }
                if verbose:
                    print(f"  seed={seed:2d} {name:<22}  ERROR: {exc}")
            if on_result is not None:
                on_result(name, results[name])

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

EXCLUDED = {
    "matlap": (
        "O(m·n³) compute even with batching — at 10k×1k each row needs an n=1000 "
        "Cholesky (~10⁹ FLOPs); 10k rows per iter ≈ 10¹³ FLOPs/iter (infeasible). "
        "Use `matlap_batched` at n ≲ 300."
    ),
    "matlap_grid": "Same O(m·n³) compute limit as matlap; replaced by matlap_grid_lowrank.",
    "vi_row_mvn": (
        "Guide stores m row-MVN covariances of size n×n "
        "≈ 40 GB for 10k×1k (OOM)."
    ),
    "vi_matrix_normal": (
        "Guide scale_tril_row is m×m (400 MB for m=10k); "
        "each SVI step costs O(m²·n) ≈ 10¹¹ FLOPs — impractical on CPU."
    ),
}

METHOD_DESC = {
    "proximal":             "Nuclear-norm FISTA, λ set by heuristic",
    "proximal_cv":          "Nuclear-norm FISTA, λ by entry-wise CV",
    "matlap_faem":          "Factor Analysis EM, free subspace W_r, auto-λ (EB)",
    "matlap_gradml":        "Gradient marginal LL (Adam), free subspace W_r, auto-λ",
    "matlap_lowrank":       "Low-rank CAVI (Woodbury, rank-r factor subspace), auto-λ",
    "matlap_grid_lowrank":          "Low-rank CAVI on λ-grid, warm-started path, best ELBO",
    "matlap_grid_lowrank_iso_elbo":  "Low-rank+iso CAVI on λ-grid, warm-started path, best ELBO",
    "matlap_grid_lowrank_iso_renyi": "Low-rank+iso CAVI on λ-grid, warm-started path, best Rényi α=0.5",
    "matlap_grid_lowrank_iso_ldlt":      "Low-rank+iso CAVI on λ-grid, CuPy LDL^T kernel, best ELBO",
    "matlap_grid_lowrank_iso_xla_ldlt":  "Low-rank+iso CAVI on λ-grid, XLA FFI LDL^T kernel, best ELBO",
    "matlap_batched":       "Full CAVI, batched rows (O(batch·n²) peak mem), auto-λ",
    "vi_diagonal":          "SVI, fully-factorised Gaussian guide, auto-λ",
    "vi_diagonal_approx":   "SVI, fully-factorised Gaussian + rSVD nuclear norm, auto-λ",
    "vi_matrix_factor":     "SVI, shared column-factor guide + rSVD, auto-λ; O(mn) memory",
    "vi_row_lowrank":       "SVI, per-row low-rank guide + rSVD, auto-λ; O(mnr) memory",
    "mcmc_mala":            "Proximal MALA, cold start, heuristic λ (fixed)",
    "mcmc_gibbs":           "MALA+MH Gibbs, cold start, heuristic λ init, λ sampled",
}


def build_report(
    all_results: dict[str, list[dict]],
    args: argparse.Namespace,
    devices_used: list[str],
    gpu_available: bool,
) -> str:
    # Only show methods that were actually run (MCMC excluded if --mcmc not set)
    methods = [m for m in METHOD_DESC
               if any(m in s for seed_res in all_results.values() for s in seed_res)]
    lines = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines.append("# matlap Benchmark Report")
    lines.append("")
    lines.append(
        f"Generated: {ts}  |  "
        f"Matrix: {args.rows}×{args.cols}, rank {args.rank}  |  "
        f"Missing: {args.missing*100:.0f}%  |  "
        f"Seeds: {args.seeds}"
    )
    lines.append("")

    # ── Configuration
    lines.append("## Configuration")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    lines.append(f"| Rows (m) | {args.rows:,} |")
    lines.append(f"| Columns (n) | {args.cols:,} |")
    lines.append(f"| True rank | {args.rank} |")
    lines.append(f"| Missing fraction | {args.missing:.0%} |")
    lines.append(f"| Seeds | {args.seeds} |")
    lines.append(f"| FISTA iterations | {args.proximal_iters} |")
    lines.append(f"| SVI steps | {args.vi_steps} |")
    lines.append(f"| matlap_lowrank iters | {args.lowrank_iters} |")
    lines.append(f"| matlap_lowrank rank | {args.lowrank_rank} |")
    lines.append(f"| VI guide rank | {args.guide_rank} |")
    lines.append(f"| rSVD approx rank | {args.approx_rank} |")
    if args.mcmc:
        lines.append(f"| MCMC warmup steps | {args.mcmc_warmup} |")
        lines.append(f"| MCMC sample steps | {args.mcmc_samples} |")
    lines.append(f"| Devices | {', '.join(devices_used)} |")
    lines.append("")

    if not gpu_available:
        lines.append(
            "> **Note:** No GPU detected. All results are CPU-only. "
            "Re-run with a CUDA-enabled JAX install to include GPU timings."
        )
        lines.append("")

    # ── Included methods
    lines.append("## Methods Included")
    lines.append("")
    lines.append("| Method | Description |")
    lines.append("|---|---|")
    for name, desc in METHOD_DESC.items():
        lines.append(f"| `{name}` | {desc} |")
    lines.append("")

    # ── Excluded methods
    lines.append("## Methods Excluded")
    lines.append("")
    lines.append("The following methods were excluded due to memory or compute constraints:")
    lines.append("")
    lines.append("| Method | Reason |")
    lines.append("|---|---|")
    for name, reason in EXCLUDED.items():
        lines.append(f"| `{name}` | {reason} |")
    lines.append("")

    # ── Per-device results
    for device in devices_used:
        dev_results = all_results[device]

        lines.append(f"## Results — {device}")
        lines.append("")

        # -- RMSE table
        lines.append("### Test-Set RMSE")
        lines.append("")
        lines.append("RMSE on held-out entries (lower is better).")
        lines.append("")
        lines.append("| Method | Mean RMSE | Std RMSE | Converged (%) |")
        lines.append("|---|---|---|---|")
        for m_name in methods:
            vals = [s[m_name]["rmse"] for s in dev_results
                    if m_name in s and s[m_name]["error"] is None
                    and not np.isnan(s[m_name]["rmse"])]
            conv = [s[m_name]["converged"] for s in dev_results
                    if m_name in s and s[m_name]["error"] is None]
            if vals:
                pct = 100 * sum(conv) / max(len(conv), 1)
                lines.append(f"| {m_name} | {np.mean(vals):.4f} | "
                              f"{np.std(vals):.4f} | {pct:.0f}% |")
            else:
                lines.append(f"| {m_name} | N/A | N/A | — |")
        lines.append("")

        # -- Lambda agreement
        lines.append("### Lambda Agreement")
        lines.append("")
        lines.append("Estimated regularisation strength λ per method across seeds.")
        lines.append("")
        lines.append("| Method | Mean λ | Std λ | Min λ | Max λ |")
        lines.append("|---|---|---|---|---|")
        for m_name in methods:
            lams = [s[m_name]["lambda"] for s in dev_results
                    if m_name in s and s[m_name]["error"] is None
                    and np.isfinite(s[m_name]["lambda"])]
            if lams:
                lines.append(
                    f"| {m_name} | {np.mean(lams):.3f} | {np.std(lams):.3f} | "
                    f"{np.min(lams):.3f} | {np.max(lams):.3f} |"
                )
            else:
                lines.append(f"| {m_name} | N/A | — | — | — |")
        lines.append("")

        # Per-seed lambda
        lines.append("<details><summary>Per-seed lambda values</summary>")
        lines.append("")
        header = "| Seed | " + " | ".join(methods) + " |"
        sep = "|---|" + "---|" * len(methods)
        lines.append(header)
        lines.append(sep)
        for i, sd_res in enumerate(dev_results):
            row = f"| {i} | "
            row += " | ".join(
                f"{sd_res[m]['lambda']:.3f}" if m in sd_res and sd_res[m]["error"] is None else "ERR"
                for m in methods
            ) + " |"
            lines.append(row)
        lines.append("")
        lines.append("</details>")
        lines.append("")

        # -- Runtime table
        lines.append("### Runtimes")
        lines.append("")
        lines.append("Wall-clock time per seed (seconds). "
                     "Seed 0 may include JAX JIT compilation overhead.")
        lines.append("")
        lines.append("| Method | Mean (s) | Std (s) | Min (s) | Max (s) |")
        lines.append("|---|---|---|---|---|")
        for m_name in methods:
            times = [s[m_name]["time"] for s in dev_results
                     if m_name in s and s[m_name]["error"] is None]
            if times:
                lines.append(
                    f"| {m_name} | {np.mean(times):.1f} | {np.std(times):.1f} | "
                    f"{np.min(times):.1f} | {np.max(times):.1f} |"
                )
            else:
                lines.append(f"| {m_name} | N/A | — | — | — |")
        lines.append("")

        # Per-seed timing
        lines.append("<details><summary>Per-seed runtimes (s)</summary>")
        lines.append("")
        lines.append(header)
        lines.append(sep)
        for i, sd_res in enumerate(dev_results):
            row = f"| {i} | "
            row += " | ".join(
                f"{sd_res[m]['time']:.1f}" if m in sd_res and sd_res[m]["error"] is None else "ERR"
                for m in methods
            ) + " |"
            lines.append(row)
        lines.append("")
        lines.append("</details>")
        lines.append("")

    lines.append("## Scalability Notes")
    lines.append("")
    lines.append("Memory and compute scaling at 10k×1k (m=10000, n=1000).")
    lines.append("")
    lines.append("| Method | Memory | Per-iter compute | Notes |")
    lines.append("|---|---|---|---|")
    lines.append("| matlap | O(m·n²) — **40 GB OOM** | O(m·n³) | Exact but infeasible at n=1k |")
    lines.append(f"| matlap_batched | O(B·n²), B={args.batch_size} — {args.batch_size}×4MB={args.batch_size*4} MB | O(m·n³) — **slow** at n=1k | Feasible memory; use at n≲300 |")
    lines.append(f"| matlap_faem | O(mr² + mn) at r={args.lowrank_rank} — ~44 MB | O(mnr + mr³) | Free subspace; FA EM M-step |")
    lines.append(f"| matlap_gradml | O(mr² + mn) at r={args.lowrank_rank} — ~44 MB | O(mnr + mr³) per step | Free subspace; Adam on marginal LL |")
    lines.append(f"| matlap_lowrank | O(mn + nr²) at r={args.lowrank_rank} — ~44 MB | O(mn·r) Woodbury | Exact in rank-r subspace |")
    lines.append(f"| matlap_grid_lowrank | O(mn + nr²) at r={args.lowrank_rank} | O(G·mn·r) warm path | G={args.grid_points} grid pts, warm-started |")
    lines.append(f"| matlap_grid_lowrank_iso_elbo | O(mn + nr²) at r={args.lowrank_rank} | O(G·mn·r) warm path | iso; G={args.grid_points} grid pts, ELBO scoring (not recommended — see note) |")
    lines.append(f"| matlap_grid_lowrank_iso_renyi | O(mn + nr²) at r={args.lowrank_rank} | O(G·mn·r) warm path | iso; G={args.grid_points} grid pts, Rényi α=0.5 scoring |")
    lines.append(f"| matlap_grid_lowrank_iso_ldlt | O(mn + nr²) at r={args.lowrank_rank} | O(G·mn·r) warm path | iso+CuPy LDL^T; G={args.grid_points} grid pts, Rényi scoring |")
    lines.append(f"| matlap_grid_lowrank_iso_xla_ldlt | O(mn + nr²) at r={args.lowrank_rank} | O(G·mn·r) warm path | iso+XLA FFI LDL^T (no sync barriers); G={args.grid_points} grid pts, Rényi scoring |")
    lines.append("| proximal | O(mn) — 40 MB | O(mn·min(m,n)) full SVD | ~1s/iter on CPU |")
    lines.append(f"| vi_diagonal_approx | O(mn) — 40 MB | O(mn·r) rSVD, r={args.approx_rank} | ~30× faster per step vs full SVD |")
    lines.append(f"| vi_matrix_factor | O(mn) — 40 MB | O(mn·r) rSVD, r={args.approx_rank} | Shared column-factor guide |")
    lines.append(f"| vi_row_lowrank | O(mn·r) at r={args.guide_rank} — ~600 MB | O(mn·r) rSVD | Per-row low-rank covariance |")
    lines.append("| vi_row_mvn | O(mn²) — **40 GB OOM** | O(mn³) | Infeasible |")
    lines.append("| vi_matrix_normal | O(m²+n²) — 400 MB | O(m²n) — **impractical** | 10¹¹ FLOPs/step |")
    lines.append("| mcmc_mala | O(mn) — 40 MB | O(mn·min(m,n)) × T steps full SVD | Gold standard; slow at large mn |")
    lines.append("| mcmc_gibbs | O(mn) — 40 MB | O(mn·min(m,n)) × T steps full SVD | Gold standard; also samples λ |")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-m", "--rows",           type=int,   default=10_000)
    parser.add_argument("-k", "--cols",           type=int,   default=1_000)
    parser.add_argument("-r", "--rank",           type=int,   default=15)
    parser.add_argument("-s", "--seeds",          type=int,   default=10)
    parser.add_argument("--missing",              type=float, default=0.20)
    parser.add_argument("--proximal-iters",       type=int,   default=100)
    parser.add_argument("--vi-steps",             type=int,   default=200)
    parser.add_argument("--faem-iters",           type=int,   default=100,
                        help="FA EM iterations (default 100)")
    parser.add_argument("--gradml-steps",         type=int,   default=500,
                        help="GradML Adam steps (default 500)")
    parser.add_argument("--lowrank-iters",        type=int,   default=50)
    parser.add_argument("--lowrank-rank",         type=int,   default=50)
    parser.add_argument("--guide-rank",           type=int,   default=15)
    parser.add_argument("--approx-rank",          type=int,   default=30)
    parser.add_argument("--grid-points",          type=int,   default=7,
                        help="Lambda grid size for CV / matlap_grid_lowrank (default 7)")
    parser.add_argument("--batch-size",           type=int,   default=64,
                        help="Row batch size for matlap_batched (default 64)")
    parser.add_argument("--mcmc",                 action="store_true",
                        help="Include MCMC methods (slow gold standard)")
    parser.add_argument("--mcmc-warmup",          type=int,   default=300,
                        help="MCMC warmup steps (default 300)")
    parser.add_argument("--mcmc-samples",         type=int,   default=400,
                        help="MCMC sample steps (default 400)")
    parser.add_argument("--no-gpu",               action="store_true")
    parser.add_argument("--gpu-only",             action="store_true",
                        help="Skip CPU benchmarking; GPU only")
    parser.add_argument("--output",               type=str,   default="benchmark_results")
    parser.add_argument("--resume",               action="store_true",
                        help="Load existing output CSV and skip already-completed "
                             "(device, seed, method) entries")
    args = parser.parse_args()

    print(f"matlap benchmark  {args.rows}×{args.cols} rank-{args.rank} "
          f"| seeds={args.seeds} | missing={args.missing:.0%}")
    print(f"  proximal_iters={args.proximal_iters}  vi_steps={args.vi_steps}  "
          f"faem_iters={args.faem_iters}  gradml_steps={args.gradml_steps}  "
          f"lowrank_iters={args.lowrank_iters}  lowrank_rank={args.lowrank_rank}  "
          f"guide_rank={args.guide_rank}  approx_rank={args.approx_rank}  "
          f"grid_points={args.grid_points}  batch_size={args.batch_size}")
    if args.mcmc:
        print(f"  mcmc_warmup={args.mcmc_warmup}  mcmc_samples={args.mcmc_samples}")
    print()

    devices = get_devices(args.no_gpu, args.gpu_only)
    device_names = [name for name, _ in devices]
    gpu_available = any(n == "GPU" for n in device_names)

    if not gpu_available:
        print("  [GPU not available — running CPU only]")
    print()

    # ── Load cache from existing CSV (--resume)
    csv_path = f"{args.output}.csv"
    # cache[device][seed][method] = result_dict
    cache: dict[str, dict[int, dict[str, dict]]] = {}
    if args.resume and os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                dev = row["device"]
                s = int(row["seed"])
                m_name = row["method"]
                cache.setdefault(dev, {}).setdefault(s, {})[m_name] = {
                    "rmse": float(row["rmse"]),
                    "lambda": float(row["lambda"]),
                    "time": float(row["time_s"]),
                    "converged": row["converged"] == "True",
                    "error": row["error"] or None,
                }
        n_cached = sum(len(ms) for sd in cache.values() for ms in sd.values())
        print(f"  [--resume] loaded {n_cached} cached results from {csv_path}")
        print()

    all_results: dict[str, list[dict]] = {}

    # Open CSV for incremental writing (append if resuming, write header if new)
    csv_exists = os.path.exists(csv_path)
    csv_mode = "a" if (args.resume and csv_exists) else "w"
    csv_file = open(csv_path, csv_mode, newline="")  # noqa: SIM115
    csv_writer = csv.writer(csv_file)
    if csv_mode == "w":
        csv_writer.writerow(["device", "seed", "method", "rmse", "lambda",
                             "time_s", "converged", "error"])

    try:
        for dev_name, device in devices:
            print(f"{'─'*70}")
            print(f"Device: {dev_name}")
            print(f"{'─'*70}")
            seed_results = []
            for seed in range(args.seeds):
                print(f"\n--- Seed {seed} ---")
                cached_seed = cache.get(dev_name, {}).get(seed, {})

                def _on_result(name, r, _dev=dev_name, _seed=seed):
                    csv_writer.writerow([_dev, _seed, name,
                                         r["rmse"], r["lambda"],
                                         r["time"], r["converged"], r["error"]])
                    csv_file.flush()

                res = benchmark_seed(
                    seed=seed,
                    m=args.rows,
                    n=args.cols,
                    rank=args.rank,
                    missing_frac=args.missing,
                    proximal_iters=args.proximal_iters,
                    vi_steps=args.vi_steps,
                    faem_iters=args.faem_iters,
                    gradml_steps=args.gradml_steps,
                    lowrank_iters=args.lowrank_iters,
                    lowrank_rank=args.lowrank_rank,
                    guide_rank=args.guide_rank,
                    approx_rank=args.approx_rank,
                    grid_points=args.grid_points,
                    batch_size=args.batch_size,
                    run_mcmc=args.mcmc,
                    mcmc_warmup=args.mcmc_warmup,
                    mcmc_samples=args.mcmc_samples,
                    device=device,
                    verbose=True,
                    cached=cached_seed,
                    on_result=_on_result,
                )
                seed_results.append(res)
            all_results[dev_name] = seed_results
    finally:
        csv_file.close()

    methods = [m for m in METHOD_DESC
               if any(m in s for seed_res in all_results.values() for s in seed_res)]

    # ── Rewrite full CSV (merges cache + new results cleanly)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["device", "seed", "method", "rmse", "lambda",
                         "time_s", "converged", "error"])
        for dev_name, seed_results in all_results.items():
            for seed, sd_res in enumerate(seed_results):
                for m_name in methods:
                    if m_name not in sd_res:
                        continue
                    r = sd_res[m_name]
                    writer.writerow([dev_name, seed, m_name,
                                     r["rmse"], r["lambda"],
                                     r["time"], r["converged"], r["error"]])
    print(f"\nCSV saved to {csv_path}")

    # ── Write Markdown report
    md_path = f"{args.output}.md"
    report = build_report(all_results, args, device_names, gpu_available)
    with open(md_path, "w") as f:
        f.write(report)
    print(f"Report saved to {md_path}")

    # ── Print summary to stdout
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for dev_name, seed_results in all_results.items():
        print(f"\n{dev_name}:")
        print(f"  {'Method':<24}  {'RMSE':>10}  {'λ':>8}  {'Time(s)':>9}")
        print(f"  {'─'*57}")
        for m_name in methods:
            vals_r = [s[m_name]["rmse"] for s in seed_results
                      if m_name in s and s[m_name]["error"] is None]
            vals_l = [s[m_name]["lambda"] for s in seed_results
                      if m_name in s and s[m_name]["error"] is None]
            vals_t = [s[m_name]["time"] for s in seed_results
                      if m_name in s and s[m_name]["error"] is None]
            if vals_r:
                print(f"  {m_name:<24}  "
                      f"{np.mean(vals_r):.4f}±{np.std(vals_r):.4f}  "
                      f"{np.mean(vals_l):8.3f}  "
                      f"{np.mean(vals_t):>9.1f}")
            else:
                print(f"  {m_name:<24}  {'FAILED':>10}")
    print("=" * 72)
    print(f"\nFull report: {md_path}")


if __name__ == "__main__":
    main()
