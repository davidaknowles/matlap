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
matlap_lowrank      Low-rank CAVI (Woodbury rank-r updates), auto-λ
vi_matrix_factor    SVI with shared column-factor guide + rSVD; O(mn) memory
vi_row_lowrank      SVI with per-row low-rank guide + rSVD; O(mnr) memory

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
    --lowrank-iters      matlap_lowrank iters    (default 50)
    --lowrank-rank       Rank for matlap_lowrank (default 50)
    --guide-rank         Rank for low-rank VI guides (default 15)
    --approx-rank        Rank for rSVD nuclear norm  (default 30)
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

def get_devices(force_cpu: bool) -> list[tuple[str, object]]:
    """Return list of (name, jax_device) to benchmark."""
    devices = [("CPU", jax.devices("cpu")[0])]
    if not force_cpu:
        try:
            gpu = jax.devices("gpu")[0]
            devices.append(("GPU", gpu))
        except RuntimeError:
            pass
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
    lowrank_iters: int,
    lowrank_rank: int,
    guide_rank: int,
    approx_rank: int,
    device,
    verbose: bool = True,
) -> dict:
    """Run all methods on one seed and one device; return metrics dict."""
    with jax.default_device(device):
        X_true, Y, S, mask = simulate(seed, m, n, rank, missing_frac)
        lam_heuristic = heuristic_lambda(S)

        # Lambda grid for CV: 5 points spanning ±1.5 decades from heuristic
        lam_grid = float(lam_heuristic) * np.logspace(-1.5, 1.5, 5)

        results = {}

        methods = [
            ("proximal",
             run_proximal,
             (Y, S, lam_heuristic, proximal_iters)),
            ("proximal_cv",
             run_proximal_cv,
             (Y, S, lam_grid, proximal_iters // 2)),
            ("vi_diagonal",
             run_vi_diagonal,
             (Y, S, vi_steps)),
            ("vi_diagonal_approx",
             run_vi_diagonal_approx,
             (Y, S, vi_steps, approx_rank)),
            ("matlap_lowrank",
             run_matlap_lowrank,
             (Y, S, lowrank_rank, lowrank_iters)),
            ("vi_matrix_factor",
             run_vi_matrix_factor,
             (Y, S, vi_steps, guide_rank, approx_rank)),
            ("vi_row_lowrank",
             run_vi_row_lowrank,
             (Y, S, vi_steps, guide_rank, approx_rank)),
        ]

        for name, fn, args in methods:
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

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

EXCLUDED = {
    "matlap": (
        "Stores O(m·n²) posterior covariances. At 10k×1k that is "
        "10 000 × 1 000 × 1 000 × 4 bytes ≈ **40 GB** (OOM)."
    ),
    "matlap_grid": "Same O(m·n²) memory requirement as matlap.",
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
    "proximal":           "Nuclear-norm FISTA, λ set by heuristic",
    "proximal_cv":        "Nuclear-norm FISTA, λ by 2-fold entry-wise CV",
    "vi_diagonal":        "SVI, fully-factorised Gaussian guide, auto-λ",
    "vi_diagonal_approx": "SVI, fully-factorised Gaussian + rSVD nuclear norm, auto-λ",
    "matlap_lowrank":     "Low-rank CAVI (Woodbury, rank-r factor subspace), auto-λ",
    "vi_matrix_factor":   "SVI, shared column-factor guide + rSVD, auto-λ; O(mn) memory",
    "vi_row_lowrank":     "SVI, per-row low-rank guide + rSVD, auto-λ; O(mnr) memory",
}


def build_report(
    all_results: dict[str, list[dict]],
    args: argparse.Namespace,
    devices_used: list[str],
    gpu_available: bool,
) -> str:
    methods = list(METHOD_DESC)
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

    # ── Scalability table
    lines.append("## Scalability Notes")
    lines.append("")
    lines.append("Memory and compute scaling at 10k×1k (m=10000, n=1000).")
    lines.append("")
    lines.append("| Method | Memory | Per-iter compute | Notes |")
    lines.append("|---|---|---|---|")
    lines.append("| matlap (CAVI) | O(m·n²) — **40 GB OOM** | O(m·n³) | Exact but infeasible |")
    lines.append("| proximal | O(mn) — 40 MB | O(mn·min(m,n)) full SVD | ~1s/iter on CPU |")
    lines.append("| vi_diagonal | O(mn) — 40 MB | O(mn·min(m,n)) full SVD | ~1s/step on CPU |")
    lines.append(f"| vi_diagonal_approx | O(mn) — 40 MB | O(mn·r) rSVD, r={args.approx_rank} | ~30× faster per step |")
    lines.append(f"| matlap_lowrank | O(mn + nr²) — ~44 MB at r={args.lowrank_rank} | O(mn·r) Woodbury | Exact in rank-r subspace |")
    lines.append(f"| vi_matrix_factor | O(mn) — 40 MB | O(mn·r) rSVD, r={args.approx_rank} | Shared column-factor guide |")
    lines.append(f"| vi_row_lowrank | O(mn·r) — ~600 MB at r={args.guide_rank} | O(mn·r) rSVD | Per-row low-rank covariance |")
    lines.append("| vi_row_mvn | O(mn²) — **40 GB OOM** | O(mn³) | Infeasible |")
    lines.append("| vi_matrix_normal | O(m²+n²) — 400 MB | O(m²n) — **impractical** | 10¹¹ FLOPs/step |")
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
    parser.add_argument("--lowrank-iters",        type=int,   default=50)
    parser.add_argument("--lowrank-rank",         type=int,   default=50)
    parser.add_argument("--guide-rank",           type=int,   default=15)
    parser.add_argument("--approx-rank",          type=int,   default=30)
    parser.add_argument("--no-gpu",               action="store_true")
    parser.add_argument("--output",               type=str,   default="benchmark_results")
    args = parser.parse_args()

    print(f"matlap benchmark  {args.rows}×{args.cols} rank-{args.rank} "
          f"| seeds={args.seeds} | missing={args.missing:.0%}")
    print(f"  proximal_iters={args.proximal_iters}  vi_steps={args.vi_steps}  "
          f"lowrank_iters={args.lowrank_iters}  lowrank_rank={args.lowrank_rank}  "
          f"guide_rank={args.guide_rank}  approx_rank={args.approx_rank}")
    print()

    devices = get_devices(args.no_gpu)
    device_names = [name for name, _ in devices]
    gpu_available = any(n == "GPU" for n in device_names)

    if not gpu_available:
        print("  [GPU not available — running CPU only]")
    print()

    all_results: dict[str, list[dict]] = {}

    for dev_name, device in devices:
        print(f"{'─'*70}")
        print(f"Device: {dev_name}")
        print(f"{'─'*70}")
        seed_results = []
        for seed in range(args.seeds):
            print(f"\n--- Seed {seed} ---")
            res = benchmark_seed(
                seed=seed,
                m=args.rows,
                n=args.cols,
                rank=args.rank,
                missing_frac=args.missing,
                proximal_iters=args.proximal_iters,
                vi_steps=args.vi_steps,
                lowrank_iters=args.lowrank_iters,
                lowrank_rank=args.lowrank_rank,
                guide_rank=args.guide_rank,
                approx_rank=args.approx_rank,
                device=device,
                verbose=True,
            )
            seed_results.append(res)
        all_results[dev_name] = seed_results

    methods = list(METHOD_DESC)

    # ── Write CSV
    csv_path = f"{args.output}.csv"
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
