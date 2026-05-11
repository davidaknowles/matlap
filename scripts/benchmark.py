#!/usr/bin/env python
"""
Large-scale benchmark: matrix imputation on a 10 000 × 1 000 rank-15 matrix.

Tests three methods that scale to this size, repeated over multiple seeds,
then writes a Markdown report with:

  1. Test-set RMSE (mean ± std across seeds)
  2. Lambda agreement across methods (mean ± std across seeds)
  3. Wall-clock runtimes on CPU (and GPU if available)

Methods included
----------------
proximal       Nuclear-norm FISTA with a heuristic lambda
proximal_cv    Same with lambda selected by 2-fold entry-wise CV (3-point grid)
vi_diagonal    Numpyro SVI with fully-factorised Gaussian guide; auto-lambda

Methods excluded (with reason documented in report)
---------------------------------------------------
matlap          Stores O(m·n²) posterior covariances → 40 GB for 10k×1k (OOM)
matlap_grid     Same memory issue as matlap
vi_row_mvn      Same O(m·n²) memory for the guide
vi_matrix_normal  O(m²) parameters (400 MB) + O(m²·n) per-step cost, impractical

Usage
-----
    python scripts/benchmark.py [options]

    -m, --rows        Matrix rows m          (default 10000)
    -k, --cols        Matrix columns n       (default 1000)
    -r, --rank        True rank              (default 15)
    -s, --seeds       Number of random seeds (default 10)
    --missing         Fraction missing       (default 0.20)
    --proximal-iters  FISTA iterations       (default 200)
    --vi-steps        SVI gradient steps     (default 500)
    --no-gpu          Force CPU even if GPU available
    --output          Path prefix for .md and .csv outputs
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np

# Suppress JAX startup warnings on stderr
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
             missing_frac: float) -> tuple[jnp.Array, jnp.Array, jnp.Array,
                                           jnp.Array]:
    """
    Returns (X_true, Y_full, S_train, test_mask).

    Y_full  : noisy observations everywhere, shape (m, n)
    S_train : noise std; set to jnp.inf on ~missing_frac entries (test set)
    test_mask : bool array True at held-out entries
    """
    key = jax.random.PRNGKey(seed)
    scale = float(rank) ** 0.5
    U = jax.random.normal(key, (m, rank)) / scale
    V = jax.random.normal(jax.random.fold_in(key, 1), (n, rank)) / scale
    X_true = U @ V.T

    # Heteroscedastic noise: s ~ Uniform(0.3, 0.7)
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
    """Root-mean-square error on masked (held-out) entries."""
    diff = jnp.where(mask, pred - truth, 0.0)
    return float(jnp.sqrt(jnp.sum(diff ** 2) / jnp.sum(mask)))


def heuristic_lambda(S_train: jnp.Array) -> float:
    """Simple heuristic: median(1/s²) * sqrt(max(m,n)) / mn."""
    m, n = S_train.shape
    prec = jnp.where(jnp.isfinite(S_train), 1.0 / S_train ** 2, 0.0)
    med_prec = float(jnp.median(prec[prec > 0]))
    return float(jnp.sqrt(max(m, n)) / (med_prec ** 0.5))


# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------

def run_proximal(Y, S, lam, max_iter):
    from matlap.proximal import proximal_gradient
    r = proximal_gradient(Y, S, lam, max_iter=max_iter, tol=1e-6)
    return r.X, lam, r.converged


def run_proximal_cv(Y, S, lam_grid, max_iter, n_folds=2):
    from matlap.proximal import proximal_cv
    best_lam, r = proximal_cv(Y, S, jnp.array(lam_grid),
                               n_folds=n_folds,
                               max_iter=max_iter, tol=1e-6)
    return r.X, float(best_lam), r.converged


def run_vi_diagonal(Y, S, n_steps, lr=3e-3):
    from matlap.vi import fit_vi
    r = fit_vi(Y, S, guide_type="diagonal", n_steps=n_steps, lr=lr)
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
            ("proximal",    run_proximal,    (Y, S, lam_heuristic, proximal_iters)),
            ("proximal_cv", run_proximal_cv, (Y, S, lam_grid, proximal_iters // 2)),
            ("vi_diagonal", run_vi_diagonal, (Y, S, vi_steps)),
        ]

        for name, fn, args in methods:
            t0 = time.perf_counter()
            try:
                mu_hat, lam_est, converged = fn(*args)
                # Force completion of any pending JAX computation
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
                    print(f"  seed={seed:2d} {name:15s}  "
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
                    print(f"  seed={seed:2d} {name:15s}  ERROR: {exc}")

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
    "proximal":    "Nuclear-norm FISTA, λ set by heuristic",
    "proximal_cv": "Nuclear-norm FISTA, λ by 2-fold entry-wise CV",
    "vi_diagonal": "Numpyro SVI, fully-factorised Gaussian guide, auto-λ",
}


def build_report(
    all_results: dict[str, list[dict]],
    args: argparse.Namespace,
    devices_used: list[str],
    gpu_available: bool,
) -> str:
    methods = list(METHOD_DESC)
    device_names = devices_used  # e.g. ["CPU"] or ["CPU", "GPU"]

    lines = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines.append(f"# matlap Benchmark Report")
    lines.append(f"")
    lines.append(f"Generated: {ts}  |  "
                 f"Matrix: {args.rows}×{args.cols}, rank {args.rank}  |  "
                 f"Missing: {args.missing*100:.0f}%  |  "
                 f"Seeds: {args.seeds}")
    lines.append(f"")

    # ── Configuration
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"| Parameter | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Rows (m) | {args.rows:,} |")
    lines.append(f"| Columns (n) | {args.cols:,} |")
    lines.append(f"| True rank | {args.rank} |")
    lines.append(f"| Missing fraction | {args.missing:.0%} |")
    lines.append(f"| Seeds | {args.seeds} |")
    lines.append(f"| FISTA iterations | {args.proximal_iters} |")
    lines.append(f"| SVI steps | {args.vi_steps} |")
    lines.append(f"| Devices | {', '.join(device_names)} |")
    lines.append("")

    if not gpu_available:
        lines.append("> **Note:** No GPU detected in this environment "
                     "(cuSPARSE/CUDA library not found). "
                     "All results are CPU-only. "
                     "Re-run with a CUDA-enabled JAX install to include GPU timings.")
        lines.append("")

    # ── Excluded methods
    lines.append("## Methods Excluded")
    lines.append("")
    lines.append("The following methods were excluded due to memory or compute "
                 "constraints at the requested matrix size:")
    lines.append("")
    lines.append("| Method | Reason |")
    lines.append("|---|---|")
    for name, reason in EXCLUDED.items():
        lines.append(f"| `{name}` | {reason} |")
    lines.append("")

    # ── Per-device results
    for device in device_names:
        dev_results = all_results[device]  # list of seed dicts

        lines.append(f"## Results — {device}")
        lines.append("")

        # -- RMSE table
        lines.append("### Test-Set RMSE")
        lines.append("")
        lines.append("Root-mean-square error on held-out 20% of entries "
                     "(lower is better).")
        lines.append("")
        lines.append("| Method | Mean RMSE | Std RMSE | Converged (%) |")
        lines.append("|---|---|---|---|")

        for m_name in methods:
            vals = [s[m_name]["rmse"] for s in dev_results
                    if s[m_name]["error"] is None and not np.isnan(s[m_name]["rmse"])]
            conv = [s[m_name]["converged"] for s in dev_results
                    if s[m_name]["error"] is None]
            if vals:
                pct_conv = 100 * sum(conv) / len(conv)
                lines.append(f"| {m_name} | {np.mean(vals):.4f} | "
                              f"{np.std(vals):.4f} | {pct_conv:.0f}% |")
            else:
                lines.append(f"| {m_name} | N/A | N/A | — |")

        lines.append("")

        # Naive baseline (predict with observed mean)
        naive_rmses = []
        for sd_res in dev_results:
            # We don't have access to raw data here; skip naive baseline
            pass

        # -- Lambda agreement
        lines.append("### Lambda Agreement")
        lines.append("")
        lines.append("Estimated regularisation strength λ per method across seeds.  "
                     "`proximal` uses a heuristic λ (not optimised), included as reference.  "
                     "`vi_diagonal` estimates λ from data (auto).  "
                     "`proximal_cv` selects λ by cross-validation.")
        lines.append("")
        lines.append("| Method | Mean λ | Std λ | Min λ | Max λ |")
        lines.append("|---|---|---|---|---|")

        for m_name in methods:
            lams = [s[m_name]["lambda"] for s in dev_results
                    if s[m_name]["error"] is None and np.isfinite(s[m_name]["lambda"])]
            if lams:
                lines.append(f"| {m_name} | {np.mean(lams):.3f} | "
                              f"{np.std(lams):.3f} | {np.min(lams):.3f} | "
                              f"{np.max(lams):.3f} |")
            else:
                lines.append(f"| {m_name} | N/A | — | — | — |")

        lines.append("")
        # Per-seed lambda table
        lines.append("<details><summary>Per-seed lambda values</summary>")
        lines.append("")
        header = "| Seed | " + " | ".join(methods) + " |"
        sep    = "|---|" + "---|" * len(methods)
        lines.append(header)
        lines.append(sep)
        for i, sd_res in enumerate(dev_results):
            row = f"| {i} | "
            row += " | ".join(
                f"{sd_res[m_name]['lambda']:.3f}"
                if sd_res[m_name]["error"] is None
                else "ERR"
                for m_name in methods
            ) + " |"
            lines.append(row)
        lines.append("")
        lines.append("</details>")
        lines.append("")

        # -- Runtime table
        lines.append("### Runtimes")
        lines.append("")
        lines.append("Wall-clock time per seed (seconds).  "
                     "First seed may include JAX JIT compilation overhead.")
        lines.append("")
        lines.append("| Method | Mean (s) | Std (s) | Min (s) | Max (s) |")
        lines.append("|---|---|---|---|---|")

        for m_name in methods:
            times = [s[m_name]["time"] for s in dev_results
                     if s[m_name]["error"] is None]
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
        lines.append(header.replace("lambda", "time"))
        lines.append(sep)
        for i, sd_res in enumerate(dev_results):
            row = f"| {i} | "
            row += " | ".join(
                f"{sd_res[m_name]['time']:.1f}"
                if sd_res[m_name]["error"] is None
                else "ERR"
                for m_name in methods
            ) + " |"
            lines.append(row)
        lines.append("")
        lines.append("</details>")
        lines.append("")

    # ── Scalability note
    lines.append("## Scalability Notes")
    lines.append("")
    lines.append("| Method | Memory complexity | Per-iter cost |")
    lines.append("|---|---|---|")
    lines.append("| matlap (CAVI)      | O(m·n²) — OOM for n≥200 on 16 GB | O(m·n³) |")
    lines.append("| proximal gradient  | O(mn) — 40 MB for 10k×1k | O(mn·min(m,n)) SVD |")
    lines.append("| vi_diagonal (SVI)  | O(mn) — 40 MB for 10k×1k | O(mn·min(m,n)) SVD |")
    lines.append("| vi_matrix_normal   | O(m²+n²) ≈ 400 MB for 10k×1k | O(m²n) |")
    lines.append("| vi_row_mvn         | O(mn²) — OOM for n≥200 on 16 GB | O(mn³) |")
    lines.append("")
    lines.append("On GPU, the SVD and gradient steps parallelise efficiently, "
                 "typically giving 10–50× speedup over CPU for large matrices.")
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
    parser.add_argument("-m", "--rows",    type=int, default=10_000)
    parser.add_argument("-k", "--cols",    type=int, default=1_000)
    parser.add_argument("-r", "--rank",    type=int, default=15)
    parser.add_argument("-s", "--seeds",   type=int, default=10)
    parser.add_argument("--missing",       type=float, default=0.20)
    # Defaults tuned for ~3 h total on CPU at 10k×1k.
    # Use higher values if time permits (e.g. --proximal-iters 500 --vi-steps 1000).
    parser.add_argument("--proximal-iters", type=int, default=100)
    parser.add_argument("--vi-steps",       type=int, default=200)
    parser.add_argument("--no-gpu",         action="store_true")
    parser.add_argument("--output",         type=str, default="benchmark_results")
    args = parser.parse_args()

    print(f"matlap benchmark  {args.rows}×{args.cols} rank-{args.rank} "
          f"| seeds={args.seeds} | missing={args.missing:.0%}")
    print()

    devices = get_devices(args.no_gpu)
    device_names = [name for name, _ in devices]
    gpu_available = any(n == "GPU" for n in device_names)

    if not gpu_available:
        print("  [GPU not available — running CPU only]")
    print()

    # Rough time estimate: proximal + CV-fits + vi steps, at ~1.1s per iter on CPU at 10k×1k
    cv_fits = 5 * 2  # 5-point grid × 2 folds
    _iters_per_seed = (
        args.proximal_iters                          # proximal
        + (args.proximal_iters // 2) * cv_fits       # proximal_cv CV phase
        + args.proximal_iters                        # proximal_cv final refit
        + args.vi_steps                              # vi_diagonal
    )
    _secs_estimate = _iters_per_seed * 1.1 * args.seeds  # rough, assuming 1.1s/iter
    print(f"  Estimated runtime (CPU): ~{_secs_estimate/3600:.1f} h  "
          f"({_secs_estimate/args.seeds:.0f}s/seed)")
    print()

    all_results: dict[str, list[dict]] = {}

    for dev_name, device in devices:
        print(f"{'─'*60}")
        print(f"Device: {dev_name}")
        print(f"{'─'*60}")
        seed_results = []
        for seed in range(args.seeds):
            res = benchmark_seed(
                seed=seed,
                m=args.rows,
                n=args.cols,
                rank=args.rank,
                missing_frac=args.missing,
                proximal_iters=args.proximal_iters,
                vi_steps=args.vi_steps,
                device=device,
                verbose=True,
            )
            seed_results.append(res)
        all_results[dev_name] = seed_results

    # ── Write CSV
    csv_path = f"{args.output}.csv"
    methods = ["proximal", "proximal_cv", "vi_diagonal"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["device", "seed", "method", "rmse", "lambda",
                         "time_s", "converged", "error"])
        for dev_name, seed_results in all_results.items():
            for seed, sd_res in enumerate(seed_results):
                for m_name in methods:
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
        print(f"  {'Method':<18}  {'RMSE':>10}  {'λ':>8}  {'Time(s)':>9}")
        print(f"  {'─'*55}")
        for m_name in methods:
            vals_r = [s[m_name]["rmse"] for s in seed_results
                      if s[m_name]["error"] is None]
            vals_l = [s[m_name]["lambda"] for s in seed_results
                      if s[m_name]["error"] is None]
            vals_t = [s[m_name]["time"] for s in seed_results
                      if s[m_name]["error"] is None]
            if vals_r:
                print(f"  {m_name:<18}  "
                      f"{np.mean(vals_r):.4f}±{np.std(vals_r):.4f}  "
                      f"{np.mean(vals_l):8.3f}  "
                      f"{np.mean(vals_t):>9.1f}")
            else:
                print(f"  {m_name:<18}  {'FAILED':>10}")
    print("=" * 72)
    print(f"\nFull report: {md_path}")


if __name__ == "__main__":
    main()
