#!/usr/bin/env python
"""Benchmark exact proximal SVT vs fixed/adaptive randomized SVT.

This is intentionally separate from ``benchmark_lambda.py``: it evaluates the
proximal solver approximation at a fixed lambda rather than mixing in CV,
MCMC, or other model families.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matlap import proximal_gradient, sample_nnd


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_int_list(name: str, default: str) -> list[int]:
    raw = os.environ.get(name, default)
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


SHAPES = _env_int_list("PROX_RSVD_SHAPES", "40,100,200")
N_SEEDS = _env_int("PROX_RSVD_N_SEEDS", 3)
MAX_ITER = _env_int("PROX_RSVD_MAX_ITER", 80)
LAMBDA_TRUE = _env_float("PROX_RSVD_LAMBDA_TRUE", 0.05)
LAMBDA_FIT = _env_float("PROX_RSVD_LAMBDA_FIT", 3.0)
SIGMA_NOISE = _env_float("PROX_RSVD_SIGMA_NOISE", 1.0)
RSVD_N_ITER = _env_int("PROX_RSVD_N_ITER", 2)
RSVD_OVERSAMPLE = _env_int("PROX_RSVD_OVERSAMPLE", 10)
FIXED_RANKS = _env_int_list("PROX_RSVD_FIXED_RANKS", "20,50")
ADAPTIVE_START_RANK = _env_int("PROX_RSVD_ADAPT_START_RANK", 10)
ADAPTIVE_MIN_RANK = _env_int("PROX_RSVD_ADAPT_MIN_RANK", 5)
ADAPTIVE_MAX_RANK = _env_int("PROX_RSVD_ADAPT_MAX_RANK", 80)
ADAPTIVE_STEP = _env_int("PROX_RSVD_ADAPT_STEP", 10)
OUTPUT_PREFIX = os.environ.get(
    "PROX_RSVD_OUTPUT", "results/benchmark_prox_rsvd_adaptive"
)


def _device():
    gpus = jax.devices("gpu")
    return gpus[0] if gpus else jax.devices()[0]


DEVICE = _device()


def _to_device(x):
    return jax.device_put(jnp.asarray(x, dtype=jnp.float32), DEVICE)


def _rmse(A, B) -> float:
    return float(jnp.sqrt(jnp.mean((jnp.asarray(A) - jnp.asarray(B)) ** 2)))


def _rank_summary(trace: list[int]) -> str:
    if not trace:
        return ""
    arr = np.asarray(trace, dtype=int)
    return f"{arr[0]}->{arr[-1]} [{arr.min()},{arr.max()}]"


def _run_fit(method: str, Y, S, seed: int, rank: int | None = None):
    kwargs = {
        "max_iter": MAX_ITER,
        "tol": 1e-6,
        "random_seed": seed,
        "svd_n_iter": RSVD_N_ITER,
        "svd_oversample": RSVD_OVERSAMPLE,
    }
    if method == "exact_scan":
        kwargs["fixed_iter"] = True
    elif method == "rsvd_fixed_scan":
        kwargs.update({"fixed_iter": True, "svd_rank": rank})
    elif method == "rsvd_adaptive":
        kwargs.update({
            "fixed_iter": False,
            "svd_rank": rank,
            "svd_rank_adaptive": True,
            "svd_rank_min": ADAPTIVE_MIN_RANK,
            "svd_rank_max": min(ADAPTIVE_MAX_RANK, min(Y.shape)),
            "svd_rank_step": ADAPTIVE_STEP,
            "svd_rank_shrink_fraction": 0.5,
        })
    else:
        raise ValueError(f"unknown method {method!r}")

    t0 = time.perf_counter()
    res = proximal_gradient(Y, S, LAMBDA_FIT, **kwargs)
    res.X.block_until_ready()
    elapsed = time.perf_counter() - t0
    return res, elapsed


def _run_one(seed: int, n: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    X_true, _ = sample_nnd(rng, n, n, LAMBDA_TRUE)
    Y_np = X_true + rng.standard_normal((n, n)) * SIGMA_NOISE
    S_np = np.full((n, n), SIGMA_NOISE, dtype=np.float32)
    X_true_j = _to_device(X_true)
    Y = _to_device(Y_np)
    S = _to_device(S_np)

    rows = []

    exact, exact_time = _run_fit("exact_scan", Y, S, seed)
    rows.append({
        "shape": f"{n}x{n}",
        "n": n,
        "seed": seed,
        "method": "exact_scan",
        "rmse_true": _rmse(exact.X, X_true_j),
        "rmse_vs_exact": 0.0,
        "time_s": exact_time,
        "final_rank": "",
        "kept_rank": "",
        "rank_trace": "",
    })

    for fixed_rank in FIXED_RANKS:
        rank = min(fixed_rank, n)
        res, elapsed = _run_fit("rsvd_fixed_scan", Y, S, seed, rank=rank)
        rows.append({
            "shape": f"{n}x{n}",
            "n": n,
            "seed": seed,
            "method": f"rsvd_fixed_{fixed_rank}",
            "rmse_true": _rmse(res.X, X_true_j),
            "rmse_vs_exact": _rmse(res.X, exact.X),
            "time_s": elapsed,
            "final_rank": res.svd_rank,
            "kept_rank": res.svd_kept_rank,
            "rank_trace": _rank_summary(res.svd_rank_trace),
        })

    adapt_rank = min(ADAPTIVE_START_RANK, n)
    res, elapsed = _run_fit("rsvd_adaptive", Y, S, seed, rank=adapt_rank)
    rows.append({
        "shape": f"{n}x{n}",
        "n": n,
        "seed": seed,
        "method": "rsvd_adaptive",
        "rmse_true": _rmse(res.X, X_true_j),
        "rmse_vs_exact": _rmse(res.X, exact.X),
        "time_s": elapsed,
        "final_rank": res.svd_rank,
        "kept_rank": res.svd_kept_rank,
        "rank_trace": _rank_summary(res.svd_rank_trace),
    })

    return rows


def _aggregate(rows: list[dict]) -> list[dict]:
    out = []
    keys = []
    for row in rows:
        key = (row["shape"], row["method"])
        if key not in keys:
            keys.append(key)
    for shape, method in keys:
        sub = [r for r in rows if r["shape"] == shape and r["method"] == method]
        out.append({
            "shape": shape,
            "method": method,
            "rmse_true_mean": float(np.mean([r["rmse_true"] for r in sub])),
            "rmse_vs_exact_mean": float(np.mean([r["rmse_vs_exact"] for r in sub])),
            "time_median_s": float(np.median([r["time_s"] for r in sub])),
            "final_rank_median": _maybe_median([r["final_rank"] for r in sub]),
            "kept_rank_median": _maybe_median([r["kept_rank"] for r in sub]),
            "n_seeds": len(sub),
            "rank_trace_example": sub[-1]["rank_trace"],
        })
    return out


def _maybe_median(values):
    nums = [float(v) for v in values if v != "" and v is not None]
    if not nums:
        return ""
    return float(np.median(nums))


def _write_outputs(rows: list[dict], agg: list[dict]) -> tuple[Path, Path]:
    prefix = Path(OUTPUT_PREFIX)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    md_path = prefix.with_suffix(".md")

    raw_path = prefix.with_name(prefix.name + "_raw").with_suffix(".csv")
    with raw_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fieldnames = list(agg[0].keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(agg)

    lines = [
        "# Proximal rSVD Adaptive Benchmark",
        "",
        f"- device: `{DEVICE}`",
        f"- shapes: `{SHAPES}`",
        f"- n_seeds: `{N_SEEDS}`",
        f"- max_iter: `{MAX_ITER}`",
        f"- lambda_true: `{LAMBDA_TRUE}`",
        f"- lambda_fit: `{LAMBDA_FIT}`",
        f"- sigma_noise: `{SIGMA_NOISE}`",
        f"- fixed_ranks: `{FIXED_RANKS}`",
        f"- adaptive: start `{ADAPTIVE_START_RANK}`, min `{ADAPTIVE_MIN_RANK}`, "
        f"max `{ADAPTIVE_MAX_RANK}`, step `{ADAPTIVE_STEP}`",
        f"- raw CSV: `{raw_path}`",
        "",
        "| shape | method | RMSE true | RMSE vs exact | median time s | final rank | kept rank | rank trace |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in agg:
        lines.append(
            f"| {row['shape']} | {row['method']} | "
            f"{row['rmse_true_mean']:.6f} | {row['rmse_vs_exact_mean']:.6f} | "
            f"{row['time_median_s']:.4f} | {row['final_rank_median']} | "
            f"{row['kept_rank_median']} | {row['rank_trace_example']} |"
        )

    md_path.write_text("\n".join(lines) + "\n")
    return csv_path, md_path


def main():
    print("Proximal rSVD adaptive benchmark")
    print(f"  device={DEVICE}")
    print(f"  shapes={SHAPES}, seeds={N_SEEDS}, max_iter={MAX_ITER}")
    print(f"  lambda_true={LAMBDA_TRUE}, lambda_fit={LAMBDA_FIT}, sigma={SIGMA_NOISE}")

    rows = []
    for n in SHAPES:
        print(f"\nRunning {n}x{n}")
        for seed in range(N_SEEDS):
            t0 = time.perf_counter()
            seed_rows = _run_one(seed, n)
            rows.extend(seed_rows)
            print(f"  seed {seed}: {time.perf_counter() - t0:.2f}s")

    agg = _aggregate(rows)
    csv_path, md_path = _write_outputs(rows, agg)

    print("\nSummary")
    for row in agg:
        print(
            f"  {row['shape']:>7s} {row['method']:>16s} "
            f"rmse={row['rmse_true_mean']:.4f} "
            f"vs_exact={row['rmse_vs_exact_mean']:.4f} "
            f"time={row['time_median_s']:.3f}s "
            f"rank={row['rank_trace_example']}"
        )
    print(f"\nSaved CSV to {csv_path}")
    print(f"Saved Markdown to {md_path}")


if __name__ == "__main__":
    main()
