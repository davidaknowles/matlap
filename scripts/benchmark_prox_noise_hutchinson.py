#!/usr/bin/env python
"""Benchmark exact vs Hutchinson-CG updates for proximal-noise g fitting."""

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

from matlap import proximal_noise_eb, sample_nnd


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_float_list(name: str, default: str) -> list[float]:
    raw = os.environ.get(name, default)
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def _env_shapes(name: str, default: str) -> list[tuple[int, int]]:
    shapes = []
    for part in os.environ.get(name, default).split(","):
        part = part.strip().lower()
        if not part:
            continue
        m, n = part.split("x", 1) if "x" in part else (part, part)
        shapes.append((int(m), int(n)))
    return shapes


SHAPES = _env_shapes("PROX_HUTCH_SHAPES", "40x40,80x80")
N_SEEDS = _env_int("PROX_HUTCH_N_SEEDS", 2)
LAMBDAS = _env_float_list("PROX_HUTCH_LAMBDAS", "0.5,1,2,4")
LAMBDA_TRUE = _env_float("PROX_HUTCH_LAMBDA_TRUE", 0.05)
G_TRUE = _env_float("PROX_HUTCH_G_TRUE", 0.25)
MISSING_FRAC = _env_float("PROX_HUTCH_MISSING_FRAC", 0.30)
MAX_OUTER = _env_int("PROX_HUTCH_MAX_OUTER", 6)
PROX_MAX_ITER = _env_int("PROX_HUTCH_PROX_MAX_ITER", 30)
EXACT_GAMMA_MAX_ITER = _env_int("PROX_HUTCH_EXACT_GAMMA_MAX_ITER", 8)
HUTCH_GAMMA_MAX_ITER = _env_int("PROX_HUTCH_GAMMA_MAX_ITER", 12)
HUTCH_PROBES = _env_int("PROX_HUTCH_PROBES", 4)
HUTCH_CG_MAXITER = _env_int("PROX_HUTCH_CG_MAXITER", 30)
HUTCH_CG_TOL = _env_float("PROX_HUTCH_CG_TOL", 1e-3)
HUTCH_LR = _env_float("PROX_HUTCH_LR", 1e-4)
HUTCH_GRAD_CLIP = _env_float("PROX_HUTCH_GRAD_CLIP", 1e3)
WARMUP = _env_int("PROX_HUTCH_WARMUP", 1)
OUTPUT_PREFIX = os.environ.get(
    "PROX_HUTCH_OUTPUT", "results/benchmark_prox_noise_hutchinson"
)


def _device():
    gpus = jax.devices("gpu")
    return gpus[0] if gpus else jax.devices()[0]


DEVICE = _device()


def _to_device(x):
    return jax.device_put(jnp.asarray(x, dtype=jnp.float32), DEVICE)


def _sample_matrix(rng: np.random.Generator, m: int, n: int) -> np.ndarray:
    if m >= n:
        X, _ = sample_nnd(rng, m, n, LAMBDA_TRUE)
        return X
    X_t, _ = sample_nnd(rng, n, m, LAMBDA_TRUE)
    return X_t.T


def _make_data(seed: int, m: int, n: int):
    rng = np.random.default_rng(seed)
    X_true = _sample_matrix(rng, m, n).astype(np.float32)
    S_known = rng.uniform(0.3, 0.9, size=(m, n)).astype(np.float32)
    Y_full = X_true + rng.standard_normal((m, n)).astype(np.float32) * np.sqrt(
        S_known ** 2 + G_TRUE
    )
    train_mask = rng.random((m, n)) >= MISSING_FRAC
    Y_train = np.where(train_mask, Y_full, np.nan).astype(np.float32)
    S_train = np.where(train_mask, S_known, np.inf).astype(np.float32)
    return (
        _to_device(Y_train),
        _to_device(S_train),
        _to_device(Y_full),
        _to_device(X_true),
        jnp.asarray(train_mask),
        jnp.asarray(~train_mask),
    )


def _rmse(A, B, mask=None) -> float:
    diff2 = (jnp.asarray(A) - jnp.asarray(B)) ** 2
    if mask is not None:
        diff2 = jnp.where(mask, diff2, jnp.nan)
        return float(jnp.sqrt(jnp.nanmean(diff2)))
    return float(jnp.sqrt(jnp.mean(diff2)))


def _fit(method: str, Y, S, lam: float, seed: int):
    common = {
        "lambda_val": lam,
        "update_lambda": False,
        "max_outer": MAX_OUTER,
        "prox_max_iter": PROX_MAX_ITER,
        "random_seed": seed,
        "score_exact_objective": False,
        "recover_sigma_diag": False,
    }
    if method == "exact":
        return proximal_noise_eb(
            Y,
            S,
            gamma_update="exact",
            gamma_max_iter=EXACT_GAMMA_MAX_ITER,
            **common,
        )
    if method == "hutchinson":
        return proximal_noise_eb(
            Y,
            S,
            gamma_update="hutchinson",
            gamma_max_iter=HUTCH_GAMMA_MAX_ITER,
            hutchinson_probes=HUTCH_PROBES,
            hutchinson_lr=HUTCH_LR,
            hutchinson_cg_tol=HUTCH_CG_TOL,
            hutchinson_cg_maxiter=HUTCH_CG_MAXITER,
            hutchinson_grad_clip=HUTCH_GRAD_CLIP,
            hutchinson_seed=seed + 50_000,
            **common,
        )
    raise ValueError(f"unknown method {method!r}")


def _run_one(seed: int, shape: tuple[int, int]) -> list[dict]:
    m, n = shape
    Y, S, Y_full, X_true, train_mask, test_mask = _make_data(seed, m, n)
    rows = []
    for lam in LAMBDAS:
        for method in ["exact", "hutchinson"]:
            t0 = time.perf_counter()
            res = _fit(method, Y, S, lam, seed)
            res.X.block_until_ready()
            elapsed = time.perf_counter() - t0
            rows.append({
                "shape": f"{m}x{n}",
                "m": m,
                "n": n,
                "seed": seed,
                "lambda": lam,
                "method": method,
                "rmse_x": _rmse(res.X, X_true),
                "test_y_rmse": _rmse(res.X, Y_full, test_mask),
                "g_true": G_TRUE,
                "g_hat": res.g,
                "g_abs_error": abs(res.g - G_TRUE),
                "lambda_eff": res.lambda_eff,
                "time_s": elapsed,
                "n_iter": res.n_iter,
                "converged": res.converged,
                "g_trace": ";".join(f"{x:.6g}" for x in res.g_trace),
            })
    return rows


def _aggregate(rows: list[dict]) -> list[dict]:
    out = []
    keys = []
    for row in rows:
        key = (row["shape"], row["lambda"], row["method"])
        if key not in keys:
            keys.append(key)
    for shape, lam, method in keys:
        sub = [
            r for r in rows
            if r["shape"] == shape and r["lambda"] == lam and r["method"] == method
        ]
        out.append({
            "shape": shape,
            "lambda": lam,
            "method": method,
            "rmse_x_mean": float(np.mean([r["rmse_x"] for r in sub])),
            "test_y_rmse_mean": float(np.mean([r["test_y_rmse"] for r in sub])),
            "g_hat_mean": float(np.mean([r["g_hat"] for r in sub])),
            "g_abs_error_mean": float(np.mean([r["g_abs_error"] for r in sub])),
            "lambda_eff_mean": float(np.mean([r["lambda_eff"] for r in sub])),
            "time_median_s": float(np.median([r["time_s"] for r in sub])),
            "n_seeds": len(sub),
        })
    return out


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

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        writer.writeheader()
        writer.writerows(agg)

    lines = [
        "# Proximal Noise Hutchinson Benchmark",
        "",
        f"- device: `{DEVICE}`",
        f"- shapes: `{SHAPES}`",
        f"- n_seeds: `{N_SEEDS}`",
        f"- lambdas: `{LAMBDAS}`",
        f"- g_true: `{G_TRUE}`",
        f"- hutchinson_probes: `{HUTCH_PROBES}`",
        f"- hutchinson_cg_maxiter: `{HUTCH_CG_MAXITER}`",
        f"- hutchinson_lr: `{HUTCH_LR}`",
        f"- warmup: `{WARMUP}`",
        f"- raw CSV: `{raw_path}`",
        "",
        "| shape | lambda | method | X RMSE | test Y RMSE | g hat | |g err| | lambda eff | median time s |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in agg:
        lines.append(
            f"| {row['shape']} | {row['lambda']:.4g} | {row['method']} | "
            f"{row['rmse_x_mean']:.6f} | {row['test_y_rmse_mean']:.6f} | "
            f"{row['g_hat_mean']:.6f} | {row['g_abs_error_mean']:.6f} | "
            f"{row['lambda_eff_mean']:.6f} | {row['time_median_s']:.3f} |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    return csv_path, md_path


def main() -> None:
    if WARMUP:
        for shape in SHAPES:
            Y, S, *_ = _make_data(123_456, *shape)
            for method in ["exact", "hutchinson"]:
                res = _fit(method, Y, S, LAMBDAS[0], 123_456)
                res.X.block_until_ready()

    rows = []
    for shape in SHAPES:
        for seed in range(N_SEEDS):
            rows.extend(_run_one(seed, shape))
    agg = _aggregate(rows)
    csv_path, md_path = _write_outputs(rows, agg)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
