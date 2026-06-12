#!/usr/bin/env python
"""Benchmark effective-lambda vs base-lambda grids for proximal NND noise EB."""

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
from matlap.scoring import closed_form_loo


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
        if "x" in part:
            m, n = part.split("x", 1)
            shapes.append((int(m), int(n)))
        else:
            n = int(part)
            shapes.append((n, n))
    return shapes


SHAPES = _env_shapes("PROX_NOISE_SHAPES", "40x40,80x80,50x100")
N_SEEDS = _env_int("PROX_NOISE_N_SEEDS", 2)
LAMBDA_TRUE = _env_float("PROX_NOISE_LAMBDA_TRUE", 0.05)
G_TRUE = _env_float("PROX_NOISE_G_TRUE", 0.25)
MISSING_FRAC = _env_float("PROX_NOISE_MISSING_FRAC", 0.30)
LAMBDA_INIT = _env_float("PROX_NOISE_LAMBDA_INIT", 2.0)
LAMBDA_GRID = _env_float_list("PROX_NOISE_LAMBDA_GRID", "0.5,1,2,4,8,16")
PARAMETERIZATIONS = [
    part.strip().lower()
    for part in os.environ.get("PROX_NOISE_PARAMETERIZATIONS", "effective,base").split(",")
    if part.strip()
]
SELECTORS = {
    part.strip().lower()
    for part in os.environ.get("PROX_NOISE_SELECTORS", "joint,cv,taylor_elbo,loo").split(",")
    if part.strip()
}
MAX_OUTER = _env_int("PROX_NOISE_MAX_OUTER", 8)
PROX_MAX_ITER = _env_int("PROX_NOISE_PROX_MAX_ITER", 40)
GAMMA_MAX_ITER = _env_int("PROX_NOISE_GAMMA_MAX_ITER", 12)
CV_FRAC = _env_float("PROX_NOISE_CV_FRAC", 0.20)
OUTPUT_PREFIX = os.environ.get("PROX_NOISE_OUTPUT", "results/benchmark_prox_noise")


def _device():
    gpus = jax.devices("gpu")
    return gpus[0] if gpus else jax.devices()[0]


DEVICE = _device()


def _to_device(x):
    return jax.device_put(jnp.asarray(x, dtype=jnp.float32), DEVICE)


def _rmse(A, B, mask=None) -> float:
    diff2 = (jnp.asarray(A) - jnp.asarray(B)) ** 2
    if mask is not None:
        diff2 = jnp.where(mask, diff2, jnp.nan)
        return float(jnp.sqrt(jnp.nanmean(diff2)))
    return float(jnp.sqrt(jnp.mean(diff2)))


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
    Y_full = X_true + rng.standard_normal((m, n)).astype(np.float32) * np.sqrt(S_known ** 2 + G_TRUE)
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


def _fit_joint(Y, S, seed: int):
    return proximal_noise_eb(
        Y,
        S,
        lambda_val=LAMBDA_INIT,
        update_lambda=True,
        max_outer=MAX_OUTER,
        prox_max_iter=PROX_MAX_ITER,
        gamma_max_iter=GAMMA_MAX_ITER,
        lambda_parameterization="base",
        random_seed=seed,
    )


def _fit_grid_cv(Y, S, seed: int, parameterization: str):
    rng = np.random.default_rng(seed + 100_000)
    obs = np.asarray(jnp.isfinite(Y) & jnp.isfinite(S))
    val_mask_np = obs & (rng.random(obs.shape) < CV_FRAC)
    if not np.any(val_mask_np):
        val_mask_np[np.argwhere(obs)[0][0], np.argwhere(obs)[0][1]] = True
    fit_mask_np = obs & ~val_mask_np

    Y_np = np.asarray(Y)
    S_np = np.asarray(S)
    Y_fit = _to_device(np.where(fit_mask_np, Y_np, np.nan).astype(np.float32))
    S_fit = _to_device(np.where(fit_mask_np, S_np, np.inf).astype(np.float32))
    val_mask = jnp.asarray(val_mask_np)

    scores = []
    for lam in LAMBDA_GRID:
        cv_res = proximal_noise_eb(
            Y_fit,
            S_fit,
            lambda_val=lam,
            update_lambda=False,
            max_outer=MAX_OUTER,
            prox_max_iter=PROX_MAX_ITER,
            gamma_max_iter=GAMMA_MAX_ITER,
            lambda_parameterization=parameterization,
            random_seed=seed,
        )
        scores.append(_rmse(cv_res.X, Y, val_mask))
    best_lambda = LAMBDA_GRID[int(np.argmin(scores))]

    return proximal_noise_eb(
        Y,
        S,
        lambda_val=best_lambda,
        update_lambda=False,
        max_outer=MAX_OUTER,
        prox_max_iter=PROX_MAX_ITER,
        gamma_max_iter=GAMMA_MAX_ITER,
        lambda_parameterization=parameterization,
        random_seed=seed,
    )


def _loo_score(res, Y, S) -> float:
    S_eff = jnp.sqrt(S ** 2 + jnp.asarray(res.g, dtype=jnp.float32))
    return float(closed_form_loo(res.X, res.sigma_diag, Y, S_eff))


def _fit_grid_model_score(Y, S, seed: int, score_name: str, parameterization: str):
    fits = []
    scores = []
    for lam in LAMBDA_GRID:
        res = proximal_noise_eb(
            Y,
            S,
            lambda_val=lam,
            update_lambda=False,
            max_outer=MAX_OUTER,
            prox_max_iter=PROX_MAX_ITER,
            gamma_max_iter=GAMMA_MAX_ITER,
            lambda_parameterization=parameterization,
            random_seed=seed,
        )
        fits.append(res)
        if score_name == "taylor_elbo":
            scores.append(-res.objective_trace[-1])
        elif score_name == "loo":
            scores.append(_loo_score(res, Y, S))
        else:
            raise ValueError(f"unknown score_name {score_name!r}")
    return fits[int(np.argmax(scores))]


def _fit_grid_taylor_elbo(Y, S, seed: int, parameterization: str):
    return _fit_grid_model_score(Y, S, seed, "taylor_elbo", parameterization)


def _fit_grid_loo(Y, S, seed: int, parameterization: str):
    return _fit_grid_model_score(Y, S, seed, "loo", parameterization)


def _run_one(seed: int, shape: tuple[int, int]) -> list[dict]:
    m, n = shape
    Y, S, Y_full, X_true, train_mask, test_mask = _make_data(seed, m, n)
    rows = []
    method_specs = []
    if "joint" in SELECTORS:
        method_specs.append(
            ("joint_base_lambda_g", "base", lambda y, s, sd, p: _fit_joint(y, s, sd))
        )
    for parameterization in PARAMETERIZATIONS:
        if "cv" in SELECTORS:
            method_specs.append((
                f"grid_cv_{parameterization}",
                parameterization,
                lambda y, s, sd, p: _fit_grid_cv(y, s, sd, p),
            ))
        if "taylor_elbo" in SELECTORS:
            method_specs.append((
                f"grid_taylor_elbo_{parameterization}",
                parameterization,
                lambda y, s, sd, p: _fit_grid_taylor_elbo(y, s, sd, p),
            ))
        if "loo" in SELECTORS:
            method_specs.append((
                f"grid_loo_{parameterization}",
                parameterization,
                lambda y, s, sd, p: _fit_grid_loo(y, s, sd, p),
            ))
    if not method_specs:
        raise ValueError("PROX_NOISE_SELECTORS selected no methods.")

    for method, parameterization, fitter in method_specs:
        t0 = time.perf_counter()
        res = fitter(Y, S, seed, parameterization)
        res.X.block_until_ready()
        elapsed = time.perf_counter() - t0
        rows.append({
            "shape": f"{m}x{n}",
            "m": m,
            "n": n,
            "seed": seed,
            "method": method,
            "lambda_parameterization": res.lambda_parameterization,
            "rmse_x": _rmse(res.X, X_true),
            "train_y_rmse": _rmse(res.X, Y_full, train_mask),
            "test_y_rmse": _rmse(res.X, Y_full, test_mask),
            "g_true": G_TRUE,
            "g_hat": res.g,
            "lambda_hat": res.lambda_val,
            "lambda_eff_hat": res.lambda_eff,
            "time_s": elapsed,
            "n_iter": res.n_iter,
            "converged": res.converged,
            "objective": res.objective_trace[-1],
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
            "rmse_x_mean": float(np.mean([r["rmse_x"] for r in sub])),
            "test_y_rmse_mean": float(np.mean([r["test_y_rmse"] for r in sub])),
            "g_hat_mean": float(np.mean([r["g_hat"] for r in sub])),
            "lambda_hat_mean": float(np.mean([r["lambda_hat"] for r in sub])),
            "lambda_eff_hat_mean": float(np.mean([r["lambda_eff_hat"] for r in sub])),
            "parameterization": sub[0]["lambda_parameterization"],
            "time_median_s": float(np.median([r["time_s"] for r in sub])),
            "n_iter_median": float(np.median([r["n_iter"] for r in sub])),
            "converged_count": int(np.sum([bool(r["converged"]) for r in sub])),
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
        "# Proximal Noise EB Benchmark",
        "",
        f"- device: `{DEVICE}`",
        f"- shapes: `{SHAPES}`",
        f"- n_seeds: `{N_SEEDS}`",
        f"- lambda_true: `{LAMBDA_TRUE}`",
        f"- g_true: `{G_TRUE}`",
        f"- train missing fraction: `{MISSING_FRAC}`",
        f"- lambda_init: `{LAMBDA_INIT}`",
        f"- lambda_grid: `{LAMBDA_GRID}`",
        f"- parameterizations: `{PARAMETERIZATIONS}`",
        f"- selectors: `{sorted(SELECTORS)}`",
        f"- cv_frac: `{CV_FRAC}`",
        f"- max_outer: `{MAX_OUTER}`",
        f"- prox_max_iter: `{PROX_MAX_ITER}`",
        f"- raw CSV: `{raw_path}`",
        "",
        "| shape | method | param | X RMSE | test Y RMSE | g hat | lambda grid/base | lambda eff | median time s | median outer iters | converged |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in agg:
        lines.append(
            f"| {row['shape']} | {row['method']} | {row['parameterization']} | "
            f"{row['rmse_x_mean']:.6f} | {row['test_y_rmse_mean']:.6f} | "
            f"{row['g_hat_mean']:.6f} | {row['lambda_hat_mean']:.6f} | "
            f"{row['lambda_eff_hat_mean']:.6f} | "
            f"{row['time_median_s']:.3f} | {row['n_iter_median']:.1f} | "
            f"{row['converged_count']}/{row['n_seeds']} |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    return csv_path, md_path


def main() -> None:
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
