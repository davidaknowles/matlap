#!/usr/bin/env python
"""Compare proximal nuclear-norm lambda selection against entrywise CV.

This is the plain proximal baseline:

    min_X 0.5 * sum_obs (Y_ij - X_ij)^2 + lambda * ||X||_*

There is no unknown homoskedastic noise update here.  The benchmark reports
fixed-lambda grid fits, the lambda selected by observed-entry CV, and oracle
grid choices based on inaccessible X/test-Y RMSE.
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

from matlap import sample_nnd, taylor_gradient
from matlap.proximal import proximal_gradient
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
        m, n = part.split("x", 1) if "x" in part else (part, part)
        shapes.append((int(m), int(n)))
    return shapes


SHAPES = _env_shapes("PROX_LAMBDA_SHAPES", "100x1000")
N_SEEDS = _env_int("PROX_LAMBDA_N_SEEDS", 1)
LAMBDAS = _env_float_list("PROX_LAMBDA_GRID", "0.5,1,2,4,8,16,32")
LAMBDA_TRUE = _env_float("PROX_LAMBDA_TRUE", 5.0)
GAMMA2_TRUE = _env_float("PROX_LAMBDA_GAMMA2_TRUE", 0.25)
MISSING_FRAC = _env_float("PROX_LAMBDA_MISSING_FRAC", 0.30)
MAX_ITER = _env_int("PROX_LAMBDA_MAX_ITER", 80)
CV_FOLDS = _env_int("PROX_LAMBDA_CV_FOLDS", 3)
WARMUP = _env_int("PROX_LAMBDA_WARMUP", 1)
FIXED_ITER = bool(_env_int("PROX_LAMBDA_FIXED_ITER", 1))
SVD_RANK = _env_int("PROX_LAMBDA_SVD_RANK", 0)
SVD_N_ITER = _env_int("PROX_LAMBDA_SVD_N_ITER", 2)
SVD_OVERSAMPLE = _env_int("PROX_LAMBDA_SVD_OVERSAMPLE", 10)
OUTPUT_PREFIX = os.environ.get(
    "PROX_LAMBDA_OUTPUT", "results/benchmark_proximal_lambda_cv"
)
TAYLOR_SCORES = bool(_env_int("PROX_LAMBDA_TAYLOR_SCORES", 1))


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
    Y_full = X_true + rng.standard_normal((m, n)).astype(np.float32) * np.sqrt(GAMMA2_TRUE)
    train_mask = rng.random((m, n)) >= MISSING_FRAC
    Y_train = np.where(train_mask, Y_full, np.nan).astype(np.float32)
    S_train = np.where(train_mask, 1.0, np.inf).astype(np.float32)
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


def _fit(Y, S, lam: float, *, init_X=None, init_svd_basis=None, seed: int = 0):
    kwargs = {
        "max_iter": MAX_ITER,
        "fixed_iter": FIXED_ITER,
        "init_X": init_X,
        "random_seed": seed,
    }
    if SVD_RANK > 0:
        kwargs.update({
            "svd_rank": SVD_RANK,
            "svd_n_iter": SVD_N_ITER,
            "svd_oversample": SVD_OVERSAMPLE,
            "init_svd_basis": init_svd_basis,
        })
    return proximal_gradient(Y, S, float(lam), **kwargs)


def _fit_lambda_path(Y, S, lambdas: list[float], *, seed: int):
    by_lambda = {}
    times = {}
    init_X = None
    init_basis = None
    elapsed = 0.0
    for lam in sorted(lambdas, reverse=True):
        t0 = time.perf_counter()
        res = _fit(Y, S, lam, init_X=init_X, init_svd_basis=init_basis, seed=seed)
        res.X.block_until_ready()
        fit_time = time.perf_counter() - t0
        elapsed += fit_time
        by_lambda[float(lam)] = res
        times[float(lam)] = fit_time
        init_X = res.X
        init_basis = res.svd_basis
    return by_lambda, times, elapsed


def _cv_select_lambda(Y, S, lambdas: list[float], *, seed: int):
    obs = np.asarray(jnp.isfinite(Y) & jnp.isfinite(S))
    obs_idx = np.argwhere(obs)
    if len(obs_idx) < CV_FOLDS:
        raise ValueError(f"Fewer observed entries ({len(obs_idx)}) than CV folds ({CV_FOLDS}).")

    rng = np.random.default_rng(seed + 100_000)
    perm = rng.permutation(len(obs_idx))
    fold_ids = np.arange(len(obs_idx)) % CV_FOLDS

    scores = {float(lam): [] for lam in lambdas}
    elapsed = 0.0
    for fold in range(CV_FOLDS):
        val_pos = obs_idx[perm[fold_ids == fold]]
        S_fit = S.at[val_pos[:, 0], val_pos[:, 1]].set(jnp.inf)
        fits, _, fold_elapsed = _fit_lambda_path(Y, S_fit, lambdas, seed=seed + fold)
        elapsed += fold_elapsed
        i_val, j_val = val_pos[:, 0], val_pos[:, 1]
        for lam, res in fits.items():
            pred = res.X[i_val, j_val]
            true = Y[i_val, j_val]
            scores[lam].append(float(jnp.mean((pred - true) ** 2)))

    cv_mse = {lam: float(np.mean(vals)) for lam, vals in scores.items()}
    best_lambda = min(cv_mse, key=cv_mse.get)
    return best_lambda, cv_mse, elapsed


def _score_taylor_at_prox(Y, S, lam: float, X) -> tuple[dict[str, float], float]:
    """Score a fixed proximal estimate with the Taylor delta approximation."""
    t0 = time.perf_counter()
    if Y.shape[0] < Y.shape[1]:
        res = taylor_gradient(
            Y.T,
            S.T,
            lam,
            max_iter=0,
            init_mu=X.T,
            recover_sigma=False,
        )
        res.sigma_diag.block_until_ready()
        scores = {
            "taylor_elbo": float(res.elbo),
            "taylor_loo": float(closed_form_loo(res.mu, res.sigma_diag, Y.T, S.T)),
        }
    else:
        res = taylor_gradient(
            Y,
            S,
            lam,
            max_iter=0,
            init_mu=X,
            recover_sigma=False,
        )
        res.sigma_diag.block_until_ready()
        scores = {
            "taylor_elbo": float(res.elbo),
            "taylor_loo": float(closed_form_loo(res.mu, res.sigma_diag, Y, S)),
        }
    return scores, time.perf_counter() - t0


def _row_from_result(
    *,
    shape: str,
    m: int,
    n: int,
    seed: int,
    selector: str,
    lam: float,
    res,
    X_true,
    Y_full,
    train_mask,
    test_mask,
    time_s: float,
    cv_mse: float = float("nan"),
) -> dict:
    return {
        "shape": shape,
        "m": m,
        "n": n,
        "seed": seed,
        "selector": selector,
        "lambda": lam,
        "cv_mse": cv_mse,
        "rmse_x": _rmse(res.X, X_true),
        "train_y_rmse": _rmse(res.X, Y_full, train_mask),
        "test_y_rmse": _rmse(res.X, Y_full, test_mask),
        "time_s": time_s,
        "n_iter": res.n_iter,
        "converged": res.converged,
    }


def _run_one(seed: int, shape: tuple[int, int]) -> list[dict]:
    m, n = shape
    shape_name = f"{m}x{n}"
    Y, S, Y_full, X_true, train_mask, test_mask = _make_data(seed, m, n)

    rows = []
    full_fits, full_fit_times, full_grid_time = _fit_lambda_path(Y, S, LAMBDAS, seed=seed)
    for lam in LAMBDAS:
        res = full_fits[float(lam)]
        rows.append(_row_from_result(
            shape=shape_name,
            m=m,
            n=n,
            seed=seed,
            selector="fixed_lambda",
            lam=float(lam),
            res=res,
            X_true=X_true,
            Y_full=Y_full,
            train_mask=train_mask,
            test_mask=test_mask,
            time_s=full_fit_times[float(lam)],
        ))

    oracle_x = min(
        (row for row in rows if row["selector"] == "fixed_lambda"),
        key=lambda row: row["rmse_x"],
    )
    rows.append({**oracle_x, "selector": "oracle_x_grid", "time_s": full_grid_time})

    oracle_test_y = min(
        (row for row in rows if row["selector"] == "fixed_lambda"),
        key=lambda row: row["test_y_rmse"],
    )
    rows.append({**oracle_test_y, "selector": "oracle_test_y_grid", "time_s": full_grid_time})

    if TAYLOR_SCORES:
        taylor_scores = {}
        taylor_score_time = 0.0
        for lam in LAMBDAS:
            scores, score_time = _score_taylor_at_prox(Y, S, float(lam), full_fits[float(lam)].X)
            taylor_scores[float(lam)] = scores
            taylor_score_time += score_time

        for score_name in ["taylor_elbo", "taylor_loo"]:
            best_lam = max(taylor_scores, key=lambda lam: taylor_scores[lam][score_name])
            res = full_fits[best_lam]
            rows.append(_row_from_result(
                shape=shape_name,
                m=m,
                n=n,
                seed=seed,
                selector=f"{score_name}_lambda",
                lam=best_lam,
                res=res,
                X_true=X_true,
                Y_full=Y_full,
                train_mask=train_mask,
                test_mask=test_mask,
                time_s=full_grid_time + taylor_score_time,
                cv_mse=float("nan"),
            ))

        for lam, scores in taylor_scores.items():
            for score_name, score_value in scores.items():
                rows.append({
                    "shape": shape_name,
                    "m": m,
                    "n": n,
                    "seed": seed,
                    "selector": f"{score_name}_score",
                    "lambda": lam,
                    "cv_mse": score_value,
                    "rmse_x": float("nan"),
                    "train_y_rmse": float("nan"),
                    "test_y_rmse": float("nan"),
                    "time_s": taylor_score_time,
                    "n_iter": 0,
                    "converged": False,
                })

    cv_lambda, cv_mse, cv_time = _cv_select_lambda(Y, S, LAMBDAS, seed=seed)
    t0 = time.perf_counter()
    cv_res = _fit(Y, S, cv_lambda, seed=seed + 1_000_000)
    cv_res.X.block_until_ready()
    cv_total_time = cv_time + (time.perf_counter() - t0)
    rows.append(_row_from_result(
        shape=shape_name,
        m=m,
        n=n,
        seed=seed,
        selector="cv_lambda",
        lam=cv_lambda,
        res=cv_res,
        X_true=X_true,
        Y_full=Y_full,
        train_mask=train_mask,
        test_mask=test_mask,
        time_s=cv_total_time,
        cv_mse=cv_mse[cv_lambda],
    ))

    for lam, mse in cv_mse.items():
        rows.append({
            "shape": shape_name,
            "m": m,
            "n": n,
            "seed": seed,
            "selector": "cv_score",
            "lambda": lam,
            "cv_mse": mse,
            "rmse_x": float("nan"),
            "train_y_rmse": float("nan"),
            "test_y_rmse": float("nan"),
            "time_s": cv_time,
            "n_iter": 0,
            "converged": False,
        })

    return rows


def _aggregate(rows: list[dict]) -> list[dict]:
    def finite_mean(values: list[float]) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(np.mean(arr)) if arr.size else float("nan")

    out = []
    keys = []
    for row in rows:
        key = (row["shape"], row["selector"], row["lambda"])
        if key not in keys:
            keys.append(key)
    for shape, selector, lam in keys:
        sub = [
            r for r in rows
            if r["shape"] == shape and r["selector"] == selector and r["lambda"] == lam
        ]
        out.append({
            "shape": shape,
            "selector": selector,
            "lambda": lam,
            "cv_mse_mean": finite_mean([r["cv_mse"] for r in sub]),
            "rmse_x_mean": finite_mean([r["rmse_x"] for r in sub]),
            "train_y_rmse_mean": finite_mean([r["train_y_rmse"] for r in sub]),
            "test_y_rmse_mean": finite_mean([r["test_y_rmse"] for r in sub]),
            "time_median_s": float(np.nanmedian([r["time_s"] for r in sub])),
            "n_iter_median": float(np.nanmedian([r["n_iter"] for r in sub])),
            "converged_count": int(np.sum([bool(r["converged"]) for r in sub])),
            "n_seeds": len(sub),
        })
    return out


def _write_outputs(rows: list[dict], agg: list[dict]) -> tuple[Path, Path]:
    prefix = Path(OUTPUT_PREFIX)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    raw_path = prefix.with_name(prefix.name + "_raw").with_suffix(".csv")
    md_path = prefix.with_suffix(".md")

    with raw_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        writer.writeheader()
        writer.writerows(agg)

    lines = [
        "# Proximal Lambda CV Benchmark",
        "",
        f"- device: `{DEVICE}`",
        f"- shapes: `{SHAPES}`",
        f"- n_seeds: `{N_SEEDS}`",
        f"- lambda_grid: `{LAMBDAS}`",
        f"- lambda_true_for_simulation: `{LAMBDA_TRUE}`",
        f"- gamma2_true_for_simulation: `{GAMMA2_TRUE}`",
        f"- missing_frac: `{MISSING_FRAC}`",
        f"- max_iter: `{MAX_ITER}`",
        f"- fixed_iter: `{FIXED_ITER}`",
        f"- cv_folds: `{CV_FOLDS}`",
        f"- svd_rank: `{SVD_RANK or 'exact'}`",
        f"- taylor_scores: `{TAYLOR_SCORES}`",
        f"- raw CSV: `{raw_path}`",
        "",
        "| shape | selector | lambda | CV MSE | X RMSE | train Y RMSE | test Y RMSE | median time s |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in agg:
        if row["selector"] == "cv_score":
            continue
        lines.append(
            f"| {row['shape']} | {row['selector']} | {row['lambda']:.4g} | "
            f"{row['cv_mse_mean']:.6f} | {row['rmse_x_mean']:.6f} | "
            f"{row['train_y_rmse_mean']:.6f} | {row['test_y_rmse_mean']:.6f} | "
            f"{row['time_median_s']:.3f} |"
        )

    lines.extend([
        "",
        "## Selection Scores",
        "",
        "| shape | score | lambda | value |",
        "|---|---|---:|---:|",
    ])
    for row in agg:
        if row["selector"] not in {"cv_score", "taylor_elbo_score", "taylor_loo_score"}:
            continue
        lines.append(
            f"| {row['shape']} | {row['selector']} | {row['lambda']:.4g} | "
            f"{row['cv_mse_mean']:.6f} |"
        )

    md_path.write_text("\n".join(lines) + "\n")
    return csv_path, md_path


def main() -> None:
    if WARMUP:
        for shape in SHAPES:
            Y, S, *_ = _make_data(123_456, *shape)
            fits, _, _ = _fit_lambda_path(Y, S, [LAMBDAS[0]], seed=123_456)
            fits[float(LAMBDAS[0])].X.block_until_ready()

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
