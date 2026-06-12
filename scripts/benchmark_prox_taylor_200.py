#!/usr/bin/env python
"""GPU benchmark for warm-started proximal CV vs prox-Taylor lambda scoring.

Runs medium-SNR, rank-truncated NND instances for several matrix shapes.  A
fixed subset of entries is held out as a final test set.  For each shape, the
script fits the nuclear-norm proximal estimator along a descending lambda path,
using the previous higher-lambda solution as the warm start, then scores each
fixed estimate with the Taylor delta approximation.  It also computes entry-wise
CV along the same descending warm-start trajectory inside each fold.
"""

from __future__ import annotations

import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matlap import matlap_faem, proximal_gradient, sample_nnd, taylor_gradient
from matlap.scoring import closed_form_loo


GPU_DEVICES = jax.devices("gpu")
if not GPU_DEVICES:
    raise RuntimeError("benchmark_prox_taylor_200.py requires a JAX GPU device.")
GPU_DEVICE = GPU_DEVICES[0]


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _to_gpu(x):
    return jax.device_put(jnp.asarray(x, dtype=jnp.float32), GPU_DEVICE)


def _to_gpu_bool(x):
    return jax.device_put(jnp.asarray(x, dtype=bool), GPU_DEVICE)


def _block_until_ready(x):
    if hasattr(x, "block_until_ready"):
        x.block_until_ready()


def parse_shapes(value: str) -> list[tuple[int, int]]:
    shapes = []
    for token in value.split(","):
        token = token.strip().lower()
        if not token:
            continue
        parts = token.split("x")
        if len(parts) != 2:
            raise ValueError(f"Invalid shape {token!r}; expected MxN.")
        m, n = (int(parts[0]), int(parts[1]))
        if m <= 0 or n <= 0:
            raise ValueError(f"Shape dimensions must be positive, got {token!r}.")
        shapes.append((m, n))
    if not shapes:
        raise ValueError("At least one benchmark shape is required.")
    return shapes


def parse_signal_ranks(value: str) -> list[int | None]:
    ranks: list[int | None] = []
    for token in value.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token in {"full", "none", "0"}:
            ranks.append(None)
            continue
        rank = int(token)
        if rank <= 0:
            ranks.append(None)
        else:
            ranks.append(rank)
    if not ranks:
        raise ValueError("At least one signal rank is required.")
    return ranks


def rank_spec_label(rank_spec: int | None) -> str:
    return "full" if rank_spec is None else str(rank_spec)


DEFAULT_SHAPES = "200x200,100x1000,1000x100"
SHAPES = parse_shapes(os.environ.get("PROX_TAYLOR_200_SHAPES", DEFAULT_SHAPES))
SEED = _env_int("PROX_TAYLOR_200_SEED", 0)
LAMBDA_TRUE = _env_float("PROX_TAYLOR_200_LAMBDA_TRUE", 0.05)
DEFAULT_SIGNAL_RANKS = os.environ.get(
    "PROX_TAYLOR_200_SIGNAL_RANK",
    "20",
)
SIGNAL_RANK_SPECS = parse_signal_ranks(
    os.environ.get("PROX_TAYLOR_200_SIGNAL_RANKS", DEFAULT_SIGNAL_RANKS)
)
SIGNAL_RMS = _env_float("PROX_TAYLOR_200_SIGNAL_RMS", 1.5)
SIGMA_NOISE = _env_float("PROX_TAYLOR_200_SIGMA_NOISE", 0.5)
TEST_FRAC = _env_float("PROX_TAYLOR_200_TEST_FRAC", 0.2)
TRAIN_MISSING_FRAC = _env_float("PROX_TAYLOR_200_TRAIN_MISSING_FRAC", 0.3)
PROX_ITER = _env_int("PROX_TAYLOR_200_PROX_ITER", 80)
CV_ITER = _env_int("PROX_TAYLOR_200_CV_ITER", PROX_ITER)
CV_FOLDS = _env_int("PROX_TAYLOR_200_CV_FOLDS", 3)
FAEM_RANK_OVERRIDE = os.environ.get("PROX_TAYLOR_200_FAEM_RANK")
FAEM_ITER = _env_int("PROX_TAYLOR_200_FAEM_ITER", 100)
TOL = _env_float("PROX_TAYLOR_200_TOL", 1e-5)
PROX_SOLVER = os.environ.get("PROX_TAYLOR_200_PROX_SOLVER", "monotone_fista")
PROX_OBJ_TOL = _env_float("PROX_TAYLOR_200_PROX_OBJ_TOL", 1e-6)
PROX_OBJ_PATIENCE = _env_int("PROX_TAYLOR_200_PROX_OBJ_PATIENCE", 5)
OUTPUT_PREFIX = os.environ.get("PROX_TAYLOR_200_OUTPUT", "results/prox_taylor_200")

DEFAULT_LAMBDAS = "0.5,1,2,5,10,20,50,100"
LAMBDA_GRID = [
    float(v.strip())
    for v in os.environ.get("PROX_TAYLOR_200_LAMBDAS", DEFAULT_LAMBDAS).split(",")
    if v.strip()
]
LAMBDA_PATH = sorted(LAMBDA_GRID, reverse=True)


def shape_label(m: int, n: int) -> str:
    return f"{m}x{n}"


def condition_label(m: int, n: int, rank_spec: int | None) -> str:
    return f"{shape_label(m, n)} r={rank_spec_label(rank_spec)}"


def rmse(mu, X_true) -> float:
    return float(jnp.sqrt(jnp.mean((jnp.asarray(mu) - X_true) ** 2)))


def masked_rmse(mu, X_true, mask) -> float:
    sq = (jnp.asarray(mu) - X_true) ** 2
    return float(jnp.sqrt(jnp.mean(sq[mask])))


def sample_signal(
    rng: np.random.Generator,
    m: int,
    n: int,
    rank_spec: int | None,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    if m >= n:
        X, full_singular_values = sample_nnd(rng, m, n, LAMBDA_TRUE)
    else:
        X_t, full_singular_values = sample_nnd(rng, n, m, LAMBDA_TRUE)
        X = X_t.T

    min_dim = min(m, n)
    sv_norm2 = float(np.sum(full_singular_values ** 2))
    if rank_spec is None or rank_spec >= min_dim:
        effective_rank = min_dim
        discarded_rms_fraction = 0.0
    else:
        effective_rank = rank_spec
        U, singular_values, Vt = np.linalg.svd(X, full_matrices=False)
        X = (U[:, :effective_rank] * singular_values[:effective_rank][None, :]) @ Vt[:effective_rank, :]
        discarded_rms_fraction = float(
            np.sqrt(np.sum(singular_values[effective_rank:] ** 2) / sv_norm2)
        )
        full_singular_values = singular_values

    raw_signal_rms = float(np.sqrt(np.mean(X ** 2)))
    scale = SIGNAL_RMS / raw_signal_rms
    X = X * scale
    full_singular_values = full_singular_values * scale
    signal_rms = float(np.sqrt(np.mean(X ** 2)))
    return X.astype(np.float32), {
        "rank": effective_rank,
        "rank_label": "full" if effective_rank == min_dim else str(effective_rank),
        "entry_rms": signal_rms,
        "raw_entry_rms": raw_signal_rms,
        "scale": scale,
        "noise_to_signal": SIGMA_NOISE / signal_rms,
        "nuclear_norm": float(np.sum(full_singular_values[:effective_rank])),
        "discarded_rms_fraction": discarded_rms_fraction,
    }


def taylor_scores(Y, S, lam: float, mu):
    t0 = time.time()
    res = taylor_gradient(
        Y,
        S,
        lam,
        max_iter=0,
        init_mu=mu,
        recover_sigma=False,
    )
    _block_until_ready(res.sigma_diag)
    loo = float(closed_form_loo(res.mu, res.sigma_diag, Y, S))
    return {
        "taylor_elbo": float(res.elbo),
        "taylor_loo": loo,
        "taylor_renyi": float(res.renyi_elbo),
        "score_time_s": time.time() - t0,
    }


def warm_prox_path(Y, S, lambdas: list[float], *, max_iter: int):
    init_X = None
    results = {}
    for lam in LAMBDA_PATH:
        if lam not in lambdas:
            continue
        t0 = time.time()
        prox = proximal_gradient(
            Y,
            S,
            lam,
            max_iter=max_iter,
            tol=TOL,
            init_X=init_X,
            solver=PROX_SOLVER,
            obj_tol=PROX_OBJ_TOL,
            obj_patience=PROX_OBJ_PATIENCE,
        )
        _block_until_ready(prox.X)
        fit_time = time.time() - t0
        results[lam] = (prox, fit_time)
        init_X = prox.X
    return results


def warm_cv_scores(Y, S, lambdas: list[float]) -> tuple[dict[float, float], dict[float, float], float]:
    Y = jnp.asarray(Y, dtype=jnp.float32)
    S = jnp.asarray(S, dtype=jnp.float32)
    obs_mask_np = np.array(jnp.isfinite(S) & jnp.isfinite(Y))
    obs_idx = np.argwhere(obs_mask_np)
    n_obs = len(obs_idx)
    if n_obs == 0:
        raise ValueError("No observed entries for CV.")
    if n_obs < CV_FOLDS:
        raise ValueError(f"Fewer observed entries ({n_obs}) than folds ({CV_FOLDS}).")

    rng = np.random.default_rng(42)
    perm = rng.permutation(n_obs)
    fold_ids = perm % CV_FOLDS

    cv_mse = {lam: 0.0 for lam in lambdas}
    cv_time_by_lambda = {lam: 0.0 for lam in lambdas}
    total_time = 0.0

    for fold in range(CV_FOLDS):
        val_pos = obs_idx[perm[fold_ids == fold]]
        S_fold = S.at[val_pos[:, 0], val_pos[:, 1]].set(jnp.inf)
        init_X = None
        for lam in LAMBDA_PATH:
            if lam not in cv_mse:
                continue
            t0 = time.time()
            prox = proximal_gradient(
                Y,
                S_fold,
                lam,
                max_iter=CV_ITER,
                tol=TOL,
                init_X=init_X,
                solver=PROX_SOLVER,
                obj_tol=PROX_OBJ_TOL,
                obj_patience=PROX_OBJ_PATIENCE,
            )
            _block_until_ready(prox.X)
            fit_time = time.time() - t0
            cv_time_by_lambda[lam] += fit_time
            total_time += fit_time
            init_X = prox.X

            i_val, j_val = val_pos[:, 0], val_pos[:, 1]
            pred = prox.X[i_val, j_val]
            true = Y[i_val, j_val]
            s_val = S[i_val, j_val]
            mse = float(jnp.mean(((pred - true) / s_val) ** 2))
            cv_mse[lam] += mse / CV_FOLDS

        print(f"  CV fold {fold + 1}/{CV_FOLDS} done")

    return ({lam: -mse for lam, mse in cv_mse.items()}, cv_time_by_lambda, total_time)


def faem_rank(signal: dict[str, float | int | str]) -> int:
    if FAEM_RANK_OVERRIDE:
        return int(FAEM_RANK_OVERRIDE)
    return int(signal["rank"])


def run_faem(Y, S, X_true, train_mask, test_mask, signal) -> dict[str, Any]:
    rank = faem_rank(signal)
    t0 = time.time()
    res = matlap_faem(Y, S, rank=rank, max_iter=FAEM_ITER, tol=TOL)
    _block_until_ready(res.mu)
    elapsed = time.time() - t0
    score = res.ll_trace[-1] if res.ll_trace else float("nan")
    return {
        "lambda": float(res.lambda_bar),
        "train_rmse": masked_rmse(res.mu, X_true, train_mask),
        "test_rmse": masked_rmse(res.mu, X_true, test_mask),
        "full_rmse": rmse(res.mu, X_true),
        "score": float(score),
        "time_s": elapsed,
        "n_iter": res.n_iter,
        "converged": res.converged,
        "convergence_reason": "ll" if res.converged else "",
        "n_restarts": "",
        "rank": rank,
    }


def fmt_num(x: float, digits: int = 4) -> str:
    if not math.isfinite(x):
        return "nan"
    if abs(x) >= 1000 or (abs(x) > 0 and abs(x) < 0.001):
        return f"{x:.{digits}e}"
    return f"{x:.{digits}g}"


def write_score_plot(rows: list[dict], path: Path) -> None:
    import pandas as pd
    from plotnine import (
        aes,
        element_text,
        facet_grid,
        geom_hline,
        geom_line,
        geom_point,
        geom_vline,
        ggplot,
        labs,
        scale_x_log10,
        theme,
        theme_bw,
    )

    panels = [
        ("Train RMSE", "train_rmse"),
        ("Test RMSE", "test_rmse"),
        ("Oracle RMSE", "full_rmse"),
        ("CV score", "cv_score"),
        ("Taylor ELBO", "taylor_elbo"),
        ("Taylor LOO", "taylor_loo"),
        ("Taylor Renyi", "taylor_renyi"),
    ]
    panel_order = [p[0] for p in panels]
    score_rows = []
    rows_by_shape: dict[str, list[dict]] = {}
    for row in rows:
        rows_by_shape.setdefault(row["condition"], []).append(row)
        for panel_name, key in panels:
            score_rows.append({
                "shape": row["condition"],
                "lambda": row["lambda"],
                "value": row[key],
                "panel": panel_name,
            })
    oracle_rows = []
    oracle_hlines = []
    for shape, shape_rows in rows_by_shape.items():
        oracle = min(shape_rows, key=lambda r: r["full_rmse"])
        oracle_hlines.append({
            "shape": shape,
            "panel": "Test RMSE",
            "value": oracle["test_rmse"],
        })
        oracle_hlines.append({
            "shape": shape,
            "panel": "Oracle RMSE",
            "value": oracle["full_rmse"],
        })
        for panel_name, _ in panels:
            oracle_rows.append({
                "shape": shape,
                "panel": panel_name,
                "lambda": oracle["lambda"],
            })

    scores = pd.DataFrame(score_rows)
    oracle_marks = pd.DataFrame(oracle_rows)
    oracle_test_rmse = pd.DataFrame(oracle_hlines)
    scores["panel"] = pd.Categorical(scores["panel"], categories=panel_order, ordered=True)
    oracle_marks["panel"] = pd.Categorical(oracle_marks["panel"], categories=panel_order, ordered=True)
    oracle_test_rmse["panel"] = pd.Categorical(
        oracle_test_rmse["panel"], categories=panel_order, ordered=True
    )

    plot = (
        ggplot(scores, aes("lambda", "value"))
        + geom_vline(
            aes(xintercept="lambda"),
            data=oracle_marks,
            linetype="dashed",
            color="#666666",
            alpha=0.6,
        )
        + geom_hline(
            aes(yintercept="value"),
            data=oracle_test_rmse,
            linetype="dashed",
            color="#b23a48",
            alpha=0.75,
        )
        + geom_line(size=0.9)
        + geom_point(size=2.0)
        + scale_x_log10(
            breaks=LAMBDA_GRID,
            labels=[fmt_num(lam, 3) for lam in LAMBDA_GRID],
        )
        + facet_grid("panel~shape", scales="free_y")
        + labs(
            title="Warm-started prox-Taylor lambda benchmark",
            x="lambda",
            y="score",
        )
        + theme_bw(base_size=10)
        + theme(
            figure_size=(min(24, max(13, 2.6 * len(rows_by_shape))), 13),
            strip_text=element_text(weight="bold"),
            axis_text_x=element_text(rotation=0),
        )
    )
    plot.save(str(path), verbose=False, limitsize=False)


def select_summary(
    rows: list[dict],
    prox_path_time_s: float,
    cv_total_time_s: float,
    taylor_total_time_s: float,
) -> dict[str, dict[str, Any]]:
    def selected(score_key: str, time_s: float) -> dict[str, Any]:
        row = max(rows, key=lambda r: r[score_key])
        return {
            "lambda": row["lambda"],
            "train_rmse": row["train_rmse"],
            "test_rmse": row["test_rmse"],
            "full_rmse": row["full_rmse"],
            "score": row[score_key],
            "time_s": time_s,
            "n_iter": row["prox_n_iter"],
            "converged": row["prox_converged"],
            "convergence_reason": row["prox_convergence_reason"],
            "n_restarts": row["prox_n_restarts"],
        }

    prox_taylor_time_s = prox_path_time_s + taylor_total_time_s
    return {
        "proximal_cv": selected("cv_score", cv_total_time_s + prox_path_time_s),
        "prox_taylor_elbo": selected("taylor_elbo", prox_taylor_time_s),
        "prox_taylor_loo": selected("taylor_loo", prox_taylor_time_s),
        "prox_taylor_renyi": selected("taylor_renyi", prox_taylor_time_s),
    }


def write_method_summary_csv(results: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "condition",
        "shape",
        "m",
        "n",
        "seed",
        "method",
        "method_rank",
        "lambda",
        "train_rmse",
        "test_rmse",
        "full_rmse",
        "test_improvement_vs_noisy",
        "score",
        "time_s",
        "n_iter",
        "converged",
        "convergence_reason",
        "n_restarts",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            noisy = result["noisy"]
            for method, s in result["summary"].items():
                writer.writerow({
                    "shape": result["shape"],
                    "condition": result["condition"],
                    "m": result["m"],
                    "n": result["n"],
                    "seed": result["seed"],
                    "method": method,
                    "method_rank": s.get("rank", ""),
                    "lambda": s["lambda"],
                    "train_rmse": s["train_rmse"],
                    "test_rmse": s["test_rmse"],
                    "full_rmse": s["full_rmse"],
                    "test_improvement_vs_noisy": noisy["test_rmse"] - s["test_rmse"],
                    "score": s["score"],
                    "time_s": s["time_s"],
                    "n_iter": s.get("n_iter", ""),
                    "converged": s.get("converged", ""),
                    "convergence_reason": s.get("convergence_reason", ""),
                    "n_restarts": s.get("n_restarts", ""),
                })


def write_summary(
    results: list[dict[str, Any]],
    csv_path: Path,
    methods_csv_path: Path,
    svg_path: Path,
    md_path: Path,
) -> None:
    lines = [
        "# Warm-started prox-Taylor benchmark",
        "",
        f"- conditions: {', '.join(r['condition'] for r in results)}",
        f"- seed: {SEED}",
        f"- lambda_true: {LAMBDA_TRUE:g}",
        f"- sigma_noise: {SIGMA_NOISE:g}",
        f"- target signal entry RMS: {SIGNAL_RMS:g}",
        f"- signal rank settings: {', '.join(rank_spec_label(r) for r in SIGNAL_RANK_SPECS)}",
        f"- held-out test fraction: {TEST_FRAC:g}",
        f"- additional train missing fraction: {TRAIN_MISSING_FRAC:g}",
        f"- proximal iterations: {PROX_ITER}",
        f"- proximal solver: {PROX_SOLVER}",
        f"- proximal objective tolerance: {PROX_OBJ_TOL:g}",
        f"- proximal objective patience: {PROX_OBJ_PATIENCE}",
        f"- CV folds: {CV_FOLDS}",
        f"- CV iterations per fit: {CV_ITER}",
        f"- FA EM rank: {'override ' + FAEM_RANK_OVERRIDE if FAEM_RANK_OVERRIDE else 'matches effective signal rank'}",
        f"- FA EM iterations: {FAEM_ITER}",
        f"- lambda grid: {', '.join(fmt_num(lam, 3) for lam in LAMBDA_GRID)}",
        f"- lambda path: high to low ({', '.join(fmt_num(lam, 3) for lam in LAMBDA_PATH)})",
        "",
    ]

    for result in results:
        rows = result["rows"]
        summary = result["summary"]
        noisy = result["noisy"]
        signal = result["signal"]
        oracle = min(rows, key=lambda r: r["test_rmse"])
        lines.extend([
            f"## Condition {result['condition']}",
            "",
            f"- signal rank: {signal['rank_label']}",
            f"- condition seed: {result['seed']}",
            f"- signal entry RMS: {signal['entry_rms']:.6f}",
            f"- raw signal entry RMS before scaling: {signal['raw_entry_rms']:.6f}",
            f"- signal scale factor: {signal['scale']:.6g}",
            f"- noise/signal RMS: {signal['noise_to_signal']:.6f}",
            f"- discarded full-NND RMS fraction: {signal['discarded_rms_fraction']:.6f}",
            f"- final test entries: {result['n_test']}",
            f"- observed train entries: {result['n_train_observed']}",
            f"- additional missing train entries: {result['n_train_missing']}",
            f"- noisy Y train RMSE: {noisy['train_rmse']:.6f}",
            f"- noisy Y test RMSE: {noisy['test_rmse']:.6f}",
            f"- noisy Y full RMSE: {noisy['full_rmse']:.6f}",
            f"- warm proximal path time: {result['prox_path_time_s']:.3f}s",
            f"- warm CV path time: {result['cv_total_time_s']:.3f}s",
            f"- Taylor scoring time: {result['taylor_total_time_s']:.3f}s",
            f"- oracle test-RMSE lambda: {oracle['lambda']:.6g} "
            f"(test RMSE {oracle['test_rmse']:.6f})",
            "",
            "| method | lambda | train RMSE | test RMSE | full RMSE | test improvement vs noisy | selection score | method time (s) | n iter | converged | reason | restarts |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|",
        ])
        for name in ["proximal_cv", "prox_taylor_elbo", "prox_taylor_loo", "prox_taylor_renyi", "faem"]:
            s = summary[name]
            improvement = noisy["test_rmse"] - s["test_rmse"]
            lines.append(
                f"| {name} | {s['lambda']:.6g} | {s['train_rmse']:.6f} | "
                f"{s['test_rmse']:.6f} | {s['full_rmse']:.6f} | {improvement:.6f} | "
                f"{s['score']:.6g} | {s['time_s']:.3f} | "
                f"{s.get('n_iter', '')} | {s.get('converged', '')} | "
                f"{s.get('convergence_reason', '')} | {s.get('n_restarts', '')} |"
            )
        lines.append("")

    lines.extend([
        "## Outputs",
        "",
        f"- lambda-curve CSV: `{csv_path}`",
        f"- method-summary CSV: `{methods_csv_path}`",
        f"- plot: `{svg_path}`",
    ])
    md_path.write_text("\n".join(lines) + "\n")


def run_condition(
    m: int,
    n: int,
    shape_index: int,
    rank_spec: int | None,
) -> dict[str, Any]:
    label = shape_label(m, n)
    condition = condition_label(m, n, rank_spec)
    condition_seed = SEED + 1000 * shape_index
    print(f"\n=== {condition} ===")
    print(
        f"Simulating {label}, signal_rank={rank_spec_label(rank_spec)}, "
        f"lambda_true={LAMBDA_TRUE:g}, sigma={SIGMA_NOISE:g}, seed={condition_seed}"
    )

    rng = np.random.default_rng(condition_seed)
    X_np, signal = sample_signal(rng, m, n, rank_spec)
    Y_np = X_np + SIGMA_NOISE * rng.standard_normal((m, n)).astype(np.float32)
    X_true = _to_gpu(X_np)
    Y = _to_gpu(Y_np)
    split_rng = np.random.default_rng(condition_seed + 10_000)
    test_mask_np = split_rng.random((m, n)) < TEST_FRAC
    train_pool_mask_np = ~test_mask_np
    train_missing_mask_np = (split_rng.random((m, n)) < TRAIN_MISSING_FRAC) & train_pool_mask_np
    train_mask_np = train_pool_mask_np & ~train_missing_mask_np
    if not np.any(train_mask_np):
        raise ValueError(f"No observed train entries remain for {label}.")
    if not np.any(test_mask_np):
        raise ValueError(f"No final test entries were sampled for {label}.")
    S_train_np = SIGMA_NOISE * np.ones((m, n), dtype=np.float32)
    S_train_np[test_mask_np] = np.inf
    S_train_np[train_missing_mask_np] = np.inf
    S_train = _to_gpu(S_train_np)
    train_mask = _to_gpu_bool(train_mask_np)
    test_mask = _to_gpu_bool(test_mask_np)
    _block_until_ready(Y)

    noisy = {
        "train_rmse": masked_rmse(Y, X_true, train_mask),
        "test_rmse": masked_rmse(Y, X_true, test_mask),
        "full_rmse": rmse(Y, X_true),
    }

    print(f"warm proximal path: {', '.join(fmt_num(lam, 3) for lam in LAMBDA_PATH)}")
    path_results = warm_prox_path(Y, S_train, LAMBDA_GRID, max_iter=PROX_ITER)
    prox_path_time_s = sum(fit_time for _, fit_time in path_results.values())

    rows_by_lambda = {}
    taylor_total_time_s = 0.0
    for lam in LAMBDA_GRID:
        prox, fit_time = path_results[lam]
        row = {
            "shape": label,
            "condition": condition,
            "m": m,
            "n": n,
            "seed": condition_seed,
            "signal_rank": signal["rank"],
            "lambda": lam,
            "train_rmse": masked_rmse(prox.X, X_true, train_mask),
            "test_rmse": masked_rmse(prox.X, X_true, test_mask),
            "full_rmse": rmse(prox.X, X_true),
            "prox_fit_time_s": fit_time,
            "prox_n_iter": prox.n_iter,
            "prox_converged": prox.converged,
            "prox_convergence_reason": prox.convergence_reason,
            "prox_n_restarts": prox.n_restarts,
        }
        row.update(taylor_scores(Y, S_train, lam, prox.X))
        taylor_total_time_s += row["score_time_s"]
        rows_by_lambda[lam] = row

    print("warm CV path")
    cv_scores, cv_time_by_lambda, cv_total_time_s = warm_cv_scores(Y, S_train, LAMBDA_GRID)
    for lam, row in rows_by_lambda.items():
        row["cv_score"] = cv_scores[lam]
        row["cv_time_s"] = cv_time_by_lambda[lam]

    rows = [rows_by_lambda[lam] for lam in LAMBDA_GRID]
    summary = select_summary(rows, prox_path_time_s, cv_total_time_s, taylor_total_time_s)
    print("FA EM comparator")
    summary["faem"] = run_faem(Y, S_train, X_true, train_mask, test_mask, signal)

    print(
        f"noisy_Y RMSE: train={noisy['train_rmse']:.6f}  "
        f"test={noisy['test_rmse']:.6f}  full={noisy['full_rmse']:.6f}"
    )
    print(
        f"signal: rank={signal['rank_label']}  entry_RMS={signal['entry_rms']:.6f}  "
        f"noise/signal={signal['noise_to_signal']:.6f}"
    )
    for name, s in summary.items():
        print(
            f"{name:18s} lambda={s['lambda']:7g}  "
            f"train_RMSE={s['train_rmse']:.6f}  test_RMSE={s['test_rmse']:.6f}  "
            f"score={s['score']:.6g}  method_time={s['time_s']:.3f}s"
        )

    return {
        "shape": label,
        "condition": condition,
        "m": m,
        "n": n,
        "seed": condition_seed,
        "rows": rows,
        "summary": summary,
        "noisy": noisy,
        "signal": signal,
        "prox_path_time_s": prox_path_time_s,
        "cv_total_time_s": cv_total_time_s,
        "taylor_total_time_s": taylor_total_time_s,
        "n_test": int(test_mask_np.sum()),
        "n_train_observed": int(train_mask_np.sum()),
        "n_train_missing": int(train_missing_mask_np.sum()),
    }


def main() -> None:
    output_prefix = Path(OUTPUT_PREFIX)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    methods_csv_path = output_prefix.with_name(output_prefix.name + "_methods.csv")
    svg_path = output_prefix.with_name(output_prefix.name + "_scores.svg")
    md_path = output_prefix.with_suffix(".md")

    print(f"Using GPU: {GPU_DEVICE}")
    results = [
        run_condition(m, n, shape_idx, rank_spec)
        for shape_idx, (m, n) in enumerate(SHAPES)
        for rank_spec in SIGNAL_RANK_SPECS
    ]
    all_rows = [row for result in results for row in result["rows"]]

    fieldnames = [
        "condition",
        "shape",
        "m",
        "n",
        "seed",
        "signal_rank",
        "lambda",
        "train_rmse",
        "test_rmse",
        "full_rmse",
        "prox_fit_time_s",
        "score_time_s",
        "cv_time_s",
        "cv_score",
        "taylor_elbo",
        "taylor_loo",
        "taylor_renyi",
        "prox_n_iter",
        "prox_converged",
        "prox_convergence_reason",
        "prox_n_restarts",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    write_method_summary_csv(results, methods_csv_path)
    write_score_plot(all_rows, svg_path)
    write_summary(results, csv_path, methods_csv_path, svg_path, md_path)

    print(f"\nwrote {csv_path}")
    print(f"wrote {methods_csv_path}")
    print(f"wrote {md_path}")
    print(f"wrote {svg_path}")


if __name__ == "__main__":
    main()
