#!/usr/bin/env python
"""Streaming NMF/semi-NMF on proximal low-rank denoised PSI targets.

The proximal fits are stored as ``U, s, Vt``.  This script generates clipped
PSI-scale denoised rows in batches, fits nonnegative cell program usages, then
transforms all cells in batches.  It never materializes the full
cell-by-junction denoised matrix.
"""

from __future__ import annotations

import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.decomposition import MiniBatchNMF


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analyses" / "splicing_proximal" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


DEFAULT_FITS = {
    "psi_homo": OUT / "subset10k_psi_rsvd_iter5_lambda300_homo_lambda300_factors.npz",
    "psi_hetero_q25": OUT / "subset10k_psi_hetero_q25_floor_iter5_lambda100000_factors.npz",
}


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def env_list(name: str, default: str) -> list[str]:
    return [x.strip() for x in os.environ.get(name, default).split(",") if x.strip()]


def load_lowrank(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    factors = np.load(path)
    U = factors["U"].astype(np.float32)
    s = factors["s"].astype(np.float32)
    Vt = factors["Vt"].astype(np.float32)
    return U, s, Vt


def lowrank_batch(
    U: np.ndarray,
    s: np.ndarray,
    Vt: np.ndarray,
    rows: np.ndarray,
    *,
    target_transform: str,
) -> np.ndarray:
    X = (U[rows].astype(np.float32) * s[None, :]) @ Vt
    if target_transform == "clip":
        np.clip(X, 0.0, 1.0, out=X)
    elif target_transform == "sigmoid":
        X = expit(X).astype(np.float32)
    else:
        raise ValueError("target_transform must be 'clip' or 'sigmoid'.")
    return X.astype(np.float32, copy=False)


def iterate_batches(indices: np.ndarray, batch_size: int):
    for start in range(0, indices.size, batch_size):
        yield indices[start:start + batch_size]


def normalize_usage_scale(W: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = np.maximum(W.mean(axis=0), 1e-8).astype(np.float32)
    W /= scale[None, :]
    H = H * scale[:, None]
    return W, H.astype(np.float32)


def solve_nonnegative_usage(
    X: np.ndarray,
    H: np.ndarray,
    *,
    ridge: float,
    n_iter: int,
) -> np.ndarray:
    """Solve min_W ||X - W H||^2 with W >= 0 by projected gradient."""
    H64 = H.astype(np.float64, copy=False)
    X64 = X.astype(np.float64, copy=False)
    gram = H64 @ H64.T
    rhs = X64 @ H64.T
    gram_reg = gram + ridge * np.eye(gram.shape[0])
    try:
        W = rhs @ np.linalg.inv(gram_reg)
    except np.linalg.LinAlgError:
        W = rhs @ np.linalg.pinv(gram_reg)
    W = np.maximum(W, 0.0)
    lipschitz = float(np.linalg.eigvalsh(gram).max()) + ridge
    step = 1.0 / max(lipschitz, 1e-8)
    for _ in range(n_iter):
        grad = W @ gram - rhs + ridge * W
        W -= step * grad
        np.maximum(W, 0.0, out=W)
    return W.astype(np.float32)


def fit_one_nmf(
    name: str,
    factor_path: Path,
    *,
    n_components: int,
    batch_size: int,
    epochs: int,
    transform_batch_size: int,
    target_transform: str,
    alpha: float,
    l1_ratio: float,
    random_state: int,
    max_transform_iter: int,
    eval_rows: int,
) -> dict[str, float | int | str]:
    U, s, Vt = load_lowrank(factor_path)
    n_cells = U.shape[0]
    n_junctions = Vt.shape[1]
    rng = np.random.default_rng(random_state)

    model = MiniBatchNMF(
        n_components=n_components,
        init="random",
        batch_size=batch_size,
        beta_loss="frobenius",
        alpha_W=alpha,
        alpha_H=alpha,
        l1_ratio=l1_ratio,
        random_state=random_state,
        max_iter=1,
        transform_max_iter=max_transform_iter,
    )

    t0 = time.perf_counter()
    for epoch in range(epochs):
        order = rng.permutation(n_cells)
        for batch_no, rows in enumerate(iterate_batches(order, batch_size), start=1):
            X = lowrank_batch(U, s, Vt, rows, target_transform=target_transform)
            model.partial_fit(X)
            if batch_no % 25 == 0:
                print(
                    {
                        "fit": name,
                        "epoch": epoch + 1,
                        "batch": batch_no,
                        "rows_seen": int(batch_no * batch_size),
                    },
                    flush=True,
                )
    train_seconds = time.perf_counter() - t0

    H = model.components_.astype(np.float32)
    W_path = OUT / f"{name}_nmf_W.float32.mmap"
    W = np.memmap(W_path, mode="w+", dtype=np.float32, shape=(n_cells, n_components))
    t_transform = time.perf_counter()
    all_rows = np.arange(n_cells, dtype=np.int64)
    for batch_no, rows in enumerate(iterate_batches(all_rows, transform_batch_size), start=1):
        X = lowrank_batch(U, s, Vt, rows, target_transform=target_transform)
        W[rows] = model.transform(X).astype(np.float32)
        if batch_no % 20 == 0:
            W.flush()
            print({"fit": name, "transform_batch": batch_no}, flush=True)
    W.flush()
    transform_seconds = time.perf_counter() - t_transform

    W_arr = np.asarray(W)
    W_arr, H = normalize_usage_scale(W_arr, H)
    W.flush()

    eval_n = min(eval_rows, n_cells)
    eval_idx = np.sort(rng.choice(n_cells, size=eval_n, replace=False))
    sse = 0.0
    denom = 0
    for rows in iterate_batches(eval_idx, transform_batch_size):
        X = lowrank_batch(U, s, Vt, rows, target_transform=target_transform)
        pred = np.asarray(W[rows]) @ H
        sse += float(np.sum((X - pred) ** 2))
        denom += int(X.size)
    rmse = float(np.sqrt(sse / max(denom, 1)))

    np.savez_compressed(
        OUT / f"{name}_nmf_rank{n_components}.npz",
        W=np.asarray(W, dtype=np.float32),
        H=H.astype(np.float32),
        factor_path=str(factor_path),
        target_transform=target_transform,
        n_components=np.array(n_components),
    )
    pd.DataFrame(H).to_csv(OUT / f"{name}_nmf_H_rank{n_components}.csv", index=False)

    return {
        "fit": name,
        "method": "nmf",
        "factor_path": str(factor_path),
        "target_transform": target_transform,
        "n_cells": n_cells,
        "n_junctions": n_junctions,
        "prox_rank": int(s.size),
        "nmf_rank": n_components,
        "batch_size": batch_size,
        "epochs": epochs,
        "alpha": alpha,
        "l1_ratio": l1_ratio,
        "train_seconds": train_seconds,
        "transform_seconds": transform_seconds,
        "eval_rows": eval_n,
        "eval_rmse": rmse,
        "W_path": str(W_path),
        "npz_path": str(OUT / f"{name}_nmf_rank{n_components}.npz"),
    }


def initialize_signed_loadings(
    name: str,
    *,
    n_components: int,
    n_junctions: int,
    random_state: int,
) -> np.ndarray:
    nmf_path = OUT / f"{name}_nmf_rank{n_components}.npz"
    if nmf_path.exists():
        H = np.load(nmf_path)["H"].astype(np.float32)
        print({"fit": name, "semi_nmf_init": str(nmf_path)}, flush=True)
        return H
    rng = np.random.default_rng(random_state)
    return rng.normal(0.0, 0.05, size=(n_components, n_junctions)).astype(np.float32)


def fit_one_semi_nmf(
    name: str,
    factor_path: Path,
    *,
    n_components: int,
    batch_size: int,
    epochs: int,
    transform_batch_size: int,
    target_transform: str,
    random_state: int,
    semi_ridge: float,
    semi_nnls_iter: int,
    eval_rows: int,
) -> dict[str, float | int | str]:
    """Fit W H with W >= 0 and signed H by streaming alternating least squares."""
    U, s, Vt = load_lowrank(factor_path)
    n_cells = U.shape[0]
    n_junctions = Vt.shape[1]
    rng = np.random.default_rng(random_state)
    H = initialize_signed_loadings(
        name,
        n_components=n_components,
        n_junctions=n_junctions,
        random_state=random_state,
    )

    t0 = time.perf_counter()
    for epoch in range(epochs):
        order = rng.permutation(n_cells)
        lhs = semi_ridge * np.eye(n_components, dtype=np.float64)
        rhs = np.zeros((n_components, n_junctions), dtype=np.float64)
        for batch_no, rows in enumerate(iterate_batches(order, batch_size), start=1):
            X = lowrank_batch(U, s, Vt, rows, target_transform=target_transform)
            W_batch = solve_nonnegative_usage(
                X,
                H,
                ridge=semi_ridge,
                n_iter=semi_nnls_iter,
            )
            W64 = W_batch.astype(np.float64, copy=False)
            lhs += W64.T @ W64
            rhs += W64.T @ X.astype(np.float64, copy=False)
            if batch_no % 25 == 0:
                print(
                    {
                        "fit": name,
                        "method": "semi_nmf",
                        "epoch": epoch + 1,
                        "batch": batch_no,
                        "rows_seen": int(batch_no * batch_size),
                    },
                    flush=True,
                )
        H = np.linalg.solve(lhs, rhs).astype(np.float32)
        row_norm = np.maximum(np.linalg.norm(H, axis=1), 1e-8).astype(np.float32)
        H = H / row_norm[:, None]
        print(
            {
                "fit": name,
                "method": "semi_nmf",
                "epoch": epoch + 1,
                "signed_loading_min": float(H.min()),
                "signed_loading_max": float(H.max()),
            },
            flush=True,
        )
    train_seconds = time.perf_counter() - t0

    W_path = OUT / f"{name}_semi_nmf_W.float32.mmap"
    W = np.memmap(W_path, mode="w+", dtype=np.float32, shape=(n_cells, n_components))
    t_transform = time.perf_counter()
    all_rows = np.arange(n_cells, dtype=np.int64)
    for batch_no, rows in enumerate(iterate_batches(all_rows, transform_batch_size), start=1):
        X = lowrank_batch(U, s, Vt, rows, target_transform=target_transform)
        W[rows] = solve_nonnegative_usage(
            X,
            H,
            ridge=semi_ridge,
            n_iter=semi_nnls_iter,
        )
        if batch_no % 20 == 0:
            W.flush()
            print({"fit": name, "method": "semi_nmf", "transform_batch": batch_no}, flush=True)
    W.flush()
    transform_seconds = time.perf_counter() - t_transform

    W_arr = np.asarray(W)
    W_arr, H = normalize_usage_scale(W_arr, H)
    W.flush()

    eval_n = min(eval_rows, n_cells)
    eval_idx = np.sort(rng.choice(n_cells, size=eval_n, replace=False))
    sse = 0.0
    denom = 0
    for rows in iterate_batches(eval_idx, transform_batch_size):
        X = lowrank_batch(U, s, Vt, rows, target_transform=target_transform)
        pred = np.asarray(W[rows]) @ H
        sse += float(np.sum((X - pred) ** 2))
        denom += int(X.size)
    rmse = float(np.sqrt(sse / max(denom, 1)))

    np.savez_compressed(
        OUT / f"{name}_semi_nmf_rank{n_components}.npz",
        W=np.asarray(W, dtype=np.float32),
        H=H.astype(np.float32),
        factor_path=str(factor_path),
        target_transform=target_transform,
        n_components=np.array(n_components),
    )
    pd.DataFrame(H).to_csv(OUT / f"{name}_semi_nmf_H_rank{n_components}.csv", index=False)

    return {
        "fit": name,
        "method": "semi_nmf",
        "factor_path": str(factor_path),
        "target_transform": target_transform,
        "n_cells": n_cells,
        "n_junctions": n_junctions,
        "prox_rank": int(s.size),
        "nmf_rank": n_components,
        "batch_size": batch_size,
        "epochs": epochs,
        "semi_ridge": semi_ridge,
        "semi_nnls_iter": semi_nnls_iter,
        "train_seconds": train_seconds,
        "transform_seconds": transform_seconds,
        "eval_rows": eval_n,
        "eval_rmse": rmse,
        "signed_loading_min": float(H.min()),
        "signed_loading_max": float(H.max()),
        "W_path": str(W_path),
        "npz_path": str(OUT / f"{name}_semi_nmf_rank{n_components}.npz"),
    }


def main() -> None:
    fit_names = env_list("SPLICING_NMF_FITS", "psi_homo,psi_hetero_q25")
    method = os.environ.get("SPLICING_NMF_METHOD", "nmf").lower()
    n_components = env_int("SPLICING_NMF_RANK", 20)
    batch_size = env_int("SPLICING_NMF_BATCH_SIZE", 2048)
    transform_batch_size = env_int("SPLICING_NMF_TRANSFORM_BATCH_SIZE", batch_size)
    epochs = env_int("SPLICING_NMF_EPOCHS", 3)
    target_transform = os.environ.get("SPLICING_NMF_TARGET", "clip").lower()
    alpha = env_float("SPLICING_NMF_ALPHA", 0.0)
    l1_ratio = env_float("SPLICING_NMF_L1_RATIO", 0.0)
    random_state = env_int("SPLICING_NMF_RANDOM_STATE", 20260614)
    max_transform_iter = env_int("SPLICING_NMF_TRANSFORM_MAX_ITER", 100)
    eval_rows = env_int("SPLICING_NMF_EVAL_ROWS", 5000)
    semi_ridge = env_float("SPLICING_SEMI_NMF_RIDGE", 1e-4)
    semi_nnls_iter = env_int("SPLICING_SEMI_NMF_NNLS_ITER", 30)

    rows = []
    if method not in {"nmf", "semi_nmf"}:
        raise ValueError("SPLICING_NMF_METHOD must be 'nmf' or 'semi_nmf'.")
    for fit_name in fit_names:
        fit_name = fit_name.strip()
        path = Path(os.environ.get(f"SPLICING_NMF_FACTOR_{fit_name.upper()}", DEFAULT_FITS[fit_name]))
        print({"fit": fit_name, "factor_path": str(path)}, flush=True)
        if method == "nmf":
            rows.append(
                fit_one_nmf(
                    fit_name,
                    path,
                    n_components=n_components,
                    batch_size=batch_size,
                    epochs=epochs,
                    transform_batch_size=transform_batch_size,
                    target_transform=target_transform,
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    random_state=random_state,
                    max_transform_iter=max_transform_iter,
                    eval_rows=eval_rows,
                )
            )
        else:
            rows.append(
                fit_one_semi_nmf(
                    fit_name,
                    path,
                    n_components=n_components,
                    batch_size=batch_size,
                    epochs=epochs,
                    transform_batch_size=transform_batch_size,
                    target_transform=target_transform,
                    random_state=random_state,
                    semi_ridge=semi_ridge,
                    semi_nnls_iter=semi_nnls_iter,
                    eval_rows=eval_rows,
                )
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / f"lowrank_{method}_rank{n_components}_summary.csv", index=False)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
