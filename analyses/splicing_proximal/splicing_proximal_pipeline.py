"""Helpers for the single-cell splicing proximal-model notebook."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
import urllib.request
import warnings

import anndata as ad
import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import betaln, expit, gammaln, logit
import seaborn as sns
import umap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from matlap import (
    proximal_gradient,
    proximal_noise_eb,
    taylor_proximal_homoskedastic_unknown_noise,
)


ATLAS_URL = (
    "https://zenodo.org/records/17981150/files/"
    "splice_adata_for_figures_mouse_foundation.h5ad?download=1"
)
ATLAS_FILENAME = "splice_adata_for_figures_mouse_foundation.h5ad"
COUNT_LAYER = "cell_by_junction_matrix"
CLUSTER_LAYER = "cell_by_cluster_matrix"


@dataclass
class PsiEstimate:
    """Dense PSI/logit-PSI matrices for a selected cell-junction block."""

    Y_logit: np.ndarray
    S_logit: np.ndarray
    psi_mean: np.ndarray
    psi_se: np.ndarray
    observed: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    cells: pd.DataFrame
    junctions: pd.DataFrame


def download_file(url: str, path: Path, *, overwrite: bool = False) -> Path:
    """Download ``url`` to ``path`` unless it is already present."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return path
    with urllib.request.urlopen(url) as src, path.open("wb") as out:
        out.write(src.read())
    return path


def download_default_atlas(data_dir: Path) -> Path:
    """Download the trained atlas object provided in the prompt."""
    return download_file(ATLAS_URL, Path(data_dir) / ATLAS_FILENAME)


def inspect_h5ad(path: Path) -> dict:
    """Return a compact AnnData inventory without loading the full object."""
    with h5py.File(path, "r") as f:
        shape = None
        if "X" in f and "shape" in f["X"].attrs:
            shape = tuple(int(x) for x in f["X"].attrs["shape"])
        info = {
            "path": str(path),
            "shape": shape,
            "obs_columns": _h5_dataframe_columns(f["obs"]) if "obs" in f else [],
            "var_columns": _h5_dataframe_columns(f["var"]) if "var" in f else [],
            "layers": list(f["layers"].keys()) if "layers" in f else [],
            "obsm": list(f["obsm"].keys()) if "obsm" in f else [],
            "varm": list(f["varm"].keys()) if "varm" in f else [],
            "sparse": _h5_sparse_inventory(f),
        }
    return info


def _h5_dataframe_columns(group: h5py.Group) -> list[str]:
    order = group.attrs.get("column-order")
    if order is not None:
        return [x.decode() if isinstance(x, bytes) else str(x) for x in order]
    return list(group.keys())


def _h5_sparse_inventory(f: h5py.File) -> dict[str, dict[str, object]]:
    out = {}
    for name in ["X"]:
        if name in f:
            obj = f[name]
            out[name] = {
                "encoding": obj.attrs.get("encoding-type", ""),
                "shape": tuple(int(x) for x in obj.attrs.get("shape", ())),
            }
    if "layers" in f:
        for name, obj in f["layers"].items():
            out[f"layers/{name}"] = {
                "encoding": obj.attrs.get("encoding-type", ""),
                "shape": tuple(int(x) for x in obj.attrs.get("shape", ())),
            }
    return out


def ensure_sparse_h5ad(source: Path, target: Path | None = None) -> Path:
    """Return a sparse h5ad path, writing a sparse copy only if needed.

    The full mouse splicing object already stores `X` and the count layers as
    HDF5-backed CSR matrices. This helper is mostly a guardrail for future input
    files: it avoids densifying large matrices in memory, and only uses AnnData's
    writer on objects that are actually dense.
    """
    source = Path(source)
    with h5py.File(source, "r") as f:
        dense = []
        for key in ["X"]:
            if key in f and not isinstance(f[key], h5py.Group):
                dense.append(key)
        for key in [f"layers/{COUNT_LAYER}", f"layers/{CLUSTER_LAYER}"]:
            if key in f and not isinstance(f[key], h5py.Group):
                dense.append(key)
    if not dense:
        return source
    if target is None:
        target = source.with_name(source.stem + "_sparse.h5ad")
    a = ad.read_h5ad(source)
    if a.X is not None and not sparse.issparse(a.X):
        a.X = sparse.csr_matrix(a.X)
    for layer in list(a.layers.keys()):
        if not sparse.issparse(a.layers[layer]):
            a.layers[layer] = sparse.csr_matrix(a.layers[layer])
    a.write_h5ad(target)
    return Path(target)


def require_count_layers(adata: ad.AnnData) -> None:
    """Validate that an AnnData object has the count layers needed here."""
    missing = [name for name in (COUNT_LAYER, CLUSTER_LAYER) if name not in adata.layers]
    if missing:
        raise RuntimeError(
            "The supplied AnnData does not contain the count layers required "
            "for beta-binomial PSI estimation. Missing layers: "
            f"{missing}. Expected LeafletFA/ATSEmapper-style layers "
            f"{COUNT_LAYER!r} and {CLUSTER_LAYER!r}. The small Zenodo atlas "
            "linked in the prompt is a fitted-results object, not the raw "
            "count input."
        )


def _as_csr(x) -> sparse.csr_matrix:
    if sparse.issparse(x):
        return x.tocsr()
    return sparse.csr_matrix(np.asarray(x))


def _decode_h5_values(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind in {"S", "O"}:
        return np.asarray([
            x.decode() if isinstance(x, (bytes, np.bytes_)) else x
            for x in values
        ])
    return values


def _read_h5_column(group: h5py.Group, col: str, idx: np.ndarray | None = None) -> np.ndarray:
    obj = group[col]
    if isinstance(obj, h5py.Group) and obj.attrs.get("encoding-type") == "categorical":
        codes = obj["codes"][...] if idx is None else obj["codes"][idx]
        cats = _decode_h5_values(obj["categories"][...])
        vals = np.asarray(cats)[np.asarray(codes)]
        vals[np.asarray(codes) < 0] = None
        return vals
    vals = obj[...] if idx is None else obj[idx]
    return _decode_h5_values(np.asarray(vals))


def _read_h5_dataframe(
    f: h5py.File,
    key: str,
    idx: np.ndarray,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    group = f[key]
    cols = columns or _h5_dataframe_columns(group)
    idx = np.asarray(idx, dtype=np.int64)
    data = {col: _read_h5_column(group, col, idx) for col in cols if col in group}
    index_key = group.attrs.get("_index")
    index_name = index_key.decode() if isinstance(index_key, bytes) else str(index_key)
    if index_name in group:
        index = _read_h5_column(group, index_name, idx)
    elif "_index" in group:
        index = _read_h5_column(group, "_index", idx)
    else:
        index = idx
    return pd.DataFrame(data, index=index)


def _read_h5_dataframe_full(
    f: h5py.File,
    key: str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    group = f[key]
    n = next(iter(group.values()))
    if isinstance(n, h5py.Group):
        n_rows = n["codes"].shape[0]
    else:
        n_rows = n.shape[0]
    return _read_h5_dataframe(f, key, np.arange(n_rows), columns)


def _csr_block_from_h5(group: h5py.Group, rows: np.ndarray, cols: np.ndarray) -> sparse.csr_matrix:
    """Read a row/column block from an HDF5-backed CSR matrix."""
    shape = tuple(int(x) for x in group.attrs["shape"])
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    col_map = np.full(shape[1], -1, dtype=np.int32)
    col_map[cols] = np.arange(cols.size, dtype=np.int32)

    indptr_in = group["indptr"]
    indices_in = group["indices"]
    data_in = group["data"]
    out_indptr = [0]
    out_indices = []
    out_data = []
    for r in rows:
        start = int(indptr_in[r])
        end = int(indptr_in[r + 1])
        row_cols = indices_in[start:end]
        mapped = col_map[row_cols]
        keep = mapped >= 0
        if np.any(keep):
            out_indices.extend(mapped[keep].tolist())
            out_data.extend(data_in[start:end][keep].tolist())
        out_indptr.append(len(out_indices))
    return sparse.csr_matrix(
        (
            np.asarray(out_data, dtype=data_in.dtype),
            np.asarray(out_indices, dtype=np.int32),
            np.asarray(out_indptr, dtype=np.int32),
        ),
        shape=(rows.size, cols.size),
    )


def _select_junction_indices(var: pd.DataFrame, max_junctions: int) -> np.ndarray:
    if "event_id" not in var:
        raise RuntimeError("Expected `var['event_id']` to identify LeafCutter/ATSE clusters.")
    j_score = (
        var["n_cells_detected"].to_numpy()
        if "n_cells_detected" in var
        else var.get("CountJuncs", pd.Series(np.ones(len(var)))).to_numpy()
    )
    event_sizes = var["event_id"].value_counts()
    order = np.argsort(-np.asarray(j_score, dtype=float))
    chosen = []
    event_counts: dict[str, int] = {}
    for j in order:
        event = str(var["event_id"].iloc[j])
        n_in_event = int(event_sizes.loc[event])
        if n_in_event < 2:
            continue
        chosen.append(int(j))
        event_counts[event] = event_counts.get(event, 0) + 1
        if len(chosen) >= max_junctions:
            break
    return np.sort(np.asarray(chosen, dtype=int))


def select_count_block(
    count_h5ad: Path,
    *,
    max_cells: int = 1500,
    max_junctions: int = 500,
    random_seed: int = 0,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, pd.DataFrame, pd.DataFrame]:
    """Select a manageable high-coverage cell/junction block from count layers."""
    rng = np.random.default_rng(random_seed)
    count_h5ad = ensure_sparse_h5ad(count_h5ad)
    with h5py.File(count_h5ad, "r") as f:
        layers = f.get("layers")
        if layers is None or COUNT_LAYER not in layers or CLUSTER_LAYER not in layers:
            missing = [
                name for name in (COUNT_LAYER, CLUSTER_LAYER)
                if layers is None or name not in layers
            ]
            raise RuntimeError(
                "The supplied AnnData does not contain the count layers required "
                f"for beta-binomial PSI estimation. Missing layers: {missing}."
            )
        var = _read_h5_dataframe_full(
            f,
            "var",
            [
                "junction_id",
                "event_id",
                "gene_name",
                "gene_id",
                "num_junctions",
                "CountJuncs",
                "n_cells_detected",
                "annotation_status",
            ],
        )
        obs_score = _read_h5_column(f["obs"], "total_junction_reads")
        eligible_cells = np.flatnonzero(np.isfinite(obs_score))
        if eligible_cells.size > max_cells:
            weights = np.maximum(obs_score[eligible_cells].astype(float), 0.0)
            weights = None if weights.sum() == 0 else weights / weights.sum()
            cell_idx = np.sort(rng.choice(eligible_cells, size=max_cells, replace=False, p=weights))
        else:
            cell_idx = eligible_cells
        junction_idx = _select_junction_indices(var, max_junctions)
        cells = _read_h5_dataframe(
            f,
            "obs",
            cell_idx,
            [
                "cell_id",
                "cell_clean",
                "specific_cell_type",
                "medium_cell_type",
                "broad_cell_type",
                "cell_ontology_class",
                "tissue",
                "subtissue",
                "dataset",
                "total_junction_reads",
            ],
        )
        junctions = var.iloc[junction_idx].copy()
        K = _csr_block_from_h5(layers[COUNT_LAYER], cell_idx, junction_idx)
        N = _csr_block_from_h5(layers[CLUSTER_LAYER], cell_idx, junction_idx)
    return K, N, cells, junctions


def _fit_beta_binomial_column(k: np.ndarray, n: np.ndarray) -> tuple[float, float]:
    """MLE for one beta-binomial column, parameterized by mean and concentration."""
    mask = (n > 0) & np.isfinite(k) & np.isfinite(n) & (k >= 0) & (k <= n)
    k = np.asarray(k[mask], dtype=float)
    n = np.asarray(n[mask], dtype=float)
    if k.size == 0:
        return 1.0, 1.0
    p0 = np.clip(k.sum() / max(n.sum(), 1.0), 1e-4, 1.0 - 1e-4)

    def nll(theta: np.ndarray) -> float:
        mu = expit(theta[0])
        conc = np.exp(theta[1])
        alpha = np.maximum(mu * conc, 1e-6)
        beta = np.maximum((1.0 - mu) * conc, 1e-6)
        ll = (
            gammaln(n + 1.0)
            - gammaln(k + 1.0)
            - gammaln(n - k + 1.0)
            + betaln(k + alpha, n - k + beta)
            - betaln(alpha, beta)
        )
        return float(-np.sum(ll))

    res = minimize(
        nll,
        x0=np.array([logit(p0), np.log(50.0)]),
        bounds=[(-10.0, 10.0), (-5.0, 12.0)],
        method="L-BFGS-B",
    )
    if not res.success and not np.isfinite(res.fun):
        warnings.warn(f"beta-binomial fit failed; using fallback prior: {res.message}")
        return float(p0 * 50.0), float((1.0 - p0) * 50.0)
    mu = expit(res.x[0])
    conc = np.exp(res.x[1])
    return float(max(mu * conc, 1e-6)), float(max((1.0 - mu) * conc, 1e-6))


def beta_binomial_psi(
    junction_counts: sparse.spmatrix,
    cluster_counts: sparse.spmatrix,
    cells: pd.DataFrame,
    junctions: pd.DataFrame,
    *,
    min_total: int = 1,
    clip: float = 1e-4,
) -> PsiEstimate:
    """Fit per-junction beta-binomial priors and return posterior PSI means/SEs."""
    K = _as_csr(junction_counts).astype(float).toarray()
    N = _as_csr(cluster_counts).astype(float).toarray()
    if K.shape != N.shape:
        raise ValueError(f"junction and cluster count shapes differ: {K.shape} vs {N.shape}")

    p = K.shape[1]
    alpha = np.zeros(p)
    beta = np.zeros(p)
    for j in range(p):
        alpha[j], beta[j] = _fit_beta_binomial_column(K[:, j], N[:, j])

    a_post = K + alpha[None, :]
    b_post = np.maximum(N - K, 0.0) + beta[None, :]
    total = a_post + b_post
    psi = np.clip(a_post / total, clip, 1.0 - clip)
    psi_var = (a_post * b_post) / (total ** 2 * (total + 1.0))
    psi_se = np.sqrt(np.maximum(psi_var, 0.0))
    observed = N >= min_total

    Y = logit(psi)
    S = psi_se / np.maximum(psi * (1.0 - psi), clip)
    Y = np.where(observed, Y, np.nan).astype(np.float32)
    S = np.where(observed, np.maximum(S, clip), np.inf).astype(np.float32)
    return PsiEstimate(Y, S, psi.astype(np.float32), psi_se.astype(np.float32), observed, alpha, beta, cells, junctions)


def split_observed(
    observed: np.ndarray,
    *,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    random_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split observed entries into train/validation/test masks."""
    rng = np.random.default_rng(random_seed)
    idx = np.argwhere(observed)
    rng.shuffle(idx)
    n_test = int(round(test_frac * len(idx)))
    n_val = int(round(val_frac * len(idx)))
    test = np.zeros_like(observed, dtype=bool)
    val = np.zeros_like(observed, dtype=bool)
    train = observed.copy()
    if n_test:
        test[tuple(idx[:n_test].T)] = True
    if n_val:
        val[tuple(idx[n_test:n_test + n_val].T)] = True
    train[test | val] = False
    return train, val, test


def _masked_rmse(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    d = np.asarray(a)[mask] - np.asarray(b)[mask]
    return float(np.sqrt(np.mean(d ** 2)))


def _fit_once(
    model: str,
    Y_train: np.ndarray,
    S_train: np.ndarray,
    lambda_val: float,
    *,
    max_iter: int,
    random_seed: int,
):
    if model == "homo":
        obs = np.isfinite(Y_train)
        res = taylor_proximal_homoskedastic_unknown_noise(
            jnp.asarray(Y_train),
            lambda_val=lambda_val,
            observed_mask=jnp.asarray(obs),
            max_iter=max_iter,
            mu_prox_max_iter=40,
            gamma_steps=1,
            gamma_update="hutchinson",
            hutchinson_probes=4,
            hutchinson_cg_maxiter=40,
            recover_sigma=False,
            hutchinson_seed=random_seed,
        )
        return np.asarray(res.mu), {"gamma2": float(res.gamma2), "n_iter": res.n_iter}
    if model == "hetero":
        res = proximal_gradient(
            jnp.asarray(Y_train),
            jnp.asarray(S_train),
            lambda_val=lambda_val,
            max_iter=max_iter,
            tol=1e-5,
            solver="monotone_fista",
        )
        return np.asarray(res.X), {"n_iter": res.n_iter}
    if model == "combined":
        res = proximal_noise_eb(
            jnp.asarray(Y_train),
            jnp.asarray(S_train),
            lambda_val=lambda_val,
            update_lambda=False,
            max_outer=6,
            prox_max_iter=max_iter,
            gamma_update="hutchinson",
            hutchinson_probes=4,
            hutchinson_cg_maxiter=40,
            lambda_parameterization="effective",
            random_seed=random_seed,
        )
        return np.asarray(res.X), {"g": float(res.g), "n_iter": res.n_iter}
    raise ValueError(f"unknown model {model!r}")


def fit_proximal_models(
    psi: PsiEstimate,
    *,
    lambda_grid: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0),
    max_iter: int = 80,
    random_seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Fit homoskedastic, heteroskedastic, and combined proximal models."""
    train, val, test = split_observed(psi.observed, random_seed=random_seed)
    predictions: dict[str, np.ndarray] = {}
    rows = []
    for model in ("homo", "hetero", "combined"):
        best = None
        for lam in lambda_grid:
            Y_fit = np.where(train, psi.Y_logit, np.nan)
            S_fit = np.where(train, psi.S_logit, np.inf)
            X, extra = _fit_once(model, Y_fit, S_fit, lam, max_iter=max_iter, random_seed=random_seed)
            val_rmse = _masked_rmse(X, psi.Y_logit, val)
            if best is None or val_rmse < best["val_logit_rmse"]:
                best = {"lambda": lam, "val_logit_rmse": val_rmse, "extra": extra}

        final_train = train | val
        Y_fit = np.where(final_train, psi.Y_logit, np.nan)
        S_fit = np.where(final_train, psi.S_logit, np.inf)
        X, extra = _fit_once(
            model,
            Y_fit,
            S_fit,
            float(best["lambda"]),
            max_iter=max_iter,
            random_seed=random_seed,
        )
        predictions[model] = X
        pred_psi = expit(X)
        rows.append({
            "model": model,
            "lambda": float(best["lambda"]),
            "val_logit_rmse": float(best["val_logit_rmse"]),
            "test_logit_rmse": _masked_rmse(X, psi.Y_logit, test),
            "test_psi_rmse": _masked_rmse(pred_psi, psi.psi_mean, test),
            **extra,
        })
    return pd.DataFrame(rows), predictions


def plot_atlas_umap(
    atlas_h5ad: Path,
    out_png: Path,
    *,
    color_by: str = "medium_cell_type",
    max_cells: int = 10000,
    random_seed: int = 0,
) -> pd.DataFrame:
    """UMAP of the trained atlas `X_PHI` representation, colored by metadata."""
    a = ad.read_h5ad(atlas_h5ad, backed="r")
    if "X_PHI" not in a.obsm:
        raise RuntimeError("Expected `obsm['X_PHI']` in the trained atlas.")
    rng = np.random.default_rng(random_seed)
    n = a.n_obs
    idx = np.arange(n)
    if n > max_cells:
        idx = np.sort(rng.choice(idx, size=max_cells, replace=False))
    phi = np.asarray(a.obsm["X_PHI"][idx])
    labels = a.obs.iloc[idx][color_by].astype(str).to_numpy() if color_by in a.obs else np.repeat("NA", len(idx))
    emb = umap.UMAP(n_neighbors=30, min_dist=0.3, metric="euclidean", random_state=random_seed).fit_transform(phi)
    df = pd.DataFrame({"UMAP1": emb[:, 0], "UMAP2": emb[:, 1], color_by: labels})
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 7))
    sns.scatterplot(data=df, x="UMAP1", y="UMAP2", hue=color_by, s=5, linewidth=0, alpha=0.8, legend=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    a.file.close()
    return df


def atlas_splicing_program_matrix(atlas_h5ad: Path) -> tuple[np.ndarray, pd.DataFrame, int]:
    """Load cell splicing-program scores and metadata from the trained atlas."""
    a = ad.read_h5ad(atlas_h5ad, backed="r")
    obs = a.obs.copy()
    sp_cols = [c for c in obs.columns if c.startswith("SP_")]
    if sp_cols:
        X = obs[sp_cols].to_numpy(dtype=np.float32)
    elif "X_PHI" in a.obsm:
        X = np.asarray(a.obsm["X_PHI"], dtype=np.float32)
        sp_cols = [f"SP_{i + 1}" for i in range(X.shape[1])]
    else:
        raise RuntimeError("No SP_* obs columns or obsm['X_PHI'] found in atlas.")
    n_programs = X.shape[1]
    a.file.close()
    return X, obs, n_programs


def assess_celltype_predictability(
    atlas_h5ad: Path,
    *,
    labels: tuple[str, ...] = ("broad_cell_type", "medium_cell_type", "specific_cell_type"),
    max_cells: int = 60000,
    random_seed: int = 0,
) -> pd.DataFrame:
    """Train/test logistic classifiers from splicing programs to cell labels."""
    X, obs, n_programs = atlas_splicing_program_matrix(atlas_h5ad)
    rng = np.random.default_rng(random_seed)
    rows = []
    for label in labels:
        if label not in obs:
            continue
        y_raw = obs[label].astype(str).to_numpy()
        keep = pd.Series(y_raw).notna().to_numpy() & (y_raw != "nan")
        classes, counts = np.unique(y_raw[keep], return_counts=True)
        keep_classes = set(classes[counts >= 20])
        keep &= np.isin(y_raw, list(keep_classes))
        idx_all = np.flatnonzero(keep)
        y_all = y_raw[idx_all]
        if idx_all.size > max_cells:
            idx, _ = train_test_split(
                idx_all,
                train_size=max_cells,
                random_state=random_seed,
                stratify=y_all,
            )
            idx = np.sort(idx)
        else:
            idx = idx_all
        y = y_raw[idx]
        X_sub = X[idx]
        enc = LabelEncoder()
        y_enc = enc.fit_transform(y)
        stratify = y_enc if np.min(np.bincount(y_enc)) >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_sub,
            y_enc,
            test_size=0.25,
            random_state=random_seed,
            stratify=stratify,
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        clf = LogisticRegression(
            max_iter=500,
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
        )
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        majority = np.bincount(y_train).argmax()
        majority_pred = np.full_like(y_test, majority)
        rows.append({
            "label": label,
            "n_programs": n_programs,
            "n_cells": int(idx.size),
            "n_classes": int(len(enc.classes_)),
            "accuracy": accuracy_score(y_test, pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, pred),
            "majority_accuracy": accuracy_score(y_test, majority_pred),
            "majority_balanced_accuracy": balanced_accuracy_score(y_test, majority_pred),
        })
    return pd.DataFrame(rows)


def synthetic_count_example(
    *,
    n_cells: int = 120,
    n_junctions: int = 80,
    rank: int = 4,
    random_seed: int = 0,
) -> PsiEstimate:
    """Small count-like example used to smoke-test the notebook pipeline."""
    rng = np.random.default_rng(random_seed)
    U = rng.normal(size=(n_cells, rank))
    V = rng.normal(size=(rank, n_junctions))
    true_psi = expit(U @ V / np.sqrt(rank))
    totals = rng.poisson(8.0, size=(n_cells, n_junctions))
    totals[rng.random(totals.shape) < 0.35] = 0
    counts = rng.binomial(totals, true_psi)
    cells = pd.DataFrame({"cell_id": [f"cell_{i}" for i in range(n_cells)]})
    junctions = pd.DataFrame({
        "junction_id": [f"junction_{j}" for j in range(n_junctions)],
        "event_id": [f"event_{j // 2}" for j in range(n_junctions)],
    })
    return beta_binomial_psi(sparse.csr_matrix(counts), sparse.csr_matrix(totals), cells, junctions)


def timed(label: str):
    """Tiny context manager for notebook timings."""
    class _Timer:
        def __enter__(self):
            self.t0 = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc, tb):
            self.seconds = time.perf_counter() - self.t0
            print(f"{label}: {self.seconds:.2f}s")
    return _Timer()
