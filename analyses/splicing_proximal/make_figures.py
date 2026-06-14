#!/usr/bin/env python
"""Generate figures for the splicing proximal analysis note."""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import umap


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analyses" / "splicing_proximal" / "outputs"
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
COUNT_H5AD = Path(
    "/media/david/HDD/splicing_data/model_ready_aligned_splicing_data_20251009_024406.h5ad"
)
RNG_SEED = 20260613


def savefig(name: str, *, dpi: int = 220, tight: bool = True) -> None:
    path = FIG / name
    if tight:
        plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close()
    print(path)


def plot_junction_selection(max_background: int = 80000) -> None:
    all_path = OUT / "junction_celltype_variability_medium_cell_type.csv"
    sel_path = OUT / "selected_10000_junctions_by_coverage_celltype_variability.csv"
    all_df = pd.read_csv(all_path)
    sel = pd.read_csv(sel_path)
    eligible = (
        (all_df["num_junctions"] >= 2)
        & (all_df["n_celltypes_with_total"] >= 8)
        & np.isfinite(all_df["celltype_psi_variance"])
        & (all_df["celltype_psi_variance"] > 0)
    )
    all_df = all_df.loc[eligible].copy()
    rng = np.random.default_rng(RNG_SEED)
    if len(all_df) > max_background:
        bg = all_df.sample(n=max_background, random_state=RNG_SEED)
    else:
        bg = all_df

    plt.figure(figsize=(6.4, 4.2))
    plt.scatter(
        np.log10(bg["n_cells_detected"] + 1),
        bg["celltype_psi_variance"],
        s=3,
        c="#b8b8b8",
        alpha=0.25,
        linewidths=0,
        label=f"eligible junctions, sampled {len(bg):,}",
        rasterized=True,
    )
    plt.scatter(
        np.log10(sel["n_cells_detected"] + 1),
        sel["celltype_psi_variance"],
        s=5,
        c="#c23b22",
        alpha=0.5,
        linewidths=0,
        label="selected 10,000",
        rasterized=True,
    )
    plt.xlabel("log10(cells detected + 1)")
    plt.ylabel("PSI variance across medium cell types")
    plt.legend(frameon=False, loc="upper left")
    savefig("splicing_subset_junction_selection.pdf")


def plot_model_scores() -> None:
    df = pd.read_csv(OUT / "subset10k_selected_models_allrows_iso_row_summary.csv")
    labels = {"homo": "homo", "hetero": "hetero", "combined": "combined"}
    df["label"] = df["noise_model"].map(labels)
    colors = ["#4c78a8", "#f58518", "#54a24b"]

    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.0))
    metrics = [
        ("obs_logit_rmse", "Observed logit RMSE"),
        ("iso_row_taylor_loo_rmse", "Row-iso LOO RMSE"),
        ("rank", "Retained rank"),
    ]
    for ax, (col, title) in zip(axes, metrics):
        ax.bar(df["label"], df[col], color=colors, width=0.65)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.spines[["top", "right"]].set_visible(False)
    savefig("splicing_subset_model_scores.pdf")


def plot_celltype_prediction() -> None:
    df = pd.read_csv(OUT / "full_cells_celltype_predictability_from_splicing_programs.csv")
    label_map = {
        "broad_cell_type": "broad",
        "medium_cell_type": "medium",
        "specific_cell_type": "specific",
    }
    x = np.arange(len(df))
    width = 0.25
    plt.figure(figsize=(6.4, 3.6))
    plt.bar(x - width, df["accuracy"], width, label="accuracy", color="#4c78a8")
    plt.bar(x, df["balanced_accuracy"], width, label="balanced accuracy", color="#f58518")
    plt.bar(x + width, df["majority_accuracy"], width, label="majority accuracy", color="#9d9d9d")
    plt.xticks(x, [label_map.get(v, v) for v in df["label"]])
    plt.ylabel("Held-out score")
    plt.ylim(0, 0.85)
    plt.legend(frameon=False, ncol=1)
    plt.gca().spines[["top", "right"]].set_visible(False)
    savefig("splicing_program_celltype_prediction.pdf")


def _read_categorical_obs(h5ad_path: Path, key: str) -> np.ndarray:
    with h5py.File(h5ad_path, "r") as f:
        group = f["obs"][key]
        codes = group["codes"][:]
        categories = group["categories"][:].astype(str)
    out = np.full(codes.shape, "missing", dtype=object)
    valid = codes >= 0
    out[valid] = categories[codes[valid]]
    return out


def _factor_embedding(path: Path) -> np.ndarray:
    factors = np.load(path)
    U = factors["U"].astype(np.float32)
    s = factors["s"].astype(np.float32)
    return U * s[None, :]


def _atlas_embedding() -> np.ndarray:
    atlas = ROOT / "analyses" / "splicing_proximal" / "data" / "splice_adata_for_figures_mouse_foundation.h5ad"
    with h5py.File(atlas, "r") as f:
        return f["obsm"]["X_PHI"][:].astype(np.float32)


def plot_logit_vs_psi_scores() -> None:
    logit = pd.read_csv(OUT / "subset10k_selected_models_allrows_iso_row_summary.csv")
    logit["value_space"] = "logit"
    if "obs_rmse" not in logit.columns:
        logit["obs_rmse"] = logit["obs_logit_rmse"]
    psi_paths = [
        OUT / "subset10k_psi_iso_row_homo_lambda300_allrows_scores.csv",
        OUT / "subset10k_psi_iso_row_hetero_lambda1e6_allrows_scores.csv",
        OUT / "subset10k_psi_iso_row_combined_lambda300_allrows_scores.csv",
    ]
    psi = pd.concat([pd.read_csv(path) for path in psi_paths], ignore_index=True)
    keep = [
        "value_space",
        "noise_model",
        "lambda",
        "rank",
        "obs_rmse",
        "iso_row_taylor_loo_rmse",
        "iso_row_taylor_elbo",
        "g",
    ]
    df = pd.concat([logit[keep], psi[keep]], ignore_index=True)
    df.to_csv(OUT / "subset10k_logit_vs_psi_model_summary.csv", index=False)

    order = ["homo", "hetero", "combined"]
    colors = {"logit": "#4c78a8", "psi": "#f58518"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, metric, title in [
        (axes[0], "obs_rmse", "Observed-entry RMSE"),
        (axes[1], "iso_row_taylor_loo_rmse", "Row-iso LOO RMSE"),
    ]:
        x = np.arange(len(order))
        for offset, space in [(-0.18, "logit"), (0.18, "psi")]:
            vals = [
                df.loc[(df["noise_model"] == model) & (df["value_space"] == space), metric].iloc[0]
                for model in order
            ]
            ax.bar(x + offset, vals, width=0.34, color=colors[space], label=space)
        ax.set_title(title)
        ax.set_xticks(x, order, rotation=25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    savefig("splicing_logit_vs_psi_model_scores.pdf")


def _embedding_specs() -> list[tuple[str, Path | None]]:
    return [
        ("atlas SP", None),
        ("logit homo", OUT / "subset10k_rsvd_iter5_homo_lambda1000_factors.npz"),
        ("PSI homo", OUT / "subset10k_psi_rsvd_iter5_lambda300_homo_lambda300_factors.npz"),
        ("PSI hetero q25", OUT / "subset10k_psi_hetero_q25_floor_iter5_lambda100000_factors.npz"),
    ]


def _load_embedding(spec: tuple[str, Path | None]) -> np.ndarray:
    _, path = spec
    if path is None:
        return _atlas_embedding()
    return _factor_embedding(path)


def _nmf_specs() -> list[tuple[str, Path]]:
    return [
        ("NMF PSI homo", OUT / "psi_homo_nmf_rank20.npz"),
        ("NMF PSI hetero q25", OUT / "psi_hetero_q25_nmf_rank20.npz"),
        ("Semi-NMF PSI homo", OUT / "psi_homo_semi_nmf_rank20.npz"),
        ("Semi-NMF PSI hetero q25", OUT / "psi_hetero_q25_semi_nmf_rank20.npz"),
    ]


def _load_nmf_usage(spec: tuple[str, Path]) -> np.ndarray:
    _, path = spec
    arr = np.load(path)
    return arr["W"].astype(np.float32)


def _usage_embedding(W: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(np.log1p(np.maximum(W, 0.0)))


def _label_colors(classes: list[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("turbo", len(classes))
    return {cls: cmap(i) for i, cls in enumerate(classes)}


def plot_splicing_space_umaps(max_cells: int = 20000) -> None:
    labels = _read_categorical_obs(COUNT_H5AD, "broad_cell_type")
    rng = np.random.default_rng(RNG_SEED)
    n = labels.shape[0]
    if max_cells < n:
        sample = np.sort(rng.choice(n, size=max_cells, replace=False))
    else:
        sample = np.arange(n)

    specs = _embedding_specs()
    sampled_labels = labels[sample]
    classes = pd.Series(sampled_labels).value_counts().index.tolist()
    cmap = plt.get_cmap("tab20")
    color_map = {cls: cmap(i % 20) for i, cls in enumerate(classes)}

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.2), sharex=False, sharey=False)
    for ax, spec in zip(axes.ravel(), specs):
        name, _ = spec
        Z = _load_embedding(spec)[sample]
        Z = StandardScaler().fit_transform(Z)
        reducer = umap.UMAP(
            n_neighbors=30,
            min_dist=0.25,
            metric="euclidean",
            random_state=RNG_SEED,
        )
        xy = reducer.fit_transform(Z)
        for cls in classes:
            mask = sampled_labels == cls
            ax.scatter(
                xy[mask, 0],
                xy[mask, 1],
                s=2.0,
                alpha=0.45,
                linewidths=0,
                color=color_map[cls],
                label=cls,
                rasterized=True,
            )
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    handles, labels_out = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_out,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        markerscale=4,
        fontsize=7,
    )
    savefig("splicing_space_umap_broad_cell_type.pdf")


def plot_nmf_medium_umaps(max_cells: int = 20000) -> None:
    labels = _read_categorical_obs(COUNT_H5AD, "medium_cell_type")
    rng = np.random.default_rng(RNG_SEED)
    n = labels.shape[0]
    sample = np.sort(rng.choice(n, size=min(max_cells, n), replace=False))
    sampled_labels = labels[sample]
    classes = pd.Series(sampled_labels).value_counts().index.tolist()
    color_map = _label_colors(classes)

    specs = _nmf_specs()
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.4), sharex=False, sharey=False)
    for ax, spec in zip(axes.ravel(), specs):
        name, _ = spec
        Z = _usage_embedding(_load_nmf_usage(spec)[sample])
        xy = umap.UMAP(
            n_neighbors=30,
            min_dist=0.25,
            metric="euclidean",
            random_state=RNG_SEED,
        ).fit_transform(Z)
        for cls in classes:
            mask = sampled_labels == cls
            ax.scatter(
                xy[mask, 0],
                xy[mask, 1],
                s=2.0,
                alpha=0.55,
                linewidths=0,
                color=color_map[cls],
                label=cls,
                rasterized=True,
            )
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    handles, labels_out = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_out,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        markerscale=4,
        fontsize=4.5,
        ncol=2,
    )
    savefig("nmf_medium_cell_type_umap.pdf")


def assess_latent_celltype_separation(
    max_cells: int = 30000,
    silhouette_cells: int = 12000,
    k_neighbors: int = 15,
) -> None:
    rng = np.random.default_rng(RNG_SEED)
    labels_by_key = {
        key: _read_categorical_obs(COUNT_H5AD, key)
        for key in ["broad_cell_type", "medium_cell_type", "specific_cell_type"]
    }
    n = next(iter(labels_by_key.values())).shape[0]
    sample_n = min(max_cells, n)
    sample = np.sort(rng.choice(n, size=sample_n, replace=False))
    sil_n = min(silhouette_cells, sample_n)
    sil_local = np.sort(rng.choice(sample_n, size=sil_n, replace=False))

    rows = []
    for spec in _embedding_specs():
        name, _ = spec
        Z = _load_embedding(spec)[sample]
        Z = StandardScaler().fit_transform(Z)
        nn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric="euclidean", n_jobs=-1)
        nn.fit(Z)
        neigh = nn.kneighbors(return_distance=False)[:, 1:]
        for label_key, all_labels in labels_by_key.items():
            y = all_labels[sample]
            valid = y != "missing"
            yv = y[valid]
            neigh_valid = neigh[valid]
            same = yv[:, None] == y[neigh_valid]
            local_purity = float(np.mean(same))
            props = pd.Series(yv).value_counts(normalize=True)
            random_purity = float(np.sum(props.to_numpy() ** 2))
            try:
                sil_idx = sil_local[valid[sil_local]]
                sil = float(silhouette_score(Z[sil_idx], y[sil_idx], metric="euclidean"))
            except Exception:
                sil = np.nan
            rows.append({
                "embedding": name,
                "label": label_key,
                "n_cells": int(valid.sum()),
                "n_classes": int(pd.Series(yv).nunique()),
                "knn_k": k_neighbors,
                "knn_local_purity": local_purity,
                "random_local_purity": random_purity,
                "knn_purity_enrichment": local_purity / max(random_purity, 1e-12),
                "silhouette": sil,
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "latent_celltype_separation_metrics.csv", index=False)

    label_order = ["broad_cell_type", "medium_cell_type", "specific_cell_type"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.7))
    for ax, metric, title in [
        (axes[0], "knn_purity_enrichment", "kNN purity enrichment"),
        (axes[1], "silhouette", "Silhouette"),
    ]:
        pivot = df.pivot(index="embedding", columns="label", values=metric)
        x = np.arange(len(pivot.index))
        width = 0.24
        for offset, label_key, color in zip(
            [-width, 0, width],
            label_order,
            ["#4c78a8", "#f58518", "#54a24b"],
        ):
            ax.bar(x + offset, pivot[label_key], width=width, color=color, label=label_key.replace("_cell_type", ""))
        ax.set_title(title)
        ax.set_xticks(x, pivot.index, rotation=25, ha="right")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7)
    savefig("latent_celltype_separation_metrics.pdf")


def assess_nmf_medium_celltype_separation(
    max_cells: int = 30000,
    silhouette_cells: int = 12000,
    k_neighbors: int = 15,
) -> None:
    rng = np.random.default_rng(RNG_SEED)
    labels = _read_categorical_obs(COUNT_H5AD, "medium_cell_type")
    n = labels.shape[0]
    sample = np.sort(rng.choice(n, size=min(max_cells, n), replace=False))
    sil_local = np.sort(rng.choice(sample.size, size=min(silhouette_cells, sample.size), replace=False))
    rows = []
    for spec in _nmf_specs():
        name, _ = spec
        Z = _usage_embedding(_load_nmf_usage(spec)[sample])
        y = labels[sample]
        valid = y != "missing"
        nn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric="euclidean", n_jobs=-1)
        nn.fit(Z)
        neigh = nn.kneighbors(return_distance=False)[:, 1:]
        yv = y[valid]
        same = yv[:, None] == y[neigh[valid]]
        local_purity = float(np.mean(same))
        props = pd.Series(yv).value_counts(normalize=True)
        random_purity = float(np.sum(props.to_numpy() ** 2))
        try:
            sil_idx = sil_local[valid[sil_local]]
            sil = float(silhouette_score(Z[sil_idx], y[sil_idx], metric="euclidean"))
        except Exception:
            sil = np.nan
        rows.append({
            "embedding": name,
            "label": "medium_cell_type",
            "n_cells": int(valid.sum()),
            "n_classes": int(pd.Series(yv).nunique()),
            "knn_k": k_neighbors,
            "knn_local_purity": local_purity,
            "random_local_purity": random_purity,
            "knn_purity_enrichment": local_purity / max(random_purity, 1e-12),
            "silhouette": sil,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "nmf_medium_celltype_separation_metrics.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
    for ax, metric, title, color in [
        (axes[0], "knn_purity_enrichment", "Medium kNN purity enrichment", "#4c78a8"),
        (axes[1], "silhouette", "Medium silhouette", "#f58518"),
    ]:
        ax.bar(df["embedding"], df[metric], color=color, width=0.62)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25, labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    savefig("nmf_medium_celltype_separation_metrics.pdf")


def _celltype_factor_means(spec: tuple[str, Path | None], labels: np.ndarray) -> pd.DataFrame:
    name, _ = spec
    Z = _load_embedding(spec)
    Z = StandardScaler().fit_transform(Z)
    valid = labels != "missing"
    labels_valid = labels[valid]
    Z = Z[valid]

    rows = []
    for label in pd.Series(labels_valid).value_counts().index:
        mask = labels_valid == label
        mean = Z[mask].mean(axis=0)
        rows.append(mean)
    label_order = pd.Series(labels_valid).value_counts().index.to_numpy()
    means = np.vstack(rows)

    # Orient factors so the cell type with the largest absolute mean is positive.
    for j in range(means.shape[1]):
        i = int(np.argmax(np.abs(means[:, j])))
        if means[i, j] < 0:
            means[:, j] *= -1.0
    peak_order = np.lexsort((-np.max(means, axis=0), np.argmax(means, axis=0)))
    means = means[:, peak_order]

    out_rows = []
    counts = pd.Series(labels_valid).value_counts()
    for i, label in enumerate(label_order):
        for j in range(means.shape[1]):
            out_rows.append({
                "embedding": name,
                "cell_type": label,
                "factor": f"F{j + 1}",
                "mean_z_usage": float(means[i, j]),
                "n_cells": int(counts[label]),
            })
    return pd.DataFrame(out_rows)


def plot_celltype_factor_usage_heatmap(label_key: str = "broad_cell_type") -> None:
    labels = _read_categorical_obs(COUNT_H5AD, label_key)
    specs = _embedding_specs()
    long = pd.concat(
        [_celltype_factor_means(spec, labels) for spec in specs],
        ignore_index=True,
    )
    long.to_csv(OUT / f"{label_key}_factor_usage_means.csv", index=False)

    atlas = long.loc[long["embedding"] == "atlas SP"].pivot(
        index="cell_type",
        columns="factor",
        values="mean_z_usage",
    )
    row_order = atlas.index.to_list()
    if atlas.shape[0] > 2:
        order = leaves_list(linkage(atlas.to_numpy(), method="average", metric="euclidean"))
        row_order = [row_order[i] for i in order]

    n_rows = len(row_order)
    height = max(8.2, min(28.0, 2.7 + 0.14 * n_rows))
    row_font = max(3.0, min(6.0, 72.0 / max(n_rows, 1)))
    fig, axes = plt.subplots(2, 2, figsize=(12.8, height), sharey=False)
    fig.subplots_adjust(left=0.18, right=0.86, bottom=0.06, top=0.94, wspace=0.34, hspace=0.24)
    vmax = 1.75
    for ax, spec in zip(axes.ravel(), specs):
        name, _ = spec
        mat = long.loc[long["embedding"] == name].pivot(
            index="cell_type",
            columns="factor",
            values="mean_z_usage",
        )
        mat = mat.reindex(row_order)
        factor_order = sorted(mat.columns, key=lambda x: int(x[1:]))
        mat = mat[factor_order]
        im = ax.imshow(
            mat.to_numpy(),
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(name)
        ax.set_xticks(np.arange(mat.shape[1]))
        ax.set_xticklabels(mat.columns, rotation=90, fontsize=6)
        ax.set_yticks(np.arange(mat.shape[0]))
        ax.set_yticklabels(mat.index, fontsize=row_font)
        ax.tick_params(length=0)
        ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    cbar_ax = fig.add_axes([0.89, 0.18, 0.02, 0.64])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Mean standardized factor usage")
    savefig(f"{label_key}_factor_usage_heatmap.pdf", dpi=240, tight=False)


def plot_nmf_medium_factor_usage_heatmap() -> None:
    label_key = "medium_cell_type"
    labels = _read_categorical_obs(COUNT_H5AD, label_key)
    rows = []
    for spec in _nmf_specs():
        name, _ = spec
        Z = _usage_embedding(_load_nmf_usage(spec))
        valid = labels != "missing"
        labels_valid = labels[valid]
        Z = Z[valid]
        counts = pd.Series(labels_valid).value_counts()
        label_order = counts.index.to_numpy()
        means = []
        for label in label_order:
            means.append(Z[labels_valid == label].mean(axis=0))
        means = np.vstack(means)
        peak_order = np.lexsort((-np.max(means, axis=0), np.argmax(means, axis=0)))
        means = means[:, peak_order]
        for i, label in enumerate(label_order):
            for j in range(means.shape[1]):
                rows.append({
                    "embedding": name,
                    "cell_type": label,
                    "factor": f"N{j + 1}",
                    "mean_z_usage": float(means[i, j]),
                    "n_cells": int(counts[label]),
                })
    long = pd.DataFrame(rows)
    long.to_csv(OUT / "nmf_medium_cell_type_factor_usage_means.csv", index=False)

    first = long.loc[long["embedding"] == _nmf_specs()[0][0]].pivot(
        index="cell_type",
        columns="factor",
        values="mean_z_usage",
    )
    row_order = first.index.to_list()
    if first.shape[0] > 2:
        order = leaves_list(linkage(first.to_numpy(), method="average", metric="euclidean"))
        row_order = [row_order[i] for i in order]

    specs = _nmf_specs()
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 13.2), sharey=False)
    fig.subplots_adjust(left=0.16, right=0.88, bottom=0.06, top=0.95, wspace=0.34, hspace=0.22)
    vmax = 1.75
    for ax, spec in zip(axes.ravel(), specs):
        name, _ = spec
        mat = long.loc[long["embedding"] == name].pivot(
            index="cell_type",
            columns="factor",
            values="mean_z_usage",
        ).reindex(row_order)
        factor_order = sorted(mat.columns, key=lambda x: int(x[1:]))
        mat = mat[factor_order]
        im = ax.imshow(
            mat.to_numpy(),
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(name)
        ax.set_xticks(np.arange(mat.shape[1]))
        ax.set_xticklabels(mat.columns, rotation=90, fontsize=6)
        ax.set_yticks(np.arange(mat.shape[0]))
        ax.set_yticklabels(mat.index, fontsize=4.5)
        ax.tick_params(length=0)
        ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    cbar_ax = fig.add_axes([0.91, 0.2, 0.02, 0.6])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Mean standardized NMF usage")
    savefig("nmf_medium_cell_type_factor_usage_heatmap.pdf", dpi=240, tight=False)


def main() -> None:
    plot_junction_selection()
    plot_model_scores()
    plot_logit_vs_psi_scores()
    assess_latent_celltype_separation()
    assess_nmf_medium_celltype_separation()
    for label_key in ["broad_cell_type", "medium_cell_type", "specific_cell_type"]:
        plot_celltype_factor_usage_heatmap(label_key)
    plot_nmf_medium_factor_usage_heatmap()
    plot_splicing_space_umaps()
    plot_nmf_medium_umaps()
    plot_celltype_prediction()


if __name__ == "__main__":
    main()
