#!/usr/bin/env python
"""Select high-coverage, cell-type-variable junctions and build a subset cache."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import pandas as pd


COUNT_LAYER = "cell_by_junction_matrix"
CLUSTER_LAYER = "cell_by_cluster_matrix"


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _decode(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind in {"S", "O"}:
        return np.asarray([x.decode() if isinstance(x, (bytes, np.bytes_)) else x for x in values])
    return values


def _read_categorical(group: h5py.Group, key: str) -> tuple[np.ndarray, np.ndarray]:
    obj = group[key]
    return obj["codes"][...].astype(np.int32), _decode(obj["categories"][...])


def _read_column(group: h5py.Group, key: str) -> np.ndarray:
    obj = group[key]
    if isinstance(obj, h5py.Group) and obj.attrs.get("encoding-type") == "categorical":
        codes, cats = _read_categorical(group, key)
        vals = np.empty(codes.shape[0], dtype=object)
        vals[codes >= 0] = cats[codes[codes >= 0]]
        vals[codes < 0] = None
        return vals
    return _decode(obj[...])


def _csr_group(f: h5py.File, layer: str) -> h5py.Group:
    return f["layers"][layer]


def _shape(group: h5py.Group) -> tuple[int, int]:
    return tuple(int(x) for x in group.attrs["shape"])


def compute_celltype_variability(
    count_h5ad: Path,
    out_dir: Path,
    *,
    celltype_key: str = "medium_cell_type",
    block_rows: int = 256,
    min_type_total: float = 50.0,
) -> pd.DataFrame:
    """Aggregate junction PSI by cell type and compute variability scores."""
    count_h5ad = Path(count_h5ad)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"junction_celltype_variability_{celltype_key}.csv"
    if summary_path.exists():
        return pd.read_csv(summary_path)

    with h5py.File(count_h5ad, "r") as f:
        K = _csr_group(f, COUNT_LAYER)
        N = _csr_group(f, CLUSTER_LAYER)
        m, n = _shape(K)
        type_codes, type_names = _read_categorical(f["obs"], celltype_key)
        n_types = len(type_names)
        sum_k = np.zeros((n_types, n), dtype=np.float64)
        sum_n = np.zeros((n_types, n), dtype=np.float64)
        indptr = K["indptr"]
        t0 = time.perf_counter()
        for r0 in range(0, m, block_rows):
            r1 = min(r0 + block_rows, m)
            start = int(indptr[r0])
            end = int(indptr[r1])
            if end <= start:
                continue
            row_counts = np.diff(indptr[r0:r1 + 1])
            types = np.repeat(type_codes[r0:r1], row_counts)
            cols = K["indices"][start:end]
            k = K["data"][start:end].astype(np.float64)
            denom = N["data"][start:end].astype(np.float64)
            valid = (types >= 0) & (denom > 0)
            key = types[valid].astype(np.int64) * n + cols[valid].astype(np.int64)
            sum_k += np.bincount(key, weights=k[valid], minlength=n_types * n).reshape(n_types, n)
            sum_n += np.bincount(key, weights=denom[valid], minlength=n_types * n).reshape(n_types, n)
            if r1 % (block_rows * 20) == 0 or r1 == m:
                print(f"aggregated rows {r1:,}/{m:,} in {time.perf_counter() - t0:.1f}s", flush=True)

        total_n = sum_n.sum(axis=0)
        total_k = sum_k.sum(axis=0)
        psi_global = np.divide(total_k, np.maximum(total_n, 1.0))
        valid_type = sum_n >= min_type_total
        psi_type = np.divide(sum_k, np.maximum(sum_n, 1.0))
        weights = np.where(valid_type, sum_n, 0.0)
        weight_sum = np.maximum(weights.sum(axis=0), 1.0)
        mean_type = (weights * psi_type).sum(axis=0) / weight_sum
        var_type = (weights * (psi_type - mean_type[None, :]) ** 2).sum(axis=0) / weight_sum
        n_types_detected = valid_type.sum(axis=0)

        var_group = f["var"]
        junction_id = _read_column(var_group, "junction_id") if "junction_id" in var_group else np.arange(n)
        event_id = _read_column(var_group, "event_id") if "event_id" in var_group else np.repeat("", n)
        gene_name = _read_column(var_group, "gene_name") if "gene_name" in var_group else np.repeat("", n)
        n_cells_detected = (
            var_group["n_cells_detected"][...] if "n_cells_detected" in var_group else np.asarray((total_n > 0), dtype=int)
        )
        count_juncs = var_group["CountJuncs"][...] if "CountJuncs" in var_group else total_k
        num_junctions = var_group["num_junctions"][...] if "num_junctions" in var_group else np.ones(n, dtype=int)

    df = pd.DataFrame({
        "junction_index": np.arange(n, dtype=np.int32),
        "junction_id": junction_id,
        "event_id": event_id,
        "gene_name": gene_name,
        "num_junctions": num_junctions,
        "n_cells_detected": n_cells_detected,
        "CountJuncs": count_juncs,
        "total_cluster_count": total_n,
        "total_junction_count": total_k,
        "psi_global": psi_global,
        "celltype_psi_variance": var_type,
        "n_celltypes_with_total": n_types_detected,
    })
    df.to_csv(summary_path, index=False)
    return df


def select_junctions(df: pd.DataFrame, out_dir: Path, *, n_select: int, min_celltypes: int) -> np.ndarray:
    """Select junctions using coverage and between-cell-type variability."""
    d = df.copy()
    eligible = (
        (d["num_junctions"].to_numpy() >= 2)
        & (d["n_celltypes_with_total"].to_numpy() >= min_celltypes)
        & np.isfinite(d["celltype_psi_variance"].to_numpy())
        & (d["celltype_psi_variance"].to_numpy() > 0)
    )
    d = d.loc[eligible].copy()
    coverage = np.log1p(d["n_cells_detected"].astype(float).to_numpy())
    variability = d["celltype_psi_variance"].astype(float).to_numpy()
    cov_rank = pd.Series(coverage).rank(pct=True).to_numpy()
    var_rank = pd.Series(variability).rank(pct=True).to_numpy()
    d["coverage_rank"] = cov_rank
    d["variability_rank"] = var_rank
    d["selection_score"] = cov_rank + var_rank
    d = d.sort_values("selection_score", ascending=False).head(n_select)
    out = Path(out_dir) / f"selected_{n_select}_junctions_by_coverage_celltype_variability.csv"
    d.to_csv(out, index=False)
    selected = np.sort(d["junction_index"].to_numpy(dtype=np.int32))
    np.save(Path(out_dir) / f"selected_{n_select}_junction_indices.npy", selected)
    print(f"selected {selected.size} junctions; table {out}", flush=True)
    return selected


def build_subset_cache(
    full_cache_dir: Path,
    subset_cache_dir: Path,
    selected_cols: np.ndarray,
    *,
    block_rows: int = 1024,
) -> None:
    """Build a remapped sparse PSI cache for selected columns from full cache."""
    full_cache_dir = Path(full_cache_dir)
    subset_cache_dir = Path(subset_cache_dir)
    subset_cache_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((full_cache_dir / "sparse_psi_meta.json").read_text())
    full_shape = tuple(meta["shape"])
    m, n = full_shape
    selected_cols = np.asarray(selected_cols, dtype=np.int32)
    p = selected_cols.size
    col_map = np.full(n, -1, dtype=np.int32)
    col_map[selected_cols] = np.arange(p, dtype=np.int32)

    full_indptr = np.memmap(full_cache_dir / "indptr.int32.mmap", mode="r", dtype=np.int32, shape=(m + 1,))
    full_indices = np.memmap(full_cache_dir / "indices.int32.mmap", mode="r", dtype=np.int32, shape=(int(meta["nnz"]),))
    row_counts = np.zeros(m, dtype=np.int32)
    for r0 in range(0, m, block_rows):
        r1 = min(r0 + block_rows, m)
        for i in range(r0, r1):
            mapped = col_map[full_indices[int(full_indptr[i]):int(full_indptr[i + 1])]]
            row_counts[i] = int(np.count_nonzero(mapped >= 0))
        if r1 % (block_rows * 20) == 0 or r1 == m:
            print(f"subset count rows {r1:,}/{m:,}", flush=True)
    subset_indptr_np = np.empty(m + 1, dtype=np.int32)
    subset_indptr_np[0] = 0
    np.cumsum(row_counts, out=subset_indptr_np[1:])
    nnz = int(subset_indptr_np[-1])
    print(f"subset nnz {nnz:,}", flush=True)

    indptr = np.memmap(subset_cache_dir / "indptr.int32.mmap", mode="w+", dtype=np.int32, shape=(m + 1,))
    indices = np.memmap(subset_cache_dir / "indices.int32.mmap", mode="w+", dtype=np.int32, shape=(nnz,))
    y = np.memmap(subset_cache_dir / "y_logit.float32.mmap", mode="w+", dtype=np.float32, shape=(nnz,))
    se = np.memmap(subset_cache_dir / "se_logit.float32.mmap", mode="w+", dtype=np.float32, shape=(nnz,))
    indptr[:] = subset_indptr_np
    indptr.flush()
    full_y = np.memmap(full_cache_dir / "y_logit.float32.mmap", mode="r", dtype=np.float32, shape=(int(meta["nnz"]),))
    full_se = np.memmap(full_cache_dir / "se_logit.float32.mmap", mode="r", dtype=np.float32, shape=(int(meta["nnz"]),))
    for i in range(m):
        start = int(full_indptr[i])
        end = int(full_indptr[i + 1])
        mapped = col_map[full_indices[start:end]]
        keep = mapped >= 0
        out0 = int(subset_indptr_np[i])
        out1 = int(subset_indptr_np[i + 1])
        if out1 > out0:
            indices[out0:out1] = mapped[keep]
            y[out0:out1] = full_y[start:end][keep]
            se[out0:out1] = full_se[start:end][keep]
        if (i + 1) % (block_rows * 20) == 0 or i + 1 == m:
            print(f"subset write rows {i + 1:,}/{m:,}", flush=True)
    indices.flush()
    y.flush()
    se.flush()

    full_alpha = np.memmap(full_cache_dir / "alpha.float32.mmap", mode="r", dtype=np.float32, shape=(n,))
    full_beta = np.memmap(full_cache_dir / "beta.float32.mmap", mode="r", dtype=np.float32, shape=(n,))
    alpha = np.memmap(subset_cache_dir / "alpha.float32.mmap", mode="w+", dtype=np.float32, shape=(p,))
    beta = np.memmap(subset_cache_dir / "beta.float32.mmap", mode="w+", dtype=np.float32, shape=(p,))
    alpha[:] = full_alpha[selected_cols]
    beta[:] = full_beta[selected_cols]
    alpha.flush()
    beta.flush()
    (subset_cache_dir / "sparse_psi_meta.json").write_text(json.dumps({
        "shape": [m, p],
        "nnz": nnz,
        "source_shape": list(full_shape),
        "selected_columns_file": str(subset_cache_dir / "selected_original_junction_indices.npy"),
    }, indent=2))
    np.save(subset_cache_dir / "selected_original_junction_indices.npy", selected_cols)


def main() -> None:
    count_h5ad = Path(os.environ.get(
        "SPLICING_COUNT_H5AD",
        "/media/david/HDD/splicing_data/model_ready_aligned_splicing_data_20251009_024406.h5ad",
    ))
    out_dir = Path(os.environ.get("SPLICING_SUBSET_OUT", "analyses/splicing_proximal/outputs"))
    full_cache_dir = Path(os.environ.get(
        "SPLICING_FULL_CACHE",
        "/media/david/HDD/splicing_data/matlap_sparse_cache",
    ))
    subset_cache_dir = Path(os.environ.get(
        "SPLICING_SUBSET_CACHE",
        "/media/david/HDD/splicing_data/matlap_sparse_cache_junction_subset_10000",
    ))
    n_select = env_int("SPLICING_SUBSET_N_JUNCTIONS", 10_000)
    celltype_key = os.environ.get("SPLICING_CELLTYPE_KEY", "medium_cell_type")
    min_celltypes = env_int("SPLICING_SUBSET_MIN_CELLTYPES", 8)
    block_rows = env_int("SPLICING_SUBSET_BLOCK_ROWS", 256)

    df = compute_celltype_variability(
        count_h5ad,
        out_dir,
        celltype_key=celltype_key,
        block_rows=block_rows,
    )
    selected = select_junctions(df, out_dir, n_select=n_select, min_celltypes=min_celltypes)
    build_subset_cache(
        full_cache_dir,
        subset_cache_dir,
        selected,
        block_rows=env_int("SPLICING_SUBSET_CACHE_BLOCK_ROWS", 1024),
    )
    print(f"subset cache ready: {subset_cache_dir}", flush=True)


if __name__ == "__main__":
    main()
