#!/usr/bin/env python
"""Run the real mouse splicing proximal analysis on a selected sparse block."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from splicing_proximal_pipeline import (
    ATLAS_FILENAME,
    assess_celltype_predictability,
    beta_binomial_psi,
    download_default_atlas,
    fit_proximal_models,
    inspect_h5ad,
    select_count_block,
    timed,
)


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def env_float_list(name: str, default: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in os.environ.get(name, default).split(",") if x.strip())


def main() -> None:
    here = Path(__file__).resolve().parent
    data_dir = here / "data"
    out_dir = here / "outputs"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    count_h5ad = Path(os.environ.get(
        "SPLICING_COUNT_H5AD",
        "/media/david/HDD/splicing_data/model_ready_aligned_splicing_data_20251009_024406.h5ad",
    ))
    atlas_h5ad = Path(os.environ.get("SPLICING_ATLAS_H5AD", str(data_dir / ATLAS_FILENAME)))
    if not atlas_h5ad.exists():
        atlas_h5ad = download_default_atlas(data_dir)

    max_cells = env_int("SPLICING_MAX_CELLS", 600)
    max_junctions = env_int("SPLICING_MAX_JUNCTIONS", 200)
    prox_iter = env_int("SPLICING_PROX_ITER", 50)
    seed = env_int("SPLICING_SEED", 0)
    lambda_grid = env_float_list("SPLICING_LAMBDA_GRID", "0.25,0.5,1,2")

    print("count_h5ad:", count_h5ad)
    print("atlas_h5ad:", atlas_h5ad)
    print("block:", max_cells, "cells x", max_junctions, "junctions")
    print("lambda_grid:", lambda_grid, "prox_iter:", prox_iter)
    print("count inventory:", inspect_h5ad(count_h5ad))

    with timed("select sparse count block"):
        K, N, cells, junctions = select_count_block(
            count_h5ad,
            max_cells=max_cells,
            max_junctions=max_junctions,
            random_seed=seed,
        )
    print("K shape/nnz:", K.shape, K.nnz, "N nnz:", N.nnz)
    cells.to_csv(out_dir / "selected_cells.csv")
    junctions.to_csv(out_dir / "selected_junctions.csv")

    with timed("beta-binomial PSI"):
        psi = beta_binomial_psi(K, N, cells, junctions)
    pd.DataFrame({
        "junction_id": junctions.get("junction_id", pd.Series(np.arange(K.shape[1]))).to_numpy(),
        "event_id": junctions.get("event_id", pd.Series([""] * K.shape[1])).to_numpy(),
        "alpha": psi.alpha,
        "beta": psi.beta,
        "observed_fraction": psi.observed.mean(axis=0),
        "mean_psi": np.nanmean(np.where(psi.observed, psi.psi_mean, np.nan), axis=0),
        "median_logit_se": np.nanmedian(np.where(psi.observed, psi.S_logit, np.nan), axis=0),
    }).to_csv(out_dir / "junction_beta_binomial_summary.csv", index=False)

    with timed("proximal model comparison"):
        scores, _ = fit_proximal_models(
            psi,
            lambda_grid=lambda_grid,
            max_iter=prox_iter,
            random_seed=seed + 1,
        )
    scores.to_csv(out_dir / "real_proximal_model_scores.csv", index=False)
    print(scores)

    print("atlas inventory:", inspect_h5ad(atlas_h5ad))
    with timed("cell-type prediction from splicing programs"):
        celltype = assess_celltype_predictability(
            atlas_h5ad,
            max_cells=env_int("SPLICING_CELLTYPE_MAX_CELLS", 60000),
            random_seed=seed + 2,
        )
    celltype.to_csv(out_dir / "celltype_predictability_from_splicing_programs.csv", index=False)
    print(celltype)


if __name__ == "__main__":
    main()
