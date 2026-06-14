"""Generate the single-cell splicing proximal analysis notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "splicing_proximal_models.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {
        "display_name": "jax",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}

nb.cells = [
    md(
        """
        # Proximal NND models for single-cell splicing PSI

        This notebook downloads the mouse LeafletFA atlas linked in the prompt,
        estimates junction-level PSI and uncertainty from beta-binomial
        count models when raw count layers are available, and compares proximal
        NND denoisers:

        - **Homoskedastic**: ignores per-entry standard errors and estimates one
          global noise level.
        - **Heteroskedastic**: uses beta-binomial delta-method standard errors.
        - **Combined**: uses the beta-binomial standard errors plus an unknown
          homoskedastic residual variance.

        The linked `splice_adata_for_figures_mouse_foundation.h5ad` is a trained
        atlas object. It has metadata, `obsm["X_PHI"]`, and
        `varm["psi_learned"]`, but no raw count matrix. The beta-binomial
        section therefore requires a SplicingDataset-style h5ad with
        `layers["cell_by_junction_matrix"]` and
        `layers["cell_by_cluster_matrix"]`.
        """
    ),
    code(
        """
        from pathlib import Path
        import os
        import sys

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns

        ROOT = Path.cwd()
        if ROOT.name == "splicing_proximal":
            REPO = ROOT.parents[1]
            ANALYSIS = ROOT
        else:
            REPO = ROOT
            ANALYSIS = REPO / "analyses" / "splicing_proximal"
        sys.path.insert(0, str(REPO))
        sys.path.insert(0, str(ANALYSIS))

        from splicing_proximal_pipeline import (
            ATLAS_FILENAME,
            beta_binomial_psi,
            download_default_atlas,
            fit_proximal_models,
            inspect_h5ad,
            plot_atlas_umap,
            select_count_block,
            synthetic_count_example,
            timed,
        )

        DATA = ANALYSIS / "data"
        OUT = ANALYSIS / "outputs"
        DATA.mkdir(parents=True, exist_ok=True)
        OUT.mkdir(parents=True, exist_ok=True)
        """
    ),
    md(
        """
        ## Download and inspect the atlas

        The prompt URL is downloaded to `data/`. We inspect the file before doing
        any modeling so we know which slots are actually present.
        """
    ),
    code(
        """
        atlas_path = download_default_atlas(DATA)
        atlas_info = inspect_h5ad(atlas_path)
        atlas_info
        """
    ),
    code(
        """
        print("shape:", atlas_info["shape"])
        print("layers:", atlas_info["layers"])
        print("obsm:", atlas_info["obsm"])
        print("varm:", atlas_info["varm"])
        print("obs columns:", atlas_info["obs_columns"][:20])
        print("var columns:", atlas_info["var_columns"][:20])
        """
    ),
    md(
        """
        ## Downstream atlas view

        The trained atlas contains `X_PHI`, so we can still make a cell-level
        UMAP colored by cell type. This is downstream analysis of the provided
        object, independent of the count-based proximal fits.
        """
    ),
    code(
        """
        with timed("UMAP from X_PHI"):
            umap_df = plot_atlas_umap(
                atlas_path,
                OUT / "atlas_x_phi_umap_medium_cell_type.png",
                color_by="medium_cell_type",
                max_cells=10000,
                random_seed=1,
            )
        umap_df.head()
        """
    ),
    code(
        """
        from IPython.display import Image, display
        display(Image(filename=str(OUT / "atlas_x_phi_umap_medium_cell_type.png")))
        """
    ),
    md(
        """
        ## Count-layer input

        Set `COUNT_H5AD` to a SplicingDataset-style AnnData object with:

        - `layers["cell_by_junction_matrix"]`: junction-supporting reads.
        - `layers["cell_by_cluster_matrix"]`: total reads in the same
          LeafCutter/ATSE cluster for each junction/cell pair.

        The default below deliberately points at the downloaded atlas, so the
        next cell documents the missing-count-layer issue for the prompt file.
        """
    ),
    code(
        """
        COUNT_H5AD = DATA / os.environ.get("SPLICING_COUNT_H5AD", ATLAS_FILENAME)
        MAX_CELLS = int(os.environ.get("SPLICING_MAX_CELLS", "1500"))
        MAX_JUNCTIONS = int(os.environ.get("SPLICING_MAX_JUNCTIONS", "500"))
        COUNT_H5AD
        """
    ),
    code(
        """
        count_block = None
        try:
            with timed("select count block"):
                K_counts, N_cluster, cell_meta, junction_meta = select_count_block(
                    COUNT_H5AD,
                    max_cells=MAX_CELLS,
                    max_junctions=MAX_JUNCTIONS,
                    random_seed=2,
                )
            count_block = (K_counts, N_cluster, cell_meta, junction_meta)
            print(K_counts.shape, N_cluster.shape)
        except Exception as exc:
            print(type(exc).__name__ + ":", exc)
        """
    ),
    md(
        """
        ## Beta-binomial PSI and standard errors

        For each junction `j`, fit a beta-binomial model across cells:

        \\[
        k_{ij} \\mid n_{ij}, p_{ij} \\sim \\mathrm{Binomial}(n_{ij}, p_{ij}),
        \\qquad
        p_{ij} \\sim \\mathrm{Beta}(\\alpha_j, \\beta_j).
        \\]

        The posterior mean and standard error are then:

        \\[
        \\hat p_{ij} = {k_{ij}+\\alpha_j \\over n_{ij}+\\alpha_j+\\beta_j},
        \\qquad
        \\widehat{\\mathrm{se}}(p_{ij}) =
        \\sqrt{ {a_{ij} b_{ij} \\over (a_{ij}+b_{ij})^2(a_{ij}+b_{ij}+1)} }.
        \\]

        The proximal models are fit on `logit(PSI)` with delta-method standard
        errors, because the Gaussian approximation is much more sensible on the
        unconstrained scale.
        """
    ),
    code(
        """
        psi = None
        if count_block is not None:
            K_counts, N_cluster, cell_meta, junction_meta = count_block
            with timed("beta-binomial PSI"):
                psi = beta_binomial_psi(K_counts, N_cluster, cell_meta, junction_meta)
            print("Y shape:", psi.Y_logit.shape)
            print("observed fraction:", psi.observed.mean())
        """
    ),
    md(
        """
        ## Fit proximal models on real counts

        This cell runs only when `psi` was created from real count layers.
        Lambda is selected by validation RMSE on observed held-out entries, and
        final performance is reported on a separate test split.
        """
    ),
    code(
        """
        scores = None
        preds = None
        if psi is not None:
            with timed("proximal model grid"):
                scores, preds = fit_proximal_models(
                    psi,
                    lambda_grid=(0.25, 0.5, 1.0, 2.0, 4.0),
                    max_iter=80,
                    random_seed=3,
                )
            scores.to_csv(OUT / "proximal_model_scores.csv", index=False)
            display(scores)
        """
    ),
    code(
        """
        if scores is not None:
            plt.figure(figsize=(5, 3))
            sns.barplot(data=scores, x="model", y="test_psi_rmse")
            plt.tight_layout()
            plt.savefig(OUT / "proximal_model_test_psi_rmse.png", dpi=180)
            plt.show()
        """
    ),
    md(
        """
        ## Synthetic smoke test

        The current prompt file lacks raw counts, so this small synthetic
        count-like block confirms that the beta-binomial preprocessing and all
        three proximal model calls execute end-to-end. Treat this only as a code
        smoke test, not a scientific result.
        """
    ),
    code(
        """
        RUN_SYNTHETIC_SMOKE = True
        if RUN_SYNTHETIC_SMOKE:
            synth = synthetic_count_example(n_cells=100, n_junctions=60, rank=4, random_seed=4)
            with timed("synthetic proximal model grid"):
                synth_scores, synth_preds = fit_proximal_models(
                    synth,
                    lambda_grid=(0.25, 0.5, 1.0, 2.0),
                    max_iter=40,
                    random_seed=5,
                )
            synth_scores.to_csv(OUT / "synthetic_proximal_model_scores.csv", index=False)
            display(synth_scores)
        """
    ),
    md(
        """
        ## Notes for full-data runs

        - Keep the proximal matrix block modest at first (`MAX_CELLS`,
          `MAX_JUNCTIONS`) because the dense logit-PSI matrix is what gets passed
          to the proximal solvers.
        - For larger blocks, prefer fewer lambda values initially and increase
          `max_iter` only after the selected model looks stable.
        - The combined model estimates an extra variance `g`; if it is near
          zero, the beta-binomial standard errors are already explaining the
          residual scale. If it is large, either the beta-binomial SEs are too
          optimistic or the low-rank NND model is underfitting real structure.
        """
    ),
]

nbf.write(nb, NOTEBOOK)
print(NOTEBOOK)
