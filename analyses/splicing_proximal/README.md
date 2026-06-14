# Single-cell splicing proximal analysis

This directory contains a notebook pipeline for benchmarking the proximal NND
models on single-cell splicing PSI estimates.

The linked Zenodo file,
`splice_adata_for_figures_mouse_foundation.h5ad`, is the trained LeafletFA atlas:
it contains cell metadata, `obsm/X_PHI`, and `varm/psi_learned`, but no count
matrix or count layers. The beta-binomial preprocessing requires a
SplicingDataset-style AnnData object with:

- `layers["cell_by_junction_matrix"]`: per-cell junction read counts.
- `layers["cell_by_cluster_matrix"]`: per-cell total count for the
  corresponding LeafCutter/ATSE cluster.

The notebook downloads and inspects the provided atlas, makes metadata/UMAP
figures from the available latent representation, and runs the model-fitting
sections whenever a count-layer h5ad is supplied.

The full local object at
`/media/david/HDD/splicing_data/model_ready_aligned_splicing_data_20251009_024406.h5ad`
already stores `X`, `layers["cell_by_junction_matrix"]`, and
`layers["cell_by_cluster_matrix"]` as HDF5-backed CSR matrices. The helper code
therefore reads selected sparse row/column blocks directly with `h5py`; do not
load this file with plain `ad.read_h5ad(...)`, which can materialize the full
1.28B-nonzero sparse layer.

Run with:

```bash
~/venvs/jax/bin/python -m pip install -r analyses/splicing_proximal/requirements.txt
~/venvs/jax/bin/jupyter lab analyses/splicing_proximal/splicing_proximal_models.ipynb
```

Command-line debug run:

```bash
SPLICING_MAX_CELLS=600 \
SPLICING_MAX_JUNCTIONS=200 \
SPLICING_PROX_ITER=40 \
SPLICING_LAMBDA_GRID=0.01,0.03,0.1,0.3,1 \
SPLICING_CELLTYPE_MAX_CELLS=60000 \
~/venvs/jax/bin/python analyses/splicing_proximal/run_full_analysis.py
```

Literal full observed-matrix sparse proximal run:

```bash
SPLICING_SPARSE_RANK=20 \
SPLICING_SPARSE_MAX_ITER=3 \
SPLICING_SPARSE_LAMBDA=10000 \
~/venvs/jax/bin/python analyses/splicing_proximal/sparse_full_prox.py
```

This builds a sparse logit-PSI cache at
`/media/david/HDD/splicing_data/matlap_sparse_cache` and then runs SVT on a
`sparse correction + low-rank current estimate` linear operator.  It operates on
all `142315 x 89831` features and all `1,280,029,347` observed entries without
forming a dense matrix.

Randomized SVD backend and model scoring:

```bash
SPLICING_SPARSE_SVD=rsvd \
SPLICING_SPARSE_POWER_ITER=1 \
SPLICING_SPARSE_OVERSAMPLE=10 \
SPLICING_SPARSE_RANK=20 \
SPLICING_SPARSE_MAX_ITER=3 \
SPLICING_SPARSE_LAMBDA_GRID=5000 \
SPLICING_SPARSE_NOISE_MODELS=homo,hetero,combined \
SPLICING_SPARSE_SE_FLOOR=0.25 \
SPLICING_SPARSE_PREFIX=sparse_full_rsvd_lambda5000_iter3_models \
~/venvs/jax/bin/python analyses/splicing_proximal/sparse_full_prox.py
```

The scorer writes diagonal Taylor approximations to
`*_scores.csv`.  The full row-covariance Taylor terms are not tractable at
`n=89831`, so the scalable score uses the low-rank singular vectors to form a
diagonal prior-curvature approximation and computes ELBO/LOO surrogates over all
observed entries.

Low-rank-plus-isotropic row covariance scoring on existing factors:

```bash
SPLICING_SCORE_ONLY=1 \
SPLICING_ISO_ROW_SCORE=1 \
SPLICING_ISO_ROW_MAX_ROWS=10000 \
SPLICING_SPARSE_LAMBDA=5000 \
SPLICING_SPARSE_NOISE_MODELS=homo \
SPLICING_SPARSE_PREFIX=iso_row_score_homo_lambda5000_rank10_rows10000 \
SPLICING_SPARSE_INIT_FACTORS=analyses/splicing_proximal/outputs/sparse_full_rsvd_lambda5000_iter3_homo_gamma_scorefix_factors.npz \
~/venvs/jax/bin/python analyses/splicing_proximal/sparse_full_prox.py
```

Select 10,000 junctions by coverage and medium-cell-type variability, then build
an all-cell subset cache:

```bash
SPLICING_SUBSET_N_JUNCTIONS=10000 \
SPLICING_CELLTYPE_KEY=medium_cell_type \
SPLICING_SUBSET_MIN_CELLTYPES=8 \
~/venvs/jax/bin/python analyses/splicing_proximal/select_junction_subset.py
```

The resulting cache is
`/media/david/HDD/splicing_data/matlap_sparse_cache_junction_subset_10000`.
It has all 142,315 cells, 10,000 selected junctions, and 263,931,532 observed
entries.  Example selected-model run:

```bash
SPLICING_SPARSE_CACHE=/media/david/HDD/splicing_data/matlap_sparse_cache_junction_subset_10000 \
SPLICING_SPARSE_SVD=rsvd \
SPLICING_SPARSE_RANK=50 \
SPLICING_SPARSE_MAX_ITER=5 \
SPLICING_SPARSE_LAMBDA_GRID=1000 \
SPLICING_SPARSE_NOISE_MODELS=homo \
SPLICING_SPARSE_PREFIX=subset10k_rsvd_iter5_homo_lambda1000 \
~/venvs/jax/bin/python analyses/splicing_proximal/sparse_full_prox.py
```

For heteroskedastic PSI-space fits, the global proximal-gradient step is
limited by the largest observation precision. A very small PSI standard-error
floor can make this step tiny. On the selected 10,000-junction cache, sampled
PSI-SE quantiles are roughly 0.003 at 1%, 0.052 at 25%, and 0.101 at 50%. Use a
quantile floor to cap extreme precisions and record the effective floor in the
trace/scores:

```bash
SPLICING_VALUE_SPACE=psi \
SPLICING_SPARSE_CACHE=/media/david/HDD/splicing_data/matlap_sparse_cache_junction_subset_10000 \
SPLICING_SPARSE_SVD=rsvd \
SPLICING_SPARSE_RANK=50 \
SPLICING_SPARSE_MAX_ITER=5 \
SPLICING_SPARSE_LAMBDA=100000 \
SPLICING_SPARSE_NOISE_MODELS=hetero \
SPLICING_SPARSE_SE_FLOOR=0.01 \
SPLICING_SPARSE_SE_FLOOR_QUANTILE=0.25 \
SPLICING_SPARSE_PREFIX=subset10k_psi_hetero_q25_floor_iter5_lambda100000 \
~/venvs/jax/bin/python analyses/splicing_proximal/sparse_full_prox.py
```

This first-quartile SE floor gives an effective step of about 0.00272 instead
of 0.0001 from the raw 0.01 floor. A median floor gives a larger step but caps
half of the observed entries, so the first quartile is a less aggressive
default precision cap.

Fit nonnegative factors to denoised proximal PSI targets without materializing
the full all-cell by 10,000-junction matrix:

```bash
SPLICING_NMF_FITS=psi_homo,psi_hetero_q25 \
SPLICING_NMF_RANK=20 \
SPLICING_NMF_EPOCHS=3 \
SPLICING_NMF_BATCH_SIZE=2048 \
SPLICING_NMF_TRANSFORM_BATCH_SIZE=2048 \
SPLICING_NMF_EVAL_ROWS=5000 \
~/venvs/jax/bin/python analyses/splicing_proximal/fit_lowrank_nmf.py
```

This streams clipped denoised PSI row blocks from the saved proximal low-rank
factors into `MiniBatchNMF`, then computes all-cell NMF usage matrices in
batches.  The rank-20 run writes `*_nmf_rank20.npz`, `*_nmf_W.float32.mmap`,
and `lowrank_nmf_rank20_summary.csv`.

To constrain only the cell usages while allowing signed junction loadings, run
the semi-NMF variant:

```bash
SPLICING_NMF_METHOD=semi_nmf \
SPLICING_NMF_FITS=psi_homo,psi_hetero_q25 \
SPLICING_NMF_RANK=20 \
SPLICING_NMF_EPOCHS=3 \
SPLICING_NMF_BATCH_SIZE=2048 \
SPLICING_NMF_TRANSFORM_BATCH_SIZE=2048 \
SPLICING_NMF_EVAL_ROWS=5000 \
SPLICING_SEMI_NMF_NNLS_ITER=30 \
~/venvs/jax/bin/python analyses/splicing_proximal/fit_lowrank_nmf.py
```

This fits `W H` with nonnegative cell usages `W` and signed junction loadings
`H`, writing `*_semi_nmf_rank20.npz`, `*_semi_nmf_W.float32.mmap`, and
`lowrank_semi_nmf_rank20_summary.csv`.
