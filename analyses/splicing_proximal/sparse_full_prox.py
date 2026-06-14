#!/usr/bin/env python
"""Sparse full-matrix proximal NND fitting for the splicing count object.

This is the scalable counterpart to the dense notebook runner.  It never forms
the cell-by-junction matrix densely.  Observed logit-PSI values are cached as a
CSR matrix backed by memmap files, and each proximal step computes the SVT of a
``sparse correction + low-rank current estimate`` LinearOperator.  The cached
values can be used either on the logit-PSI scale or on the raw PSI scale.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np
from numba import njit, prange
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, svds
from scipy.special import expit, logit


COUNT_LAYER = "cell_by_junction_matrix"
CLUSTER_LAYER = "cell_by_cluster_matrix"


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def env_float_list(name: str, default: str) -> list[float]:
    return [float(x.strip()) for x in os.environ.get(name, default).split(",") if x.strip()]


def env_str_list(name: str, default: str) -> list[str]:
    return [x.strip().lower() for x in os.environ.get(name, default).split(",") if x.strip()]


@dataclass
class SparsePsiCache:
    shape: tuple[int, int]
    nnz: int
    indptr: np.memmap
    indices: np.memmap
    y: np.memmap
    se_logit: np.memmap
    alpha: np.memmap
    beta: np.memmap
    cache_dir: Path
    value_space: str = "logit"


def _csr_group(f: h5py.File, layer: str) -> h5py.Group:
    return f["layers"][layer]


def _shape(group: h5py.Group) -> tuple[int, int]:
    return tuple(int(x) for x in group.attrs["shape"])


def _copy_dataset_to_memmap(ds: h5py.Dataset, path: Path, dtype, *, chunk_n: int = 20_000_000):
    arr = np.memmap(path, mode="w+", dtype=dtype, shape=ds.shape)
    for start in range(0, ds.shape[0], chunk_n):
        end = min(start + chunk_n, ds.shape[0])
        arr[start:end] = ds[start:end]
        arr.flush()
    return arr


def _compute_beta_moments(
    count_h5ad: Path,
    cache_dir: Path,
    *,
    chunk_n: int,
    min_conc: float = 2.0,
    max_conc: float = 1e6,
    clip: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Moment beta fit for every junction from sparse count arrays."""
    with h5py.File(count_h5ad, "r") as f:
        K = _csr_group(f, COUNT_LAYER)
        N = _csr_group(f, CLUSTER_LAYER)
        n_vars = _shape(K)[1]
        nnz = K["data"].shape[0]
        sum_k = np.zeros(n_vars, dtype=np.float64)
        sum_n = np.zeros(n_vars, dtype=np.float64)
        sum_p = np.zeros(n_vars, dtype=np.float64)
        sum_p2 = np.zeros(n_vars, dtype=np.float64)
        count = np.zeros(n_vars, dtype=np.float64)
        for start in range(0, nnz, chunk_n):
            end = min(start + chunk_n, nnz)
            cols = K["indices"][start:end]
            k = K["data"][start:end].astype(np.float64)
            n = N["data"][start:end].astype(np.float64)
            valid = n > 0
            cols = cols[valid]
            k = k[valid]
            n = n[valid]
            p = np.clip(k / n, clip, 1.0 - clip)
            sum_k += np.bincount(cols, weights=k, minlength=n_vars)
            sum_n += np.bincount(cols, weights=n, minlength=n_vars)
            sum_p += np.bincount(cols, weights=p, minlength=n_vars)
            sum_p2 += np.bincount(cols, weights=p * p, minlength=n_vars)
            count += np.bincount(cols, minlength=n_vars)
            print(f"moments {end:,}/{nnz:,}", flush=True)

    mean_count = np.divide(sum_k, np.maximum(sum_n, 1.0))
    mean_unweighted = np.divide(sum_p, np.maximum(count, 1.0))
    second = np.divide(sum_p2, np.maximum(count, 1.0))
    var = np.maximum(second - mean_unweighted ** 2, 0.0)
    mu = np.clip(mean_count, clip, 1.0 - clip)
    conc = np.where(
        (count > 1) & (var > 0),
        mu * (1.0 - mu) / np.maximum(var, 1e-12) - 1.0,
        50.0,
    )
    conc = np.clip(conc, min_conc, max_conc)
    alpha = np.maximum(mu * conc, 1e-6)
    beta = np.maximum((1.0 - mu) * conc, 1e-6)
    np.save(cache_dir / "alpha_moments.npy", alpha.astype(np.float32))
    np.save(cache_dir / "beta_moments.npy", beta.astype(np.float32))
    return alpha.astype(np.float32), beta.astype(np.float32)


def _ensure_psi_value_cache(cache_dir: Path, nnz: int, *, chunk_n: int) -> None:
    y_psi_path = cache_dir / "y_psi.float32.mmap"
    se_psi_path = cache_dir / "se_psi.float32.mmap"
    if y_psi_path.exists() and se_psi_path.exists():
        return
    print("deriving PSI-space values from logit cache", flush=True)
    y_logit = np.memmap(cache_dir / "y_logit.float32.mmap", mode="r", dtype=np.float32, shape=(nnz,))
    se_logit = np.memmap(cache_dir / "se_logit.float32.mmap", mode="r", dtype=np.float32, shape=(nnz,))
    y_psi = np.memmap(y_psi_path, mode="w+", dtype=np.float32, shape=(nnz,))
    se_psi = np.memmap(se_psi_path, mode="w+", dtype=np.float32, shape=(nnz,))
    for start in range(0, nnz, chunk_n):
        end = min(start + chunk_n, nnz)
        psi = expit(np.asarray(y_logit[start:end], dtype=np.float64)).astype(np.float32)
        y_psi[start:end] = psi
        se_psi[start:end] = (
            np.asarray(se_logit[start:end], dtype=np.float32)
            * np.maximum(psi * (1.0 - psi), 1e-8)
        ).astype(np.float32)
        y_psi.flush()
        se_psi.flush()
        print(f"psi-space cache {end:,}/{nnz:,}", flush=True)


def build_sparse_psi_cache(
    count_h5ad: Path,
    cache_dir: Path,
    *,
    chunk_n: int = 20_000_000,
    clip: float = 1e-4,
    force: bool = False,
    value_space: str = "logit",
) -> SparsePsiCache:
    """Build or load a full sparse PSI cache in logit or raw-PSI space."""
    value_space = value_space.lower()
    if value_space not in {"logit", "psi"}:
        raise ValueError("value_space must be 'logit' or 'psi'.")
    count_h5ad = Path(count_h5ad)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "sparse_psi_meta.json"

    if meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text())
        shape = tuple(meta["shape"])
        nnz = int(meta["nnz"])
        if value_space == "psi":
            _ensure_psi_value_cache(cache_dir, nnz, chunk_n=chunk_n)
        value_name = "psi" if value_space == "psi" else "logit"
        return SparsePsiCache(
            shape=shape,
            nnz=nnz,
            indptr=np.memmap(cache_dir / "indptr.int32.mmap", mode="r", dtype=np.int32, shape=(shape[0] + 1,)),
            indices=np.memmap(cache_dir / "indices.int32.mmap", mode="r", dtype=np.int32, shape=(nnz,)),
            y=np.memmap(cache_dir / f"y_{value_name}.float32.mmap", mode="r", dtype=np.float32, shape=(nnz,)),
            se_logit=np.memmap(cache_dir / f"se_{value_name}.float32.mmap", mode="r", dtype=np.float32, shape=(nnz,)),
            alpha=np.memmap(cache_dir / "alpha.float32.mmap", mode="r", dtype=np.float32, shape=(shape[1],)),
            beta=np.memmap(cache_dir / "beta.float32.mmap", mode="r", dtype=np.float32, shape=(shape[1],)),
            cache_dir=cache_dir,
            value_space=value_space,
        )

    with h5py.File(count_h5ad, "r") as f:
        K = _csr_group(f, COUNT_LAYER)
        shape = _shape(K)
        nnz = int(K["data"].shape[0])
        print("shape", shape, "nnz", f"{nnz:,}", flush=True)
        indptr = _copy_dataset_to_memmap(K["indptr"], cache_dir / "indptr.int32.mmap", np.int32, chunk_n=chunk_n)
        indices = _copy_dataset_to_memmap(K["indices"], cache_dir / "indices.int32.mmap", np.int32, chunk_n=chunk_n)

    alpha, beta = _compute_beta_moments(count_h5ad, cache_dir, chunk_n=chunk_n, clip=clip)
    alpha_mm = np.memmap(cache_dir / "alpha.float32.mmap", mode="w+", dtype=np.float32, shape=(shape[1],))
    beta_mm = np.memmap(cache_dir / "beta.float32.mmap", mode="w+", dtype=np.float32, shape=(shape[1],))
    alpha_mm[:] = alpha
    beta_mm[:] = beta
    alpha_mm.flush()
    beta_mm.flush()

    y = np.memmap(cache_dir / "y_logit.float32.mmap", mode="w+", dtype=np.float32, shape=(nnz,))
    se = np.memmap(cache_dir / "se_logit.float32.mmap", mode="w+", dtype=np.float32, shape=(nnz,))
    with h5py.File(count_h5ad, "r") as f:
        K = _csr_group(f, COUNT_LAYER)
        N = _csr_group(f, CLUSTER_LAYER)
        for start in range(0, nnz, chunk_n):
            end = min(start + chunk_n, nnz)
            cols = indices[start:end]
            k = K["data"][start:end].astype(np.float32)
            n = N["data"][start:end].astype(np.float32)
            a = k + alpha[cols]
            b = np.maximum(n - k, 0.0) + beta[cols]
            total = a + b
            psi = np.clip(a / total, clip, 1.0 - clip)
            var = (a * b) / np.maximum(total * total * (total + 1.0), 1e-12)
            y[start:end] = logit(psi).astype(np.float32)
            se[start:end] = (np.sqrt(np.maximum(var, 0.0)) / np.maximum(psi * (1.0 - psi), clip)).astype(np.float32)
            y.flush()
            se.flush()
            print(f"psi {end:,}/{nnz:,}", flush=True)

    meta_path.write_text(json.dumps({"shape": shape, "nnz": nnz}, indent=2))
    return build_sparse_psi_cache(count_h5ad, cache_dir, chunk_n=chunk_n, force=False, value_space=value_space)


@njit(parallel=True)
def lowrank_observed_values(indptr, indices, U, s, Vt, out):
    for i in prange(indptr.shape[0] - 1):
        start = indptr[i]
        end = indptr[i + 1]
        for p in range(start, end):
            j = indices[p]
            val = 0.0
            for r in range(s.shape[0]):
                val += U[i, r] * s[r] * Vt[r, j]
            out[p] = val


class SparsePlusLowRank(LinearOperator):
    """LinearOperator for C + U diag(s) Vt."""

    def __init__(self, correction: sparse.csr_matrix, U=None, s=None, Vt=None):
        self.correction = correction
        self.U = U
        self.s = s
        self.Vt = Vt
        super().__init__(dtype=np.float32, shape=correction.shape)

    def _matvec(self, x):
        x = np.asarray(x).reshape(-1)
        y = self.correction @ x
        if self.U is not None and self.s.size:
            y = y + self.U @ (self.s * (self.Vt @ x))
        return y

    def _rmatvec(self, x):
        x = np.asarray(x).reshape(-1)
        y = self.correction.T @ x
        if self.U is not None and self.s.size:
            y = y + self.Vt.T @ (self.s * (self.U.T @ x))
        return y

    def _matmat(self, X):
        Y = self.correction @ X
        if self.U is not None and self.s.size:
            Y = Y + self.U @ (self.s[:, None] * (self.Vt @ X))
        return Y

    def _rmatmat(self, X):
        Y = self.correction.T @ X
        if self.U is not None and self.s.size:
            Y = Y + self.Vt.T @ (self.s[:, None] * (self.U.T @ X))
        return Y


def randomized_svd_operator(
    op: LinearOperator,
    rank: int,
    *,
    oversample: int = 10,
    n_iter: int = 1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomized SVD for a LinearOperator using block matmat products."""
    rng = np.random.default_rng(seed)
    m, n = op.shape
    ell = min(min(m, n), rank + oversample)
    omega = rng.standard_normal((n, ell), dtype=np.float32)
    Q, _ = np.linalg.qr(op.matmat(omega), mode="reduced")
    Q = Q.astype(np.float32, copy=False)
    for _ in range(n_iter):
        Z, _ = np.linalg.qr(op.rmatmat(Q), mode="reduced")
        Q, _ = np.linalg.qr(op.matmat(Z.astype(np.float32, copy=False)), mode="reduced")
        Q = Q.astype(np.float32, copy=False)
    B = op.rmatmat(Q).T
    Ub, sv, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ Ub
    return U[:, :rank].astype(np.float32), sv[:rank].astype(np.float32), Vt[:rank, :].astype(np.float32)


def truncated_svd_operator(
    op: LinearOperator,
    rank: int,
    *,
    backend: str,
    oversample: int,
    n_iter: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a rank-capped SVD using ARPACK or randomized SVD."""
    backend = backend.lower()
    if backend == "arpack":
        u, sv, vt = svds(op, k=rank, which="LM", tol=1e-3, maxiter=300)
        order = np.argsort(sv)[::-1]
        return u[:, order].astype(np.float32), sv[order].astype(np.float32), vt[order, :].astype(np.float32)
    if backend == "rsvd":
        return randomized_svd_operator(
            op,
            rank,
            oversample=oversample,
            n_iter=n_iter,
            seed=seed,
        )
    raise ValueError("backend must be 'arpack' or 'rsvd'.")


def _noise_variance(
    cache: SparsePsiCache,
    *,
    noise_model: str,
    g: float,
    se_floor: float,
) -> np.ndarray | float:
    if noise_model == "homo":
        return max(float(g), 1e-6)
    se2 = np.maximum(np.asarray(cache.se_logit, dtype=np.float32), se_floor) ** 2
    if noise_model == "hetero":
        return se2
    if noise_model == "combined":
        return se2 + np.float32(g)
    raise ValueError("noise_model must be homo, hetero, or combined.")


def _se_floor_from_quantile(
    cache: SparsePsiCache,
    *,
    base_floor: float,
    quantile: float,
    max_sample: int = 5_000_000,
) -> float:
    """Optionally raise the SE floor to an empirical quantile.

    The heteroskedastic proximal step is limited by the largest precision,
    i.e. by the smallest allowed SE.  A few ultra-precise entries can therefore
    make global-step proximal gradient crawl.  Quantile flooring is a pragmatic
    precision cap; it should be reported because it changes the noise model.
    """
    if quantile <= 0.0:
        return float(base_floor)
    if not 0.0 < quantile < 1.0:
        raise ValueError("SPLICING_SPARSE_SE_FLOOR_QUANTILE must be in (0, 1).")
    stride = max(1, cache.nnz // max_sample)
    sample = np.asarray(cache.se_logit[::stride], dtype=np.float64)
    q_floor = float(np.quantile(sample, quantile))
    floor = max(float(base_floor), q_floor)
    print(
        {
            "value_space": cache.value_space,
            "base_se_floor": float(base_floor),
            "se_floor_quantile": float(quantile),
            "quantile_se_floor": q_floor,
            "effective_se_floor": floor,
            "hetero_step_if_used": floor * floor,
            "sample_n": int(sample.size),
        },
        flush=True,
    )
    return floor


def _max_precision(cache: SparsePsiCache, *, noise_model: str, g: float, se_floor: float) -> float:
    if noise_model == "homo":
        return 1.0 / max(float(g), 1e-6)
    return float(1.0 / (se_floor ** 2 + (g if noise_model == "combined" else 0.0)))


def _weighted_correction(
    cache: SparsePsiCache,
    x_obs: np.ndarray,
    *,
    noise_model: str,
    g: float,
    se_floor: float,
    step: float,
) -> np.ndarray:
    resid = np.asarray(cache.y, dtype=np.float32) - x_obs
    if noise_model == "homo":
        return (step * resid).astype(np.float32, copy=False)
    var = _noise_variance(cache, noise_model=noise_model, g=g, se_floor=se_floor)
    return (step * resid / var).astype(np.float32, copy=False)


def _estimate_g(cache: SparsePsiCache, x_obs: np.ndarray, *, se_floor: float) -> float:
    resid2 = (np.asarray(cache.y, dtype=np.float32) - x_obs) ** 2
    se2 = np.maximum(np.asarray(cache.se_logit, dtype=np.float32), se_floor) ** 2
    return float(np.maximum(np.mean(resid2 - se2), 1e-6))


def _estimate_gamma2(cache: SparsePsiCache, x_obs: np.ndarray) -> float:
    resid2 = (np.asarray(cache.y, dtype=np.float32) - x_obs) ** 2
    return float(np.maximum(np.mean(resid2), 1e-6))


def score_lowrank_diag_taylor(
    cache: SparsePsiCache,
    U: np.ndarray,
    s: np.ndarray,
    Vt: np.ndarray,
    *,
    lambda_val: float,
    noise_model: str,
    g: float,
    se_floor: float,
) -> dict[str, float]:
    """Full-observed-entry diagonal Taylor ELBO/LOO surrogate."""
    if s.size == 0:
        return {
            "diag_taylor_elbo": float("-inf"),
            "diag_taylor_loo_rmse": float("inf"),
            "obs_logit_rmse": float("inf"),
            "obs_rmse": float("inf"),
            "data_nll": float("inf"),
        }
    x_obs = np.empty(cache.nnz, dtype=np.float32)
    lowrank_observed_values(cache.indptr, cache.indices, U, s, Vt, x_obs)
    resid = np.asarray(cache.y, dtype=np.float32) - x_obs
    var = _noise_variance(cache, noise_model=noise_model, g=g, se_floor=se_floor)
    if np.isscalar(var):
        var_obs = np.float32(var)
        prec = np.float32(1.0 / var_obs)
        data_loss = 0.5 * float(np.sum(resid ** 2) * prec)
        data_log = 0.5 * cache.nnz * float(np.log(var_obs))
        prec_obs = np.full(cache.nnz, prec, dtype=np.float32)
    else:
        var_obs = np.asarray(var, dtype=np.float32)
        prec_obs = 1.0 / var_obs
        data_loss = 0.5 * float(np.sum((resid ** 2) * prec_obs))
        data_log = 0.5 * float(np.sum(np.log(var_obs)))

    inv_s = 1.0 / np.maximum(s.astype(np.float32), 1e-6)
    prior_diag = lambda_val * np.sum((Vt.T ** 2) * inv_s[None, :], axis=1)
    prior_at_obs = prior_diag[np.asarray(cache.indices)]
    post_prec_diag = prec_obs + prior_at_obs
    h = np.clip(prec_obs / np.maximum(post_prec_diag, 1e-12), 0.0, 0.99)
    loo_resid = resid / np.maximum(1.0 - h, 1e-3)
    logdet_diag = 0.5 * float(np.sum(np.log(np.maximum(post_prec_diag, 1e-12))))
    nuc = float(np.sum(s))
    loss = data_loss + data_log + logdet_diag + lambda_val * nuc
    return {
        "diag_taylor_elbo": -loss,
        "diag_taylor_loo_rmse": float(np.sqrt(np.mean(loo_resid ** 2))),
        "obs_logit_rmse": float(np.sqrt(np.mean(resid ** 2))),
        "obs_rmse": float(np.sqrt(np.mean(resid ** 2))),
        "data_nll": data_loss + data_log,
        "diag_logdet": logdet_diag,
        "nuclear_norm": nuc,
        "value_space": cache.value_space,
    }


def score_lowrank_isotropic_row_taylor(
    cache: SparsePsiCache,
    U: np.ndarray,
    s: np.ndarray,
    Vt: np.ndarray,
    *,
    lambda_val: float,
    noise_model: str,
    g: float,
    se_floor: float,
    delta: float | None = None,
    max_rows: int | None = None,
    seed: int = 0,
) -> dict[str, float]:
    """Row-wise Taylor score with low-rank plus isotropic row covariance.

    This keeps the exact row-wise Woodbury correction in the fitted right
    singular-vector subspace, while using an isotropic curvature outside that
    subspace.  It can be evaluated on a random row subset and scaled to the full
    matrix, which is often the only practical way to run it on the mouse atlas.
    """
    if s.size == 0:
        return {
            "iso_row_taylor_elbo": float("-inf"),
            "iso_row_taylor_loo_rmse": float("inf"),
            "iso_row_obs_logit_rmse": float("inf"),
            "iso_row_obs_rmse": float("inf"),
        }
    rng = np.random.default_rng(seed)
    m, n = cache.shape
    if max_rows is None or max_rows <= 0 or max_rows >= m:
        rows = np.arange(m, dtype=np.int64)
    else:
        rows = np.sort(rng.choice(m, size=max_rows, replace=False)).astype(np.int64)

    V = Vt.T.astype(np.float64, copy=False)
    s64 = np.maximum(s.astype(np.float64), 1e-8)
    if delta is None:
        delta = float(np.max(s64))
    delta = max(float(delta), 1e-8)
    tau = float(lambda_val / delta)
    c = lambda_val / s64 - tau
    active = np.abs(c) > 1e-10
    V_active = V[:, active]
    c_active = c[active]
    r = c_active.size
    c_inv = 1.0 / c_active if r else np.zeros(0)

    x_obs = np.empty(cache.nnz, dtype=np.float32)
    lowrank_observed_values(cache.indptr, cache.indices, U, s, Vt, x_obs)
    y_obs = np.asarray(cache.y, dtype=np.float32)
    se_obs = np.asarray(cache.se_logit, dtype=np.float32)

    data_term = 0.0
    logdet_term = 0.0
    loo_sse = 0.0
    rmse_sse = 0.0
    n_obs = 0
    for i in rows:
        start = int(cache.indptr[i])
        end = int(cache.indptr[i + 1])
        if end <= start:
            continue
        cols = np.asarray(cache.indices[start:end])
        resid = (y_obs[start:end] - x_obs[start:end]).astype(np.float64)
        if noise_model == "homo":
            var = np.full(end - start, max(float(g), 1e-6), dtype=np.float64)
        else:
            se = np.maximum(se_obs[start:end].astype(np.float64), se_floor)
            var = se ** 2
            if noise_model == "combined":
                var = var + max(float(g), 1e-6)
        prec = 1.0 / var
        rmse_sse += float(np.sum(resid ** 2))
        data_term += 0.5 * float(np.sum(np.log(var) + resid ** 2 * prec))

        m_inv = 1.0 / (tau + prec)
        logdet = n * np.log(tau) + float(np.sum(np.log1p(prec / tau)))
        if r:
            Vobs = V_active[cols, :]
            coeff = m_inv - 1.0 / tau
            S = (1.0 / tau) * np.eye(r) + Vobs.T @ (coeff[:, None] * Vobs)
            H = np.diag(c_inv) + S
            sign, logabs = np.linalg.slogdet(H)
            sign_c, logabs_c = np.linalg.slogdet(np.diag(c_inv))
            if sign > 0 and sign_c != 0:
                logdet += logabs - logabs_c
                K = np.linalg.inv(H)
                VK = Vobs @ K
                diag_sigma = m_inv - (m_inv ** 2) * np.sum(VK * Vobs, axis=1)
            else:
                diag_sigma = m_inv
        else:
            diag_sigma = m_inv
        logdet_term += 0.5 * float(logdet)
        h = np.clip(diag_sigma / var, 0.0, 0.99)
        loo = resid / np.maximum(1.0 - h, 1e-3)
        loo_sse += float(np.sum(loo ** 2))
        n_obs += end - start

    scale_rows = m / max(len(rows), 1)
    nuc = float(np.sum(s))
    estimated_loss = scale_rows * (data_term + logdet_term) + lambda_val * nuc
    return {
        "iso_row_taylor_elbo": -estimated_loss,
        "iso_row_taylor_loo_rmse": float(np.sqrt(loo_sse / max(n_obs, 1))),
        "iso_row_obs_logit_rmse": float(np.sqrt(rmse_sse / max(n_obs, 1))),
        "iso_row_obs_rmse": float(np.sqrt(rmse_sse / max(n_obs, 1))),
        "iso_row_data_logdet_loss_est": scale_rows * (data_term + logdet_term),
        "iso_row_n_rows": int(len(rows)),
        "iso_row_n_obs": int(n_obs),
        "iso_row_delta": float(delta),
        "iso_row_tau": float(tau),
        "value_space": cache.value_space,
    }


def sparse_soft_impute(
    cache: SparsePsiCache,
    *,
    lambda_val: float,
    rank: int,
    max_iter: int,
    out_dir: Path,
    noise_model: str = "homo",
    g_init: float = 1.0,
    se_floor: float = 0.25,
    svd_backend: str = "rsvd",
    svd_oversample: int = 10,
    svd_n_iter: int = 1,
    iso_row_score: bool = False,
    iso_row_max_rows: int | None = None,
    iso_row_delta: float | None = None,
    output_prefix: str = "sparse_full_softimpute",
    init_factors: Path | None = None,
    tol: float = 1e-4,
) -> pd.DataFrame:
    """Run sparse homoskedastic proximal gradient / SoftImpute."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    indptr = cache.indptr
    indices = cache.indices
    y_obs = cache.y
    shape = cache.shape
    noise_model = noise_model.lower()
    if noise_model not in {"homo", "hetero", "combined"}:
        raise ValueError("noise_model must be homo, hetero, or combined.")
    g = float(g_init)
    if init_factors is not None and Path(init_factors).exists():
        factors = np.load(init_factors)
        U = factors["U"].astype(np.float32)
        s = factors["s"].astype(np.float32)
        Vt = factors["Vt"].astype(np.float32)
        print(f"loaded init factors {init_factors} rank={s.size}", flush=True)
    else:
        U = None
        s = np.zeros(0, dtype=np.float32)
        Vt = None
    x_obs = np.zeros(cache.nnz, dtype=np.float32)
    rows = []
    for it in range(max_iter):
        t0 = time.perf_counter()
        if U is None or s.size == 0:
            x_obs.fill(0.0)
        else:
            lowrank_observed_values(indptr, indices, U.astype(np.float32), s.astype(np.float32), Vt.astype(np.float32), x_obs)
        obs_rmse = float(np.sqrt(np.mean((x_obs - y_obs) ** 2)))
        if noise_model == "combined" and it > 0:
            g = _estimate_g(cache, x_obs, se_floor=se_floor)
        fit_g = 1.0 if noise_model == "homo" else g
        L = _max_precision(cache, noise_model=noise_model, g=fit_g, se_floor=se_floor)
        step = 1.0 / max(L, 1e-12)
        corr_data = _weighted_correction(
            cache,
            x_obs,
            noise_model=noise_model,
            g=fit_g,
            se_floor=se_floor,
            step=step,
        )
        C = sparse.csr_matrix((corr_data, indices, indptr), shape=shape)
        op = SparsePlusLowRank(C, U, s, Vt)
        k = min(rank, min(shape) - 1)
        u, sv, vt = truncated_svd_operator(
            op,
            k,
            backend=svd_backend,
            oversample=svd_oversample,
            n_iter=svd_n_iter,
            seed=it + 104729,
        )
        threshold = lambda_val * step
        keep = sv > threshold
        sv_new = (sv[keep] - threshold).astype(np.float32)
        U_new = u[:, keep].astype(np.float32)
        Vt_new = vt[keep, :].astype(np.float32)
        rel = np.nan
        if s.size and sv_new.size == s.size:
            rel = float(np.linalg.norm(sv_new - s) / max(np.linalg.norm(s), 1.0))
        U, s, Vt = U_new, sv_new, Vt_new
        elapsed = time.perf_counter() - t0
        row = {
            "iter": it + 1,
            "noise_model": noise_model,
            "value_space": cache.value_space,
            "lambda": lambda_val,
            "se_floor": se_floor,
            "threshold": threshold,
            "g": g,
            "step": step,
            "svd_backend": svd_backend,
            "rank_cap": rank,
            "kept_rank": int(s.size),
            "obs_logit_rmse_before_update": obs_rmse,
            "top_sv": float(sv[0]) if sv.size else np.nan,
            "min_captured_sv": float(sv[-1]) if sv.size else np.nan,
            "rel_s_change": rel,
            "seconds": elapsed,
        }
        rows.append(row)
        print(row, flush=True)
        pd.DataFrame(rows).to_csv(out_dir / f"{output_prefix}_trace.csv", index=False)
        np.savez_compressed(out_dir / f"{output_prefix}_factors.npz", U=U, s=s, Vt=Vt)
        if s.size == 0 or (np.isfinite(rel) and rel < tol):
            break

    if U is not None and s.size:
        lowrank_observed_values(indptr, indices, U.astype(np.float32), s.astype(np.float32), Vt.astype(np.float32), x_obs)
        if noise_model == "homo":
            g = _estimate_gamma2(cache, x_obs)
        elif noise_model == "combined":
            g = _estimate_g(cache, x_obs, se_floor=se_floor)
        final_rmse = float(np.sqrt(np.mean((x_obs - y_obs) ** 2)))
        final_row = {
            "iter": "final",
            "noise_model": noise_model,
            "value_space": cache.value_space,
            "lambda": lambda_val,
            "se_floor": se_floor,
            "threshold": lambda_val / _max_precision(cache, noise_model=noise_model, g=g, se_floor=se_floor),
            "g": g,
            "step": 1.0 / _max_precision(cache, noise_model=noise_model, g=g, se_floor=se_floor),
            "svd_backend": svd_backend,
            "rank_cap": rank,
            "kept_rank": int(s.size),
            "obs_logit_rmse_before_update": final_rmse,
            "top_sv": np.nan,
            "min_captured_sv": np.nan,
            "rel_s_change": np.nan,
            "seconds": 0.0,
        }
        rows.append(final_row)
        print(final_row, flush=True)
        score = score_lowrank_diag_taylor(
            cache,
            U.astype(np.float32),
            s.astype(np.float32),
            Vt.astype(np.float32),
            lambda_val=lambda_val,
            noise_model=noise_model,
            g=g,
            se_floor=se_floor,
        )
        score.update({
            "noise_model": noise_model,
            "value_space": cache.value_space,
            "lambda": lambda_val,
            "se_floor": se_floor,
            "g": g,
            "rank": int(s.size),
        })
        if iso_row_score:
            t_score = time.perf_counter()
            iso_score = score_lowrank_isotropic_row_taylor(
                cache,
                U.astype(np.float32),
                s.astype(np.float32),
                Vt.astype(np.float32),
                lambda_val=lambda_val,
                noise_model=noise_model,
                g=g,
                se_floor=se_floor,
                delta=iso_row_delta,
                max_rows=iso_row_max_rows,
                seed=17,
            )
            iso_score["iso_row_seconds"] = time.perf_counter() - t_score
            score.update(iso_score)
        pd.DataFrame([score]).to_csv(out_dir / f"{output_prefix}_scores.csv", index=False)
        print(score, flush=True)

    pd.DataFrame(rows).to_csv(out_dir / f"{output_prefix}_trace.csv", index=False)
    np.savez_compressed(out_dir / f"{output_prefix}_factors.npz", U=U, s=s, Vt=Vt)
    return pd.DataFrame(rows)


def main() -> None:
    count_h5ad = Path(os.environ.get(
        "SPLICING_COUNT_H5AD",
        "/media/david/HDD/splicing_data/model_ready_aligned_splicing_data_20251009_024406.h5ad",
    ))
    cache_dir = Path(os.environ.get(
        "SPLICING_SPARSE_CACHE",
        "/media/david/HDD/splicing_data/matlap_sparse_cache",
    ))
    out_dir = Path(os.environ.get(
        "SPLICING_SPARSE_OUT",
        "analyses/splicing_proximal/outputs",
    ))
    chunk_n = env_int("SPLICING_CHUNK_NNZ", 20_000_000)
    rank = env_int("SPLICING_SPARSE_RANK", 20)
    max_iter = env_int("SPLICING_SPARSE_MAX_ITER", 5)
    lambda_val = env_float("SPLICING_SPARSE_LAMBDA", 1.0)
    lambda_grid = env_float_list("SPLICING_SPARSE_LAMBDA_GRID", "")
    noise_models = env_str_list("SPLICING_SPARSE_NOISE_MODELS", os.environ.get("SPLICING_SPARSE_NOISE_MODEL", "homo"))
    se_floor = env_float("SPLICING_SPARSE_SE_FLOOR", 0.25)
    se_floor_quantile = env_float("SPLICING_SPARSE_SE_FLOOR_QUANTILE", 0.0)
    g_init = env_float("SPLICING_SPARSE_G_INIT", 1.0)
    svd_backend = os.environ.get("SPLICING_SPARSE_SVD", "rsvd")
    svd_oversample = env_int("SPLICING_SPARSE_OVERSAMPLE", 10)
    svd_n_iter = env_int("SPLICING_SPARSE_POWER_ITER", 1)
    iso_row_score = bool(env_int("SPLICING_ISO_ROW_SCORE", 0))
    iso_row_max_rows = env_int("SPLICING_ISO_ROW_MAX_ROWS", 0)
    iso_row_delta_raw = os.environ.get("SPLICING_ISO_ROW_DELTA", "")
    iso_row_delta = float(iso_row_delta_raw) if iso_row_delta_raw else None
    force = bool(env_int("SPLICING_FORCE_CACHE", 0))
    output_prefix = os.environ.get("SPLICING_SPARSE_PREFIX", "sparse_full_softimpute")
    init_factors_raw = os.environ.get("SPLICING_SPARSE_INIT_FACTORS", "")
    init_factors = Path(init_factors_raw) if init_factors_raw else None
    score_only = bool(env_int("SPLICING_SCORE_ONLY", 0))
    value_space = os.environ.get("SPLICING_VALUE_SPACE", "logit").lower()

    cache = build_sparse_psi_cache(count_h5ad, cache_dir, chunk_n=chunk_n, force=force, value_space=value_space)
    se_floor = _se_floor_from_quantile(
        cache,
        base_floor=se_floor,
        quantile=se_floor_quantile,
    )
    if score_only:
        if init_factors is None or not init_factors.exists():
            raise ValueError("SPLICING_SCORE_ONLY=1 requires SPLICING_SPARSE_INIT_FACTORS.")
        factors = np.load(init_factors)
        U = factors["U"].astype(np.float32)
        s = factors["s"].astype(np.float32)
        Vt = factors["Vt"].astype(np.float32)
        rows = []
        for model in noise_models:
            for lam in (lambda_grid or [lambda_val]):
                x_obs = np.empty(cache.nnz, dtype=np.float32)
                lowrank_observed_values(cache.indptr, cache.indices, U, s, Vt, x_obs)
                if model == "homo":
                    g_score = _estimate_gamma2(cache, x_obs)
                elif model == "combined":
                    g_score = _estimate_g(cache, x_obs, se_floor=se_floor)
                else:
                    g_score = g_init
                score = score_lowrank_diag_taylor(
                    cache,
                    U,
                    s,
                    Vt,
                    lambda_val=lam,
                    noise_model=model,
                    g=g_score,
                    se_floor=se_floor,
                )
                score.update({
                    "noise_model": model,
                    "value_space": cache.value_space,
                    "lambda": lam,
                    "se_floor": se_floor,
                    "g": g_score,
                    "rank": int(s.size),
                })
                if iso_row_score:
                    t_score = time.perf_counter()
                    iso_score = score_lowrank_isotropic_row_taylor(
                        cache,
                        U,
                        s,
                        Vt,
                        lambda_val=lam,
                        noise_model=model,
                        g=g_score,
                        se_floor=se_floor,
                        delta=iso_row_delta,
                        max_rows=iso_row_max_rows,
                        seed=17,
                    )
                    iso_score["iso_row_seconds"] = time.perf_counter() - t_score
                    score.update(iso_score)
                rows.append(score)
                print(score, flush=True)
        scores = pd.DataFrame(rows)
        scores.to_csv(out_dir / f"{output_prefix}_scores.csv", index=False)
        return
    if not lambda_grid:
        lambda_grid = [lambda_val]
    all_scores = []
    for model in noise_models:
        for lam in lambda_grid:
            prefix = output_prefix
            if len(noise_models) > 1 or len(lambda_grid) > 1:
                prefix = f"{output_prefix}_{model}_lambda{lam:g}".replace(".", "p")
            trace = sparse_soft_impute(
                cache,
                lambda_val=lam,
                rank=rank,
                max_iter=max_iter,
                out_dir=out_dir,
                noise_model=model,
                g_init=g_init,
                se_floor=se_floor,
                svd_backend=svd_backend,
                svd_oversample=svd_oversample,
                svd_n_iter=svd_n_iter,
                iso_row_score=iso_row_score,
                iso_row_max_rows=iso_row_max_rows if iso_row_max_rows > 0 else None,
                iso_row_delta=iso_row_delta,
                output_prefix=prefix,
                init_factors=init_factors if len(noise_models) == 1 and len(lambda_grid) == 1 else None,
            )
            print(trace, flush=True)
            score_path = out_dir / f"{prefix}_scores.csv"
            if score_path.exists():
                all_scores.append(pd.read_csv(score_path))
    if all_scores:
        scores = pd.concat(all_scores, ignore_index=True)
        scores.to_csv(out_dir / f"{output_prefix}_all_scores.csv", index=False)
        print(scores, flush=True)


if __name__ == "__main__":
    main()
