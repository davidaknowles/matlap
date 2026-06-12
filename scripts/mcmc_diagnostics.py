#!/usr/bin/env python
"""MCMC trace diagnostics for lambda-sampling proximal MALA.

The script runs several independent chains on one simulated NND instance and
reports scalar diagnostics for lambda, nuclear norm, and log posterior.  It is
intended to answer: did warmup settle, and are post-warmup chains long enough?
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

from matlap import mcmc_proximal_mala, proximal_gradient, sample_nnd


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


M = _env_int("MCMC_DIAG_M", 60)
N = _env_int("MCMC_DIAG_N", 60)
LAMBDA_TRUE = _env_float("MCMC_DIAG_LAMBDA_TRUE", 0.05)
SIGMA_NOISE = _env_float("MCMC_DIAG_SIGMA_NOISE", 1.0)
N_CHAINS = _env_int("MCMC_DIAG_CHAINS", 4)
N_WARMUP = _env_int("MCMC_DIAG_WARMUP", 200)
N_SAMPLES = _env_int("MCMC_DIAG_SAMPLES", 500)
SEED = _env_int("MCMC_DIAG_SEED", 0)
PROPOSAL_SVD_RANK = _env_int("MCMC_DIAG_PROPOSAL_SVD_RANK", 90)
PROPOSAL_SVD_N_ITER = _env_int("MCMC_DIAG_PROPOSAL_SVD_N_ITER", 1)
PROPOSAL_SVD_OVERSAMPLE = _env_int("MCMC_DIAG_PROPOSAL_SVD_OVERSAMPLE", 5)
INIT_ITER = _env_int("MCMC_DIAG_INIT_ITER", 60)
RM_STEP = _env_float("MCMC_DIAG_RM_STEP", 0.5)
STEP_SIZE_INIT_ENV = os.environ.get("MCMC_DIAG_STEP_SIZE_INIT")
STEP_SIZE_INIT = None if STEP_SIZE_INIT_ENV in {None, ""} else float(STEP_SIZE_INIT_ENV)
LAMBDA_LOG_STEP_ENV = os.environ.get("MCMC_DIAG_LAMBDA_LOG_STEP")
LAMBDA_LOG_STEP = None if LAMBDA_LOG_STEP_ENV in {None, ""} else float(LAMBDA_LOG_STEP_ENV)
OUTPUT_PREFIX = os.environ.get("MCMC_DIAG_OUTPUT", "results/mcmc_diagnostics")


def _device():
    gpus = jax.devices("gpu")
    return gpus[0] if gpus else jax.devices()[0]


DEVICE = _device()


def _to_device(x):
    return jax.device_put(jnp.asarray(x, dtype=jnp.float32), DEVICE)


def _split_rhat(chains: np.ndarray) -> float:
    """Split-Rhat for an array with shape (chains, draws)."""
    chains = np.asarray(chains, dtype=float)
    n_chains, n_draws = chains.shape
    half = n_draws // 2
    if n_chains < 2 or half < 2:
        return float("nan")
    split = np.concatenate([chains[:, :half], chains[:, -half:]], axis=0)
    m, n = split.shape
    chain_means = split.mean(axis=1)
    chain_vars = split.var(axis=1, ddof=1)
    within = chain_vars.mean()
    if within <= 0 or not np.isfinite(within):
        return float("nan")
    between = n * chain_means.var(ddof=1)
    var_hat = ((n - 1) / n) * within + between / n
    return float(np.sqrt(var_hat / within))


def _rough_ess(chains: np.ndarray) -> float:
    """Simple positive-sequence ESS estimate for scalar chains."""
    chains = np.asarray(chains, dtype=float)
    n_chains, n_draws = chains.shape
    if n_draws < 4:
        return float("nan")
    centered = chains - chains.mean(axis=1, keepdims=True)
    var = np.mean(centered ** 2)
    if var <= 0 or not np.isfinite(var):
        return float("nan")

    max_lag = min(n_draws - 1, 200)
    rhos = []
    for lag in range(1, max_lag + 1):
        cov = np.mean(centered[:, :-lag] * centered[:, lag:])
        rho = float(cov / var)
        if rho <= 0:
            break
        rhos.append(rho)
    tau = 1.0 + 2.0 * float(np.sum(rhos))
    return float(n_chains * n_draws / tau)


def _running_stability(draws: np.ndarray) -> float:
    """Relative difference between first-half and second-half means."""
    draws = np.asarray(draws, dtype=float)
    half = draws.shape[1] // 2
    if half == 0:
        return float("nan")
    first = draws[:, :half].mean()
    second = draws[:, half:].mean()
    return float(abs(second - first) / max(abs(second), 1e-12))


def _summarize(name: str, chains: np.ndarray) -> dict[str, float | str]:
    flat = np.asarray(chains, dtype=float).reshape(-1)
    return {
        "variable": name,
        "mean": float(np.mean(flat)),
        "sd": float(np.std(flat)),
        "median": float(np.median(flat)),
        "q05": float(np.quantile(flat, 0.05)),
        "q95": float(np.quantile(flat, 0.95)),
        "split_rhat": _split_rhat(chains),
        "rough_ess": _rough_ess(chains),
        "running_half_rel_change": _running_stability(chains),
    }


def _initial_lambda(Y, S) -> float:
    obs_mask = jnp.isfinite(S)
    X0 = jnp.where(obs_mask, Y, 0.0)
    nuc_X0 = jnp.linalg.svd(X0, compute_uv=False).sum()
    return float((Y.shape[0] * Y.shape[1]) / jnp.maximum(nuc_X0, 1e-10))


def _run_chain(Y, S, chain_id: int):
    rank = PROPOSAL_SVD_RANK
    max_approx_rank = min(M, N) - PROPOSAL_SVD_OVERSAMPLE - 1
    proposal_rank = None if rank <= 0 else min(rank, max_approx_rank)
    if proposal_rank is not None and proposal_rank <= 0:
        proposal_rank = None
    lambda_init = _initial_lambda(Y, S)
    x_init = None
    if INIT_ITER > 0:
        init = proximal_gradient(Y, S, lambda_init, max_iter=INIT_ITER, tol=1e-5)
        x_init = init.X
    return mcmc_proximal_mala(
        Y,
        S,
        lambda_val=lambda_init,
        x_init=x_init,
        sample_lambda=True,
        n_warmup=N_WARMUP,
        n_samples=N_SAMPLES,
        step_size_init=STEP_SIZE_INIT,
        rm_step=RM_STEP,
        proposal_svd_rank=proposal_rank,
        proposal_svd_n_iter=PROPOSAL_SVD_N_ITER,
        proposal_svd_oversample=PROPOSAL_SVD_OVERSAMPLE,
        lambda_log_step=LAMBDA_LOG_STEP,
        return_trace=True,
        key=jax.random.PRNGKey(SEED + 1000 * chain_id),
    )


def main():
    rng = np.random.default_rng(SEED)
    X_true, _ = sample_nnd(rng, M, N, LAMBDA_TRUE)
    Y_np = X_true + rng.standard_normal((M, N)) * SIGMA_NOISE
    S_np = np.full((M, N), SIGMA_NOISE, dtype=np.float32)
    Y = _to_device(Y_np)
    S = _to_device(S_np)
    X_true_j = _to_device(X_true)

    print("MCMC diagnostics")
    print(f"  device={DEVICE}")
    print(f"  shape={M}x{N}, lambda_true={LAMBDA_TRUE}, sigma_noise={SIGMA_NOISE}")
    print(f"  chains={N_CHAINS}, warmup={N_WARMUP}, samples={N_SAMPLES}")
    print(
        f"  proposal_svd_rank={PROPOSAL_SVD_RANK}, "
        f"n_iter={PROPOSAL_SVD_N_ITER}, oversample={PROPOSAL_SVD_OVERSAMPLE}, "
        f"init_iter={INIT_ITER}, rm_step={RM_STEP}, "
        f"step_size_init={STEP_SIZE_INIT}, lambda_log_step={LAMBDA_LOG_STEP}"
    )

    results = []
    for chain_id in range(N_CHAINS):
        t0 = time.perf_counter()
        res = _run_chain(Y, S, chain_id)
        res.mu.block_until_ready()
        elapsed = time.perf_counter() - t0
        rmse = float(jnp.sqrt(jnp.mean((res.mu - X_true_j) ** 2)))
        print(
            f"  chain {chain_id}: lambda_bar={res.lambda_bar:.4g} "
            f"accept={res.accept_rate:.3f} rmse={rmse:.4f} time={elapsed:.1f}s"
        )
        results.append((res, rmse, elapsed))

    traces = {
        "lambda": np.stack([np.asarray(r.lambda_trace) for r, _, _ in results]),
        "nuclear": np.stack([np.asarray(r.nuclear_trace) for r, _, _ in results]),
        "logpost": np.stack([np.asarray(r.logpost_trace) for r, _, _ in results]),
        "mala_accept": np.stack([np.asarray(r.accept_trace) for r, _, _ in results]),
        "lambda_accept": np.stack([np.asarray(r.lambda_accept_trace) for r, _, _ in results]),
    }
    warmup_traces = {
        "warmup_lambda": np.stack([np.asarray(r.warmup_lambda_trace) for r, _, _ in results]),
        "warmup_nuclear": np.stack([np.asarray(r.warmup_nuclear_trace) for r, _, _ in results]),
        "warmup_logpost": np.stack([np.asarray(r.warmup_logpost_trace) for r, _, _ in results]),
        "warmup_step_size": np.stack([np.asarray(r.warmup_step_size_trace) for r, _, _ in results]),
    }

    rows = [_summarize(name, vals) for name, vals in traces.items()]
    rows.extend(_summarize(name, vals) for name, vals in warmup_traces.items())

    prefix = Path(OUTPUT_PREFIX)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    md_path = prefix.with_suffix(".md")

    fieldnames = [
        "variable", "mean", "sd", "median", "q05", "q95", "split_rhat",
        "rough_ess", "running_half_rel_change",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# MCMC Diagnostics",
        "",
        f"- device: `{DEVICE}`",
        f"- shape: `{M}x{N}`",
        f"- lambda_true: `{LAMBDA_TRUE}`",
        f"- sigma_noise: `{SIGMA_NOISE}`",
        f"- chains: `{N_CHAINS}`",
        f"- warmup: `{N_WARMUP}`",
        f"- samples: `{N_SAMPLES}`",
        f"- proposal_svd_rank: `{PROPOSAL_SVD_RANK}`",
        f"- proposal_svd_n_iter: `{PROPOSAL_SVD_N_ITER}`",
        f"- proposal_svd_oversample: `{PROPOSAL_SVD_OVERSAMPLE}`",
        f"- init_iter: `{INIT_ITER}`",
        f"- rm_step: `{RM_STEP}`",
        f"- step_size_init: `{STEP_SIZE_INIT}`",
        f"- lambda_log_step: `{LAMBDA_LOG_STEP}`",
        "",
        "| variable | mean | sd | Rhat | rough ESS | half-change |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variable']} | {row['mean']:.6g} | {row['sd']:.6g} | "
            f"{row['split_rhat']:.4f} | {row['rough_ess']:.1f} | "
            f"{row['running_half_rel_change']:.4g} |"
        )
    lines.extend([
        "",
        "## Per-chain",
        "",
        "| chain | lambda_bar | MALA accept | lambda accept | RMSE | time s |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for chain_id, (res, rmse, elapsed) in enumerate(results):
        lambda_accept = float(np.mean(np.asarray(res.lambda_accept_trace)))
        lines.append(
            f"| {chain_id} | {res.lambda_bar:.6g} | {res.accept_rate:.4f} | "
            f"{lambda_accept:.4f} | {rmse:.6f} | {elapsed:.2f} |"
        )

    md_path.write_text("\n".join(lines) + "\n")
    print(f"\nSaved diagnostics CSV to {csv_path}")
    print(f"Saved diagnostics Markdown to {md_path}")


if __name__ == "__main__":
    main()
