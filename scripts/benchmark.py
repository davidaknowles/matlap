#!/usr/bin/env python
"""
Benchmark: imputation performance on simulated low-rank data.

Simulates a rank-3 50×20 matrix, masks ~20% of entries as a held-out test set,
fits six methods on the training entries, and reports test RMSE and wall-clock
runtime.  Results are averaged over ``N_SEEDS`` random seeds.

Usage
-----
    python scripts/benchmark.py [--seeds N] [--rows M] [--cols K]

Methods compared
----------------
- matlap          : CAVI with automatic λ (empirical Bayes)
- matlap_grid     : CAVI with grid-search λ (best ELBO)
- proximal_cv     : Nuclear-norm FISTA with entry-wise CV λ selection
- vi_diagonal     : Numpyro SVI, fully-factorised Gaussian guide
- vi_row_mvn      : Numpyro SVI, row-MVN guide
- vi_matrix_normal: Numpyro SVI, Matrix Normal guide
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np

from matlap import matlap, matlap_grid
from matlap.proximal import proximal_cv
from matlap.vi import fit_vi

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_SEEDS = 5
DEFAULT_M = 50
DEFAULT_N = 20
RANK = 3
NOISE_LEVEL = 0.5
MISSING_FRAC = 0.20

LAMBDA_GRID = jnp.logspace(-1, 2, 15)
N_CV_FOLDS = 5
VI_N_STEPS = 3000
VI_LR = 3e-3

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def simulate(seed: int, m: int, n: int) -> tuple[jnp.Array, jnp.Array, jnp.Array]:
    """Return (X_true, Y_train, S_train) with ~MISSING_FRAC entries masked."""
    key = jax.random.PRNGKey(seed)
    U = jax.random.normal(key, (m, RANK)) / RANK ** 0.5
    V = jax.random.normal(jax.random.fold_in(key, 1), (n, RANK)) / RANK ** 0.5
    X_true = U @ V.T

    # Heteroscedastic noise: s ~ Uniform(0.3, 0.7)
    s_vals = 0.3 + 0.4 * jax.random.uniform(jax.random.fold_in(key, 2), (m, n))
    Y_full = X_true + s_vals * jax.random.normal(jax.random.fold_in(key, 3), (m, n))

    # Random ~20% test mask
    test_mask = jax.random.uniform(jax.random.fold_in(key, 4), (m, n)) < MISSING_FRAC
    S_train = jnp.where(test_mask, jnp.inf, s_vals)
    return X_true, Y_full, S_train


def rmse(pred: jnp.Array, truth: jnp.Array, mask: jnp.Array) -> float:
    """RMSE over masked entries only."""
    diff = jnp.where(mask, pred - truth, 0.0)
    n_obs = jnp.sum(mask)
    return float(jnp.sqrt(jnp.sum(diff ** 2) / n_obs))


def fit_timed(fn, *args, **kwargs):
    """Run fn(*args, **kwargs) and return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    # Force JAX to complete pending computation
    if hasattr(result, "mu"):
        _ = result.mu.block_until_ready()
    elif hasattr(result, "X"):
        _ = result.X.block_until_ready()
    return result, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_benchmark(seeds: int, m: int, n: int) -> None:
    stats: dict[str, list[float]] = defaultdict(list)

    for seed in range(seeds):
        print(f"\n=== Seed {seed} ===")
        X_true, Y_full, S_train = simulate(seed, m, n)
        test_mask = ~jnp.isfinite(S_train)

        methods = {
            "matlap": lambda: matlap(Y_full, S_train),
            "matlap_grid": lambda: matlap_grid(Y_full, S_train, LAMBDA_GRID).best_result,
            "proximal_cv": lambda: proximal_cv(
                Y_full, S_train, LAMBDA_GRID, n_folds=N_CV_FOLDS
            )[1],
            "vi_diagonal": lambda: fit_vi(
                Y_full, S_train, guide_type="diagonal",
                n_steps=VI_N_STEPS, lr=VI_LR
            ),
            "vi_row_mvn": lambda: fit_vi(
                Y_full, S_train, guide_type="row_mvn",
                n_steps=VI_N_STEPS, lr=VI_LR
            ),
            "vi_matrix_normal": lambda: fit_vi(
                Y_full, S_train, guide_type="matrix_normal",
                n_steps=VI_N_STEPS, lr=VI_LR
            ),
        }

        for name, fn in methods.items():
            try:
                result, elapsed = fit_timed(fn)
                mu = result.mu if hasattr(result, "mu") else result.X
                test_rmse = rmse(mu, X_true, test_mask)
                stats[f"{name}_rmse"].append(test_rmse)
                stats[f"{name}_time"].append(elapsed)
                conv = getattr(result, "converged", None)
                print(f"  {name:20s}  RMSE={test_rmse:.4f}  t={elapsed:.1f}s"
                      + (f"  conv={conv}" if conv is not None else ""))
            except Exception as exc:  # noqa: BLE001
                print(f"  {name:20s}  ERROR: {exc}")

    # Summary table
    print("\n" + "=" * 72)
    print(f"{'Method':<22} {'Test RMSE':>14} {'Runtime (s)':>14}")
    print("-" * 72)
    for name in ["matlap", "matlap_grid", "proximal_cv",
                 "vi_diagonal", "vi_row_mvn", "vi_matrix_normal"]:
        rmse_vals = stats.get(f"{name}_rmse", [])
        time_vals = stats.get(f"{name}_time", [])
        if rmse_vals:
            r_mu = np.mean(rmse_vals)
            r_sd = np.std(rmse_vals)
            t_mu = np.mean(time_vals)
            t_sd = np.std(time_vals)
            print(f"{name:<22} {r_mu:.4f} ± {r_sd:.4f}   {t_mu:6.1f} ± {t_sd:.1f}")
        else:
            print(f"{name:<22} {'FAILED':>14}")
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                        help=f"Number of random seeds (default {DEFAULT_SEEDS})")
    parser.add_argument("--rows", type=int, default=DEFAULT_M,
                        help=f"Matrix rows m (default {DEFAULT_M})")
    parser.add_argument("--cols", type=int, default=DEFAULT_N,
                        help=f"Matrix columns n (default {DEFAULT_N})")
    args = parser.parse_args()
    run_benchmark(args.seeds, args.rows, args.cols)
