"""Adaptive golden-ratio lambda search.

Provides a generic :func:`adaptive_lambda_search` loop that decouples the
search strategy from the fitting and scoring functions.  Convenience
warm-state extractors are provided for the standard model variants.

Typical usage::

    from matlap.adaptive import adaptive_lambda_search, iso_warm_state
    from matlap.scoring import make_renyi_scorer
    from matlap.core import matlap_lowrank_isotropic

    def fit_fn(Y, S, lam, **warm):
        return matlap_lowrank_isotropic(Y, S, lam, rank=50, **warm)

    best_lam, best_res, results = adaptive_lambda_search(
        Y, S,
        fit_fn=fit_fn,
        score_fn=make_renyi_scorer(alpha=0.5),
        extract_warm_state=iso_warm_state,
    )
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

GOLDEN_RATIO: float = (1.0 + 5.0 ** 0.5) / 2.0
"""Golden ratio φ ≈ 1.618; each adaptive step multiplies λ by 1/φ ≈ 0.618."""


def adaptive_lambda_search(
    Y: jax.Array,
    S: jax.Array,
    fit_fn: Callable,
    score_fn: Callable,
    *,
    extract_warm_state: Callable[[Any], dict] | None = None,
    lambda_start: float | None = None,
    lambda_min: float = 1e-4,
    patience: int = 2,
    verbose: bool = False,
) -> tuple[float, Any, list[tuple[float, Any]]]:
    """Generic adaptive golden-ratio lambda search.

    Starts at ``lambda_start`` and repeatedly multiplies λ by 1/φ (≈ 0.618).
    Stops after ``patience`` consecutive non-improving steps or when λ drops
    below ``lambda_min``.  Each CAVI run is warm-started from the previous
    result via ``extract_warm_state``.

    ``fit_fn`` and ``score_fn`` are callables with the following signatures::

        result   = fit_fn(Y, S, lam, **warm_kwargs)
        score    = score_fn(result, Y, S, lam)   # higher = better
        warm_kws = extract_warm_state(result)    # passed to next fit_fn call

    The scoring function receives the *already-fitted* result so that
    analytical scores (ELBO, LOO, Rényi) are cheap.  CV scorers may ignore
    the ``result`` and re-fit internally.

    Args:
        Y:                  Observed matrix, shape (m, n).
        S:                  Noise std matrix, shape (m, n). ``jnp.inf`` where missing.
        fit_fn:             ``(Y, S, lam, **warm_kwargs) -> result``
        score_fn:           ``(result, Y, S, lam) -> float`` — higher is better.
        extract_warm_state: ``result -> dict`` of kwargs forwarded to the next
                            ``fit_fn`` call for warm-starting. ``None`` disables
                            warm starts.
        lambda_start:       Starting λ. If None, auto-set to 100 × data heuristic
                            (``sqrt(max(m,n)) / sqrt(median precision)``).
        lambda_min:         Hard lower bound; search stops when λ < lambda_min.
        patience:           Stop after this many consecutive non-improving steps.
        verbose:            Print per-step progress.

    Returns:
        ``(best_lambda, best_result, results)`` where ``results`` is a list of
        ``(lambda, result)`` pairs sorted ascending by λ.
    """
    if patience < 1:
        raise ValueError(f"patience must be >= 1, got {patience}")

    Y = jnp.asarray(Y, dtype=jnp.float32)
    S = jnp.asarray(S, dtype=jnp.float32)

    if lambda_start is None:
        prec = jnp.where(jnp.isfinite(S), 1.0 / S ** 2, 0.0)
        med_prec = float(jnp.median(prec[prec > 0]))
        heuristic = float(jnp.sqrt(max(Y.shape))) / max(med_prec ** 0.5, 1e-12)
        lambda_start = 100.0 * heuristic

    reduction = 1.0 / GOLDEN_RATIO
    warm_kwargs: dict = {}
    results: list[tuple[float, Any]] = []
    best_score = -float("inf")
    best_lam = float(lambda_start)
    best_res: Any = None
    no_improve = 0
    lam = float(lambda_start)

    while lam >= lambda_min:
        result = fit_fn(Y, S, lam, **warm_kwargs)
        if extract_warm_state is not None:
            warm_kwargs = extract_warm_state(result)
        score = float(score_fn(result, Y, S, lam))
        results.append((lam, result))

        improved = score > best_score
        if improved:
            best_score = score
            best_lam = lam
            best_res = result
            no_improve = 0
        else:
            no_improve += 1

        if verbose:
            marker = "  *" if improved else ""
            print(
                f"  lambda={lam:.4f}  score={score:.4f}  "
                f"no_improve={no_improve}{marker}"
            )

        if no_improve >= patience:
            break

        lam *= reduction

    results.sort(key=lambda x: x[0])
    return best_lam, best_res, results


# ---------------------------------------------------------------------------
# Warm-state extractors for standard model types
# ---------------------------------------------------------------------------


def iso_warm_state(result: Any) -> dict:
    """Extract warm-start kwargs from a :class:`~matlap.core.LowRankIsotropicResult`.

    Returns:
        dict with ``V_r_init``, ``d_r_init``, and ``delta_init`` keys.
    """
    return {
        "V_r_init": result.V_r,
        "d_r_init": result.d_r,
        "delta_init": result.delta,
    }


def lowrank_warm_state(result: Any) -> dict:
    """Extract warm-start kwargs from a :class:`~matlap.core.LowRankCAVIResult`.

    Returns:
        dict with ``V_r_init`` and ``d_r_init`` keys.
    """
    return {
        "V_r_init": result.V_r,
        "d_r_init": result.d_r,
    }


def batched_warm_state(result: Any) -> dict:
    """Extract warm-start kwargs from a :class:`~matlap.core.BatchedCAVIResult`.

    Returns:
        dict with ``mu_init`` key (posterior mean to warm-start the next run).
    """
    return {"mu_init": result.mu}
