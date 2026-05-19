"""
General entry-wise K-fold cross-validation for lambda selection.

Works with any fitting function that follows the protocol::

    result = fit_fn(Y, S, lambda_val, **fit_kwargs)

where ``Y`` and ``S`` are the (m, n) data/noise arrays (``S[i,j]=inf``
signals a missing entry) and ``lambda_val`` is a scalar regularisation
strength.  The result must expose the matrix estimate either as a
``.mu`` attribute (CAVI / VI style) or a ``.X`` attribute (proximal style),
or you can supply a custom ``get_mu`` extractor.

Cross-validation is performed only over the *observed* entries
(``S[i,j] < inf``).  Missing entries are never used to select lambda.
"""

from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jnp
import numpy as np


def cv_lambda(
    Y,
    S,
    lambda_grid,
    fit_fn: Callable,
    get_mu: Callable | None = None,
    *,
    n_folds: int = 5,
    verbose: bool = False,
    **fit_kwargs,
) -> tuple[float, Any]:
    """Select regularisation strength by entry-wise K-fold cross-validation.

    For each value in ``lambda_grid``, fits the model K times (each time
    holding out one fold of the observed entries) and records the average
    prediction error on the held-out fold.  Returns the lambda with the
    lowest average error together with a final model refitted on **all**
    observed entries.

    The folds are constructed from the *observed* entries only; missing
    entries (``S[i,j] = inf``) are excluded from both training and
    evaluation.

    Args:
        Y:           Observed matrix, shape (m, n).  Values at missing entries
                     are ignored.
        S:           Known standard errors, shape (m, n).  ``jnp.inf`` marks
                     missing entries.
        lambda_grid: 1-D iterable of lambda values to evaluate.
        fit_fn:      Callable with signature
                     ``fit_fn(Y, S, lambda_val, **fit_kwargs) -> result``.
                     Missing entries in S are passed as ``jnp.inf``.
        get_mu:      Callable ``get_mu(result) -> jax.Array`` of shape (m, n).
                     If *None*, the function tries ``result.mu`` then
                     ``result.X`` in that order.
        n_folds:     Number of CV folds (default 5).
        verbose:     Print per-fold and per-lambda progress.
        **fit_kwargs: Extra keyword arguments forwarded to ``fit_fn``.

    Returns:
        ``(best_lambda, final_result)`` where ``final_result`` is the output
        of ``fit_fn`` called on **all** observed entries with ``best_lambda``.
    """
    Y = jnp.asarray(Y, dtype=jnp.float32)
    S = jnp.asarray(S, dtype=jnp.float32)
    lambda_grid = list(lambda_grid)

    if get_mu is None:
        def get_mu(result):
            if hasattr(result, "mu"):
                return result.mu
            if hasattr(result, "X"):
                return result.X
            raise AttributeError(
                "Result has neither .mu nor .X attribute; "
                "supply a custom get_mu extractor."
            )

    # Collect observed entry coordinates
    obs_mask_np = np.array(jnp.isfinite(S) & jnp.isfinite(Y))
    obs_idx = np.argwhere(obs_mask_np)  # (n_obs, 2)
    n_obs = len(obs_idx)

    if n_obs == 0:
        raise ValueError("No observed entries (all S are inf).")
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2.")
    if n_obs < n_folds:
        raise ValueError(
            f"Fewer observed entries ({n_obs}) than folds ({n_folds})."
        )

    # Shuffle and assign fold IDs
    rng = np.random.default_rng(42)
    perm = rng.permutation(n_obs)
    fold_ids = perm % n_folds  # entry index -> fold

    n_lambdas = len(lambda_grid)
    cv_mse = np.zeros(n_lambdas)

    for fold in range(n_folds):
        val_pos = obs_idx[perm[fold_ids == fold]]    # held-out (row, col) pairs
        train_pos = obs_idx[perm[fold_ids != fold]]  # training (row, col) pairs  # noqa: F841

        # Mark held-out entries as missing during training
        S_train = S.at[val_pos[:, 0], val_pos[:, 1]].set(jnp.inf)

        for k, lam in enumerate(lambda_grid):
            res = fit_fn(Y, S_train, float(lam), **fit_kwargs)
            mu_hat = get_mu(res)

            # Normalised MSE on held-out entries (scaled by noise std)
            i_val, j_val = val_pos[:, 0], val_pos[:, 1]
            pred = mu_hat[i_val, j_val]
            true = Y[i_val, j_val]
            s_val = S[i_val, j_val]
            mse = float(jnp.mean(((pred - true) / s_val) ** 2))
            cv_mse[k] += mse / n_folds

        if verbose:
            print(f"  fold {fold + 1}/{n_folds} done")

    best_k = int(np.argmin(cv_mse))
    best_lambda = float(lambda_grid[best_k])

    if verbose:
        for k, lam in enumerate(lambda_grid):
            marker = " <-- best" if k == best_k else ""
            print(f"  lambda={float(lam):.4g}  cv_mse={cv_mse[k]:.6f}{marker}")

    # Final fit on all observed entries
    final_result = fit_fn(Y, S, best_lambda, **fit_kwargs)
    return best_lambda, final_result


def cv_score_single(
    Y,
    S,
    lam: float,
    fit_fn: Callable,
    get_mu: Callable | None = None,
    *,
    n_folds: int = 3,
    **fit_kwargs,
) -> float:
    """Cross-validation score for a **single** lambda value (higher = better).

    Runs K-fold CV with the given ``fit_fn`` at the fixed ``lam`` and returns
    the negative mean squared prediction error averaged over held-out folds.
    A higher (less negative) value indicates a better fit.

    Args:
        Y:          Observed matrix, shape (m, n).
        S:          Noise std matrix, shape (m, n). ``jnp.inf`` marks missing.
        lam:        Lambda value to evaluate.
        fit_fn:     ``fit_fn(Y, S, lam, **fit_kwargs) -> result``
        get_mu:     Extracts the ``(m, n)`` estimate from ``result``.
                    Defaults to ``result.mu`` then ``result.X``.
        n_folds:    Number of CV folds (default 3).
        **fit_kwargs: Forwarded to ``fit_fn``.

    Returns:
        Negative mean normalised MSE over held-out folds (higher = better).
    """
    Y = jnp.asarray(Y, dtype=jnp.float32)
    S = jnp.asarray(S, dtype=jnp.float32)

    if get_mu is None:
        def get_mu(result):
            if hasattr(result, "mu"):
                return result.mu
            if hasattr(result, "X"):
                return result.X
            raise AttributeError(
                "Result has neither .mu nor .X; supply a custom get_mu extractor."
            )

    obs_mask_np = np.array(jnp.isfinite(S) & jnp.isfinite(Y))
    obs_idx = np.argwhere(obs_mask_np)
    n_obs = len(obs_idx)

    if n_obs == 0:
        raise ValueError("No observed entries (all S are inf).")
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2.")

    rng = np.random.default_rng(42)
    perm = rng.permutation(n_obs)
    fold_ids = perm % n_folds

    total_mse = 0.0
    for fold in range(n_folds):
        val_pos = obs_idx[perm[fold_ids == fold]]
        S_train = S.at[val_pos[:, 0], val_pos[:, 1]].set(jnp.inf)
        res = fit_fn(Y, S_train, float(lam), **fit_kwargs)
        mu_hat = get_mu(res)
        i_val, j_val = val_pos[:, 0], val_pos[:, 1]
        mse = float(jnp.mean(((mu_hat[i_val, j_val] - Y[i_val, j_val]) / S[i_val, j_val]) ** 2))
        total_mse += mse / n_folds

    return -total_mse


def make_cv_scorer(
    fit_fn: Callable,
    n_folds: int = 3,
    get_mu: Callable | None = None,
    **fit_kwargs,
) -> Callable:
    """Return a CV scorer compatible with :func:`~matlap.adaptive.adaptive_lambda_search`.

    The returned ``score_fn(result, Y, S, lam)`` ignores ``result`` and instead
    re-fits ``fit_fn`` from scratch across ``n_folds`` folds.  This is expensive
    (K fits per λ step) but unbiased.

    Args:
        fit_fn:     ``fit_fn(Y, S, lam, **fit_kwargs) -> result``
        n_folds:    Number of CV folds (default 3).
        get_mu:     Custom estimator extractor; defaults to ``.mu`` / ``.X``.
        **fit_kwargs: Extra kwargs forwarded to ``fit_fn``.

    Returns:
        ``score_fn(result, Y, S, lam) -> float`` (higher = better).
    """
    def score(result: Any, Y, S, lam: float) -> float:
        return cv_score_single(Y, S, lam, fit_fn, get_mu=get_mu, n_folds=n_folds, **fit_kwargs)
    return score
