Quickstart
==========

Basic usage
-----------

.. code-block:: python

    import jax.numpy as jnp
    import jax.random as jr
    from matlap import matlap_batched

    rng = jr.PRNGKey(0)
    Y = jr.normal(rng, (50, 100))   # noisy observations
    S = jnp.ones_like(Y)            # observation noise std devs

    result = matlap_batched(Y, S)
    X_hat = result.mean             # posterior mean (denoised matrix)

Missing data
------------

Set ``S[i, j] = jnp.inf`` (or a very large value) for missing entries:

.. code-block:: python

    import jax.numpy as jnp
    S_missing = S.at[0, 0].set(jnp.inf)
    result = matlap_batched(Y, S_missing)

Grid search over λ
------------------

Use ``matlap_grid_batched`` to score a range of regularisation strengths
with LOO, ELBO, or Rényi:

.. code-block:: python

    from matlap import matlap_grid_batched

    lam_grid = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    result, best_lam = matlap_grid_batched(Y, S, lam_grid=lam_grid,
                                           score_fn="loo")

Simulating from the NND prior
------------------------------

.. code-block:: python

    from matlap import sample_nnd

    rng = jr.PRNGKey(42)
    X_true, sigma = sample_nnd(rng, m=50, n=100, lam=0.05)
