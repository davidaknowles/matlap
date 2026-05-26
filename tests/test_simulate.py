"""Tests for matlap.simulate — NND sampler."""

from __future__ import annotations

import numpy as np
import pytest

from matlap.simulate import sample_nnd


class TestSampleNND:
    """Tests for the NND SVD sampler."""

    def test_output_shape(self):
        rng = np.random.default_rng(0)
        X, sigma = sample_nnd(rng, m=30, n=20, lam=1.0)
        assert X.shape == (30, 20)
        assert sigma.shape == (20,)

    def test_square_shape(self):
        rng = np.random.default_rng(1)
        X, sigma = sample_nnd(rng, m=25, n=25, lam=0.5)
        assert X.shape == (25, 25)
        assert sigma.shape == (25,)

    def test_singular_values_descending(self):
        rng = np.random.default_rng(2)
        _, sigma = sample_nnd(rng, m=40, n=30, lam=1.0)
        assert np.all(sigma[:-1] >= sigma[1:]), "SVs should be sorted descending"

    def test_singular_values_positive(self):
        rng = np.random.default_rng(3)
        _, sigma = sample_nnd(rng, m=20, n=20, lam=0.1)
        assert np.all(sigma > 0)

    def test_singular_values_match_svd(self):
        """SVD of X should recover the sampled singular values."""
        rng = np.random.default_rng(4)
        X, sigma = sample_nnd(rng, m=30, n=20, lam=0.5)
        _, svd_vals, _ = np.linalg.svd(X, full_matrices=False)
        np.testing.assert_allclose(svd_vals, sigma, rtol=1e-4)

    def test_mean_sv_matches_gamma_expectation(self):
        """Mean SV should be ≈ (m-n+1)/lam (Gamma expectation, ignoring repulsion)."""
        rng = np.random.default_rng(5)
        m, n, lam = 50, 30, 2.0
        expected_mean = (m - n + 1) / lam  # = 21/2 = 10.5
        svs = []
        for _ in range(200):
            _, sigma = sample_nnd(rng, m=m, n=n, lam=lam)
            svs.extend(sigma.tolist())
        actual_mean = np.mean(svs)
        # Allow 5% relative tolerance for 200*n=6000 samples
        assert abs(actual_mean - expected_mean) / expected_mean < 0.05, (
            f"Mean SV {actual_mean:.2f} deviates from expected {expected_mean:.2f}"
        )

    def test_square_svs_exponential(self):
        """For m=n, SVs should be ~ Exp(lam); check mean ≈ 1/lam."""
        rng = np.random.default_rng(6)
        lam = 0.1
        svs = []
        for _ in range(500):
            _, sigma = sample_nnd(rng, m=20, n=20, lam=lam)
            svs.extend(sigma.tolist())
        actual_mean = np.mean(svs)
        expected_mean = 1.0 / lam  # = 10
        assert abs(actual_mean - expected_mean) / expected_mean < 0.05, (
            f"Mean SV {actual_mean:.2f} vs expected {expected_mean:.2f}"
        )

    def test_requires_m_ge_n(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="m >= n"):
            sample_nnd(rng, m=10, n=20, lam=1.0)

    def test_requires_positive_lam(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="lam must be positive"):
            sample_nnd(rng, m=20, n=20, lam=0.0)

    def test_different_seeds_give_different_results(self):
        X1, _ = sample_nnd(np.random.default_rng(0), m=20, n=20, lam=1.0)
        X2, _ = sample_nnd(np.random.default_rng(1), m=20, n=20, lam=1.0)
        assert not np.allclose(X1, X2)


class TestNNDBenchmarkSmoke:
    """Smoke tests for the NND benchmark: run a tiny version and check RMSE improves."""

    def test_batched_better_than_lowrank_on_nnd(self):
        """On full-rank NND data, batched LOO should not be much worse than lowrank.
        With sufficient rank, lowrank approaches batched as rank → n."""
        import jax.numpy as jnp
        from matlap import matlap_batched, matlap_grid_lowrank, matlap_grid_batched

        rng = np.random.default_rng(42)
        m, n, lam_true, sigma_noise = 50, 50, 0.05, 1.0
        X_true, _ = sample_nnd(rng, m=m, n=n, lam=lam_true)
        Y = jnp.array(X_true + rng.standard_normal((m, n)) * sigma_noise)
        S = sigma_noise * jnp.ones((m, n))
        rmse = lambda mu: float(jnp.sqrt(jnp.mean((jnp.array(mu) - X_true) ** 2)))

        lam_grid = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]

        res_batched = matlap_grid_batched(Y, S, lam_grid, max_iter=50, score_fn="loo")
        res_lowrank_r5 = matlap_grid_lowrank(
            Y, S, lam_grid, rank=5, max_iter=50, score_fn="loo"
        )
        res_lowrank_r30 = matlap_grid_lowrank(
            Y, S, lam_grid, rank=30, max_iter=50, score_fn="loo"
        )

        rmse_batched = rmse(res_batched.best_result.mu)
        rmse_lr5 = rmse(res_lowrank_r5.best_result.mu)
        rmse_lr30 = rmse(res_lowrank_r30.best_result.mu)

        # Batched should beat low rank with too-small rank
        assert rmse_batched < rmse_lr5, (
            f"Batched RMSE ({rmse_batched:.4f}) should beat lowrank_r5 ({rmse_lr5:.4f})"
        )
        # Higher rank lowrank should be closer to batched
        assert rmse_lr30 <= rmse_lr5 + 0.05, (
            f"rank=30 ({rmse_lr30:.4f}) should be better than rank=5 ({rmse_lr5:.4f})"
        )
