"""Tests for GARCH(1,1) fitting and simulation."""

import numpy as np
import pandas as pd

from src.models import fit_garch, simulate_garch, simulate_gbm

OMEGA, ALPHA, BETA = 2e-6, 0.15, 0.80
LONG_RUN_VAR = OMEGA / (1 - ALPHA - BETA)


def _daily_log_returns(paths):
    return np.diff(np.log(paths), axis=0).ravel()


def test_garch_shape_start_and_positive_prices():
    paths = simulate_garch(100, 0.0003, OMEGA, ALPHA, BETA, LONG_RUN_VAR,
                           n_days=252, n_sims=500)
    assert paths.shape == (253, 500)
    assert np.all(paths[0] == 100)
    assert np.all(paths > 0)


def test_garch_with_zero_alpha_beta_matches_plain_gbm():
    sigma = 0.01
    garch = simulate_garch(100, 0.0005, omega=sigma**2, alpha=0.0, beta=0.0,
                           init_variance=sigma**2, n_days=100, n_sims=50, seed=9)
    gbm = simulate_gbm(100, 0.0005, sigma, n_days=100, n_sims=50, seed=9)
    assert np.allclose(garch, gbm)


def test_garch_creates_volatility_clustering_fat_tails():
    def kurtosis(x):
        x = x - x.mean()
        return (x**4).mean() / (x**2).mean() ** 2

    paths = simulate_garch(100, 0.0, OMEGA, ALPHA, BETA, LONG_RUN_VAR,
                           n_days=252, n_sims=2000, seed=4)
    assert kurtosis(_daily_log_returns(paths)) > 4.0   # normal GBM is about 3


def test_fit_garch_returns_valid_parameters():
    paths = simulate_garch(100, 0.0003, OMEGA, ALPHA, BETA, LONG_RUN_VAR,
                           n_days=1500, n_sims=1, seed=1)
    result = fit_garch(pd.Series(paths[:, 0]))
    assert 0 <= result["alpha"] <= 1
    assert 0 <= result["beta"] <= 1
    assert result["next_variance"] > 0
    assert np.isfinite(result["omega"])