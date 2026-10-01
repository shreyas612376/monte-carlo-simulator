"""Tests for parameter estimation, GBM simulation and analytics."""

import numpy as np
import pandas as pd

from src.analytics import summarize_final_prices
from src.models import estimate_ewma_volatility, estimate_parameters, simulate_gbm


def test_path_shape_and_start_price():
    paths = simulate_gbm(s0=100, drift=0.0005, volatility=0.01, n_days=252, n_sims=1000)
    assert paths.shape == (253, 1000)          # 252 days + day 0
    assert np.all(paths[0] == 100)             # every path starts at today's price


def test_prices_always_positive():
    paths = simulate_gbm(s0=100, drift=0.0, volatility=0.05)
    assert np.all(paths > 0)                   # exp() guarantees positive prices


def test_zero_drift_zero_vol_is_flat():
    paths = simulate_gbm(s0=100, drift=0.0, volatility=0.0, n_days=10, n_sims=5)
    assert np.allclose(paths, 100)


def test_same_seed_is_reproducible():
    a = simulate_gbm(100, 0.0005, 0.01, seed=7)
    b = simulate_gbm(100, 0.0005, 0.01, seed=7)
    assert np.array_equal(a, b)


def test_mean_final_price_matches_theory():
    """
    Under GBM, E[S_T] = S_0 * exp(mu * T). The -0.5*sigma^2 Ito term
    must cancel exactly against the lognormal convexity effect.
    """
    s0, mu, sigma, days = 100, 0.0005, 0.01, 252
    paths = simulate_gbm(s0, mu, sigma, n_days=days, n_sims=20000, seed=1)
    expected = s0 * np.exp(mu * days)
    assert abs(paths[-1].mean() - expected) / expected < 0.01   # within 1%


def test_estimate_parameters_constant_prices():
    prices = pd.Series([100.0] * 30)
    drift, vol = estimate_parameters(prices)
    assert drift == 0.0
    assert vol == 0.0


def test_percentiles_are_ordered():
    paths = simulate_gbm(100, 0.0005, 0.02)
    stats = summarize_final_prices(paths)
    assert stats["p10"] <= stats["p50"] <= stats["p90"]


def _daily_log_returns(paths):
    return np.diff(np.log(paths), axis=0).ravel()


def test_student_t_shocks_have_unit_variance():
    paths = simulate_gbm(100, 0.0, 0.01, n_days=252, n_sims=2000,
                         seed=5, shock="student_t", df=5)
    assert abs(_daily_log_returns(paths).std() - 0.01) / 0.01 < 0.03


def test_student_t_has_fatter_tails_than_normal():
    def kurtosis(x):
        x = x - x.mean()
        return (x**4).mean() / (x**2).mean() ** 2

    normal = simulate_gbm(100, 0.0, 0.01, n_sims=2000, seed=5, shock="normal")
    fat = simulate_gbm(100, 0.0, 0.01, n_sims=2000, seed=5, shock="student_t", df=5)
    assert kurtosis(_daily_log_returns(normal)) < 3.3   # normal is about 3
    assert kurtosis(_daily_log_returns(fat)) > 4.0      # fat tails



def _prices_from_returns(returns):
    return pd.Series(100 * np.cumprod(1 + np.asarray(returns)))


def test_ewma_equals_sigma_for_constant_magnitude_returns():
    returns = [0.01, -0.01] * 100
    vol = estimate_ewma_volatility(_prices_from_returns(returns))
    assert abs(vol - 0.01) < 1e-4


def test_ewma_reacts_to_recent_shock_more_than_equal_weight():
    calm = [0.005, -0.005] * 122          # 244 calm days
    spike = [0.05, -0.05, 0.05, -0.05, 0.05]   # 5 volatile recent days
    prices = _prices_from_returns(calm + spike)

    _, equal_weight_vol = estimate_parameters(prices)
    ewma_vol = estimate_ewma_volatility(prices)
    assert ewma_vol > 1.5 * equal_weight_vol