"""Tests for the portfolio walk-forward backtest."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import summarize_backtest
from src.portfolio_backtest import MODELS, run_portfolio_backtest

CORR = np.array([[1.0, 0.6, 0.2],
                 [0.6, 1.0, 0.3],
                 [0.2, 0.3, 1.0]])
SIGMAS = np.array([0.010, 0.015, 0.012])
WEIGHTS = [0.5, 0.3, 0.2]


def _synthetic_prices(n=2000, seed=0):
    """Correlated zero-drift GBM prices, so a GBM model should be calibrated."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 3)) @ np.linalg.cholesky(CORR).T
    log_returns = -0.5 * SIGMAS**2 + SIGMAS * z
    return pd.DataFrame(100 * np.exp(np.cumsum(log_returns, axis=0)), columns=["A", "B", "C"])


def test_row_count_columns_and_percentile_order():
    prices = _synthetic_prices(n=700)
    results = run_portfolio_backtest(prices, WEIGHTS, horizon=20, lookback=300,
                                     step=20, n_sims=100)
    assert len(results) == 19 * len(MODELS)   # 19 forecast origins x 3 models
    for col in ["origin_date", "target_date", "model", "s0", "actual",
                "p05", "p10", "p50", "p90", "p95", "pit"]:
        assert col in results.columns
    assert (results["p05"] <= results["p50"]).all()
    assert (results["p50"] <= results["p95"]).all()


def test_gbm_is_calibrated_on_data_generated_by_correlated_gbm():
    prices = _synthetic_prices(n=2000)
    results = run_portfolio_backtest(prices, WEIGHTS, horizon=20, lookback=300, step=20,
                                     n_sims=300, drift_mode="zero")
    summary = summarize_backtest(results)
    assert abs(summary.loc["Correlated GBM (normal)", "Coverage 80%"] - 0.80) < 0.15


def test_invalid_weights_raise():
    prices = _synthetic_prices(n=700)
    with pytest.raises(ValueError):
        run_portfolio_backtest(prices, [0.7, 0.7, 0.2], horizon=20, lookback=300)
    with pytest.raises(ValueError):
        run_portfolio_backtest(prices, [0.5, 0.5], horizon=20, lookback=300)


def test_not_enough_history_raises():
    prices = _synthetic_prices(n=200)
    with pytest.raises(ValueError):
        run_portfolio_backtest(prices, WEIGHTS, horizon=63, lookback=504)