"""Tests for the walk-forward backtest."""

import numpy as np
import pandas as pd

from src.backtest import kupiec_pof_pvalue, run_backtest, summarize_backtest


def _synthetic_gbm_prices(n=2000, sigma=0.01, seed=0):
    """Prices that truly follow zero-drift GBM, so a GBM model should be calibrated."""
    rng = np.random.default_rng(seed)
    log_returns = -0.5 * sigma**2 + sigma * rng.standard_normal(n)
    return pd.Series(100 * np.exp(np.cumsum(log_returns)))


def test_kupiec_accepts_expected_breach_rate():
    assert kupiec_pof_pvalue(5, 100, 0.05) > 0.99


def test_kupiec_rejects_far_too_many_breaches():
    assert kupiec_pof_pvalue(30, 100, 0.05) < 0.001


def test_summary_on_handcrafted_results():
    results = pd.DataFrame({
        "model": ["A"] * 4,
        "actual": [100, 50, 150, 100],
        "p05": [60] * 4, "p10": [70] * 4, "p50": [100] * 4,
        "p90": [130] * 4, "p95": [140] * 4,
        "pit": [0.5, 0.01, 0.99, 0.5],
    })
    summary = summarize_backtest(results)
    assert summary.loc["A", "Observations"] == 4
    assert summary.loc["A", "Coverage 80%"] == 0.5
    assert summary.loc["A", "Below 5th pct"] == 0.25
    assert summary.loc["A", "Above 95th pct"] == 0.25
    assert np.isclose(summary.loc["A", "Mean PIT"], 0.5)


def test_origin_count_and_columns():
    prices = _synthetic_gbm_prices(n=600)
    results = run_backtest(prices, horizon=20, lookback=300, step=20,
                           n_sims=100, include_garch=False)
    assert len(results) == 14 * 3   # 14 forecast origins x 3 models
    for col in ["origin_date", "target_date", "model", "s0", "actual",
                "p05", "p10", "p50", "p90", "p95", "pit"]:
        assert col in results.columns
    assert (results["p05"] <= results["p50"]).all()
    assert (results["p50"] <= results["p95"]).all()


def test_gbm_is_calibrated_on_data_generated_by_gbm():
    prices = _synthetic_gbm_prices(n=2000)
    results = run_backtest(prices, horizon=20, lookback=300, step=20, n_sims=300,
                           drift_mode="zero", include_garch=False)
    summary = summarize_backtest(results)
    assert abs(summary.loc["GBM (normal)", "Coverage 80%"] - 0.80) < 0.15