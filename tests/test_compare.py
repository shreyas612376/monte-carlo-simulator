"""Tests for the model comparison table."""

import numpy as np
import pandas as pd

from src.analytics import summarize_final_prices
from src.compare import PERCENT_COLUMNS, PRICE_COLUMNS, compare_models
from src.models import estimate_parameters, simulate_gbm

FAKE_GARCH = {"omega": 2e-6, "alpha": 0.1, "beta": 0.85, "next_variance": 1e-4}


def _prices(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n)))


def test_three_models_without_garch_four_with_garch():
    prices = _prices()
    assert len(compare_models(prices, 0.0003, n_days=50, n_sims=200)) == 3
    assert len(compare_models(prices, 0.0003, n_days=50, n_sims=200, garch=FAKE_GARCH)) == 4


def test_gbm_row_matches_direct_simulation():
    prices = _prices()
    table = compare_models(prices, 0.0003, n_days=60, n_sims=300, seed=7)
    _, vol = estimate_parameters(prices)
    direct = summarize_final_prices(
        simulate_gbm(float(prices.iloc[-1]), 0.0003, vol, n_days=60, n_sims=300, seed=7)
    )
    assert np.isclose(table.loc["GBM (normal)", "Median"], direct["p50"])


def test_expected_columns_and_cvar_at_least_var():
    table = compare_models(_prices(), 0.0, n_days=50, n_sims=300, garch=FAKE_GARCH)
    for col in PRICE_COLUMNS + PERCENT_COLUMNS:
        assert col in table.columns
    assert (table["CVaR 95%"] >= table["VaR 95%"]).all()