"""Side-by-side comparison of the available simulation models."""

import numpy as np
import pandas as pd

from src.analytics import compute_risk_metrics, summarize_final_prices
from src.config import N_SIMULATIONS, TRADING_DAYS
from src.models import (
    estimate_ewma_volatility,
    estimate_parameters,
    simulate_garch,
    simulate_gbm,
)

PRICE_COLUMNS = ["10th pct", "Median", "90th pct", "Mean"]
PERCENT_COLUMNS = [
    "Start volatility (ann.)",
    "VaR 95%",
    "CVaR 95%",
    "Prob. of loss",
    "Median max drawdown",
    "Worst-5% max drawdown",
]


def _summarize(paths: np.ndarray, start_daily_vol: float) -> dict:
    """One table row: final-price percentiles plus risk metrics."""
    stats = summarize_final_prices(paths)
    risk = compute_risk_metrics(paths)
    return {
        "Start volatility (ann.)": start_daily_vol * np.sqrt(TRADING_DAYS),
        "10th pct": stats["p10"],
        "Median": stats["p50"],
        "90th pct": stats["p90"],
        "Mean": stats["mean"],
        "VaR 95%": risk["var"],
        "CVaR 95%": risk["cvar"],
        "Prob. of loss": risk["prob_loss"],
        "Median max drawdown": risk["median_max_drawdown"],
        "Worst-5% max drawdown": risk["worst_5pct_max_drawdown"],
    }


def compare_models(
    prices: pd.Series,
    drift: float,
    n_days: int = TRADING_DAYS,
    n_sims: int = N_SIMULATIONS,
    seed: int = 42,
    t_df: int = 5,
    ewma_lambda: float = 0.94,
    garch: dict | None = None,
) -> pd.DataFrame:
    """
    Run every model with the SAME daily drift, horizon, simulation count and
    seed, so differences in the table come from the model assumptions only.

    Models:
    - GBM (normal): equal-weight volatility, normal shocks
    - GBM + Student-t: same volatility, fat-tailed shocks
    - GBM + EWMA volatility: recent-weighted volatility, normal shocks
    - GARCH(1,1): volatility evolves daily (only if fitted parameters are given)

    `drift` is the DAILY drift to use. Returns one row per model.
    """
    s0 = float(prices.iloc[-1])
    _, hist_vol = estimate_parameters(prices)
    ewma_vol = estimate_ewma_volatility(prices, ewma_lambda)
    common = dict(n_days=n_days, n_sims=n_sims, seed=seed)

    rows = {}

    paths = simulate_gbm(s0, drift, hist_vol, **common)
    rows["GBM (normal)"] = _summarize(paths, hist_vol)

    paths = simulate_gbm(s0, drift, hist_vol, shock="student_t", df=t_df, **common)
    rows["GBM + Student-t"] = _summarize(paths, hist_vol)

    paths = simulate_gbm(s0, drift, ewma_vol, **common)
    rows["GBM + EWMA volatility"] = _summarize(paths, ewma_vol)

    if garch is not None:
        start_vol = float(np.sqrt(garch["next_variance"]))
        paths = simulate_garch(
            s0, drift, garch["omega"], garch["alpha"], garch["beta"],
            garch["next_variance"], **common,
        )
        rows["GARCH(1,1)"] = _summarize(paths, start_vol)

    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("Model")