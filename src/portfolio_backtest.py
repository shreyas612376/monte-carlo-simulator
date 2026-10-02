"""Walk-forward backtest for portfolio forecasts (correlated GBM vs historical bootstrap)."""

import numpy as np
import pandas as pd

from src.portfolio import (
    estimate_portfolio_parameters,
    portfolio_value_paths,
    simulate_correlated_gbm,
)

MODELS = ("Correlated GBM (normal)", "Correlated GBM + Student-t", "Historical bootstrap")


def bootstrap_final_values(
    window: pd.DataFrame,
    weights: np.ndarray,
    horizon: int,
    n_sims: int,
    seed: int,
    demean: bool = False,
) -> np.ndarray:
    """
    Historical bootstrap (the standard benchmark).

    Each simulated day is a randomly chosen REAL day from the estimation window,
    and ALL stocks take that day's returns together, so correlations and fat
    tails come straight from the data with no distribution assumption.
    demean=True removes the historical mean return (the "zero drift" version).

    Returns the portfolio value after `horizon` days per simulation, starting
    from 1.0 with the given start-of-period weights.
    """
    returns = window.pct_change().dropna().values          # (days, assets)
    if demean:
        returns = returns - returns.mean(axis=0)

    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(returns), size=(horizon, n_sims))
    sampled = returns[picks]                                # (horizon, sims, assets)
    growth = np.prod(1 + sampled, axis=0)                   # (sims, assets)
    return growth @ weights


def _forecast_final_values(
    window: pd.DataFrame,
    weights: np.ndarray,
    horizon: int,
    n_sims: int,
    seed: int,
    drift_mode: str,
    t_df: int,
) -> dict:
    """Simulate every model from one forecast origin; return final portfolio values."""
    params = estimate_portfolio_parameters(window)
    s0 = window.iloc[-1].values
    drift = np.zeros(len(s0)) if drift_mode == "zero" else params["mean"].values
    vol = params["vol"].values
    corr = params["corr"].values
    common = dict(n_days=horizon, n_sims=n_sims, seed=seed)

    normal_paths = simulate_correlated_gbm(s0, drift, vol, corr, **common)
    t_paths = simulate_correlated_gbm(s0, drift, vol, corr, shock="student_t",
                                      df=t_df, **common)
    return {
        MODELS[0]: portfolio_value_paths(normal_paths, weights)[-1],
        MODELS[1]: portfolio_value_paths(t_paths, weights)[-1],
        MODELS[2]: bootstrap_final_values(window, weights, horizon, n_sims, seed,
                                          demean=(drift_mode == "zero")),
    }


def run_portfolio_backtest(
    prices: pd.DataFrame,
    weights,
    horizon: int = 63,
    lookback: int = 504,
    step: int = 21,
    n_sims: int = 500,
    drift_mode: str = "historical",
    t_df: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Walk-forward portfolio backtest. At every forecast origin i (every `step` days):
      1. use ONLY prices[i - lookback : i + 1] to estimate parameters (no look-ahead),
      2. simulate the portfolio's value after `horizon` days with each model,
      3. compare with the ACTUAL portfolio value after `horizon` days.

    The portfolio starts every forecast at value 1.0 with the target `weights`
    (rebalanced at each origin). The actual value is sum_i w_i * P_i(t+h) / P_i(t).

    The output uses the same columns as src.backtest.run_backtest, so
    summarize_backtest works on it directly.
    """
    weights = np.asarray(weights, dtype=float)
    if prices.shape[1] != len(weights):
        raise ValueError("Need exactly one weight per stock.")
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("Weights must be non-negative and sum to 1.")
    if drift_mode not in ("historical", "zero"):
        raise ValueError("drift_mode must be 'historical' or 'zero'.")

    origins = range(lookback, len(prices) - horizon, step)
    if len(origins) == 0:
        raise ValueError(
            f"Not enough history: need more than {lookback + horizon} common trading "
            f"days, got {len(prices)}. Use a longer history or a smaller window/horizon."
        )

    values = prices.values
    rows = []
    for k, i in enumerate(origins):
        window = prices.iloc[i - lookback: i + 1]
        actual = float((values[i + horizon] / values[i]) @ weights)
        finals = _forecast_final_values(window, weights, horizon, n_sims, seed + k,
                                        drift_mode, t_df)
        for model, final in finals.items():
            p05, p10, p50, p90, p95 = np.percentile(final, [5, 10, 50, 90, 95])
            rows.append({
                "origin_date": prices.index[i],
                "target_date": prices.index[i + horizon],
                "model": model,
                "s0": 1.0,
                "actual": actual,
                "p05": p05, "p10": p10, "p50": p50, "p90": p90, "p95": p95,
                "pit": float((final < actual).mean()),
            })
    return pd.DataFrame(rows)