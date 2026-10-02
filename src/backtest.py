"""Walk-forward backtest: how well calibrated are the simulation models?"""

import math

import numpy as np
import pandas as pd

from src.models import (
    estimate_ewma_volatility,
    estimate_parameters,
    fit_garch,
    simulate_garch,
    simulate_gbm,
)


def kupiec_pof_pvalue(n_breaches: int, n_obs: int, expected_rate: float = 0.05) -> float:
    """
    Kupiec proportion-of-failures test for a VaR / tail-breach rate.

    H0: the true breach probability equals `expected_rate`.
    LR = -2 * ln[ L(expected_rate) / L(observed_rate) ], asymptotically
    chi-square with 1 degree of freedom, whose survival function is
    erfc(sqrt(LR / 2)). A small p-value means the breach rate is
    inconsistent with the model.
    """
    x, n, p = n_breaches, n_obs, expected_rate
    if n == 0:
        return float("nan")

    def log_likelihood(prob: float) -> float:
        # Bernoulli log-likelihood, treating 0 * log(0) as 0
        ll = 0.0
        if n - x > 0:
            ll += (n - x) * math.log(1 - prob)
        if x > 0:
            ll += x * math.log(prob)
        return ll

    observed_rate = x / n
    lr = max(-2 * (log_likelihood(p) - log_likelihood(observed_rate)), 0.0)
    return float(math.erfc(math.sqrt(lr / 2)))


def _forecast_final_prices(
    window: pd.Series,
    horizon: int,
    n_sims: int,
    seed: int,
    drift_mode: str,
    t_df: int,
    ewma_lambda: float,
    include_garch: bool,
) -> dict:
    """Simulate every model from one forecast origin; return final prices per model."""
    s0 = float(window.iloc[-1])
    hist_drift, hist_vol = estimate_parameters(window)
    drift = 0.0 if drift_mode == "zero" else hist_drift
    ewma_vol = estimate_ewma_volatility(window, ewma_lambda)
    common = dict(n_days=horizon, n_sims=n_sims, seed=seed)

    finals = {
        "GBM (normal)": simulate_gbm(s0, drift, hist_vol, **common)[-1],
        "GBM + Student-t": simulate_gbm(s0, drift, hist_vol, shock="student_t",
                                        df=t_df, **common)[-1],
        "GBM + EWMA volatility": simulate_gbm(s0, drift, ewma_vol, **common)[-1],
    }
    if include_garch:
        try:
            g = fit_garch(window)
            if g["persistence"] < 1:   # otherwise the variance process can explode
                finals["GARCH(1,1)"] = simulate_garch(
                    s0, drift, g["omega"], g["alpha"], g["beta"],
                    g["next_variance"], **common,
                )[-1]
        except Exception:
            pass  # skip GARCH at origins where the fit fails
    return finals


def run_backtest(
    prices: pd.Series,
    horizon: int = 63,
    lookback: int = 504,
    step: int = 21,
    n_sims: int = 500,
    drift_mode: str = "historical",
    t_df: int = 5,
    ewma_lambda: float = 0.94,
    include_garch: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Walk-forward backtest. At every forecast origin i (every `step` days):
      1. use ONLY prices[i - lookback : i] to estimate parameters (no look-ahead),
      2. simulate `n_sims` paths over `horizon` days with each model,
      3. compare the actual price at i + horizon with the simulated distribution.

    PIT (probability integral transform) = share of simulated final prices
    below the actual price. A well-calibrated model gives PIT values spread
    uniformly between 0 and 1.

    Returns one row per (forecast origin, model).
    """
    if drift_mode not in ("historical", "zero"):
        raise ValueError("drift_mode must be 'historical' or 'zero'.")

    origins = range(lookback, len(prices) - horizon, step)
    if len(origins) == 0:
        raise ValueError(
            f"Not enough history: need more than {lookback + horizon} trading days, "
            f"got {len(prices)}. Use a longer history or a smaller window/horizon."
        )

    rows = []
    for k, i in enumerate(origins):
        window = prices.iloc[i - lookback: i + 1]
        actual = float(prices.iloc[i + horizon])
        finals = _forecast_final_prices(window, horizon, n_sims, seed + k,
                                        drift_mode, t_df, ewma_lambda, include_garch)
        for model, final in finals.items():
            p05, p10, p50, p90, p95 = np.percentile(final, [5, 10, 50, 90, 95])
            rows.append({
                "origin_date": prices.index[i],
                "target_date": prices.index[i + horizon],
                "model": model,
                "s0": float(window.iloc[-1]),
                "actual": actual,
                "p05": p05, "p10": p10, "p50": p50, "p90": p90, "p95": p95,
                "pit": float((final < actual).mean()),
            })
    return pd.DataFrame(rows)


def summarize_backtest(results: pd.DataFrame) -> pd.DataFrame:
    """Calibration statistics per model (see run_backtest for the definitions)."""
    rows = {}
    for model, grp in results.groupby("model", sort=False):
        n = len(grp)
        below = int((grp["actual"] < grp["p05"]).sum())
        above = int((grp["actual"] > grp["p95"]).sum())
        inside_80 = ((grp["actual"] >= grp["p10"]) & (grp["actual"] <= grp["p90"])).mean()
        inside_90 = ((grp["actual"] >= grp["p05"]) & (grp["actual"] <= grp["p95"])).mean()
        rows[model] = {
            "Observations": n,
            "Coverage 80%": float(inside_80),
            "Coverage 90%": float(inside_90),
            "Below 5th pct": below / n,
            "Above 95th pct": above / n,
            "Mean PIT": float(grp["pit"].mean()),
            "Kupiec p-value": kupiec_pof_pvalue(below, n, 0.05),
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("Model")
