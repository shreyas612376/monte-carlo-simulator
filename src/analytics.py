"""Statistics on simulated price paths."""

import numpy as np


def summarize_final_prices(paths: np.ndarray) -> dict:
    """Percentiles of the final-day price distribution across simulations."""
    final_prices = paths[-1]
    return {
        "p10": np.percentile(final_prices, 10),   # pessimistic scenario
        "p50": np.percentile(final_prices, 50),   # median scenario
        "p90": np.percentile(final_prices, 90),   # optimistic scenario
        "mean": final_prices.mean(),
    }


def compute_risk_metrics(paths: np.ndarray, confidence: float = 0.95) -> dict:
    """
    Risk metrics on the simulated paths (losses are reported as positive numbers).

    - VaR: loss level that is exceeded in only (1 - confidence) of scenarios.
    - CVaR (Expected Shortfall): average loss in the scenarios beyond VaR.
    - Max drawdown: worst peak-to-trough fall along each path.
    """
    s0 = paths[0, 0]
    final_returns = paths[-1] / s0 - 1

    # Worst (1 - confidence) tail of final returns
    cutoff = np.percentile(final_returns, (1 - confidence) * 100)
    tail = final_returns[final_returns <= cutoff]

    # Drawdown = current price / highest price so far - 1 (always <= 0)
    running_max = np.maximum.accumulate(paths, axis=0)
    max_drawdowns = (paths / running_max - 1).min(axis=0)

    return {
        "var": float(-cutoff),
        "cvar": float(-tail.mean()),
        "prob_loss": float((final_returns < 0).mean()),
        "median_max_drawdown": float(np.median(max_drawdowns)),
        "worst_5pct_max_drawdown": float(np.percentile(max_drawdowns, 5)),
    }