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