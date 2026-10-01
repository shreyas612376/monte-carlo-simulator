"""Matplotlib charts for simulation results."""

import matplotlib.pyplot as plt
import numpy as np


def plot_results(ticker: str, paths: np.ndarray, stats: dict) -> None:
    """Plot all simulated paths and the histogram of final prices."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: all simulated price paths
    ax1.plot(paths, linewidth=0.5, alpha=0.3)
    ax1.set_title(f"{ticker}: {paths.shape[1]} Simulated Price Paths")
    ax1.set_xlabel("Trading Days")
    ax1.set_ylabel("Price")
    ax1.grid(alpha=0.3)

    # Right: histogram of final prices with percentile markers
    final_prices = paths[-1]
    ax2.hist(final_prices, bins=50, color="steelblue", alpha=0.7, edgecolor="white")

    markers = [
        ("10th pct", stats["p10"], "red"),
        ("Median", stats["p50"], "black"),
        ("90th pct", stats["p90"], "green"),
    ]
    for label, value, color in markers:
        ax2.axvline(value, color=color, linestyle="--", linewidth=2,
                    label=f"{label}: {value:,.2f}")

    ax2.set_title(f"{ticker}: Distribution of Price at Day {paths.shape[0] - 1}")
    ax2.set_xlabel("Final Price")
    ax2.set_ylabel("Frequency")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()