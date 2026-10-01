"""
Monte Carlo stock price simulator using Geometric Brownian Motion (GBM).

Pipeline:
    fetch prices -> estimate drift & volatility -> simulate paths -> plot results
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

TRADING_DAYS = 252      # Trading days in one year
N_SIMULATIONS = 1000    # Number of independent price paths
HIST_PERIOD = "1y"      # Historical window used for parameter estimation


# ---------------------------------------------------------------------------
# Step 1: Data
# ---------------------------------------------------------------------------
EXCHANGE_SUFFIXES = (".NS", ".BO")  # NSE first, then BSE as fallback


def _download_close(symbol: str, period: str) -> pd.Series:
    """Download adjusted closing prices for one exact Yahoo symbol."""
    data = yf.download(symbol, period=period, auto_adjust=True, progress=False)
    if data.empty:
        return pd.Series(dtype=float)
    return data["Close"].squeeze().dropna()


def fetch_prices(ticker: str, period: str = HIST_PERIOD) -> tuple[str, pd.Series]:
    """
    Fetch prices for an Indian stock.

    - If the user already typed a suffix (.NS / .BO), use it as is.
    - Otherwise try NSE (.NS) first, then BSE (.BO).
    Returns the resolved Yahoo symbol and the closing price series.
    """
    ticker = ticker.strip().upper()

    if ticker.endswith(EXCHANGE_SUFFIXES):
        candidates = [ticker]
    else:
        candidates = [ticker + suffix for suffix in EXCHANGE_SUFFIXES]

    for symbol in candidates:
        prices = _download_close(symbol, period)
        if not prices.empty:
            return symbol, prices

    raise ValueError(
        f"No data found for '{ticker}' on NSE/BSE. "
        "Check the symbol (e.g. RELIANCE, TCS, HDFCBANK, INFY)."
    )


# ---------------------------------------------------------------------------
# Step 2: Parameter estimation
# ---------------------------------------------------------------------------
def estimate_parameters(prices: pd.Series) -> tuple[float, float]:
    """
    Estimate daily drift (mu) and daily volatility (sigma) from history.

    Daily simple return: r_t = P_t / P_{t-1} - 1
    - drift (mu)      = mean of r_t
    - volatility (σ)  = standard deviation of r_t

    We use SIMPLE returns here because the GBM formula below subtracts
    0.5 * σ^2 (the Ito correction) itself. Using mean LOG returns as drift
    would apply that correction twice.
    """
    returns = prices.pct_change().dropna()
    drift = float(returns.mean())
    volatility = float(returns.std())
    return drift, volatility


# ---------------------------------------------------------------------------
# Step 3: GBM simulation
# ---------------------------------------------------------------------------
def simulate_gbm(
    s0: float,
    drift: float,
    volatility: float,
    n_days: int = TRADING_DAYS,
    n_sims: int = N_SIMULATIONS,
    seed: int | None = 42,
) -> np.ndarray:
    """
    Simulate price paths with Geometric Brownian Motion.

    Discrete GBM step (dt = 1 day, so parameters are already daily):
        S_t = S_{t-1} * exp((mu - 0.5 * σ^2) + σ * Z),   Z ~ N(0, 1)

    - (mu - 0.5 * σ^2): drift of the LOG price. The -0.5σ^2 term is the Ito
      correction: volatility drags down the median compounded growth.
    - σ * Z: random shock; Z is a standard normal draw each day.
    - exp(...): keeps prices strictly positive and makes returns compound.

    Returns an array of shape (n_days + 1, n_sims); row 0 is today's price.
    """
    rng = np.random.default_rng(seed)

    # One standard normal shock per day per simulation
    shocks = rng.standard_normal((n_days, n_sims))

    # Daily log-return for every day and every path
    log_returns = (drift - 0.5 * volatility**2) + volatility * shocks

    # Summing log-returns over time == multiplying the exp() factors
    # in the formula day after day (vectorized, no Python loops)
    log_paths = np.cumsum(log_returns, axis=0)
    paths = s0 * np.exp(log_paths)

    # Prepend the starting price as Day 0
    start = np.full((1, n_sims), s0)
    return np.vstack([start, paths])


# ---------------------------------------------------------------------------
# Step 4: Statistics
# ---------------------------------------------------------------------------
def summarize_final_prices(paths: np.ndarray) -> dict:
    """Percentiles of the Day-252 price distribution across all simulations."""
    final_prices = paths[-1]
    return {
        "p10": np.percentile(final_prices, 10),   # pessimistic scenario
        "p50": np.percentile(final_prices, 50),   # median scenario
        "p90": np.percentile(final_prices, 90),   # optimistic scenario
        "mean": final_prices.mean(),
    }


# ---------------------------------------------------------------------------
# Step 5: Plotting
# ---------------------------------------------------------------------------
def plot_results(ticker: str, paths: np.ndarray, stats: dict) -> None:
    """Plot all simulated paths and the histogram of final prices."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Left: all 1,000 price paths ---
    ax1.plot(paths, linewidth=0.5, alpha=0.3)
    ax1.set_title(f"{ticker}: {paths.shape[1]} Simulated Price Paths")
    ax1.set_xlabel("Trading Days")
    ax1.set_ylabel("Price")
    ax1.grid(alpha=0.3)

    # --- Right: histogram of Day-252 prices with percentile markers ---
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    user_input = input("Enter Indian stock symbol (e.g. RELIANCE, TCS, HDFCBANK): ")

    ticker, prices = fetch_prices(user_input)
    s0 = float(prices.iloc[-1])  # most recent closing price = simulation start

    drift, volatility = estimate_parameters(prices)
    print(f"\nLast price:        Rs {s0:,.2f}")
    print(f"Daily drift:       {drift:.5%}  (annualized ~ {drift * TRADING_DAYS:.2%})")
    print(f"Daily volatility:  {volatility:.5%}  (annualized ~ {volatility * np.sqrt(TRADING_DAYS):.2%})")

    paths = simulate_gbm(s0, drift, volatility)
    stats = summarize_final_prices(paths)

    print(f"\nDay {TRADING_DAYS} forecast over {N_SIMULATIONS} simulations:")
    print(f"  10th percentile: {stats['p10']:,.2f}")
    print(f"  Median:          {stats['p50']:,.2f}")
    print(f"  90th percentile: {stats['p90']:,.2f}")
    print(f"  Mean:            {stats['mean']:,.2f}")

    plot_results(ticker, paths, stats)


if __name__ == "__main__":
    main()