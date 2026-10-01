"""Entry point: Monte Carlo stock price simulator for Indian stocks."""

import numpy as np

from src.analytics import summarize_final_prices
from src.config import N_SIMULATIONS, TRADING_DAYS
from src.data import fetch_prices
from src.models import estimate_parameters, simulate_gbm
from src.plots import plot_results


def main() -> None:
    user_input = input("Enter Indian stock symbol (e.g. RELIANCE, TCS, HDFCBANK): ")

    ticker, prices = fetch_prices(user_input)
    s0 = float(prices.iloc[-1])  # most recent close = simulation start

    drift, volatility = estimate_parameters(prices)
    print(f"\nLast price:        Rs {s0:,.2f}")
    print(f"Daily drift:       {drift:.5%}  (annualized ~ {drift * TRADING_DAYS:.2%})")
    print(f"Daily volatility:  {volatility:.5%}  (annualized ~ {volatility * np.sqrt(TRADING_DAYS):.2%})")

    paths = simulate_gbm(s0, drift, volatility)
    stats = summarize_final_prices(paths)

    print(f"\nDay {TRADING_DAYS} forecast over {N_SIMULATIONS} simulations:")
    print(f"  10th percentile: Rs {stats['p10']:,.2f}")
    print(f"  Median:          Rs {stats['p50']:,.2f}")
    print(f"  90th percentile: Rs {stats['p90']:,.2f}")
    print(f"  Mean:            Rs {stats['mean']:,.2f}")

    plot_results(ticker, paths, stats)


if __name__ == "__main__":
    main()