"""Market data loading (Yahoo Finance)."""

import pandas as pd
import yfinance as yf

from src.config import EXCHANGE_SUFFIXES, HIST_PERIOD


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