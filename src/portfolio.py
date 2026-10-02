"""Multi-asset (portfolio) Monte Carlo: correlated GBM, portfolio value and risk."""

import numpy as np
import pandas as pd

from src.analytics import compute_risk_metrics
from src.config import N_SIMULATIONS, TRADING_DAYS
from src.data import fetch_prices

MIN_ASSETS = 2
MAX_ASSETS = 10


def fetch_multi_prices(symbols: list[str], period: str) -> pd.DataFrame:
    """
    Download several Indian stocks and align them on their common trading days.
    Returns a DataFrame with one column per resolved Yahoo symbol (e.g. TCS.NS).
    """
    unique = list(dict.fromkeys(s.strip().upper() for s in symbols if s.strip()))
    if not MIN_ASSETS <= len(unique) <= MAX_ASSETS:
        raise ValueError(f"Enter between {MIN_ASSETS} and {MAX_ASSETS} different stocks.")

    series = {}
    for symbol in unique:
        ticker, prices = fetch_prices(symbol, period)
        series[ticker] = prices
    if len(series) < MIN_ASSETS:
        raise ValueError("These symbols point to the same stock. Enter different stocks.")

    prices = pd.concat(series, axis=1).dropna()   # keep only days where all stocks traded
    if len(prices) < 60:
        raise ValueError("Fewer than 60 common trading days. Use a longer lookback.")
    return prices


def estimate_portfolio_parameters(prices: pd.DataFrame) -> dict:
    """
    Daily simple-return statistics for every stock:
    mean (drift), standard deviation (volatility), correlation and covariance.
    """
    returns = prices.pct_change().dropna()
    return {
        "mean": returns.mean(),
        "vol": returns.std(),
        "corr": returns.corr(),
        "cov": returns.cov(),
    }


def _safe_cholesky(corr: np.ndarray) -> np.ndarray:
    """
    Cholesky factor L with L @ L.T = corr. If the matrix is singular (for example
    two identical stocks), a tiny diagonal jitter is added so it still factorizes.
    """
    n = corr.shape[0]
    jitter = 0.0
    for _ in range(8):
        try:
            return np.linalg.cholesky(corr + jitter * np.eye(n))
        except np.linalg.LinAlgError:
            jitter = 1e-10 if jitter == 0 else jitter * 10
    raise ValueError("Correlation matrix is not positive semi-definite.")


def simulate_correlated_gbm(
    s0,
    drift,
    vol,
    corr,
    n_days: int = TRADING_DAYS,
    n_sims: int = N_SIMULATIONS,
    seed: int | None = 42,
    shock: str = "normal",
    df: int = 5,
) -> np.ndarray:
    """
    Simulate several stocks at once with correlated GBM.

    1. Draw independent standard normals Z, shaped (days, sims, assets).
    2. Impose the correlation: Z_corr = Z @ L.T, where L is the Cholesky factor
       of the correlation matrix (so Cov(Z_corr) = L @ L.T = corr).
    3. Apply the usual GBM step to every stock with its own drift and volatility:
           S_t = S_(t-1) * exp((mu - 0.5*sigma^2) + sigma * Z_corr)

    shock="student_t": every correlated normal is multiplied by sqrt((df-2) / W),
    where W ~ chi-square(df) is ONE draw shared by all stocks on that day. The result
    is a multivariate Student-t with unit variance: fat tails, and extreme moves
    happen in several stocks at the same time (tail dependence).

    Returns shape (n_days + 1, n_sims, n_assets); row 0 holds the start prices.
    """
    s0 = np.asarray(s0, dtype=float)
    drift = np.asarray(drift, dtype=float)
    vol = np.asarray(vol, dtype=float)
    n_assets = len(s0)

    rng = np.random.default_rng(seed)
    chol = _safe_cholesky(np.asarray(corr, dtype=float))

    z = rng.standard_normal((n_days, n_sims, n_assets))
    correlated = z @ chol.T

    if shock == "student_t":
        if df <= 2:
            raise ValueError("Student-t needs df > 2 for a finite variance.")
        w = rng.chisquare(df, size=(n_days, n_sims, 1))
        correlated = correlated * np.sqrt((df - 2) / w)
    elif shock != "normal":
        raise ValueError("shock must be 'normal' or 'student_t'.")

    log_returns = (drift - 0.5 * vol**2) + vol * correlated   # broadcasts over assets
    paths = s0 * np.exp(np.cumsum(log_returns, axis=0))

    start = np.broadcast_to(s0, (1, n_sims, n_assets))
    return np.concatenate([start, paths], axis=0)


def portfolio_value_paths(asset_paths: np.ndarray, weights, investment: float = 1.0) -> np.ndarray:
    """
    Portfolio value over time for a buy-and-hold portfolio (no rebalancing).

    weights are the capital shares on day 0 (non-negative, sum to 1). Each stock's
    value grows with its own price relative to day 0:
        V_t = investment * sum_i w_i * S_i,t / S_i,0
    Returns shape (n_days + 1, n_sims); row 0 equals `investment`.
    """
    weights = np.asarray(weights, dtype=float)
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("Weights must be non-negative and sum to 1.")
    relative = asset_paths / asset_paths[0]
    return investment * (relative @ weights)


def risk_contributions(weights, cov) -> np.ndarray:
    """
    Share of portfolio variance caused by each stock (sums to 1):
        RC_i = w_i * (Cov @ w)_i / (w' Cov w)
    A stock with a small weight can still dominate risk if it is volatile and
    highly correlated with the rest.
    """
    w = np.asarray(weights, dtype=float)
    cov = np.asarray(cov, dtype=float)
    portfolio_variance = float(w @ cov @ w)
    if portfolio_variance == 0:
        return np.zeros_like(w)
    return w * (cov @ w) / portfolio_variance


def diversification_summary(asset_paths, weights, portfolio_paths, cov, vol,
                            confidence: float = 0.95) -> dict:
    """
    Compare the portfolio with its stocks taken one by one.

    - diversification_ratio = weighted-average volatility / portfolio volatility
      (1.0 = no benefit, higher = more benefit)
    - VaR benefit = weighted-average standalone VaR - portfolio VaR
    All volatilities here are DAILY.
    """
    weights = np.asarray(weights, dtype=float)
    vol = np.asarray(vol, dtype=float)
    cov = np.asarray(cov, dtype=float)

    standalone_var = np.array([
        compute_risk_metrics(asset_paths[:, :, i], confidence)["var"]
        for i in range(asset_paths.shape[2])
    ])
    portfolio_var = compute_risk_metrics(portfolio_paths, confidence)["var"]

    portfolio_vol = float(np.sqrt(weights @ cov @ weights))
    weighted_avg_vol = float(weights @ vol)

    return {
        "standalone_var": standalone_var,
        "weighted_avg_var": float(weights @ standalone_var),
        "portfolio_var": float(portfolio_var),
        "var_diversification_benefit": float(weights @ standalone_var - portfolio_var),
        "portfolio_vol_daily": portfolio_vol,
        "weighted_avg_vol_daily": weighted_avg_vol,
        "diversification_ratio": weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1.0,
    }