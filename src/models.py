"""Parameter estimation and Geometric Brownian Motion simulation."""

import numpy as np
import pandas as pd

from src.config import N_SIMULATIONS, TRADING_DAYS


def estimate_parameters(prices: pd.Series) -> tuple[float, float]:
    """
    Estimate daily drift (mu) and daily volatility (sigma) from history.

    Daily simple return: r_t = P_t / P_{t-1} - 1
    - drift (mu)      = mean of r_t
    - volatility (sigma) = standard deviation of r_t

    SIMPLE returns are used because the GBM step below subtracts
    0.5 * sigma^2 (the Ito correction) itself. Using mean LOG returns
    as drift would apply that correction twice.
    """
    returns = prices.pct_change().dropna()
    return float(returns.mean()), float(returns.std())


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

    Discrete GBM step (dt = 1 day, parameters already daily):
        S_t = S_{t-1} * exp((mu - 0.5 * sigma^2) + sigma * Z),  Z ~ N(0, 1)

    - (mu - 0.5 * sigma^2): drift of the LOG price (Ito correction).
    - sigma * Z: random shock; Z is a standard normal draw each day.
    - exp(...): keeps prices positive and makes returns compound.

    Returns an array of shape (n_days + 1, n_sims); row 0 is today's price.
    """
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((n_days, n_sims))

    log_returns = (drift - 0.5 * volatility**2) + volatility * shocks
    log_paths = np.cumsum(log_returns, axis=0)
    paths = s0 * np.exp(log_paths)

    start = np.full((1, n_sims), s0)
    return np.vstack([start, paths])