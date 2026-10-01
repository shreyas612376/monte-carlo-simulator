"""Parameter estimation and Geometric Brownian Motion simulation."""

import numpy as np
import pandas as pd

from src.config import N_SIMULATIONS, TRADING_DAYS

SHOCK_TYPES = ("normal", "student_t")


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


def generate_shocks(
    n_days: int,
    n_sims: int,
    rng: np.random.Generator,
    shock: str = "normal",
    df: int = 5,
) -> np.ndarray:
    """
    Draw the random shocks Z, shaped (n_days, n_sims), with mean 0 and variance 1.

    - "normal":    Z ~ N(0, 1), the classic GBM assumption.
    - "student_t": Z ~ t(df) rescaled to unit variance. A t-distribution has
      fatter tails, so extreme daily moves occur more often. Its variance is
      df / (df - 2), so we divide by sqrt(df / (df - 2)). Needs df > 2.
    """
    if shock == "normal":
        return rng.standard_normal((n_days, n_sims))
    if shock == "student_t":
        if df <= 2:
            raise ValueError("Student-t needs df > 2 for a finite variance.")
        raw = rng.standard_t(df, size=(n_days, n_sims))
        return raw / np.sqrt(df / (df - 2))
    raise ValueError(f"Unknown shock type '{shock}'. Use one of {SHOCK_TYPES}.")


def simulate_gbm(
    s0: float,
    drift: float,
    volatility: float,
    n_days: int = TRADING_DAYS,
    n_sims: int = N_SIMULATIONS,
    seed: int | None = 42,
    shock: str = "normal",
    df: int = 5,
) -> np.ndarray:
    """
    Simulate price paths with Geometric Brownian Motion.

    Discrete GBM step (dt = 1 day, parameters already daily):
        S_t = S_{t-1} * exp((mu - 0.5 * sigma^2) + sigma * Z)

    - (mu - 0.5 * sigma^2): drift of the LOG price (Ito correction).
    - sigma * Z: random shock; Z has mean 0 and variance 1 (see generate_shocks).
    - exp(...): keeps prices positive and makes returns compound.

    Returns an array of shape (n_days + 1, n_sims); row 0 is today's price.
    """
    rng = np.random.default_rng(seed)
    shocks = generate_shocks(n_days, n_sims, rng, shock, df)

    log_returns = (drift - 0.5 * volatility**2) + volatility * shocks
    log_paths = np.cumsum(log_returns, axis=0)
    paths = s0 * np.exp(log_paths)

    start = np.full((1, n_sims), s0)
    return np.vstack([start, paths])


def estimate_ewma_volatility(prices: pd.Series, lam: float = 0.94) -> float:
    """
    Exponentially weighted daily volatility (RiskMetrics style).

        sigma_t^2 = lam * sigma_(t-1)^2 + (1 - lam) * r_(t-1)^2

    A lower lam forgets the past faster (more weight on recent days). The
    return of the last observation gives the latest volatility estimate,
    which is used as the forward-looking sigma.
    """
    if not 0 < lam < 1:
        raise ValueError("lam must be between 0 and 1.")
    returns = prices.pct_change().dropna()
    ewma_variance = (returns**2).ewm(alpha=1 - lam, adjust=False).mean()
    return float(np.sqrt(ewma_variance.iloc[-1]))


def fit_garch(prices: pd.Series) -> dict:
    """
    Fit a GARCH(1,1) model to daily returns by maximum likelihood.

        sigma_t^2 = omega + alpha * eps_(t-1)^2 + beta * sigma_(t-1)^2

    Returns omega, alpha, beta (in decimal units, not percent), the
    one-step-ahead variance forecast to start the simulation from, and the
    long-run volatility the model mean-reverts to (needs alpha + beta < 1).
    """
    from arch import arch_model  # imported here so the rest of the module works without it

    # The optimizer behaves better with returns in percent
    returns_pct = prices.pct_change().dropna() * 100
    model = arch_model(returns_pct, mean="Constant", vol="GARCH", p=1, q=1,
                       dist="normal", rescale=False)
    result = model.fit(disp="off")

    omega = float(result.params["omega"]) / 1e4        # percent^2 -> decimal^2
    alpha = float(result.params["alpha[1]"])
    beta = float(result.params["beta[1]"])
    next_variance = float(result.forecast(horizon=1).variance.values[-1, 0]) / 1e4

    persistence = alpha + beta
    long_run_vol = float(np.sqrt(omega / (1 - persistence))) if persistence < 1 else float("nan")

    return {
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "persistence": persistence,
        "next_variance": next_variance,
        "long_run_vol": long_run_vol,
    }


def simulate_garch(
    s0: float,
    drift: float,
    omega: float,
    alpha: float,
    beta: float,
    init_variance: float,
    n_days: int = TRADING_DAYS,
    n_sims: int = N_SIMULATIONS,
    seed: int | None = 42,
    shock: str = "normal",
    df: int = 5,
) -> np.ndarray:
    """
    Simulate price paths where volatility evolves every day (GARCH(1,1)).

    Each day, for every path:
        eps_t      = sigma_t * Z_t                       (today's innovation)
        log S_t    = log S_(t-1) + (drift - 0.5*sigma_t^2) + eps_t
        sigma_(t+1)^2 = omega + alpha * eps_t^2 + beta * sigma_t^2

    A big shock raises tomorrow's variance (volatility clustering); the
    variance then decays back toward its long-run level. With alpha = beta = 0
    this reduces exactly to plain GBM.

    Returns an array of shape (n_days + 1, n_sims); row 0 is today's price.
    """
    rng = np.random.default_rng(seed)
    shocks = generate_shocks(n_days, n_sims, rng, shock, df)

    paths = np.empty((n_days + 1, n_sims))
    paths[0] = s0

    variance = np.full(n_sims, init_variance)
    log_price = np.full(n_sims, np.log(s0))

    # Variance depends on the previous day, so we step through time
    # (252 iterations; each one is vectorized across all simulations).
    for t in range(n_days):
        eps = np.sqrt(variance) * shocks[t]
        log_price = log_price + (drift - 0.5 * variance) + eps
        paths[t + 1] = np.exp(log_price)
        variance = omega + alpha * eps**2 + beta * variance

    return paths