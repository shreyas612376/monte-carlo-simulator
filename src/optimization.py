"""Portfolio optimization: minimum variance, maximum Sharpe, risk parity, efficient frontier."""

import numpy as np
from scipy.optimize import minimize

from src.config import TRADING_DAYS
from src.portfolio import risk_contributions


def annualize(mean_daily, cov_daily, periods: int = TRADING_DAYS):
    """Scale daily mean returns and the daily covariance matrix to annual figures."""
    return np.asarray(mean_daily, dtype=float) * periods, np.asarray(cov_daily, dtype=float) * periods


def shrink_returns(mu, shrinkage: float) -> np.ndarray:
    """
    Shrink expected returns toward their cross-sectional average.

        mu_shrunk = (1 - s) * mu + s * mean(mu)

    Historical mean returns are extremely noisy (a 2-year mean has a huge error
    bar), and optimizers amplify that noise. Shrinkage pulls the extreme
    estimates toward the middle: s = 0 keeps history, s = 1 treats every stock
    as having the same expected return.
    """
    if not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must be between 0 and 1.")
    mu = np.asarray(mu, dtype=float)
    return (1 - shrinkage) * mu + shrinkage * mu.mean()


def portfolio_performance(weights, mu, cov, risk_free: float = 0.0) -> dict:
    """Expected return, volatility and Sharpe ratio of a portfolio (same units as mu, cov)."""
    w = np.asarray(weights, dtype=float)
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    sharpe = (ret - risk_free) / vol if vol > 0 else float("nan")
    return {"return": ret, "volatility": vol, "sharpe": sharpe}


def _check_cap(n_assets: int, max_weight: float) -> None:
    if not 0 < max_weight <= 1:
        raise ValueError("max_weight must be between 0 and 1.")
    if max_weight * n_assets < 1 - 1e-9:
        raise ValueError(
            f"A {max_weight:.0%} cap per stock is too low for {n_assets} stocks "
            f"(the weights could not add up to 100%). Use at least {1 / n_assets:.0%}."
        )


def _solve(objective, n_assets, max_weight, extra_constraints=(), lower=0.0, x0=None):
    """Minimise `objective` over long-only weights that sum to 1 and respect the cap."""
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}] + list(extra_constraints)
    start = np.full(n_assets, 1.0 / n_assets) if x0 is None else x0
    result = minimize(
        objective, start, method="SLSQP",
        bounds=[(lower, max_weight)] * n_assets, constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        raise ValueError(f"Optimizer did not converge: {result.message}")
    weights = np.clip(result.x, 0.0, max_weight)
    return weights / weights.sum()


def min_variance_weights(cov, max_weight: float = 1.0) -> np.ndarray:
    """Long-only weights with the lowest possible portfolio variance."""
    cov = np.asarray(cov, dtype=float)
    _check_cap(cov.shape[0], max_weight)
    return _solve(lambda w: float(w @ cov @ w), cov.shape[0], max_weight)


def max_sharpe_weights(mu, cov, risk_free: float = 0.0, max_weight: float = 1.0) -> np.ndarray:
    """
    Long-only weights with the highest Sharpe ratio (the tangency portfolio):
        maximise (w'mu - rf) / sqrt(w' cov w)

    Only defined when at least one stock is expected to beat the risk-free rate.
    """
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    _check_cap(len(mu), max_weight)
    if not np.any(mu > risk_free):
        raise ValueError(
            "No stock has an expected return above the risk-free rate, so a maximum-Sharpe "
            "portfolio is not defined. Increase shrinkage, lower the risk-free rate, "
            "or use another strategy."
        )

    def negative_sharpe(w):
        return -float((w @ mu - risk_free) / np.sqrt(w @ cov @ w))

    return _solve(negative_sharpe, len(mu), max_weight)


def risk_parity_weights(cov, max_weight: float = 1.0) -> np.ndarray:
    """
    Equal risk contribution: every stock causes the same share of portfolio variance.
    Solved by minimising the squared gap between each risk share and 1/n.
    """
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    _check_cap(n, max_weight)
    target = np.full(n, 1.0 / n)

    def objective(w):
        return float(((risk_contributions(w, cov) - target) ** 2).sum())

    return _solve(objective, n, max_weight, lower=1e-6)


def max_return_weights(mu, max_weight: float = 1.0) -> np.ndarray:
    """Highest-return long-only portfolio under the cap: fill the best stocks first."""
    mu = np.asarray(mu, dtype=float)
    _check_cap(len(mu), max_weight)
    weights = np.zeros(len(mu))
    remaining = 1.0
    for i in np.argsort(-mu):
        take = min(max_weight, remaining)
        weights[i] = take
        remaining -= take
        if remaining <= 1e-12:
            break
    return weights


def efficient_frontier(mu, cov, n_points: int = 30, max_weight: float = 1.0) -> list[dict]:
    """
    Points on the efficient frontier: for each target return between the
    minimum-variance portfolio and the highest feasible return, the weights
    with the lowest volatility. Returns a list of dicts sorted by return.
    """
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    n = len(mu)
    _check_cap(n, max_weight)

    low = float(min_variance_weights(cov, max_weight) @ mu)
    high = float(max_return_weights(mu, max_weight) @ mu)

    points = []
    for target in np.linspace(low, high, n_points):
        constraint = {"type": "eq", "fun": lambda w, r=target: float(w @ mu - r)}
        try:
            w = _solve(lambda x: float(x @ cov @ x), n, max_weight, extra_constraints=[constraint])
        except ValueError:
            continue   # the very edge of the feasible set can fail numerically
        perf = portfolio_performance(w, mu, cov)
        points.append({"return": perf["return"], "volatility": perf["volatility"], "weights": w})
    return points


def random_portfolios(n_assets: int, n_samples: int = 3000, max_weight: float = 1.0,
                      seed: int = 0) -> np.ndarray:
    """Random long-only weight vectors (that respect the cap), for the background cloud."""
    rng = np.random.default_rng(seed)
    samples = rng.dirichlet(np.ones(n_assets), size=n_samples)
    return samples[samples.max(axis=1) <= max_weight + 1e-12]


def build_strategies(mu, cov, risk_free: float = 0.0, max_weight: float = 1.0):
    """
    Weights for the four standard strategies: equal weight, minimum variance,
    maximum Sharpe and risk parity. Returns (strategies, notes); a strategy that
    is not defined for the inputs (for example maximum Sharpe when no stock beats
    the risk-free rate) is left out and explained in `notes`.
    """
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    n = len(mu)
    _check_cap(n, max_weight)

    notes = []
    max_sharpe = None
    try:
        max_sharpe = max_sharpe_weights(mu, cov, risk_free, max_weight)
    except ValueError as err:
        notes.append(str(err))

    strategies = {
        "Equal weight": np.full(n, 1.0 / n),
        "Minimum variance": min_variance_weights(cov, max_weight),
    }
    if max_sharpe is not None:
        strategies["Maximum Sharpe"] = max_sharpe
    strategies["Risk parity"] = risk_parity_weights(cov, max_weight)
    return strategies, notes