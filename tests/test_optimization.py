"""Tests for portfolio optimization."""

import numpy as np
import pytest

from src.optimization import (
    build_strategies,
    efficient_frontier,
    max_return_weights,
    max_sharpe_weights,
    min_variance_weights,
    portfolio_performance,
    random_portfolios,
    risk_parity_weights,
    shrink_returns,
)
from src.portfolio import risk_contributions

# Uncorrelated stocks with 10% and 20% volatility
COV_2 = np.diag([0.01, 0.04])
# Three correlated stocks
VOLS = np.array([0.15, 0.25, 0.20])
CORR_3 = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]])
COV_3 = np.outer(VOLS, VOLS) * CORR_3
MU_3 = np.array([0.08, 0.14, 0.10])


def test_min_variance_known_solution():
    # Uncorrelated: weights are proportional to 1 / variance -> 0.8 and 0.2
    w = min_variance_weights(COV_2)
    assert np.allclose(w, [0.8, 0.2], atol=1e-3)


def test_min_variance_respects_cap_and_sums_to_one():
    w = min_variance_weights(COV_2, max_weight=0.6)
    assert np.isclose(w.sum(), 1.0)
    assert np.allclose(w, [0.6, 0.4], atol=1e-3)


def test_max_sharpe_known_solution():
    # Uncorrelated, rf = 0: tangency weights are proportional to mu / variance
    mu = np.array([0.10, 0.05])
    w = max_sharpe_weights(mu, COV_2, risk_free=0.0)
    expected = np.array([10.0, 1.25]) / 11.25
    assert np.allclose(w, expected, atol=5e-3)


def test_max_sharpe_beats_equal_weight_and_min_variance():
    sharpe = lambda w: portfolio_performance(w, MU_3, COV_3, 0.04)["sharpe"]
    best = max_sharpe_weights(MU_3, COV_3, risk_free=0.04)
    assert sharpe(best) >= sharpe(np.full(3, 1 / 3)) - 1e-9
    assert sharpe(best) >= sharpe(min_variance_weights(COV_3)) - 1e-9


def test_max_sharpe_undefined_when_nothing_beats_risk_free():
    with pytest.raises(ValueError):
        max_sharpe_weights(np.array([0.01, 0.02]), COV_2, risk_free=0.05)


def test_risk_parity_known_solution_for_uncorrelated_stocks():
    # Equal risk contribution with zero correlation -> weights proportional to 1 / volatility
    w = risk_parity_weights(COV_2)
    assert np.allclose(w, [2 / 3, 1 / 3], atol=1e-2)


def test_risk_parity_equalises_risk_contributions():
    w = risk_parity_weights(COV_3)
    assert np.allclose(risk_contributions(w, COV_3), 1 / 3, atol=1e-2)


def test_cap_too_low_raises():
    with pytest.raises(ValueError):
        min_variance_weights(np.eye(4) * 0.04, max_weight=0.2)


def test_max_return_weights_fills_best_stocks_first():
    w = max_return_weights(MU_3, max_weight=0.6)
    assert np.allclose(w, [0.0, 0.6, 0.4])


def test_efficient_frontier_is_ordered_and_valid():
    points = efficient_frontier(MU_3, COV_3, n_points=15, max_weight=0.7)
    assert len(points) >= 10
    returns = [p["return"] for p in points]
    vols = [p["volatility"] for p in points]
    assert returns == sorted(returns)
    assert all(b >= a - 1e-6 for a, b in zip(vols, vols[1:]))   # more return costs more risk
    for p in points:
        assert np.isclose(p["weights"].sum(), 1.0)
        assert p["weights"].max() <= 0.7 + 1e-6
    # The first point is the minimum-variance portfolio
    min_vol = portfolio_performance(min_variance_weights(COV_3, 0.7), MU_3, COV_3)["volatility"]
    assert np.isclose(vols[0], min_vol, atol=1e-4)


def test_shrink_returns():
    mu = np.array([0.0, 0.1, 0.2])
    assert np.allclose(shrink_returns(mu, 0.0), mu)
    assert np.allclose(shrink_returns(mu, 1.0), 0.1)
    assert np.allclose(shrink_returns(mu, 0.5), [0.05, 0.1, 0.15])
    with pytest.raises(ValueError):
        shrink_returns(mu, 1.5)


def test_random_portfolios_respect_cap():
    samples = random_portfolios(4, n_samples=2000, max_weight=0.4)
    assert len(samples) > 0
    assert np.allclose(samples.sum(axis=1), 1.0)
    assert samples.max() <= 0.4 + 1e-9


def test_build_strategies_returns_all_four():
    strategies, notes = build_strategies(MU_3, COV_3, risk_free=0.04, max_weight=0.7)
    assert list(strategies) == ["Equal weight", "Minimum variance", "Maximum Sharpe", "Risk parity"]
    assert notes == []
    for w in strategies.values():
        assert np.isclose(w.sum(), 1.0)
        assert w.max() <= 0.7 + 1e-6


def test_build_strategies_skips_max_sharpe_when_undefined():
    strategies, notes = build_strategies(np.array([-0.05, -0.10, -0.02]), COV_3,
                                         risk_free=0.065, max_weight=0.6)
    assert "Maximum Sharpe" not in strategies
    assert len(strategies) == 3
    assert len(notes) == 1