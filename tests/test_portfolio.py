"""Tests for correlated simulation and portfolio risk."""

import numpy as np
import pytest

from src.portfolio import (
    _safe_cholesky,
    diversification_summary,
    portfolio_value_paths,
    risk_contributions,
    simulate_correlated_gbm,
)

S0 = [100.0, 50.0]
DRIFT = [0.0003, 0.0002]
VOL = [0.01, 0.02]


def _corr(rho):
    return np.array([[1.0, rho], [rho, 1.0]])


def _log_returns(paths):
    return np.diff(np.log(paths), axis=0).reshape(-1, paths.shape[2])


def test_shape_start_prices_and_positivity():
    paths = simulate_correlated_gbm(S0, DRIFT, VOL, _corr(0.5), n_days=30, n_sims=100)
    assert paths.shape == (31, 100, 2)
    assert np.all(paths[0] == np.array(S0))
    assert np.all(paths > 0)


def test_simulated_correlation_matches_target():
    paths = simulate_correlated_gbm(S0, DRIFT, VOL, _corr(0.8), n_days=252, n_sims=500, seed=1)
    assert abs(np.corrcoef(_log_returns(paths).T)[0, 1] - 0.8) < 0.02


def test_student_t_keeps_target_correlation():
    paths = simulate_correlated_gbm(S0, DRIFT, VOL, _corr(0.8), n_days=252, n_sims=500,
                                    seed=2, shock="student_t", df=5)
    assert abs(np.corrcoef(_log_returns(paths).T)[0, 1] - 0.8) < 0.05


def test_each_stock_matches_gbm_expected_value():
    s0, mu, sigma, days = 100.0, 0.0005, 0.01, 252
    paths = simulate_correlated_gbm([s0, s0], [mu, mu], [sigma, sigma], _corr(0.3),
                                    n_days=days, n_sims=5000, seed=3)
    expected = s0 * np.exp(mu * days)
    for i in range(2):
        assert abs(paths[-1, :, i].mean() - expected) / expected < 0.01


def test_portfolio_start_value_and_single_stock_case():
    asset_paths = simulate_correlated_gbm(S0, DRIFT, VOL, _corr(0.5), n_days=40, n_sims=200)
    portfolio = portfolio_value_paths(asset_paths, [0.6, 0.4], investment=1000)
    assert np.allclose(portfolio[0], 1000)

    only_first = portfolio_value_paths(asset_paths, [1.0, 0.0], investment=1000)
    assert np.allclose(only_first, 1000 * asset_paths[:, :, 0] / S0[0])


def test_weights_must_sum_to_one():
    asset_paths = simulate_correlated_gbm(S0, DRIFT, VOL, _corr(0.5), n_days=10, n_sims=20)
    with pytest.raises(ValueError):
        portfolio_value_paths(asset_paths, [0.7, 0.7])


def test_lower_correlation_means_lower_portfolio_risk():
    def final_std(rho):
        paths = simulate_correlated_gbm([100, 100], [0, 0], [0.02, 0.02], _corr(rho),
                                        n_days=252, n_sims=3000, seed=5)
        return portfolio_value_paths(paths, [0.5, 0.5])[-1].std()

    assert final_std(0.0) < final_std(0.95)


def test_risk_contributions_sum_to_one_and_split_equally_for_identical_stocks():
    cov = np.array([[0.0004, 0.0001], [0.0001, 0.0004]])
    contrib = risk_contributions([0.5, 0.5], cov)
    assert np.isclose(contrib.sum(), 1.0)
    assert np.allclose(contrib, [0.5, 0.5])


def test_riskier_stock_contributes_more_risk():
    cov = np.diag([0.0001, 0.0009])
    contrib = risk_contributions([0.5, 0.5], cov)
    assert np.allclose(contrib, [0.1, 0.9])


def test_cholesky_handles_singular_matrix():
    corr = np.ones((2, 2))   # two perfectly identical stocks
    chol = _safe_cholesky(corr)
    assert np.allclose(chol @ chol.T, corr, atol=1e-3)


def test_diversification_summary():
    sigma, rho = 0.015, 0.3
    paths = simulate_correlated_gbm([100, 100], [0.0003] * 2, [sigma] * 2, _corr(rho),
                                    n_days=126, n_sims=2000, seed=7)
    weights = np.array([0.5, 0.5])
    portfolio = portfolio_value_paths(paths, weights)
    cov = sigma**2 * _corr(rho)
    summary = diversification_summary(paths, weights, portfolio, cov, np.array([sigma, sigma]))
    assert summary["diversification_ratio"] > 1.0
    assert summary["portfolio_var"] < summary["weighted_avg_var"]