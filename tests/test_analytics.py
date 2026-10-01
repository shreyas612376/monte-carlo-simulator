"""Tests for risk metrics and agreement with analytical GBM results."""

import numpy as np

from src.analytics import compute_risk_metrics, summarize_final_prices
from src.models import simulate_gbm


def test_prob_loss_known_case():
    # Two rows: day 0 and final day, four simulations. Two of four end below 100.
    paths = np.array([[100, 100, 100, 100],
                      [90, 95, 105, 110]], dtype=float)
    metrics = compute_risk_metrics(paths)
    assert metrics["prob_loss"] == 0.5


def test_cvar_is_at_least_var_and_drawdown_negative():
    paths = simulate_gbm(100, 0.0005, 0.02, seed=3)
    metrics = compute_risk_metrics(paths)
    assert metrics["cvar"] >= metrics["var"]
    assert metrics["median_max_drawdown"] <= 0


def test_simulated_percentile_matches_analytical_gbm():
    """
    Under GBM, ln(S_T / S_0) is normal with mean (mu - 0.5*sigma^2)*T and
    std sigma*sqrt(T). So the 10th percentile is known in closed form.
    """
    s0, mu, sigma, days = 100, 0.0005, 0.01, 252
    z_10 = -1.2815515655446004  # standard normal 10th percentile
    analytical_p10 = s0 * np.exp((mu - 0.5 * sigma**2) * days
                                 + sigma * np.sqrt(days) * z_10)

    paths = simulate_gbm(s0, mu, sigma, n_days=days, n_sims=50000, seed=11)
    simulated_p10 = summarize_final_prices(paths)["p10"]

    assert abs(simulated_p10 - analytical_p10) / analytical_p10 < 0.01