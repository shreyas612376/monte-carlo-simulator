"""Streamlit page: correlated multi-stock portfolio simulation."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics import compute_risk_metrics, summarize_final_prices
from src.config import TRADING_DAYS
from src.portfolio import (
    diversification_summary,
    estimate_portfolio_parameters,
    fetch_multi_prices,
    portfolio_value_paths,
    risk_contributions,
    simulate_correlated_gbm,
)

MAX_PATHS_SHOWN = 200

st.set_page_config(page_title="Portfolio | Monte Carlo Simulator", layout="wide")


@st.cache_data(ttl=3600, show_spinner="Downloading price data...")
def load_multi_prices(symbols: tuple, period: str) -> pd.DataFrame:
    """Cached download and alignment of several stocks."""
    return fetch_multi_prices(list(symbols), period)


def rs(value: float) -> str:
    return f"Rs {value:,.0f}"


def build_corr_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=list(corr.columns), y=list(corr.index),
        zmin=-1, zmax=1, colorscale="RdBu_r",
        text=np.round(corr.values, 2), texttemplate="%{text}",
    ))
    fig.update_layout(title="Correlation of daily returns", height=420)
    return fig


def build_contribution_chart(tickers, weights, contrib) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=tickers, y=weights, name="Capital weight"))
    fig.add_trace(go.Bar(x=tickers, y=contrib, name="Share of portfolio risk"))
    fig.update_layout(barmode="group", title="Capital weight vs share of portfolio risk",
                      yaxis_tickformat=".0%", height=400)
    return fig


def build_value_chart(paths: np.ndarray) -> go.Figure:
    days = np.arange(paths.shape[0])
    fig = go.Figure()
    for i in range(min(MAX_PATHS_SHOWN, paths.shape[1])):
        fig.add_trace(go.Scattergl(
            x=days, y=paths[:, i], mode="lines",
            line=dict(width=0.6), opacity=0.25, showlegend=False, hoverinfo="skip",
        ))
    for q, name, color in [(10, "10th pct", "red"), (50, "Median", "orange"), (90, "90th pct", "green")]:
        fig.add_trace(go.Scatter(x=days, y=np.percentile(paths, q, axis=1),
                                 mode="lines", name=name, line=dict(color=color, width=3)))
    fig.update_layout(
        title=f"Simulated portfolio value (showing {min(MAX_PATHS_SHOWN, paths.shape[1])} of {paths.shape[1]} paths)",
        xaxis_title="Trading Days", yaxis_title="Portfolio value (Rs)", height=500,
    )
    return fig


def build_value_histogram(paths: np.ndarray, stats: dict) -> go.Figure:
    fig = go.Figure(go.Histogram(x=paths[-1], nbinsx=50, marker_color="steelblue"))
    for name, value, color in [("10th pct", stats["p10"], "red"),
                               ("Median", stats["p50"], "orange"),
                               ("90th pct", stats["p90"], "green")]:
        fig.add_vline(x=value, line_dash="dash", line_color=color, line_width=2,
                      annotation_text=f"{name}: {rs(value)}")
    fig.update_layout(title=f"Distribution of portfolio value at Day {paths.shape[0] - 1}",
                      xaxis_title="Final value (Rs)", yaxis_title="Frequency", height=450)
    return fig


st.title("Portfolio Simulator")
st.caption("Correlated multi-stock Monte Carlo (NSE/BSE) | buy-and-hold, no rebalancing")

# ---- Sidebar inputs ----
st.sidebar.header("Portfolio inputs")
symbols_text = st.sidebar.text_area("Stocks (comma separated, 2 to 10)",
                                    value="RELIANCE, TCS, HDFCBANK, INFY")
period = st.sidebar.selectbox("Historical lookback", ["1y", "2y", "5y"], index=1)
investment = st.sidebar.number_input("Investment amount (Rs)", min_value=1000.0,
                                     value=100000.0, step=10000.0)
n_sims = st.sidebar.slider("Number of simulations", 100, 3000, 1000, step=100)
n_days = st.sidebar.slider("Forecast horizon (trading days)", 21, 504, TRADING_DAYS)
seed = int(st.sidebar.number_input("Random seed", value=42, step=1))
drift_label = st.sidebar.selectbox("Drift assumption", ["Historical", "Zero drift"])
shock_label = st.sidebar.selectbox("Shock distribution", ["Normal", "Student-t (fat tails)"],
                                   help="Student-t also makes stocks crash together (tail dependence).")
shock = "student_t" if shock_label.startswith("Student") else "normal"
t_df = 5
if shock == "student_t":
    t_df = st.sidebar.slider("Degrees of freedom", 3, 30, 5)

unique_symbols = list(dict.fromkeys(
    s.strip().upper() for s in symbols_text.replace("\n", ",").split(",") if s.strip()
))

weight_mode = st.sidebar.radio("Weights", ["Equal weight", "Custom weights"])
if weight_mode == "Custom weights" and len(unique_symbols) >= 2:
    raw_weights = np.array([
        st.sidebar.number_input(f"{sym} weight (%)", 0.0, 100.0,
                                round(100 / len(unique_symbols), 1), step=5.0, key=f"w_{sym}")
        for sym in unique_symbols
    ])
    if raw_weights.sum() == 0:
        st.sidebar.error("At least one weight must be above 0.")
        st.stop()
    weights = raw_weights / raw_weights.sum()   # normalised to 100%
else:
    weights = np.full(max(len(unique_symbols), 1), 1 / max(len(unique_symbols), 1))

if len(unique_symbols) < 2:
    st.info("Enter at least 2 stock symbols in the sidebar.")
    st.stop()

# ---- Data + model ----
try:
    prices = load_multi_prices(tuple(unique_symbols), period)
except ValueError as err:
    st.error(str(err))
    st.stop()

tickers = list(prices.columns)
params = estimate_portfolio_parameters(prices)
s0 = prices.iloc[-1].values
hist_drift = params["mean"].values
drift = np.zeros_like(hist_drift) if drift_label == "Zero drift" else hist_drift
vol = params["vol"].values
cov = params["cov"].values

asset_paths = simulate_correlated_gbm(
    s0, drift, vol, params["corr"].values,
    n_days=n_days, n_sims=n_sims, seed=seed, shock=shock, df=t_df,
)
portfolio_paths = portfolio_value_paths(asset_paths, weights, investment)
stats = summarize_final_prices(portfolio_paths)
risk = compute_risk_metrics(portfolio_paths)
div = diversification_summary(asset_paths, weights, portfolio_paths, cov, vol)
contrib = risk_contributions(weights, cov)

# ---- Headline metrics ----
st.subheader(f"Portfolio: {', '.join(tickers)}")
st.caption(f"{len(prices)} common trading days used to estimate drift, volatility and correlation.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Investment", rs(investment))
c2.metric("10th percentile", rs(stats["p10"]))
c3.metric("Median", rs(stats["p50"]))
c4.metric("90th percentile", rs(stats["p90"]))

c5, c6, c7, c8 = st.columns(4)
c5.metric("VaR 95%", f"{risk['var']:.2%}",
          help=f"Loss exceeded in only 5% of scenarios: about {rs(investment * risk['var'])}")
c6.metric("CVaR 95%", f"{risk['cvar']:.2%}",
          help=f"Average loss in the worst 5% of scenarios: about {rs(investment * risk['cvar'])}")
c7.metric("Probability of loss", f"{risk['prob_loss']:.1%}")
c8.metric("Median max drawdown", f"{risk['median_max_drawdown']:.1%}")

# ---- Diversification ----
st.subheader("Diversification")
ann = np.sqrt(TRADING_DAYS)
d1, d2, d3, d4 = st.columns(4)
d1.metric("Portfolio volatility (ann.)", f"{div['portfolio_vol_daily'] * ann:.2%}")
d2.metric("Weighted avg stock volatility", f"{div['weighted_avg_vol_daily'] * ann:.2%}")
d3.metric("Diversification ratio", f"{div['diversification_ratio']:.2f}x",
          help="Weighted average volatility divided by portfolio volatility. 1.00x means no benefit.")
d4.metric("VaR reduction", f"{div['var_diversification_benefit']:.2%}",
          help=f"Weighted average standalone VaR {div['weighted_avg_var']:.2%} "
               f"vs portfolio VaR {div['portfolio_var']:.2%}")

asset_table = pd.DataFrame({
    "Weight": weights,
    "Last price (Rs)": s0,
    "Ann. drift (historical)": hist_drift * TRADING_DAYS,
    "Ann. volatility": vol * ann,
    "Standalone VaR 95%": div["standalone_var"],
    "Share of portfolio risk": contrib,
}, index=tickers)
shown = asset_table.copy()
shown["Last price (Rs)"] = shown["Last price (Rs)"].map(lambda v: f"{v:,.2f}")
for col in ["Weight", "Ann. drift (historical)", "Ann. volatility",
            "Standalone VaR 95%", "Share of portfolio risk"]:
    shown[col] = shown[col].map(lambda v: f"{v:.1%}")
st.dataframe(shown, use_container_width=True)

left, right = st.columns(2)
left.plotly_chart(build_corr_heatmap(params["corr"]), use_container_width=True)
right.plotly_chart(build_contribution_chart(tickers, weights, contrib), use_container_width=True)

# ---- Charts ----
st.plotly_chart(build_value_chart(portfolio_paths), use_container_width=True)
st.plotly_chart(build_value_histogram(portfolio_paths, stats), use_container_width=True)

st.caption("Correlations are estimated from history and kept constant. In market crashes "
           "correlations usually rise, so real portfolio risk can be higher than shown. "
           "Scenario analysis, not investment advice.")