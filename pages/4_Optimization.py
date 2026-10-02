"""Streamlit page: portfolio optimization (minimum variance, maximum Sharpe, risk parity)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics import compute_risk_metrics
from src.config import TRADING_DAYS
from src.optimization import (
    annualize,
    build_strategies,
    efficient_frontier,
    portfolio_performance,
    random_portfolios,
    shrink_returns,
)
from src.portfolio import (
    estimate_portfolio_parameters,
    fetch_multi_prices,
    portfolio_value_paths,
    simulate_correlated_gbm,
)

PERCENT_COLUMNS = ["Expected return", "Volatility", "VaR 95%", "CVaR 95%",
                   "Prob. of loss", "Median max drawdown"]

st.set_page_config(page_title="Optimization | Monte Carlo Simulator", layout="wide")


@st.cache_data(ttl=3600, show_spinner="Downloading price data...")
def load_multi_prices(symbols: tuple, period: str):
    """Cached download and alignment of several stocks."""
    return fetch_multi_prices(list(symbols), period)


def build_frontier_chart(tickers, mu, cov, frontier, strategies, max_weight) -> go.Figure:
    """Risk-return plane: random portfolios, efficient frontier, strategies and single stocks."""
    fig = go.Figure()

    cloud = random_portfolios(len(tickers), n_samples=3000, max_weight=max_weight)
    if len(cloud) > 0:
        cloud_vol = np.sqrt(np.einsum("ij,jk,ik->i", cloud, cov, cloud))
        fig.add_trace(go.Scatter(
            x=cloud_vol, y=cloud @ mu, mode="markers", name="Random portfolios",
            marker=dict(size=4, color="rgba(150,150,150,0.35)"), hoverinfo="skip",
        ))

    if frontier:
        fig.add_trace(go.Scatter(
            x=[p["volatility"] for p in frontier], y=[p["return"] for p in frontier],
            mode="lines", name="Efficient frontier", line=dict(color="orange", width=3),
        ))

    fig.add_trace(go.Scatter(
        x=np.sqrt(np.diag(cov)), y=mu, mode="markers+text", name="Single stocks",
        text=tickers, textposition="top center",
        marker=dict(size=10, symbol="diamond", color="lightskyblue"),
    ))

    for name, w in strategies.items():
        perf = portfolio_performance(w, mu, cov)
        fig.add_trace(go.Scatter(
            x=[perf["volatility"]], y=[perf["return"]], mode="markers+text",
            name=name, text=[name], textposition="bottom center",
            marker=dict(size=13, line=dict(width=1, color="white")),
        ))

    fig.update_layout(
        title="Risk and return of each strategy",
        xaxis_title="Volatility (annualized)", yaxis_title="Expected return (annualized)",
        xaxis_tickformat=".0%", yaxis_tickformat=".0%", height=560,
    )
    return fig


def build_weights_chart(tickers, strategies) -> go.Figure:
    """Stacked bars: how each strategy splits the capital."""
    fig = go.Figure()
    names = list(strategies)
    for i, ticker in enumerate(tickers):
        fig.add_trace(go.Bar(name=ticker, x=names, y=[strategies[n][i] for n in names]))
    fig.update_layout(barmode="stack", title="Weights by strategy",
                      yaxis_tickformat=".0%", height=420)
    return fig


st.title("Portfolio Optimization")
st.caption("Minimum variance, maximum Sharpe and risk parity | long-only | NSE/BSE")

# ---- Sidebar inputs ----
st.sidebar.header("Optimization inputs")
symbols_text = st.sidebar.text_area("Stocks (comma separated, 2 to 10)",
                                    value="RELIANCE, TCS, HDFCBANK, INFY")
period = st.sidebar.selectbox(
    "Historical lookback", ["2y", "5y", "10y"], index=2,
    help="Expected returns are very noisy over short windows. Longer history is more stable.",
)
risk_free = st.sidebar.number_input(
    "Risk-free rate (% per year)", 0.0, 20.0, 6.5, step=0.25,
    help="Used for the Sharpe ratio. Set it to the current government bond yield.",
) / 100
max_weight = st.sidebar.slider(
    "Max weight per stock (%)", 10, 100, 40, step=5,
    help="Caps concentration. Without a cap, optimizers often put everything in one or two stocks.",
) / 100
shrinkage = st.sidebar.slider(
    "Return shrinkage", 0.0, 1.0, 0.5, step=0.1,
    help="0 = trust historical average returns fully. 1 = assume every stock has the same "
         "expected return. Higher values reduce the effect of noisy return estimates.",
)
n_sims = st.sidebar.slider("Simulations (risk comparison)", 100, 3000, 1000, step=100)
n_days = st.sidebar.slider("Forecast horizon (trading days)", 21, 504, TRADING_DAYS)
seed = int(st.sidebar.number_input("Random seed", value=42, step=1))
sim_drift_label = st.sidebar.selectbox(
    "Drift in the risk simulation", ["Zero drift", "Same expected returns as optimizer"],
    help="Zero drift isolates pure risk differences between strategies.",
)

unique_symbols = list(dict.fromkeys(
    s.strip().upper() for s in symbols_text.replace("\n", ",").split(",") if s.strip()
))
if len(unique_symbols) < 2:
    st.info("Enter at least 2 stock symbols in the sidebar.")
    st.stop()

# ---- Data + estimates ----
try:
    prices = load_multi_prices(tuple(unique_symbols), period)
except ValueError as err:
    st.error(str(err))
    st.stop()

tickers = list(prices.columns)
if max_weight * len(tickers) < 1:
    st.sidebar.error(f"With {len(tickers)} stocks the cap must be at least "
                     f"{100 / len(tickers):.0f}%.")
    st.stop()

params = estimate_portfolio_parameters(prices)
mu_hist, cov = annualize(params["mean"].values, params["cov"].values)
mu = shrink_returns(mu_hist, shrinkage)

try:
    strategies, notes = build_strategies(mu, cov, risk_free, max_weight)
    frontier = efficient_frontier(mu, cov, n_points=30, max_weight=max_weight)
except ValueError as err:
    st.error(str(err))
    st.stop()

# ---- Risk simulation shared by all strategies ----
daily_drift = np.zeros(len(tickers)) if sim_drift_label == "Zero drift" else mu / TRADING_DAYS
asset_paths = simulate_correlated_gbm(
    prices.iloc[-1].values, daily_drift, params["vol"].values, params["corr"].values,
    n_days=n_days, n_sims=n_sims, seed=seed,
)

rows = {}
for name, w in strategies.items():
    perf = portfolio_performance(w, mu, cov, risk_free)
    risk = compute_risk_metrics(portfolio_value_paths(asset_paths, w))
    row = {ticker: weight for ticker, weight in zip(tickers, w)}
    row.update({
        "Expected return": perf["return"],
        "Volatility": perf["volatility"],
        "Sharpe": perf["sharpe"],
        "VaR 95%": risk["var"],
        "CVaR 95%": risk["cvar"],
        "Prob. of loss": risk["prob_loss"],
        "Median max drawdown": risk["median_max_drawdown"],
    })
    rows[name] = row
table = pd.DataFrame.from_dict(rows, orient="index")

# ---- Output ----
st.subheader(", ".join(tickers))
st.caption(f"{len(prices)} common trading days used. Risk columns come from "
           f"{n_sims} correlated simulations over {n_days} days, identical for every strategy.")

for note in notes:
    st.warning(note)

shown = table.copy()
for ticker in tickers:
    shown[ticker] = shown[ticker].map(lambda v: f"{v:.1%}")
for col in PERCENT_COLUMNS:
    shown[col] = shown[col].map(lambda v: f"{v:.1%}")
shown["Sharpe"] = shown["Sharpe"].map(lambda v: f"{v:.2f}")
st.dataframe(shown, use_container_width=True)

st.plotly_chart(build_frontier_chart(tickers, mu, cov, frontier, strategies, max_weight),
                use_container_width=True)
st.plotly_chart(build_weights_chart(tickers, strategies), use_container_width=True)

with st.expander("Return and volatility inputs used"):
    inputs = pd.DataFrame({
        "Historical avg return": mu_hist,
        "Return used (after shrinkage)": mu,
        "Volatility": np.sqrt(np.diag(cov)),
    }, index=tickers)
    st.dataframe(inputs.map(lambda v: f"{v:.1%}"), use_container_width=True)

with st.expander("How to read this page"):
    st.markdown(
        "- **Minimum variance**: the long-only mix with the lowest possible volatility.\n"
        "- **Maximum Sharpe**: the best return per unit of risk above the risk-free rate. "
        "It depends heavily on expected returns, which are the noisiest input.\n"
        "- **Risk parity**: every stock contributes the same share of portfolio risk.\n"
        "- **Equal weight**: the naive benchmark that optimizers must beat.\n"
        "- **Efficient frontier**: for each return level, the lowest volatility reachable "
        "with the weight cap.\n"
        "- Weights are fitted on past data (in-sample). That is not evidence they will beat "
        "equal weight in the future."
    )

st.caption("Optimized weights are estimates from historical data and are very sensitive to "
           "the inputs. Scenario analysis, not investment advice.")