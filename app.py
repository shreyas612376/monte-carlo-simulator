"""Streamlit UI for the Monte Carlo stock simulator (Indian stocks)."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.analytics import summarize_final_prices
from src.config import TRADING_DAYS
from src.data import fetch_prices
from src.export import build_excel_report
from src.models import estimate_parameters, simulate_gbm

MAX_PATHS_SHOWN = 200  # drawing all paths in the browser is slow; stats use all of them

st.set_page_config(page_title="Monte Carlo Stock Simulator", layout="wide")


@st.cache_data(ttl=3600, show_spinner="Downloading price data...")
def load_prices(ticker: str, period: str):
    """Cached wrapper so moving a slider does not re-download data."""
    return fetch_prices(ticker, period)


def build_path_chart(ticker: str, paths: np.ndarray) -> go.Figure:
    """Sample of simulated paths plus 10th / median / 90th percentile bands."""
    days = np.arange(paths.shape[0])
    fig = go.Figure()

    for i in range(min(MAX_PATHS_SHOWN, paths.shape[1])):
        fig.add_trace(go.Scattergl(
            x=days, y=paths[:, i], mode="lines",
            line=dict(width=0.6), opacity=0.25,
            showlegend=False, hoverinfo="skip",
        ))

    bands = [(10, "10th pct", "red"), (50, "Median", "orange"), (90, "90th pct", "green")]
    for q, name, color in bands:
        fig.add_trace(go.Scatter(
            x=days, y=np.percentile(paths, q, axis=1),
            mode="lines", name=name, line=dict(color=color, width=3),
        ))

    fig.update_layout(
        title=f"{ticker}: Simulated Price Paths (showing {min(MAX_PATHS_SHOWN, paths.shape[1])} of {paths.shape[1]})",
        xaxis_title="Trading Days", yaxis_title="Price (Rs)", height=500,
    )
    return fig


def build_histogram(ticker: str, paths: np.ndarray, stats: dict) -> go.Figure:
    """Distribution of final-day prices with percentile markers."""
    fig = go.Figure(go.Histogram(x=paths[-1], nbinsx=50, marker_color="steelblue"))

    markers = [("10th pct", stats["p10"], "red"),
               ("Median", stats["p50"], "orange"),
               ("90th pct", stats["p90"], "green")]
    for name, value, color in markers:
        fig.add_vline(x=value, line_dash="dash", line_color=color, line_width=2,
                      annotation_text=f"{name}: {value:,.2f}")

    fig.update_layout(
        title=f"{ticker}: Distribution of Price at Day {paths.shape[0] - 1}",
        xaxis_title="Final Price (Rs)", yaxis_title="Frequency", height=500,
    )
    return fig


def main() -> None:
    st.title("Monte Carlo Stock Simulator")
    st.caption("Indian stocks (NSE/BSE) | Geometric Brownian Motion")

    # ---- Sidebar inputs ----
    st.sidebar.header("Inputs")
    user_input = st.sidebar.text_input("Stock symbol", value="RELIANCE",
                                       help="Examples: RELIANCE, TCS, HDFCBANK, PNB")
    period = st.sidebar.selectbox("Historical lookback", ["6mo", "1y", "2y", "5y"], index=1)
    n_sims = st.sidebar.slider("Number of simulations", 100, 5000, 1000, step=100)
    n_days = st.sidebar.slider("Forecast horizon (trading days)", 21, 504, TRADING_DAYS)
    seed = int(st.sidebar.number_input("Random seed", value=42, step=1))
    drift_mode = st.sidebar.selectbox(
        "Drift assumption",
        ["Historical", "Zero drift", "Custom"],
        help="Historical drift from 1 year is noisy. Compare how much the forecast depends on it.",
    )
    custom_annual_drift = 0.0
    if drift_mode == "Custom":
        custom_annual_drift = st.sidebar.number_input(
            "Custom annual drift (%)", value=10.0, step=1.0
        ) / 100

    if not user_input.strip():
        st.info("Enter a stock symbol in the sidebar.")
        return

    # ---- Data + model ----
    try:
        ticker, prices = load_prices(user_input, period)
    except ValueError as err:
        st.error(str(err))
        return

    s0 = float(prices.iloc[-1])
    drift, volatility = estimate_parameters(prices) 
    hist_drift = drift  # keep the historical value for display
    if drift_mode == "Zero drift":
        drift = 0.0
    elif drift_mode == "Custom":
        drift = custom_annual_drift / TRADING_DAYS  # annual -> daily
    paths = simulate_gbm(s0, drift, volatility, n_days=n_days, n_sims=n_sims, seed=seed)
    stats = summarize_final_prices(paths)
    report = build_excel_report(
        ticker,
        inputs={
            "Ticker": ticker,
            "Last price (Rs)": s0,
            "Historical lookback": period,
            "Simulations": n_sims,
            "Forecast horizon (trading days)": n_days,
            "Random seed": seed,
            "Drift assumption": drift_mode,
            "Annualized drift used (%)": drift * TRADING_DAYS * 100,
            "Annualized historical drift (%)": hist_drift * TRADING_DAYS * 100,
            "Annualized volatility (%)": volatility * np.sqrt(TRADING_DAYS) * 100,
        },
        stats=stats,
        paths=paths,
    )

    # ---- Summary metrics ----
    st.subheader(ticker)
    c1, c2, c3 = st.columns(3)
    c1.metric("Last price", f"Rs {s0:,.2f}")
    c2.metric(
        "Annualized drift (used)",
        f"{drift * TRADING_DAYS:.2%}",
        help=f"Historical drift: {hist_drift * TRADING_DAYS:.2%}",
    )
    c3.metric("Annualized volatility", f"{volatility * np.sqrt(TRADING_DAYS):.2%}")

    c4, c5, c6, c7 = st.columns(4)
    c4.metric("10th percentile", f"Rs {stats['p10']:,.2f}")
    c5.metric("Median", f"Rs {stats['p50']:,.2f}")
    c6.metric("90th percentile", f"Rs {stats['p90']:,.2f}")
    c7.metric("Mean", f"Rs {stats['mean']:,.2f}")
    st.download_button(
        "Download Excel report",
        data=report,
        file_name=f"{ticker}_monte_carlo.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    # ---- Charts ----
    st.plotly_chart(build_path_chart(ticker, paths), use_container_width=True)
    st.plotly_chart(build_histogram(ticker, paths, stats), use_container_width=True)

    st.caption("Scenario analysis based on historical drift and volatility. "
               "This is not a price prediction or investment advice.")


main()