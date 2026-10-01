"""Streamlit UI for the Monte Carlo stock simulator (Indian stocks)."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.analytics import compute_risk_metrics, summarize_final_prices
from src.config import TRADING_DAYS
from src.data import fetch_prices
from src.export import build_excel_report
from src.models import (
    estimate_ewma_volatility,
    estimate_parameters,
    fit_garch,
    simulate_garch,
    simulate_gbm,
)

MAX_PATHS_SHOWN = 200  # drawing all paths in the browser is slow; stats use all of them

st.set_page_config(page_title="Monte Carlo Stock Simulator", layout="wide")


@st.cache_data(ttl=3600, show_spinner="Downloading price data...")
def load_prices(ticker: str, period: str):
    """Cached wrapper so moving a slider does not re-download data."""
    return fetch_prices(ticker, period)


@st.cache_data(ttl=3600, show_spinner="Fitting GARCH(1,1)...")
def load_garch(ticker: str, period: str) -> dict:
    """Cached GARCH fit so moving a slider does not refit the model."""
    _, prices = load_prices(ticker, period)
    return fit_garch(prices)


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

    shock_label = st.sidebar.selectbox(
        "Shock distribution",
        ["Normal", "Student-t (fat tails)"],
        help="Student-t makes extreme daily moves more likely than the normal distribution.",
    )
    shock = "student_t" if shock_label.startswith("Student") else "normal"
    t_df = 5
    if shock == "student_t":
        t_df = st.sidebar.slider("Degrees of freedom", 3, 30, 5,
                                 help="Lower = fatter tails. 30 is almost normal.")

    vol_label = st.sidebar.selectbox(
        "Volatility estimate",
        [
            "Historical (equal weight)",
            "EWMA (recent days weighted more)",
            "GARCH(1,1) (volatility evolves daily)",
        ],
        help="EWMA weights recent days more. GARCH also lets volatility cluster and "
             "mean-revert during the simulation (use a 2y or 5y lookback).",
    )
    use_ewma = vol_label.startswith("EWMA")
    use_garch = vol_label.startswith("GARCH")
    ewma_lambda = 0.94
    if use_ewma:
        ewma_lambda = st.sidebar.slider(
            "EWMA lambda", 0.80, 0.99, 0.94, step=0.01,
            help="Higher = longer memory. 0.94 is the RiskMetrics standard for daily data.",
        )

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

    hist_volatility = volatility  # keep the equal-weight value for display
    garch = None
    if use_ewma:
        volatility = estimate_ewma_volatility(prices, ewma_lambda)
    elif use_garch:
        if len(prices) < 500:
            st.warning("GARCH needs about 2 years of data for stable estimates. "
                       "Select a 2y or 5y lookback.")
        try:
            garch = load_garch(user_input, period)
        except Exception as err:
            st.error(f"GARCH fit failed: {err}")
            return
        volatility = float(np.sqrt(garch["next_variance"]))  # starting daily volatility

    hist_drift = drift  # keep the historical value for display
    if drift_mode == "Zero drift":
        drift = 0.0
    elif drift_mode == "Custom":
        drift = custom_annual_drift / TRADING_DAYS  # annual -> daily

    if use_garch:
        paths = simulate_garch(
            s0, drift, garch["omega"], garch["alpha"], garch["beta"],
            garch["next_variance"], n_days=n_days, n_sims=n_sims,
            seed=seed, shock=shock, df=t_df,
        )
    else:
        paths = simulate_gbm(s0, drift, volatility, n_days=n_days, n_sims=n_sims,
                             seed=seed, shock=shock, df=t_df)
    stats = summarize_final_prices(paths)
    risk = compute_risk_metrics(paths)

    report_inputs = {
        "Ticker": ticker,
        "Last price (Rs)": s0,
        "Historical lookback": period,
        "Simulations": n_sims,
        "Forecast horizon (trading days)": n_days,
        "Random seed": seed,
        "Drift assumption": drift_mode,
        "Shock distribution": shock_label,
        "Annualized drift used (%)": drift * TRADING_DAYS * 100,
        "Annualized historical drift (%)": hist_drift * TRADING_DAYS * 100,
        "Annualized volatility (%)": volatility * np.sqrt(TRADING_DAYS) * 100,
        "Volatility estimate": vol_label,
        "Annualized equal-weight volatility (%)": hist_volatility * np.sqrt(TRADING_DAYS) * 100,
    }
    if garch is not None:
        report_inputs.update({
            "GARCH alpha": garch["alpha"],
            "GARCH beta": garch["beta"],
            "GARCH alpha + beta": garch["persistence"],
            "GARCH long-run volatility (%)": garch["long_run_vol"] * np.sqrt(TRADING_DAYS) * 100,
        })
    report = build_excel_report(ticker, inputs=report_inputs, stats=stats, paths=paths)

    # ---- Summary metrics ----
    st.subheader(ticker)
    vol_help = f"Equal-weight historical volatility: {hist_volatility * np.sqrt(TRADING_DAYS):.2%}"
    if use_garch:
        vol_help += ". Under GARCH this is the starting volatility; it evolves daily."

    c1, c2, c3 = st.columns(3)
    c1.metric("Last price", f"Rs {s0:,.2f}")
    c2.metric(
        "Annualized drift (used)",
        f"{drift * TRADING_DAYS:.2%}",
        help=f"Historical drift: {hist_drift * TRADING_DAYS:.2%}",
    )
    c3.metric(
        "Annualized volatility (used)",
        f"{volatility * np.sqrt(TRADING_DAYS):.2%}",
        help=vol_help,
    )

    c4, c5, c6, c7 = st.columns(4)
    c4.metric("10th percentile", f"Rs {stats['p10']:,.2f}")
    c5.metric("Median", f"Rs {stats['p50']:,.2f}")
    c6.metric("90th percentile", f"Rs {stats['p90']:,.2f}")
    c7.metric("Mean", f"Rs {stats['mean']:,.2f}")

    c8, c9, c10, c11 = st.columns(4)
    c8.metric("VaR 95%", f"{risk['var']:.2%}",
              help="Loss exceeded in only 5% of scenarios (vs today's price)")
    c9.metric("CVaR 95%", f"{risk['cvar']:.2%}",
              help="Average loss in the worst 5% of scenarios")
    c10.metric("Probability of loss", f"{risk['prob_loss']:.1%}",
               help="Share of scenarios ending below today's price")
    c11.metric("Median max drawdown", f"{risk['median_max_drawdown']:.1%}",
               help="Typical worst peak-to-trough fall along a path")

    if garch is not None:
        with st.expander("GARCH(1,1) parameters"):
            g1, g2, g3, g4 = st.columns(4)
            g1.metric("alpha (reaction to shocks)", f"{garch['alpha']:.3f}")
            g2.metric("beta (volatility persistence)", f"{garch['beta']:.3f}")
            g3.metric("alpha + beta", f"{garch['persistence']:.3f}",
                      help="Close to 1 means volatility clusters last a long time.")
            long_run = garch["long_run_vol"]
            g4.metric("Long-run volatility",
                      f"{long_run * np.sqrt(TRADING_DAYS):.2%}" if np.isfinite(long_run) else "n/a",
                      help="Annualized level volatility mean-reverts to (needs alpha + beta < 1).")

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