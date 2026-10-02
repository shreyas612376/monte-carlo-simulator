"""Streamlit page: walk-forward backtest of portfolio forecasts."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.backtest import summarize_backtest
from src.portfolio import fetch_multi_prices
from src.portfolio_backtest import run_portfolio_backtest

PERCENT_COLUMNS = ["Coverage 80%", "Coverage 90%", "Below 5th pct", "Above 95th pct"]

st.set_page_config(page_title="Portfolio Backtest | Monte Carlo Simulator", layout="wide")


@st.cache_data(ttl=3600, show_spinner="Downloading price data...")
def load_multi_prices(symbols: tuple, period: str):
    """Cached download and alignment of several stocks."""
    return fetch_multi_prices(list(symbols), period)


@st.cache_data(ttl=3600, show_spinner="Running portfolio backtest...")
def cached_portfolio_backtest(prices, weights, horizon, lookback, step,
                              n_sims, drift_mode, t_df, seed):
    """Cached backtest so identical inputs return instantly."""
    return run_portfolio_backtest(prices, np.array(weights), horizon=horizon,
                                  lookback=lookback, step=step, n_sims=n_sims,
                                  drift_mode=drift_mode, t_df=t_df, seed=seed)


def build_band_chart(grp, model: str) -> go.Figure:
    """Forecast 10th-90th band vs the portfolio value that actually happened."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grp["target_date"], y=grp["p90"], mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=grp["target_date"], y=grp["p10"], mode="lines",
                             line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(70,130,180,0.3)",
                             name="10th-90th percentile band"))
    fig.add_trace(go.Scatter(x=grp["target_date"], y=grp["p50"], mode="lines",
                             line=dict(color="orange", dash="dash"), name="Median forecast"))
    fig.add_trace(go.Scatter(x=grp["target_date"], y=grp["actual"], mode="lines+markers",
                             line=dict(color="red", width=1.5), marker=dict(size=4),
                             name="Actual portfolio value"))
    fig.update_layout(title=f"{model}: forecast range vs actual portfolio value",
                      xaxis_title="Target date",
                      yaxis_title="Portfolio value (start of each forecast = 1.00)",
                      height=450)
    return fig


def build_pit_chart(grp, model: str) -> go.Figure:
    """Histogram of PIT values. A calibrated model gives a flat histogram."""
    fig = go.Figure(go.Histogram(x=grp["pit"], xbins=dict(start=0, end=1, size=0.1),
                                 histnorm="probability", marker_color="steelblue"))
    fig.add_hline(y=0.1, line_dash="dash", line_color="orange", annotation_text="Ideal (flat)")
    fig.update_layout(title=f"{model}: where the actual portfolio value fell in the forecast",
                      xaxis_title="PIT (0 = below all simulations, 1 = above all)",
                      yaxis_title="Share of forecasts", height=400)
    return fig


st.title("Portfolio Backtest")
st.caption("Did the actual portfolio value land inside each model's forecast range?")

# ---- Sidebar inputs ----
st.sidebar.header("Backtest inputs")
symbols_text = st.sidebar.text_area("Stocks (comma separated, 2 to 10)",
                                    value="RELIANCE, TCS, HDFCBANK, INFY")
history = st.sidebar.selectbox("Price history", ["5y", "10y"], index=1)
unique_symbols = list(dict.fromkeys(
    s.strip().upper() for s in symbols_text.replace("\n", ",").split(",") if s.strip()
))

weight_mode = st.sidebar.radio("Weights", ["Equal weight", "Custom weights"])
if weight_mode == "Custom weights" and len(unique_symbols) >= 2:
    raw_weights = np.array([
        st.sidebar.number_input(f"{sym} weight (%)", 0.0, 100.0,
                                round(100 / len(unique_symbols), 1), step=5.0, key=f"pbw_{sym}")
        for sym in unique_symbols
    ])
    if raw_weights.sum() == 0:
        st.sidebar.error("At least one weight must be above 0.")
        st.stop()
    weights = raw_weights / raw_weights.sum()
else:
    weights = np.full(max(len(unique_symbols), 1), 1 / max(len(unique_symbols), 1))

lookback = st.sidebar.slider("Estimation window (trading days)", 252, 756, 504, step=21)
horizon = st.sidebar.slider("Forecast horizon (trading days)", 21, 126, 63, step=21)
step = st.sidebar.slider("Days between forecast dates", 5, 126, 21,
                         help="If smaller than the horizon, consecutive forecasts overlap. "
                              "Set it equal to the horizon for independent forecasts.")
n_sims = st.sidebar.slider("Simulations per forecast", 100, 1000, 500, step=100)
drift_label = st.sidebar.selectbox("Drift assumption", ["Historical", "Zero drift"])
t_df = st.sidebar.slider("Student-t degrees of freedom", 3, 30, 5)
seed = int(st.sidebar.number_input("Random seed", value=42, step=1))
run = st.sidebar.button("Run backtest", type="primary")

if len(unique_symbols) < 2:
    st.info("Enter at least 2 stock symbols in the sidebar.")
    st.stop()

# ---- Run ----
if run:
    try:
        prices = load_multi_prices(tuple(unique_symbols), history)
        results = cached_portfolio_backtest(
            prices, tuple(weights), horizon, lookback, step, n_sims,
            "zero" if drift_label == "Zero drift" else "historical", t_df, seed,
        )
    except ValueError as err:
        st.error(str(err))
        st.stop()
    st.session_state["pbt_results"] = results
    st.session_state["pbt_meta"] = {
        "tickers": list(prices.columns), "horizon": horizon, "step": step,
        "weights": [float(w) for w in weights], "days": len(prices),
    }

if "pbt_results" not in st.session_state:
    st.info("Set the inputs in the sidebar and click **Run backtest**.")
    st.stop()

results = st.session_state["pbt_results"]
meta = st.session_state["pbt_meta"]

# ---- Summary ----
n_origins = results["origin_date"].nunique()
weights_text = ", ".join(f"{t} {w:.0%}" for t, w in zip(meta["tickers"], meta["weights"]))
st.subheader(f"{n_origins} forecast dates, {meta['horizon']}-day horizon")
st.caption(f"Portfolio: {weights_text} | {meta['days']} common trading days of history")

summary = summarize_backtest(results)
display = summary.copy()
for col in PERCENT_COLUMNS:
    display[col] = display[col].map(lambda v: f"{v:.1%}")
display["Mean PIT"] = display["Mean PIT"].map(lambda v: f"{v:.3f}")
display["Kupiec p-value"] = display["Kupiec p-value"].map(lambda v: f"{v:.3f}")
st.dataframe(display, use_container_width=True)

if meta["step"] < meta["horizon"]:
    st.warning("Forecast windows overlap (days between forecasts < horizon), so observations "
               "are correlated and p-values look better than they should. Set the step equal "
               "to the horizon for a stricter test.")

with st.expander("How to read this table"):
    st.markdown(
        "- **Coverage 80% / 90%**: share of times the actual portfolio value fell inside the "
        "10th-90th / 5th-95th percentile range. Ideal: 80% / 90%.\n"
        "- **Below 5th pct**: how often the portfolio ended below the model's 5th percentile. "
        "This is the VaR 95% breach rate. Ideal: 5%. Much higher means the model "
        "underestimates portfolio risk.\n"
        "- **Mean PIT**: average position of the actual value inside the forecast. Ideal: 0.5. "
        "Below 0.5 means the model was too optimistic.\n"
        "- **Kupiec p-value**: tests whether the 5% breach rate matches the model. "
        "Below 0.05 means the lower tail is statistically miscalibrated.\n"
        "- **Historical bootstrap** is the industry benchmark: it resamples real past days "
        "(all stocks together), so it needs no distribution assumption."
    )

# ---- Charts for one model ----
model = st.selectbox("Inspect model", list(summary.index))
grp = results[results["model"] == model]
st.plotly_chart(build_band_chart(grp, model), use_container_width=True)
st.plotly_chart(build_pit_chart(grp, model), use_container_width=True)

st.caption("Scenario analysis, not investment advice. Each forecast uses only data "
           "available at its forecast date.")