"""Streamlit page: walk-forward backtest of the simulation models."""

import plotly.graph_objects as go
import streamlit as st

from src.backtest import run_backtest, summarize_backtest
from src.data import fetch_prices

PERCENT_COLUMNS = ["Coverage 80%", "Coverage 90%", "Below 5th pct", "Above 95th pct"]

st.set_page_config(page_title="Backtest | Monte Carlo Simulator", layout="wide")


@st.cache_data(ttl=3600, show_spinner="Downloading price data...")
def load_prices(ticker: str, period: str):
    """Cached download so re-running the backtest does not re-fetch data."""
    return fetch_prices(ticker, period)


@st.cache_data(ttl=3600, show_spinner="Running walk-forward backtest (can take up to a minute with GARCH)...")
def cached_backtest(prices, horizon, lookback, step, n_sims, drift_mode,
                    t_df, ewma_lambda, include_garch, seed):
    """Cached backtest so identical inputs return instantly."""
    return run_backtest(prices, horizon=horizon, lookback=lookback, step=step,
                        n_sims=n_sims, drift_mode=drift_mode, t_df=t_df,
                        ewma_lambda=ewma_lambda, include_garch=include_garch, seed=seed)


def build_band_chart(grp, model: str) -> go.Figure:
    """Forecast 10th-90th band vs the price that actually happened."""
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
                             name="Actual price"))
    fig.update_layout(title=f"{model}: forecast range vs actual price",
                      xaxis_title="Target date", yaxis_title="Price (Rs)", height=450)
    return fig


def build_pit_chart(grp, model: str) -> go.Figure:
    """Histogram of PIT values. A calibrated model gives a flat histogram."""
    fig = go.Figure(go.Histogram(x=grp["pit"], xbins=dict(start=0, end=1, size=0.1),
                                 histnorm="probability", marker_color="steelblue"))
    fig.add_hline(y=0.1, line_dash="dash", line_color="orange",
                  annotation_text="Ideal (flat)")
    fig.update_layout(title=f"{model}: where the actual price fell in the forecast distribution",
                      xaxis_title="PIT (0 = below all simulations, 1 = above all)",
                      yaxis_title="Share of forecasts", height=400)
    return fig


st.title("Walk-forward Backtest")
st.caption("How often did the actual price land inside each model's forecast range?")

# ---- Sidebar inputs ----
st.sidebar.header("Backtest inputs")
user_input = st.sidebar.text_input("Stock symbol", value="RELIANCE",
                                   help="Examples: RELIANCE, TCS, HDFCBANK")
history = st.sidebar.selectbox("Price history", ["5y", "10y"], index=1)
lookback = st.sidebar.slider("Estimation window (trading days)", 252, 756, 504, step=21,
                             help="Past days used to estimate parameters at each forecast date. "
                                  "GARCH needs about 500 or more.")
horizon = st.sidebar.slider("Forecast horizon (trading days)", 21, 126, 63, step=21)
step = st.sidebar.slider("Days between forecast dates", 5, 126, 21,
                         help="If this is smaller than the horizon, consecutive forecasts overlap. "
                              "Set it equal to the horizon for independent forecasts.")
n_sims = st.sidebar.slider("Simulations per forecast", 100, 1000, 500, step=100)
drift_label = st.sidebar.selectbox("Drift assumption", ["Historical", "Zero drift"])
t_df = st.sidebar.slider("Student-t degrees of freedom", 3, 30, 5)
ewma_lambda = st.sidebar.slider("EWMA lambda", 0.80, 0.99, 0.94, step=0.01)
include_garch = st.sidebar.checkbox("Include GARCH(1,1) (slower)", value=True)
seed = int(st.sidebar.number_input("Random seed", value=42, step=1))
run = st.sidebar.button("Run backtest", type="primary")

if include_garch and lookback < 500:
    st.sidebar.warning("GARCH is unstable with fewer than about 500 days of data.")

# ---- Run ----
if run:
    try:
        ticker, prices = load_prices(user_input, history)
        results = cached_backtest(
            prices, horizon, lookback, step, n_sims,
            "zero" if drift_label == "Zero drift" else "historical",
            t_df, ewma_lambda, include_garch, seed,
        )
    except ValueError as err:
        st.error(str(err))
        st.stop()
    st.session_state["bt_results"] = results
    st.session_state["bt_meta"] = {"ticker": ticker, "horizon": horizon, "step": step}

if "bt_results" not in st.session_state:
    st.info("Set the inputs in the sidebar and click **Run backtest**.")
    st.stop()

results = st.session_state["bt_results"]
meta = st.session_state["bt_meta"]

# ---- Summary ----
n_origins = results["origin_date"].nunique()
st.subheader(f"{meta['ticker']}: {n_origins} forecast dates, {meta['horizon']}-day horizon")

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
        "- **Coverage 80% / 90%**: share of times the actual price fell inside the "
        "10th-90th / 5th-95th percentile range. Ideal: 80% / 90%.\n"
        "- **Below 5th / Above 95th**: how often the price breached each tail. Ideal: 5% each. "
        "Too many breaches on the downside means the model underestimates risk.\n"
        "- **Mean PIT**: average position of the actual price inside the forecast distribution. "
        "Ideal: 0.5. Below 0.5 means the model was too optimistic, above 0.5 too pessimistic.\n"
        "- **Kupiec p-value**: tests whether the 5% lower-tail breach rate matches the model. "
        "Below 0.05 means the model's lower tail is statistically miscalibrated."
    )

# ---- Charts for one model ----
model = st.selectbox("Inspect model", list(summary.index))
grp = results[results["model"] == model]
st.plotly_chart(build_band_chart(grp, model), use_container_width=True)
st.plotly_chart(build_pit_chart(grp, model), use_container_width=True)

st.caption("Scenario analysis, not investment advice. Backtest uses only data available "
           "at each forecast date.")