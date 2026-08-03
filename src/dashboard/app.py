"""Streamlit dashboard: portfolio summary, risk metrics, GARCH volatility
forecast with anomaly flags, and a simulated live-tick panel that shows the
incremental graph recomputing only the affected node.

Run with: uv run streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.market_data import fetch_history
from src.data.simulator import simulate_path
from src.ml.anomaly import flag_anomalies, realized_vol
from src.ml.volatility import fit_garch
from src.risk_engine import metrics
from src.risk_engine.graph import Node
from src.risk_engine.types import Portfolio, Position

st.set_page_config(page_title="Portfolio Risk Engine", layout="wide")
st.title("Real-Time Portfolio Risk & Analytics Engine")

with st.sidebar:
    st.header("Portfolio")
    tickers_input = st.text_input("Tickers (comma-separated)", "AAPL,MSFT,GOOGL,JPM,XOM")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    shares_input = st.text_input("Shares (same order as tickers)", ",".join(["10"] * len(tickers)))
    shares = [float(x) for x in shares_input.split(",")]
    confidence = st.slider("VaR confidence level", 0.90, 0.99, 0.95, 0.01)
    period = st.selectbox("History period", ["6mo", "1y", "2y", "5y"], index=2)

SECTOR_BY_TICKER = {
    "AAPL": "Tech",
    "MSFT": "Tech",
    "GOOGL": "Tech",
    "JPM": "Financials",
    "XOM": "Energy",
}


@st.cache_data(ttl=3600)
def load_prices(tickers_key: tuple[str, ...], period_key: str) -> pd.DataFrame:
    return fetch_history(list(tickers_key), period=period_key)


prices = load_prices(tuple(tickers), period)
prices = prices.dropna(axis=1, how="all")
available = [t for t in tickers if t in prices.columns]

if not available:
    st.error("No valid price data for the given tickers.")
    st.stop()

last_prices = prices.iloc[-1]
positions = tuple(
    Position(t, shares[i], float(last_prices[t]), SECTOR_BY_TICKER.get(t, "Other"))
    for i, t in enumerate(tickers)
    if t in available
)
portfolio = Portfolio(positions)

st.subheader("Portfolio Summary")
returns = prices[available].pct_change().dropna()
weights = portfolio.weights()
port_returns = metrics.portfolio_returns(weights, {t: returns[t].to_numpy() for t in available})

var = metrics.historical_var(port_returns, confidence)
cvar = metrics.expected_shortfall(port_returns, confidence)
cum_value = (1 + pd.Series(port_returns)).cumprod()
mdd = metrics.max_drawdown(cum_value.to_numpy())

col1, col2, col3 = st.columns(3)
col1.metric("Total Value", f"${portfolio.total_value:,.2f}")
col2.metric(f"{int(confidence * 100)}% Historical VaR (daily)", f"{var:.2%}")
col3.metric("Expected Shortfall (CVaR)", f"{cvar:.2%}")
st.caption(f"Max drawdown over period: {mdd:.2%}")

st.subheader("Sector Exposure")
exposure = portfolio.exposure_by_sector()
fig_exp = go.Figure(data=[go.Pie(labels=list(exposure.keys()), values=list(exposure.values()))])
st.plotly_chart(fig_exp, use_container_width=True)

st.subheader("Price History")
fig_price = go.Figure()
for t in available:
    fig_price.add_trace(go.Scatter(x=prices.index, y=prices[t], name=t))
st.plotly_chart(fig_price, use_container_width=True)

st.subheader("Volatility Forecast vs Realized (GARCH)")
asset_for_vol = st.selectbox("Asset", available)
asset_returns_pct = returns[asset_for_vol].to_numpy() * 100
fitted = fit_garch(asset_returns_pct)
cond_vol = fitted.conditional_volatility / 100
realized = realized_vol(returns[asset_for_vol].to_numpy(), window=20)
anomalies = flag_anomalies(realized, cond_vol)

fig_vol = go.Figure()
fig_vol.add_trace(go.Scatter(x=returns.index, y=cond_vol, name="GARCH forecast vol"))
fig_vol.add_trace(go.Scatter(x=returns.index, y=realized, name="Realized vol (20d)"))
fig_vol.add_trace(
    go.Scatter(
        x=returns.index[anomalies],
        y=realized[anomalies],
        mode="markers",
        marker=dict(color="red", size=10),
        name="Anomaly",
    )
)
st.plotly_chart(fig_vol, use_container_width=True)

st.subheader("Live Risk Monitor (Simulated Ticks, Incremental Recompute)")
st.caption(
    "Simulates a streaming price for one asset and recomputes rolling VaR incrementally "
    "through a dependency graph — each tick only recomputes the nodes that actually depend on it."
)

sim_asset = st.selectbox("Simulate ticks for", available, key="sim_asset")
n_ticks = st.slider("Number of ticks", 10, 200, 50)

if st.button("Run simulation"):
    start_price = float(last_prices[sim_asset])
    sigma = float(returns[sim_asset].std() * np.sqrt(252))
    mu = float(returns[sim_asset].mean() * 252)
    path = simulate_path(start_price, mu, sigma, n_ticks, dt=1 / 252)

    price_node: Node[float] = Node.source("price", start_price)
    window: list[float] = [start_price]

    def recompute_var(p: float) -> float:
        window.append(p)
        if len(window) > 30:
            window.pop(0)
        w = np.array(window)
        rets = w[1:] / w[:-1] - 1.0
        return metrics.historical_var(rets, confidence) if rets.size > 1 else 0.0

    var_node: Node[float] = Node.derived("rolling_var", recompute_var, price_node)

    live_vars = []
    for tick in path:
        price_node.set(float(tick))
        live_vars.append(var_node.value())

    live_col1, live_col2 = st.columns(2)
    with live_col1:
        fig_live_price = go.Figure()
        fig_live_price.add_trace(go.Scatter(y=path, name="Simulated price"))
        st.plotly_chart(fig_live_price, use_container_width=True)
    with live_col2:
        fig_live_var = go.Figure()
        fig_live_var.add_trace(go.Scatter(y=live_vars, name="Rolling VaR"))
        st.plotly_chart(fig_live_var, use_container_width=True)

    st.caption(
        f"VaR node recompute count: {var_node.recompute_count} "
        f"(one per tick — only this node recomputed, not the full portfolio)."
    )
