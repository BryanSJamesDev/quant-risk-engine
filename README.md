# Portfolio Risk & Analytics Engine

A small, real-ish system for monitoring portfolio risk: a functional risk
core, a GARCH volatility model with anomaly detection, and a live dashboard —
built the way I'd want risk infrastructure at a trading firm or bank to be
built: pure, typed, and tested.

## Why this architecture

Risk numbers that can't be trusted are worse than no risk numbers at all. So
the design optimizes for correctness and predictability over cleverness:

- **Functional core.** [`src/risk_engine`](src/risk_engine) has no shared
  mutable state. Domain objects (`Position`, `Portfolio`) are frozen
  dataclasses; "updating" a portfolio's prices returns a new `Portfolio`
  rather than mutating one. Every risk metric (VaR, CVaR, drawdown, stress
  test) is a pure function of its inputs — same inputs, same output, always,
  which makes them property-testable and safe to call from anywhere.
- **Incremental recomputation.** [`src/risk_engine/graph.py`](src/risk_engine/graph.py)
  implements a small dependency graph: source nodes hold a value, derived
  nodes hold a pure function of their dependencies, and setting a source only
  invalidates the nodes that actually depend on it. The live-tick panel in
  the dashboard uses this so that one price update recomputes rolling VaR
  without recomputing the rest of the portfolio.
- **ML sits on top of the core, not inside it.** [`src/ml`](src/ml) fits a
  GARCH(1,1) model per asset and flags points where realized volatility
  diverges from the forecast — an explainable anomaly signal, not a black
  box.
- **Data ingestion is boring on purpose.** [`src/data/market_data.py`](src/data/market_data.py)
  pulls real historical prices via `yfinance` (no API key needed to run
  this). [`src/data/simulator.py`](src/data/simulator.py) generates a GBM
  price path to demo the live/streaming path without needing a real feed.

## Project layout

```
src/
  risk_engine/   # types.py, metrics.py, graph.py — the functional core
  data/          # market_data.py (real history), simulator.py (simulated ticks)
  ml/            # volatility.py (GARCH), anomaly.py (anomaly flagging)
  dashboard/     # app.py — Streamlit UI wiring it all together
tests/
  test_metrics.py  # property-based tests (Hypothesis) on the risk math
  test_graph.py    # correctness of incremental recomputation
```

## Running it

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                  # install dependencies
uv run streamlit run src/dashboard/app.py  # launch the dashboard
```

Enter any tickers you like in the sidebar (defaults to a small tech/financials/
energy mix); prices are pulled live from Yahoo Finance.

## Testing

```bash
uv run pytest       # unit + property-based tests
uv run mypy src      # strict type checking
```

The metric tests use [Hypothesis](https://hypothesis.readthedocs.io/) to
check invariants across randomly generated inputs (e.g. CVaR is always at
least as severe as VaR, drawdown is never positive) rather than just
hard-coded examples — the kind of test that actually catches edge cases in
financial math.

## Possible extensions

- Multi-asset stress scenarios (correlated shocks via a covariance matrix)
- Swap the anomaly detector for an Isolation Forest over a richer feature set
- Persist historical risk snapshots and chart risk-over-time, not just
  point-in-time
