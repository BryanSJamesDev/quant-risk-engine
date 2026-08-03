"""Pure risk-metric functions.

Every function here takes arrays/portfolios in and returns a value out with no
side effects and no shared state — the "functional core" the rest of the
system (data ingestion, ML, dashboard) sits on top of. This makes each metric
trivial to unit test and safe to recompute from any thread or event loop.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from src.risk_engine.types import Portfolio


def returns_from_prices(prices: npt.ArrayLike) -> np.ndarray:
    p = np.asarray(prices, dtype=float)
    if p.size < 2:
        return np.array([])
    return np.asarray(p[1:] / p[:-1] - 1.0)


def historical_var(returns: npt.ArrayLike, confidence: float = 0.95) -> float:
    """Historical (empirical) Value at Risk, expressed as a signed return (e.g. -0.02 = -2%)."""
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    return float(np.percentile(r, (1 - confidence) * 100))


def parametric_var(returns: npt.ArrayLike, confidence: float = 0.95) -> float:
    """Gaussian (variance-covariance) VaR, for comparison against the historical estimate."""
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1)) if r.size > 1 else 0.0
    z = float(norm.ppf(1 - confidence))
    return mu + z * sigma


def expected_shortfall(returns: npt.ArrayLike, confidence: float = 0.95) -> float:
    """Expected Shortfall / CVaR: mean return in the tail beyond the VaR threshold."""
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    var = historical_var(r, confidence)
    tail = r[r <= var]
    return float(tail.mean()) if tail.size else var


def max_drawdown(prices: npt.ArrayLike) -> float:
    """Largest peak-to-trough decline, expressed as a fraction (e.g. -0.35 = -35%)."""
    p = np.asarray(prices, dtype=float)
    if p.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(p)
    drawdowns = p / running_max - 1.0
    return float(drawdowns.min())


def portfolio_returns(
    weights: Mapping[str, float], asset_returns: Mapping[str, npt.ArrayLike]
) -> np.ndarray:
    """Combine per-asset return series into a single weighted portfolio return series."""
    tickers = list(weights)
    if not tickers:
        return np.array([])
    matrix = np.column_stack([np.asarray(asset_returns[t], dtype=float) for t in tickers])
    w = np.array([weights[t] for t in tickers])
    return np.asarray(matrix @ w)


def stress_test(portfolio: Portfolio, shocks: Mapping[str, float]) -> float:
    """Apply fractional price shocks (e.g. {"AAPL": -0.10} for -10%) and return the P&L impact."""
    shocked_prices = {
        p.ticker: p.price * (1 + shocks.get(p.ticker, 0.0)) for p in portfolio.positions
    }
    shocked = portfolio.with_prices(shocked_prices)
    return shocked.total_value - portfolio.total_value
