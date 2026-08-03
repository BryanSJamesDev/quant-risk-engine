from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from src.risk_engine import metrics
from src.risk_engine.types import Portfolio, Position

finite_returns = st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False)
finite_prices = st.floats(min_value=1.0, max_value=10_000.0, allow_nan=False, allow_infinity=False)


@given(st.lists(finite_returns, min_size=2, max_size=200))
def test_higher_confidence_var_is_at_least_as_severe(returns: list[float]) -> None:
    r = np.array(returns)
    var_95 = metrics.historical_var(r, 0.95)
    var_99 = metrics.historical_var(r, 0.99)
    assert var_99 <= var_95 + 1e-9


@given(st.lists(finite_returns, min_size=2, max_size=200))
def test_cvar_is_at_least_as_severe_as_var(returns: list[float]) -> None:
    r = np.array(returns)
    var = metrics.historical_var(r, 0.95)
    cvar = metrics.expected_shortfall(r, 0.95)
    assert cvar <= var + 1e-9


@given(st.lists(finite_prices, min_size=1, max_size=200))
def test_max_drawdown_is_never_positive(prices: list[float]) -> None:
    assert metrics.max_drawdown(np.array(prices)) <= 0.0


def test_max_drawdown_of_empty_series_is_zero() -> None:
    assert metrics.max_drawdown(np.array([])) == 0.0


def test_max_drawdown_of_monotonically_rising_prices_is_zero() -> None:
    assert metrics.max_drawdown(np.array([1.0, 2.0, 3.0, 4.0])) == 0.0


def test_stress_test_with_no_shocks_is_a_noop() -> None:
    portfolio = Portfolio((Position("AAPL", 10, 100.0, "Tech"),))
    assert metrics.stress_test(portfolio, {}) == 0.0


def test_stress_test_negative_shock_reduces_value() -> None:
    portfolio = Portfolio((Position("AAPL", 10, 100.0, "Tech"),))
    pnl = metrics.stress_test(portfolio, {"AAPL": -0.10})
    assert pnl == -100.0


def test_portfolio_returns_matches_manual_weighted_sum() -> None:
    weights = {"A": 0.5, "B": 0.5}
    asset_returns = {"A": np.array([0.1, -0.1]), "B": np.array([0.02, 0.02])}
    result = metrics.portfolio_returns(weights, asset_returns)
    expected = np.array([0.5 * 0.1 + 0.5 * 0.02, 0.5 * -0.1 + 0.5 * 0.02])
    np.testing.assert_allclose(result, expected)


def test_portfolio_weights_sum_to_one() -> None:
    portfolio = Portfolio(
        (
            Position("AAPL", 10, 100.0, "Tech"),
            Position("JPM", 5, 200.0, "Financials"),
        )
    )
    total_weight = sum(portfolio.weights().values())
    assert abs(total_weight - 1.0) < 1e-9
