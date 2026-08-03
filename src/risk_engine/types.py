"""Immutable domain model for the risk engine.

Every type here is a frozen dataclass. Nothing in this module mutates state —
"changing" a portfolio (e.g. marking to new prices) returns a new Portfolio
rather than mutating in place, so the risk calculations built on top of it
can stay pure functions of their inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class Position:
    ticker: str
    shares: float
    price: float
    sector: str = "Unknown"

    @property
    def market_value(self) -> float:
        return self.shares * self.price


@dataclass(frozen=True, slots=True)
class Portfolio:
    positions: tuple[Position, ...]

    @property
    def total_value(self) -> float:
        return sum(p.market_value for p in self.positions)

    def weights(self) -> Mapping[str, float]:
        total = self.total_value
        if total == 0:
            return {p.ticker: 0.0 for p in self.positions}
        return {p.ticker: p.market_value / total for p in self.positions}

    def exposure_by_sector(self) -> Mapping[str, float]:
        total = self.total_value
        raw: dict[str, float] = {}
        for p in self.positions:
            raw[p.sector] = raw.get(p.sector, 0.0) + p.market_value
        if total == 0:
            return raw
        return {sector: value / total for sector, value in raw.items()}

    def with_prices(self, prices: Mapping[str, float]) -> "Portfolio":
        """Return a new Portfolio marked to the given prices (unchanged tickers keep their old price)."""
        return Portfolio(
            tuple(
                Position(p.ticker, p.shares, prices.get(p.ticker, p.price), p.sector)
                for p in self.positions
            )
        )


@dataclass(frozen=True, slots=True)
class RiskSnapshot:
    var: float
    cvar: float
    max_drawdown: float
    volatility: float
    exposure_by_sector: Mapping[str, float]
