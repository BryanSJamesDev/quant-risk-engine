"""Geometric Brownian Motion tick simulator, used to demo the live/incremental
side of the dashboard without needing a real streaming market data feed.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def gbm_ticks(
    start_price: float, mu: float, sigma: float, dt: float, seed: int | None = None
) -> Iterator[float]:
    """Yield an infinite stream of simulated prices under GBM with drift `mu` and vol `sigma` (annualized)."""
    rng = np.random.default_rng(seed)
    price = start_price
    while True:
        z = rng.standard_normal()
        price *= float(np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z))
        yield price


def simulate_path(
    start_price: float,
    mu: float,
    sigma: float,
    n_steps: int,
    dt: float = 1.0 / 252,
    seed: int | None = None,
) -> np.ndarray:
    gen = gbm_ticks(start_price, mu, sigma, dt, seed)
    return np.array([next(gen) for _ in range(n_steps)])
