"""GARCH(1,1) volatility forecasting."""

from __future__ import annotations

import numpy as np
from arch import arch_model
from arch.univariate.base import ARCHModelResult


def fit_garch(returns_pct: np.ndarray) -> ARCHModelResult:
    """Fit a GARCH(1,1) model. `returns_pct` should be returns in percent (e.g. 1.0 == 1%)
    for numerical stability of the optimizer.
    """
    model = arch_model(returns_pct, vol="GARCH", p=1, q=1, dist="normal")
    return model.fit(disp="off")


def forecast_volatility(fitted: ARCHModelResult, horizon: int = 1) -> np.ndarray:
    """Forecast volatility (in percent, same scale as the fitting input) `horizon` steps ahead."""
    fc = fitted.forecast(horizon=horizon, reindex=False)
    variance = fc.variance.values[-1]
    return np.asarray(np.sqrt(variance))
