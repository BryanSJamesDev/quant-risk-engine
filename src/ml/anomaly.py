"""Anomaly detection: flag points where realized volatility diverges sharply
from the GARCH forecast (a simple, explainable signal for "the model didn't
see this coming")."""

from __future__ import annotations

import numpy as np


def realized_vol(returns: np.ndarray, window: int = 20) -> np.ndarray:
    """Rolling standard deviation of returns, same length as `returns` (NaN until the window fills)."""
    r = np.asarray(returns, dtype=float)
    out = np.full(r.shape, np.nan)
    for i in range(window, len(r) + 1):
        out[i - 1] = r[i - window : i].std(ddof=1)
    return out


def flag_anomalies(realized: np.ndarray, forecast: np.ndarray, z_thresh: float = 2.5) -> np.ndarray:
    """Flag indices where (realized - forecast) is an outlier relative to its own distribution."""
    resid = realized - forecast
    valid = ~np.isnan(resid)
    if valid.sum() < 2:
        return np.zeros_like(resid, dtype=bool)
    mu, sigma = np.nanmean(resid), np.nanstd(resid)
    if sigma == 0:
        return np.zeros_like(resid, dtype=bool)
    z = (resid - mu) / sigma
    return np.where(valid, np.abs(z) > z_thresh, False)
