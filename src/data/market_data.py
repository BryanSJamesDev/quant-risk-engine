"""Historical market data ingestion via yfinance (no API key required)."""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def fetch_history(tickers: list[str], period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Fetch adjusted close prices for the given tickers, one column per ticker."""
    raw = yf.download(tickers, period=period, interval=interval, progress=False, auto_adjust=True)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=tickers)
    data = raw["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(tickers[0])
    return data.dropna(how="all")
