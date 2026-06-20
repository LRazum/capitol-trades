"""
prices.py — decoupled daily price access for the gold layer.

``PriceProvider`` returns a daily close series (tz-naive DatetimeIndex, ascending) for a
ticker over a date range. ``YFinancePriceProvider`` is the production impl;
``SyntheticPriceProvider`` is a deterministic offline stand-in for tests/demos.

The gold builder only ever slices these series with ``.loc[:as_of]`` for features and
walks forward from the entry bar for labels, so point-in-time discipline is enforced by
construction at the layer that consumes this.
"""
from __future__ import annotations

import abc
import hashlib
import math
import requests
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd


class PriceProvider(abc.ABC):
    @abc.abstractmethod
    def get(self, ticker: str, start: date, end: date) -> pd.Series:
        """Daily close, ascending DatetimeIndex, inclusive of [start, end]. Empty if none."""
        ...


class SyntheticPriceProvider(PriceProvider):
    """Deterministic GBM per ticker (seeded by a stable hash of the symbol), so the same
    ticker yields the same series across runs/processes."""

    def __init__(
        self,
        start: str = "2017-01-01",
        end: str = "2025-12-31",
        mu: float = 0.07,
        sigma: float = 0.25,
        s0: float = 100.0,
    ):
        self.start, self.end = pd.Timestamp(start), pd.Timestamp(end)
        self.mu, self.sigma, self.s0 = mu, sigma, s0
        self._cache: dict[str, pd.Series] = {}

    def _seed(self, ticker: str) -> int:
        return int(hashlib.md5(ticker.encode()).hexdigest(), 16) % (2**32)

    def _series(self, ticker: str) -> pd.Series:
        if ticker not in self._cache:
            rng = np.random.default_rng(self._seed(ticker))
            idx = pd.bdate_range(self.start, self.end)
            dt = 1.0 / 252.0
            drift = (self.mu - 0.5 * self.sigma**2) * dt
            shocks = rng.normal(drift, self.sigma * math.sqrt(dt), len(idx))
            s0 = self.s0 * (0.5 + rng.random())  # vary starting level by ticker
            self._cache[ticker] = pd.Series(np.exp(np.log(s0) + np.cumsum(shocks)), index=idx)
        return self._cache[ticker]

    def get(self, ticker: str, start: date, end: date) -> pd.Series:
        s = self._series(ticker)
        return s.loc[pd.Timestamp(start):pd.Timestamp(end)]


class YFinancePriceProvider(PriceProvider):
    """Production provider that bypasses `yfinance` text parsing bugs. 
    It queries the Yahoo Chart API directly using integer UNIX timestamps."""

    def __init__(self, auto_adjust: bool = True):
        self.auto_adjust = auto_adjust
        self._cache: dict[tuple, pd.Series] = {}
        # A standard user-agent so Yahoo doesn't block us
        self.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    @staticmethod
    def _to_yahoo(ticker: str) -> str:
        """Map a disclosure ticker to Yahoo's symbology. Yahoo uses a dash for share
        classes and special suffixes (e.g. BRK/B -> BRK-B, BRK.B -> BRK-B, BF.B -> BF-B),
        whereas the feeds use a slash or dot. Adapters pass tickers verbatim; the Yahoo
        spelling is a price-source concern, so it's handled here, not upstream."""
        return (ticker or "").strip().upper().replace("/", "-").replace(".", "-")

    def get(self, ticker: str, start: date, end: date) -> pd.Series:
        key = (ticker, str(start), str(end))   # cache by the ORIGINAL ticker
        if key in self._cache:
            return self._cache[key]
            
        symbol = self._to_yahoo(ticker)
        
        # Convert pandas timestamps to Epoch UNIX integers
        p1 = int(pd.Timestamp(start).timestamp())
        p2 = int((pd.Timestamp(end) + pd.Timedelta(days=1)).timestamp())
        
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}?period1={p1}&period2={p2}&interval=1d"

        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code != 200:
                s = pd.Series(dtype=float)
            else:
                data = resp.json()
                res = data.get("chart", {}).get("result")
                
                if not res:
                    s = pd.Series(dtype=float)
                else:
                    ts = res[0].get("timestamp", [])
                    if not ts:
                        s = pd.Series(dtype=float)
                    else:
                        closes = None
                        # Try grabbing Adjusted Close first
                        if self.auto_adjust:
                            adj = res[0].get("indicators", {}).get("adjclose", [])
                            if adj and "adjclose" in adj[0]:
                                closes = adj[0]["adjclose"]
                        
                        # Fallback to standard close
                        if not closes:
                            quote = res[0].get("indicators", {}).get("quote", [])
                            if quote and "close" in quote[0]:
                                closes = quote[0]["close"]
                                
                        if closes and len(ts) == len(closes):
                            # Pass 's' (seconds) to bypass string parsing completely
                            s = pd.Series(closes, index=pd.to_datetime(ts, unit="s"))
                            # Normalize index to midnight and make timezone-naive
                            s.index = s.index.tz_localize(None).normalize()
                            s = s.dropna()
                        else:
                            s = pd.Series(dtype=float)
        except Exception:
            s = pd.Series(dtype=float)

        self._cache[key] = s
        return s
