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
    """Production provider. Memoizes by exact (ticker, start, end) so the gold builder's
    one-fetch-per-ticker pattern doesn't repeat downloads."""

    def __init__(self, auto_adjust: bool = True):
        self.auto_adjust = auto_adjust
        self._cache: dict[tuple, pd.Series] = {}

    def get(self, ticker: str, start: date, end: date) -> pd.Series:
        key = (ticker, str(start), str(end))
        if key in self._cache:
            return self._cache[key]
        import yfinance as yf  # lazy import

        df = yf.download(
            ticker,
            start=str(start),
            end=str(pd.Timestamp(end) + pd.Timedelta(days=1)),  # yf end is exclusive
            auto_adjust=self.auto_adjust,
            progress=False,
        )
        if df is None or df.empty:
            s = pd.Series(dtype=float)
        else:
            close = df["Close"]
            if isinstance(close, pd.DataFrame):  # multiindex when given a list
                close = close.iloc[:, 0]
            s = close.copy()
            s.index = pd.to_datetime(s.index)
        self._cache[key] = s
        return s
