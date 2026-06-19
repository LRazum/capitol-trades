"""
gold.py — silver NormalizedTrade -> ML-ready gold event table.

Everything here obeys one rule: features and the volatility estimate use only prices with
index <= the *as-of date*, and labels are walked forward from the entry bar. The as-of
date is chosen by the entry-timing switch (``use_original_filing_date``):

* False -> enter relative to the trade's current ``filing_date`` (the amendment date for
  amended records).
* True  -> enter relative to the *original* disclosure date (first actionable moment),
  while still using the corrected/amended field values carried by the silver record.

The original date per natural key comes from bronze via :func:`earliest_filing_dates`,
since silver's "what's true now" view keeps only the latest version.
"""
from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from .amounts import PTR_BRACKETS
from .models import AssetType, Chamber, NormalizedTrade, Owner, TransactionType
from .prices import PriceProvider


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class GoldConfig:
    # Triple barrier (barrier width is in units of *daily* vol; the horizon is set by the
    # vertical barrier, per López de Prado).
    pt_mult: float = 2.0          # upper (profit-taking) barrier = entry * (1 + pt_mult*σ)
    sl_mult: float = 2.0          # lower (stop) barrier        = entry * (1 - sl_mult*σ)
    max_holding_days: int = 21    # vertical barrier, in trading days after entry
    # Volatility (EWMA of daily returns, as of the as-of date).
    vol_lookback: int = 60
    vol_min_obs: int = 20
    # Features.
    momentum_windows: tuple[int, ...] = (21, 63, 126, 252)
    ma_window: int = 200
    benchmark_ticker: Optional[str] = None   # e.g. "SPY" -> adds excess-momentum features
    insider_lookback_days: int = 90
    # Entry timing.
    use_original_filing_date: bool = False
    entry_offset: int = 1          # enter this many sessions after the as-of date


# --------------------------------------------------------------------------- #
# Provider placeholders (committee relevance + Form 4 overlay)
# --------------------------------------------------------------------------- #
class CommitteeProvider(abc.ABC):
    @abc.abstractmethod
    def is_relevant(self, *, politician: str, ticker: str, as_of: date) -> bool:
        ...


class NoCommitteeData(CommitteeProvider):
    """Placeholder until historical committee assignments are wired (Stewart–Woon, etc.)."""
    def is_relevant(self, *, politician: str, ticker: str, as_of: date) -> bool:
        return False


class InsiderProvider(abc.ABC):
    @abc.abstractmethod
    def overlay(self, *, ticker: str, as_of: date, lookback_days: int) -> dict:
        ...


class NoInsiderData(InsiderProvider):
    """Placeholder structure for the Form 4 overlay. Returns the feature schema with NaNs
    and ``available=False`` so the columns exist and are honestly marked missing until a
    real SEC Form 4 source is connected."""
    def overlay(self, *, ticker: str, as_of: date, lookback_days: int) -> dict:
        return {
            "insider_net_buys": float("nan"),
            "insider_n_buyers": float("nan"),
            "insider_buy_value": float("nan"),
            "insider_available": False,
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def earliest_filing_dates(bronze_df: pd.DataFrame) -> dict[str, date]:
    """Map source_record_id -> earliest filing_date across ALL bronze versions (the
    original disclosure date), for the entry-timing switch."""
    g = bronze_df.dropna(subset=["filing_date"]).copy()
    g["filing_date"] = pd.to_datetime(g["filing_date"])
    mins = g.groupby("source_record_id")["filing_date"].min()
    return {k: v.date() for k, v in mins.items()}


def _size_rank(low: Optional[float]) -> float:
    if low is None:
        return float("nan")
    return float(sum(1 for lo, _ in PTR_BRACKETS if low >= lo))


def daily_vol(prices: pd.Series, as_of: pd.Timestamp, span: int) -> tuple[float, int]:
    """EWMA std of daily returns using only data up to ``as_of``. Returns (sigma, n_obs)."""
    rets = prices.loc[:as_of].pct_change(fill_method=None).dropna()
    if len(rets) < 2:
        return float("nan"), len(rets)
    return float(rets.ewm(span=span).std().iloc[-1]), len(rets)


@dataclass
class BarrierResult:
    label: int
    t0: pd.Timestamp
    t_end: pd.Timestamp
    entry_price: float
    exit_price: float
    ret: float
    barrier: str  # 'pt' | 'sl' | 'vertical'


def triple_barrier(
    prices: pd.Series, as_of: pd.Timestamp, sigma: float, cfg: GoldConfig
) -> Optional[BarrierResult]:
    """Walk the three barriers forward from the entry bar. Returns None if there is no
    entry bar or no forward data."""
    idx = prices.index
    after = idx[idx > as_of]
    entry_pos = cfg.entry_offset - 1
    if entry_pos < 0 or entry_pos >= len(after):
        return None
    t0 = after[entry_pos]
    entry_price = float(prices.loc[t0])
    if not math.isfinite(entry_price) or entry_price <= 0:
        return None

    upper = entry_price * (1 + cfg.pt_mult * sigma)
    lower = entry_price * (1 - cfg.sl_mult * sigma)

    pos_t0 = idx.get_loc(t0)
    vpos = min(pos_t0 + cfg.max_holding_days, len(idx) - 1)
    vdate = idx[vpos]

    label, t_end, exit_price, barrier = 0, vdate, float(prices.loc[vdate]), "vertical"
    for ts, px in prices.loc[t0:vdate].items():
        if ts == t0:
            continue
        if px >= upper:
            label, t_end, exit_price, barrier = 1, ts, float(px), "pt"
            break
        if px <= lower:
            label, t_end, exit_price, barrier = -1, ts, float(px), "sl"
            break

    return BarrierResult(label, t0, t_end, entry_price, exit_price, exit_price / entry_price - 1, barrier)


def _momentum(prices: pd.Series, as_of: pd.Timestamp, windows) -> dict:
    p = prices.loc[:as_of]
    out = {}
    last = float(p.iloc[-1]) if len(p) else float("nan")
    for w in windows:
        out[f"mom_{w}"] = (last / float(p.iloc[-w - 1]) - 1) if len(p) > w else float("nan")
    return out


# --------------------------------------------------------------------------- #
# Rejections
# --------------------------------------------------------------------------- #
@dataclass
class GoldReject:
    trade_id: str
    ticker: Optional[str]
    reason: str


@dataclass
class GoldResult:
    event_table: pd.DataFrame
    rejects: list[GoldReject]
    feature_columns: list[str] = field(default_factory=list)

    @property
    def n_events(self) -> int:
        return len(self.event_table)

    @property
    def n_rejected(self) -> int:
        return len(self.rejects)

    def rejects_to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"trade_id": r.trade_id, "ticker": r.ticker, "reason": r.reason} for r in self.rejects]
        )


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #
class GoldEventTableBuilder:
    def __init__(
        self,
        prices: PriceProvider,
        cfg: GoldConfig,
        *,
        committee: Optional[CommitteeProvider] = None,
        insider: Optional[InsiderProvider] = None,
        original_filing_dates: Optional[dict[str, date]] = None,
        log=print,
    ):
        self.prices = prices
        self.cfg = cfg
        self.committee = committee or NoCommitteeData()
        self.insider = insider or NoInsiderData()
        self.original_filing_dates = original_filing_dates or {}
        self.log = log
        self._feature_cols: list[str] = []

    # -- as-of date (the entry-timing switch lives here) --------------------- #
    def _as_of(self, trade: NormalizedTrade) -> date:
        if self.cfg.use_original_filing_date:
            return self.original_filing_dates.get(trade.source_record_id, trade.filing_date)
        return trade.filing_date

    # -- price batching ------------------------------------------------------ #
    def _fetch_ticker_windows(self, items: list[tuple[NormalizedTrade, date]]) -> dict[str, pd.Series]:
        max_lb = max(self.cfg.ma_window, max(self.cfg.momentum_windows), self.cfg.vol_lookback)
        buffer_cal = int(max_lb * 1.7) + 20
        horizon_cal = int(self.cfg.max_holding_days * 1.7) + 20
        by_ticker: dict[str, list[date]] = {}
        for trade, as_of in items:
            by_ticker.setdefault(trade.security.ticker_as_of, []).append(as_of)
        out = {}
        for tkr, as_ofs in by_ticker.items():
            lo = pd.Timestamp(min(as_ofs)) - pd.Timedelta(days=buffer_cal)
            hi = pd.Timestamp(max(as_ofs)) + pd.Timedelta(days=horizon_cal)
            out[tkr] = self.prices.get(tkr, lo.date(), hi.date())
        return out

    # -- features ------------------------------------------------------------ #
    def _features(self, trade: NormalizedTrade, prices: pd.Series, as_of: pd.Timestamp,
                  sigma: float, bench: Optional[pd.Series]) -> dict:
        feats: dict = {
            # politician / disclosure metadata
            "size_rank": _size_rank(trade.amount.low),
            "log_amount": math.log(trade.amount.point_estimate)
            if trade.amount.point_estimate and trade.amount.point_estimate > 0 else float("nan"),
            "disclosure_lag_days": float(trade.disclosure_lag_days),
            "is_purchase": float(trade.transaction_type == TransactionType.PURCHASE),
            "is_option": float(trade.asset_type == AssetType.OPTION),
            "owner_self": float(trade.owner == Owner.SELF),
            "chamber_house": float(trade.chamber == Chamber.HOUSE),
            "committee_relevant": float(
                self.committee.is_relevant(politician=trade.politician,
                                           ticker=trade.security.ticker_as_of, as_of=as_of.date())
            ),
            # market context (point-in-time)
            "realized_vol_daily": sigma,
            "realized_vol_ann": sigma * math.sqrt(252) if math.isfinite(sigma) else float("nan"),
        }
        mom = _momentum(prices, as_of, self.cfg.momentum_windows)
        feats.update(mom)
        p = prices.loc[:as_of]
        feats["dist_200ma"] = (
            float(p.iloc[-1]) / float(p.iloc[-self.cfg.ma_window:].mean()) - 1
            if len(p) >= self.cfg.ma_window else float("nan")
        )
        if bench is not None and len(bench.loc[:as_of]) > max(self.cfg.momentum_windows):
            bmom = _momentum(bench, as_of, self.cfg.momentum_windows)
            for w in self.cfg.momentum_windows:
                feats[f"xmom_{w}"] = feats[f"mom_{w}"] - bmom[f"mom_{w}"]
        # Form 4 overlay placeholder (schema present, marked unavailable until wired)
        feats.update(self.insider.overlay(ticker=trade.security.ticker_as_of,
                                          as_of=as_of.date(),
                                          lookback_days=self.cfg.insider_lookback_days))
        return feats

    # -- build --------------------------------------------------------------- #
    def build(self, trades: list[NormalizedTrade]) -> GoldResult:
        rejects: list[GoldReject] = []

        items: list[tuple[NormalizedTrade, date]] = []
        for t in trades:
            if not t.security.ticker_as_of:
                rejects.append(GoldReject(t.trade_id, None, "unresolved ticker"))
                continue
            items.append((t, self._as_of(t)))

        price_map = self._fetch_ticker_windows(items)
        bench = None
        if self.cfg.benchmark_ticker and items:
            all_as_of = [a for _, a in items]
            max_lb = max(self.cfg.ma_window, max(self.cfg.momentum_windows), self.cfg.vol_lookback)
            lo = pd.Timestamp(min(all_as_of)) - pd.Timedelta(days=int(max_lb * 1.7) + 20)
            hi = pd.Timestamp(max(all_as_of)) + pd.Timedelta(days=int(self.cfg.max_holding_days * 1.7) + 20)
            bench = self.prices.get(self.cfg.benchmark_ticker, lo.date(), hi.date())

        rows = []
        feat_cols: set[str] = set()
        for trade, as_of_d in items:
            tkr = trade.security.ticker_as_of
            prices = price_map.get(tkr)
            as_of = pd.Timestamp(as_of_d)
            if prices is None or prices.empty or prices.loc[:as_of].shape[0] < self.cfg.vol_min_obs:
                rejects.append(GoldReject(trade.trade_id, tkr, "insufficient price history as-of date"))
                continue
            sigma, n_obs = daily_vol(prices, as_of, self.cfg.vol_lookback)
            if not math.isfinite(sigma) or n_obs < self.cfg.vol_min_obs or sigma <= 0:
                rejects.append(GoldReject(trade.trade_id, tkr, f"volatility unavailable (n_obs={n_obs})"))
                continue
            br = triple_barrier(prices, as_of, sigma, self.cfg)
            if br is None:
                rejects.append(GoldReject(trade.trade_id, tkr, "no entry bar / insufficient forward data"))
                continue

            feats = self._features(trade, prices, as_of, sigma, bench)
            feat_cols.update(feats.keys())
            rows.append({
                "trade_id": trade.trade_id,
                "source_record_id": trade.source_record_id,
                "politician": trade.politician,
                "ticker": tkr,
                "as_of_date": as_of_d,
                "used_original_filing_date": self.cfg.use_original_filing_date,
                # label + interval
                "label": br.label,
                "t0": br.t0,
                "t_end": br.t_end,
                "barrier": br.barrier,
                "ret": br.ret,
                "entry_price": br.entry_price,
                "exit_price": br.exit_price,
                **feats,
            })

        self._feature_cols = sorted(feat_cols)
        table = pd.DataFrame(rows)
        if not table.empty:
            table = table.sort_values("t0").reset_index(drop=True)
        return GoldResult(table, rejects, feature_columns=self._feature_cols)
