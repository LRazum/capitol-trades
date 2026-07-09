"""
features.py — turn raw Quiver feeds + prices into a point-in-time ML dataset.

The model is a *cross-sectional ranker*: at an as-of date it scores each ticker by the
probability of outperforming the benchmark over the next H trading days. To avoid
look-ahead, every feature is computed from data dated on-or-before the as-of date, and
the label is the realized forward excess return over (as-of, as-of+H].

Providers
---------
  YFPriceProvider      — adjusted closes via yfinance (cached upstream by the app).
  SyntheticProvider    — self-contained mock feeds + prices for the demo / offline runs.
                         The synthetic data carries a *mild planted signal* so the demo
                         shows the machinery working; real data is left to tell the truth.

Column-name drift in Quiver's feeds is absorbed by `_pick`/the normalizers, mirroring
the fallback style of the existing pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Iterable, Optional

import numpy as np
import pandas as pd

BENCHMARK = "SPY"

FEATURE_COLUMNS = [
    "cong_net_buys_30d", "cong_net_buys_90d", "cong_dollar_30d",
    "cong_n_politicians_90d", "days_since_cong",
    "insider_net_30d", "insider_dollar_90d",
    "govcon_dollar_90d", "lobby_dollar_90d",
    "wsb_mentions_7d", "wsb_mentions_chg",
    "ret_21d", "ret_63d", "vol_21d", "dist_ma200",
]

_AMOUNT_MIDPOINTS = {
    "$1,001 - $15,000": 8000, "$15,001 - $50,000": 32500,
    "$50,001 - $100,000": 75000, "$100,001 - $250,000": 175000,
    "$250,001 - $500,000": 375000, "$500,001 - $1,000,000": 750000,
    "$1,000,001 - $5,000,000": 3000000, "$5,000,001 - $25,000,000": 15000000,
}


def _pick(df: pd.DataFrame, *names: str) -> Optional[pd.Series]:
    for n in names:
        if n in df.columns:
            return df[n]
    return None


def _series(df: pd.DataFrame, *names: str, fill="") -> pd.Series:
    """Like _pick but never None: missing columns become a fill-valued Series.
    (Avoids `Series or default`, which raises on truth-value ambiguity.)"""
    s = _pick(df, *names)
    return s if s is not None else pd.Series([fill] * len(df), index=df.index)


def _to_amount(s: pd.Series) -> pd.Series:
    def one(v):
        if pd.isna(v):
            return 0.0
        if isinstance(v, (int, float)):
            return float(v)
        v = str(v).strip()
        if v in _AMOUNT_MIDPOINTS:
            return float(_AMOUNT_MIDPOINTS[v])
        digits = "".join(ch for ch in v if ch.isdigit() or ch == ".")
        try:
            return float(digits) if digits else 0.0
        except ValueError:
            return 0.0
    return s.map(one)


# --------------------------------------------------------------------------- #
# Normalizers -> canonical schemas
# --------------------------------------------------------------------------- #
def normalize_congress(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["ticker", "date", "side", "amount", "politician"])
    out = pd.DataFrame()
    out["ticker"] = _series(df, "Ticker", "ticker").astype(str).str.upper()
    filed = _pick(df, "ReportDate", "Filed", "filed", "TransactionDate", "Traded")
    out["date"] = pd.to_datetime(filed, errors="coerce")
    txn = _series(df, "Transaction", "Type", "txn_type").astype(str).str.lower()
    out["side"] = np.where(txn.str.contains("purchase|buy"), 1,
                           np.where(txn.str.contains("sale|sell"), -1, 0))
    out["amount"] = _to_amount(_series(df, "Range", "Amount", "amount", fill=np.nan))
    out["politician"] = _series(df, "Representative", "Name", "politician").astype(str)
    return out.dropna(subset=["date"]).query("ticker != ''")


def normalize_insiders(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["ticker", "date", "side", "amount"])
    out = pd.DataFrame()
    out["ticker"] = _series(df, "Ticker", "ticker").astype(str).str.upper()
    out["date"] = pd.to_datetime(_pick(df, "Date", "TransactionDate", "Filed"), errors="coerce")
    code = _series(df, "AcquiredDisposedCode", "TransactionCode", "Transaction").astype(str).str.upper()
    shares = _pick(df, "Shares", "shares")
    out["side"] = np.where(code.str.contains("A|P|BUY"), 1,
                           np.where(code.str.contains("D|S|SELL"), -1, 0))
    price = _pick(df, "PricePerShare", "Price")
    if shares is not None and price is not None:
        out["amount"] = pd.to_numeric(shares, errors="coerce").fillna(0) * \
                        pd.to_numeric(price, errors="coerce").fillna(0)
    else:
        out["amount"] = 0.0
    return out.dropna(subset=["date"]).query("ticker != ''")


def normalize_value_feed(df: pd.DataFrame, value_cols: tuple[str, ...]) -> pd.DataFrame:
    """Generic feed with ticker/date/value (gov contracts, lobbying)."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["ticker", "date", "value"])
    out = pd.DataFrame()
    out["ticker"] = _series(df, "Ticker", "ticker").astype(str).str.upper()
    out["date"] = pd.to_datetime(_pick(df, "Date", "date", "Year"), errors="coerce")
    val = _pick(df, *value_cols)
    out["value"] = pd.to_numeric(val, errors="coerce").fillna(0) if val is not None else 0.0
    return out.dropna(subset=["date"]).query("ticker != ''")


def normalize_wsb(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["ticker", "date", "mentions"])
    out = pd.DataFrame()
    out["ticker"] = _series(df, "Ticker", "ticker").astype(str).str.upper()
    out["date"] = pd.to_datetime(_pick(df, "Date", "date"), errors="coerce")
    m = _pick(df, "Mentions", "mentions", "Count")
    out["mentions"] = pd.to_numeric(m, errors="coerce").fillna(0) if m is not None else 0.0
    return out.dropna(subset=["date"]).query("ticker != ''")


# --------------------------------------------------------------------------- #
# Feature construction
# --------------------------------------------------------------------------- #
def _win(df: pd.DataFrame, ticker: str, asof: pd.Timestamp, days: int) -> pd.DataFrame:
    lo = asof - pd.Timedelta(days=days)
    return df[(df["ticker"] == ticker) & (df["date"] > lo) & (df["date"] <= asof)]


def _price_features(prices: pd.Series, asof: pd.Timestamp) -> dict:
    s = prices[prices.index <= asof].dropna()
    if len(s) < 30:
        return {"ret_21d": np.nan, "ret_63d": np.nan, "vol_21d": np.nan, "dist_ma200": np.nan}
    p = float(s.iloc[-1])
    ret21 = p / float(s.iloc[-22]) - 1 if len(s) >= 22 else np.nan
    ret63 = p / float(s.iloc[-64]) - 1 if len(s) >= 64 else np.nan
    vol21 = float(s.pct_change().iloc[-21:].std() * np.sqrt(252)) if len(s) >= 22 else np.nan
    ma200 = float(s.iloc[-200:].mean()) if len(s) >= 200 else float(s.mean())
    return {"ret_21d": ret21, "ret_63d": ret63, "vol_21d": vol21,
            "dist_ma200": p / ma200 - 1 if ma200 else np.nan}


def _forward_excess(prices: pd.Series, bench: pd.Series, asof: pd.Timestamp, horizon: int) -> Optional[float]:
    ps = prices[prices.index >= asof].dropna()
    bs = bench[bench.index >= asof].dropna()
    if len(ps) <= horizon or len(bs) <= horizon:
        return None
    r = float(ps.iloc[horizon]) / float(ps.iloc[0]) - 1
    rb = float(bs.iloc[horizon]) / float(bs.iloc[0]) - 1
    return r - rb


@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.Series
    meta: pd.DataFrame                 # asof, ticker, fwd_excess
    current: pd.DataFrame             # latest-asof features for live selection (no label)
    feeds: dict = field(default_factory=dict)   # normalized feeds for the data tab


def build_dataset(
    feeds: dict[str, pd.DataFrame],
    prices: dict[str, pd.Series],
    *,
    horizon: int = 21,
    asof_step_days: int = 5,
    min_history_days: int = 120,
) -> Dataset:
    """Build the supervised dataset + the current cross-section.

    `feeds` are normalized frames keyed: congress, insiders, gov_contracts, lobbying, wsb.
    `prices` maps ticker (and BENCHMARK) -> adjusted-close Series indexed by date.
    """
    cong = feeds.get("congress", pd.DataFrame())
    ins = feeds.get("insiders", pd.DataFrame())
    gov = feeds.get("gov_contracts", pd.DataFrame())
    lob = feeds.get("lobbying", pd.DataFrame())
    wsb = feeds.get("wsb", pd.DataFrame())
    bench = prices.get(BENCHMARK)
    if bench is None or bench.empty:
        raise ValueError(f"benchmark {BENCHMARK} prices missing")

    tickers = sorted(set(cong["ticker"]) | set(prices.keys()) - {BENCHMARK})
    tickers = [t for t in tickers if t in prices and not prices[t].dropna().empty]

    # as-of grid: congressional event dates drive selection triggers
    rows, labels, meta_rows, current_rows = [], [], [], []
    today = pd.Timestamp(date.today())

    for t in tickers:
        pser = prices[t].dropna()
        if pser.empty:
            continue
        first_ok = pser.index.min() + pd.Timedelta(days=min_history_days)
        ev = cong[cong["ticker"] == t]["date"]
        grid = sorted(set(pd.to_datetime(ev)) | set(
            pd.date_range(max(first_ok, pser.index.min()), pser.index.max(), freq=f"{asof_step_days}D")))
        grid = [d for d in grid if d >= first_ok]

        for asof in grid:
            feat = _features_at(t, asof, cong, ins, gov, lob, wsb, pser)
            fe = _forward_excess(pser, bench, asof, horizon)
            if fe is None:                # no realized forward window yet
                continue
            rows.append(feat); labels.append(1 if fe > 0 else 0)
            meta_rows.append({"asof": asof, "ticker": t, "fwd_excess": fe})

        # current cross-section: latest as-of = last available price date
        asof_now = pser.index.max()
        current_rows.append({"ticker": t, "asof": asof_now,
                             **_features_at(t, asof_now, cong, ins, gov, lob, wsb, pser)})

    X = pd.DataFrame(rows, columns=FEATURE_COLUMNS).astype(float).fillna(0.0)
    y = pd.Series(labels, name="outperform", dtype=int)
    meta = pd.DataFrame(meta_rows)
    current = pd.DataFrame(current_rows)
    if not current.empty:
        for c in FEATURE_COLUMNS:
            if c not in current:
                current[c] = 0.0
        current[FEATURE_COLUMNS] = current[FEATURE_COLUMNS].astype(float).fillna(0.0)

    return Dataset(X=X, y=y, meta=meta, current=current,
                   feeds={"congress": cong, "insiders": ins, "gov_contracts": gov,
                          "lobbying": lob, "wsb": wsb})


def _features_at(t, asof, cong, ins, gov, lob, wsb, pser) -> dict:
    c30, c90 = _win(cong, t, asof, 30), _win(cong, t, asof, 90)
    i30, i90 = _win(ins, t, asof, 30), _win(ins, t, asof, 90)
    g90, l90 = _win(gov, t, asof, 90), _win(lob, t, asof, 90)
    w7 = _win(wsb, t, asof, 7)
    w7_prev = wsb[(wsb["ticker"] == t) &
                  (wsb["date"] > asof - pd.Timedelta(days=14)) &
                  (wsb["date"] <= asof - pd.Timedelta(days=7))]
    last_cong = cong[(cong["ticker"] == t) & (cong["date"] <= asof)]["date"]
    days_since = (asof - last_cong.max()).days if not last_cong.empty else 999
    feat = {
        "cong_net_buys_30d": float(c30["side"].sum()) if not c30.empty else 0.0,
        "cong_net_buys_90d": float(c90["side"].sum()) if not c90.empty else 0.0,
        "cong_dollar_30d": float((c30["side"] * c30["amount"]).sum()) if not c30.empty else 0.0,
        "cong_n_politicians_90d": float(c90["politician"].nunique()) if not c90.empty else 0.0,
        "days_since_cong": float(days_since),
        "insider_net_30d": float(i30["side"].sum()) if not i30.empty else 0.0,
        "insider_dollar_90d": float((i90["side"] * i90["amount"]).sum()) if not i90.empty else 0.0,
        "govcon_dollar_90d": float(g90["value"].sum()) if not g90.empty else 0.0,
        "lobby_dollar_90d": float(l90["value"].sum()) if not l90.empty else 0.0,
        "wsb_mentions_7d": float(w7["mentions"].sum()) if not w7.empty else 0.0,
        "wsb_mentions_chg": float((w7["mentions"].sum() if not w7.empty else 0) -
                                  (w7_prev["mentions"].sum() if not w7_prev.empty else 0)),
    }
    feat.update(_price_features(pser, asof))
    return feat


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #
class YFPriceProvider:
    """Adjusted closes via yfinance. Import is lazy so the app runs without it."""
    def get(self, tickers: Iterable[str], start: date, end: date) -> dict[str, pd.Series]:
        import yfinance as yf
        syms = sorted(set(tickers) | {BENCHMARK})
        data = yf.download(syms, start=start, end=end, auto_adjust=True,
                           progress=False, group_by="ticker", threads=True)
        out: dict[str, pd.Series] = {}
        for s in syms:
            try:
                col = data[s]["Close"] if isinstance(data.columns, pd.MultiIndex) else data["Close"]
                out[s] = col.dropna()
            except Exception:
                continue
        return out


class SyntheticProvider:
    """Self-contained mock Quiver feeds + prices for the demo / offline tests.

    A *genuine, moderate* signal is planted: congressional net-buying adds positive forward
    drift (and net-selling negative) over the next ~horizon days, so the model can actually
    detect an edge here and the dashboard shows the full machinery working. Real data is left
    to tell its own (usually humbler) story.
    """
    def __init__(self, n_tickers: int = 24, days: int = 420, seed: int = 7,
                 horizon: int = 21, alpha_per_event: float = 0.003):
        self.n_tickers = n_tickers; self.days = days; self.rng = np.random.default_rng(seed)
        self.tickers = [f"SYN{i:02d}" for i in range(n_tickers)]
        self.start = date.today() - timedelta(days=days)
        self.horizon = horizon; self.alpha = alpha_per_event
        self._idx = pd.bdate_range(self.start, date.today())
        self._market = self.rng.normal(0.0003, 0.009, len(self._idx))   # shared market returns
        # pre-generate congressional events (used by BOTH prices and feeds)
        self._events: dict[str, list[tuple[int, int]]] = {}   # ticker -> [(pos, side)]
        for t in self.tickers:
            evs = []
            for _ in range(self.rng.integers(6, 22)):
                pos = int(self.rng.integers(0, len(self._idx) - self.horizon - 2))
                side = 1 if self.rng.random() > 0.45 else -1
                evs.append((pos, side))
            self._events[t] = evs

    def prices(self) -> dict[str, pd.Series]:
        idx = self._idx
        out = {BENCHMARK: pd.Series(100 * np.exp(np.cumsum(self._market)), index=idx)}
        for t in self.tickers:
            beta = self.rng.uniform(0.7, 1.4)
            idio = self.rng.normal(0.0, 0.009, len(idx))
            alpha = np.zeros(len(idx))                       # congressional drift
            for pos, side in self._events[t]:
                alpha[pos + 1: pos + 1 + self.horizon] += side * self.alpha
            daily = beta * self._market + idio + alpha
            out[t] = pd.Series(100 * np.exp(np.cumsum(daily)), index=idx)
        self._price_index = idx
        return out

    def feeds(self) -> dict[str, pd.DataFrame]:
        idx = self._idx
        cong, ins, gov, lob, wsb = [], [], [], [], []
        ranges = list(_AMOUNT_MIDPOINTS.keys())
        for t in self.tickers:
            for pos, side in self._events[t]:                # emit the SAME events
                d = idx[pos]
                cong.append({"Ticker": t, "ReportDate": d.strftime("%Y-%m-%d"),
                             "Transaction": "Purchase" if side > 0 else "Sale (Full)",
                             "Range": self.rng.choice(ranges),
                             "Representative": f"Rep {self.rng.integers(0, 30)}"})
            for _ in range(self.rng.integers(0, 8)):
                d = idx[self.rng.integers(0, len(idx))]
                ins.append({"Ticker": t, "Date": d.strftime("%Y-%m-%d"),
                            "TransactionCode": self.rng.choice(["P", "S"]),
                            "Shares": int(self.rng.integers(100, 5000)),
                            "PricePerShare": float(self.rng.uniform(10, 300))})
            for _ in range(self.rng.integers(0, 5)):
                d = idx[self.rng.integers(0, len(idx))]
                gov.append({"Ticker": t, "Date": d.strftime("%Y-%m-%d"),
                            "Amount": float(self.rng.uniform(1e5, 5e7))})
                lob.append({"Ticker": t, "Date": d.strftime("%Y-%m-%d"),
                            "Amount": float(self.rng.uniform(1e4, 2e6))})
            for d in idx[::3]:
                wsb.append({"Ticker": t, "Date": d.strftime("%Y-%m-%d"),
                            "Mentions": int(max(0, self.rng.normal(20, 12)))})
        return {"congress": pd.DataFrame(cong), "insiders": pd.DataFrame(ins),
                "gov_contracts": pd.DataFrame(gov), "lobbying": pd.DataFrame(lob),
                "wsb": pd.DataFrame(wsb)}


def normalize_feeds(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {
        "congress": normalize_congress(raw.get("congress", pd.DataFrame())),
        "insiders": normalize_insiders(raw.get("insiders", pd.DataFrame())),
        "gov_contracts": normalize_value_feed(raw.get("gov_contracts", pd.DataFrame()),
                                              ("Amount", "amount", "Value")),
        "lobbying": normalize_value_feed(raw.get("lobbying", pd.DataFrame()),
                                         ("Amount", "amount", "Value")),
        "wsb": normalize_wsb(raw.get("wsb", pd.DataFrame())),
    }
