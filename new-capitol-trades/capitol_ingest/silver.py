"""
silver.py — the "what's true now" silver driver.

Reads the bronze store, collapses every natural key to its latest-``last_modified``
version (so amendments win even if they landed in a different filing_date partition than
their original), and transforms each surviving raw record into a validated
``NormalizedTrade`` via :func:`normalize_one` (which also resolves identity through the
injected ``SecurityMaster``). Anything that fails transformation is appended to a visible
rejections log with a reason — never silently dropped.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from .models import NormalizedTrade, RawDisclosure
from .normalize import Reject, normalize_one
from .security_master import SecurityMaster
from .storage import BronzeStore, latest_by_key


def _clean(v):
    """NaN/NaT -> None so Pydantic Optional[str] fields validate from a DataFrame row."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _reason(exc: Exception) -> str:
    """Extract a human-readable reason. Pydantic wraps validator errors in multi-line
    text; surface the underlying 'Value error, ...' line when present."""
    msg = str(exc)
    for line in msg.splitlines():
        s = line.strip()
        if s.lower().startswith("value error"):
            # Strip Pydantic's trailing "[type=..., input_value=..., input_type=...]" noise.
            return s.split(" [type=")[0].strip()
    return msg.splitlines()[0].strip()


@dataclass
class SilverResult:
    trades: list[NormalizedTrade]
    rejections: list[Reject]

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def n_rejected(self) -> int:
        return len(self.rejections)


class SilverDriver:
    def __init__(
        self,
        store: BronzeStore,
        security_master: SecurityMaster,
        *,
        amount_default: str = "midpoint",
    ):
        self.store = store
        self.sm = security_master
        self.amount_default = amount_default

    # ---------------------------------------------------------------- build --
    def _record_to_raw(self, rec: dict) -> RawDisclosure:
        """Reconstruct a RawDisclosure from a bronze row (inverse of disclosures_to_frame),
        so the existing normalize_one transform applies unchanged."""
        rp = rec.get("raw_payload")
        if isinstance(rp, str):
            try:
                rp = json.loads(rp)
            except (json.JSONDecodeError, TypeError):
                rp = {}
        elif not isinstance(rp, dict):
            rp = {}

        pulled = rec.get("pulled_at")
        if isinstance(pulled, pd.Timestamp):
            pulled = pulled.to_pydatetime()
        elif pulled is None or (isinstance(pulled, float) and pd.isna(pulled)):
            pulled = datetime.now()

        return RawDisclosure(
            source=_clean(rec.get("source")) or "unknown",
            source_record_id=_clean(rec.get("source_record_id")) or "unknown",
            pulled_at=pulled,
            politician_raw=_clean(rec.get("politician_raw")),
            chamber_raw=_clean(rec.get("chamber_raw")),
            issuer_raw=_clean(rec.get("issuer_raw")),
            ticker_raw=_clean(rec.get("ticker_raw")),
            cusip_raw=_clean(rec.get("cusip_raw")),
            transaction_date_raw=_clean(rec.get("transaction_date_raw")),
            filing_date_raw=_clean(rec.get("filing_date_raw")),
            transaction_type_raw=_clean(rec.get("transaction_type_raw")),
            owner_raw=_clean(rec.get("owner_raw")),
            asset_type_raw=_clean(rec.get("asset_type_raw")),
            amount_raw=_clean(rec.get("amount_raw")),
            raw_payload=rp,
        )

    def build(self, *, start=None, end=None) -> SilverResult:
        df = self.store.read(start=start, end=end)
        if df.empty:
            return SilverResult([], [])

        # 'What's true now': across ALL partitions, keep the latest version per natural key.
        # Independent of BronzeStore.compact() — silver is correct even if compaction lags.
        df = latest_by_key(
            df, key_col="source_record_id", recency_cols=["_last_modified", "pulled_at"]
        )

        trades: list[NormalizedTrade] = []
        rejections: list[Reject] = []
        for rec in df.to_dict("records"):
            raw = self._record_to_raw(rec)
            try:
                trades.append(normalize_one(raw, self.sm, amount_default=self.amount_default))
            except Exception as exc:  # noqa: BLE001 - record the reason, never drop silently
                rejections.append(Reject(raw.source_record_id, _reason(exc), raw))
        return SilverResult(trades, rejections)

    # ------------------------------------------------------------ flattening --
    @staticmethod
    def trades_to_frame(trades: list[NormalizedTrade]) -> pd.DataFrame:
        rows = []
        for t in trades:
            s, a = t.security, t.amount
            rows.append(
                {
                    "trade_id": t.trade_id,
                    "source": t.source,
                    "source_record_id": t.source_record_id,
                    "politician": t.politician,
                    "chamber": t.chamber.value,
                    "owner": t.owner.value,
                    "transaction_type": t.transaction_type.value,
                    "asset_type": t.asset_type.value,
                    "transaction_date": t.transaction_date,
                    "filing_date": t.filing_date,
                    "tradeable_date": t.tradeable_date,
                    "disclosure_lag_days": t.disclosure_lag_days,
                    "lag_exceeds_stock_act": t.lag_exceeds_stock_act,
                    "figi": s.figi,
                    "cik": s.cik,
                    "ticker_as_of": s.ticker_as_of,
                    "name_as_of": s.name_as_of,
                    "is_delisted": s.is_delisted,
                    "delisting_action": s.delisting_action,
                    "resolution_method": s.method.value,
                    "resolution_confidence": s.confidence,
                    "resolution_candidates": s.candidates,
                    "amount_low": a.low,
                    "amount_high": a.high,
                    "amount_point": a.point_estimate,
                    "amount_open_ended": a.is_open_ended,
                    "amount_bracket": a.bracket_label,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def rejections_to_frame(rejections: list[Reject]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "source_record_id": r.source_record_id,
                    "reason": r.reason,
                    "politician_raw": r.raw.politician_raw,
                    "ticker_raw": r.raw.ticker_raw,
                    "transaction_date_raw": r.raw.transaction_date_raw,
                    "filing_date_raw": r.raw.filing_date_raw,
                    "amount_raw": r.raw.amount_raw,
                }
                for r in rejections
            ]
        )

    def write(self, result: SilverResult, out_dir: str | Path) -> dict:
        """Persist the clean dataset (Parquet) and the rejections log (CSV, always written
        for visibility — an empty file still tells you the run was clean)."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        tf = self.trades_to_frame(result.trades)
        rf = self.rejections_to_frame(result.rejections)
        tpath, rpath = out / "silver_trades.parquet", out / "rejections.csv"
        if not tf.empty:
            tf.to_parquet(tpath, engine="pyarrow", index=False)
        rf.to_csv(rpath, index=False)
        return {
            "trades": len(tf),
            "rejections": len(rf),
            "trades_path": str(tpath),
            "rejections_path": str(rpath),
        }
