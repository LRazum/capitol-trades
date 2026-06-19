"""
daily.py — daily live screening: Quiver -> Bronze -> Silver -> Gold -> Actionable Signals.

``screen_signals`` is the pure, testable core: it scores gold events with a trained model
and filters to high-conviction names (committee-relevant AND insider-corroborated AND above
the ML probability threshold). ``DailyScreener`` is the cron entry point that wires the full
pipeline with injected (real) providers and writes a dated + ``latest`` parquet for the UI.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .bronze import ingest_bronze
from .gold import GoldConfig, GoldEventTableBuilder
from .silver import SilverDriver
from .storage import BronzeStore


def screen_signals(
    gold_table: pd.DataFrame,
    *,
    model=None,
    feature_columns: Optional[list[str]] = None,
    proba_threshold: float = 0.5,
    require_committee: bool = True,
    require_insider: bool = True,
    insider_col: str = "insider_net_60",
    recent_since=None,
    time_col: str = "as_of_date",
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """Score and filter the gold table into an actionable signals list."""
    df = gold_table.copy()
    if df.empty:
        return df
    if recent_since is not None:
        df = df[pd.to_datetime(df[time_col]) >= pd.Timestamp(recent_since)]

    if model is not None and feature_columns:
        usable = [c for c in feature_columns if c in df.columns]
        df["ml_proba"] = model.predict_proba(df[usable].to_numpy())[:, 1]
    else:
        df["ml_proba"] = np.nan

    mask = pd.Series(True, index=df.index)
    if require_committee and "committee_relevant" in df.columns:
        mask &= df["committee_relevant"] == 1.0
    if require_insider and insider_col in df.columns:
        mask &= df[insider_col] > 0
    if model is not None:
        mask &= df["ml_proba"] >= proba_threshold

    out = df[mask].copy()
    # composite conviction: model probability if available, else a simple rule sum
    if model is not None:
        out["signal_score"] = out["ml_proba"]
    else:
        ins = (out[insider_col] > 0).astype(float) if insider_col in out else 0.0
        out["signal_score"] = out.get("committee_relevant", 0.0) + ins
    out = out.sort_values("signal_score", ascending=False)
    if top_n:
        out = out.head(top_n)

    cols = [c for c in ["as_of_date", "politician", "ticker", "transaction_type",
                        "committee_relevant", insider_col, "ml_proba", "signal_score",
                        "amount_point", "t0", "t_end"] if c in out.columns]
    return out[cols].reset_index(drop=True)


class DailyScreener:
    """Cron entry point. All data sources are injected so the same object runs in production
    (real providers) and in tests (stubs)."""

    def __init__(
        self,
        *,
        adapter,                  # VendorAdapter (QuiverCongressAdapter, mode="live")
        store: BronzeStore,
        security_master,
        price_provider,
        committee_provider,
        insider_provider,
        gold_config: GoldConfig,
        model=None,
        feature_columns: Optional[list[str]] = None,
        original_filing_dates: Optional[dict] = None,
        proba_threshold: float = 0.5,
        out_dir: str | Path = "signals",
        log: Callable[[str], None] = print,
    ):
        self.adapter = adapter
        self.store = store
        self.security_master = security_master
        self.price_provider = price_provider
        self.committee_provider = committee_provider
        self.insider_provider = insider_provider
        self.gold_config = gold_config
        self.model = model
        self.feature_columns = feature_columns
        self.original_filing_dates = original_filing_dates or {}
        self.proba_threshold = proba_threshold
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log = log

    def run(
        self,
        *,
        asof: Optional[date] = None,
        pull_lookback_days: int = 14,
        signal_window_days: int = 7,
        compact: bool = False,
    ) -> pd.DataFrame:
        asof = asof or date.today()
        start = asof - timedelta(days=pull_lookback_days)

        self.log(f"[1/5] ingest Quiver {start}..{asof}")
        ingest_bronze(self.adapter, self.store, start=start, end=asof)
        if compact:
            self.store.compact()

        self.log("[2/5] silver (what's-true-now)")
        silver = SilverDriver(self.store, self.security_master).build(start=start, end=asof)

        self.log(f"[3/5] gold ({len(silver.trades)} trades) with real committee + insider providers")
        gold = GoldEventTableBuilder(
            self.price_provider, self.gold_config,
            committee=self.committee_provider, insider=self.insider_provider,
            original_filing_dates=self.original_filing_dates,
        ).build(silver.trades)

        self.log("[4/5] screen -> actionable signals")
        actionable = screen_signals(
            gold.event_table, model=self.model, feature_columns=self.feature_columns,
            proba_threshold=self.proba_threshold,
            recent_since=asof - timedelta(days=signal_window_days),
        )

        latest = self.out_dir / "actionable_latest.parquet"
        dated = self.out_dir / f"actionable_{asof.isoformat()}.parquet"
        if not actionable.empty:
            actionable.to_parquet(dated, engine="pyarrow", index=False)
        actionable.to_parquet(latest, engine="pyarrow", index=False)
        self.log(f"[5/5] {len(actionable)} actionable signals -> {latest}")
        return actionable
