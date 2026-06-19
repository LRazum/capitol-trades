#!/usr/bin/env python
"""
05_run_screener.py — live daily screen (the cron entry point).

Loads the persisted screening model, pulls recent Quiver trades through
Bronze -> Silver -> Gold with the real providers, screens to actionable signals, and writes
signals/actionable_latest.parquet (+ a dated copy) for the Streamlit dashboard.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from capitol_ingest import (
    BronzeStore,
    DailyScreener,
    QuiverCongressAdapter,
    earliest_filing_dates,
)


def main(args):
    if not config.QUIVER_API_KEY:
        raise SystemExit("Set QUIVER_API_KEY in your environment first.")
    if not config.MODEL_PATH.exists():
        raise SystemExit("No trained model. Run scripts/04_research.py first.")

    import joblib
    model = joblib.load(config.MODEL_PATH)
    feature_columns = json.loads(config.FEATURES_JSON.read_text())

    store = BronzeStore(config.BRONZE)
    originals = earliest_filing_dates(store.read()) if any(store.root.glob("**/data.parquet")) else {}

    screener = DailyScreener(
        adapter=QuiverCongressAdapter(config.QUIVER_API_KEY, mode="live"),
        store=store,
        security_master=config.get_security_master(),
        price_provider=config.get_price_provider(),
        committee_provider=config.get_committee_provider(),
        insider_provider=config.get_insider_provider(),
        gold_config=config.GOLD,
        model=model,
        feature_columns=feature_columns,
        original_filing_dates=originals,
        proba_threshold=config.PROBA_THRESHOLD,
        out_dir=config.SIGNALS,
    )
    actionable = screener.run(
        pull_lookback_days=args.pull_lookback_days,
        signal_window_days=args.signal_window_days,
        compact=args.compact,
    )
    print(f"\n{len(actionable)} actionable signal(s):")
    if not actionable.empty:
        print(actionable.to_string(index=False))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pull-lookback-days", type=int, default=14)
    ap.add_argument("--signal-window-days", type=int, default=7)
    ap.add_argument("--compact", action="store_true", help="run cross-partition compaction after ingest")
    return ap.parse_args()


if __name__ == "__main__":
    main(parse_args())
