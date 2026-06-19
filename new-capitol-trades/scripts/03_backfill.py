#!/usr/bin/env python
"""
03_backfill.py — populate Bronze from Quiver, compact, and build Silver (report only).

Pulls the Quiver bulk congressional feed once, commits to Bronze year-by-year, compacts
cross-partition amendments, then derives Silver to confirm row counts. Bronze on disk is
the source of truth; Silver is recomputed on demand by later stages.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from capitol_ingest import (
    BronzeStore,
    HistoricalBackfill,
    QuiverCongressAdapter,
    SilverDriver,
)


def main(args):
    if not config.QUIVER_API_KEY:
        raise SystemExit("Set QUIVER_API_KEY in your environment first.")
    store = BronzeStore(config.BRONZE)

    print(f"[1/3] Quiver bulk backfill {args.start} .. {args.end}")
    adapter = QuiverCongressAdapter(config.QUIVER_API_KEY, mode="bulk")
    bf = HistoricalBackfill(adapter, store, granularity="year")
    print("   ", bf.run_bulk_once(start=args.start, end=args.end))

    print("[2/3] compact bronze (resolve cross-partition amendments)")
    print("   ", store.compact())

    print("[3/3] build silver (report)")
    silver = SilverDriver(store, config.get_security_master()).build()
    print(f"    silver trades: {silver.n_trades}   rejected: {silver.n_rejected}")
    if silver.rejections:
        print("    sample rejects:", [(r.source_record_id, r.reason) for r in silver.rejections[:3]])


def _d(s):
    return date.fromisoformat(s)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=_d, default=config.BACKFILL_START)
    ap.add_argument("--end", type=_d, default=config.BACKFILL_END)
    return ap.parse_args()


if __name__ == "__main__":
    main(parse_args())
