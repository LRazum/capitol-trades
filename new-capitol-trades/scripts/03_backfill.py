#!/usr/bin/env python
"""
03_backfill.py — populate Bronze from a scraped capitoltrades CSV, compact, build Silver.

KEYLESS: this replaces the Quiver bulk pull with a local CSV (no API key required). It
reads the CSV via ``config.get_congress_adapter()``, commits to Bronze year-by-year,
compacts cross-partition amendments, then derives Silver to confirm row counts. Bronze on
disk is the source of truth; Silver is recomputed on demand by later stages.

The first run still resolves each unique ticker through OpenFIGI (keyless = rate-limited,
then cached in data/processed/secmaster_cache.sqlite), so it is slower than reruns.
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
    SilverDriver,
)


def main(args):
    csv_path = Path(config.CAPITOL_TRADES_CSV)
    if not csv_path.exists():
        raise SystemExit(
            f"capitoltrades CSV not found: {csv_path}\n"
            "Set CAPITOL_TRADES_CSV in your environment (or drop the file at that path)."
        )
    store = BronzeStore(config.BRONZE)

    print(f"[1/3] capitoltrades CSV backfill {args.start} .. {args.end}")
    print(f"      source: {csv_path}")
    adapter = config.get_congress_adapter()           # <-- keyless; was QuiverCongressAdapter
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
