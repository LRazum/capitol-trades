#!/usr/bin/env python
"""
02_build_insider.py — build the tidy Form 4 insider table from SEC quarterly datasets.

Input: a directory whose subdirectories each hold one quarter's unzipped TSVs
       (SUBMISSION.tsv, REPORTINGOWNER.tsv, NONDERIV_TRANS.tsv), e.g.:
           data/raw/sec_form4/2024q1/SUBMISSION.tsv
           data/raw/sec_form4/2024q2/...
Output: data/processed/insider_tidy.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from capitol_ingest.insider import Form4BulkLoader


def main(args):
    base = Path(args.quarters_dir)
    quarter_dirs = sorted(p for p in base.iterdir() if p.is_dir() and (p / "SUBMISSION.tsv").exists())
    if not quarter_dirs:
        raise SystemExit(f"No quarter dirs with SUBMISSION.tsv under {base}. See EXECUTION_GUIDE.md.")
    print(f"Building tidy insider table from {len(quarter_dirs)} quarter(s) ...")
    tidy = Form4BulkLoader.build_tidy(quarter_dirs, config.INSIDER_TIDY_PARQUET)
    print(f"  {len(tidy):,} P/S officer-eligible rows -> {config.INSIDER_TIDY_PARQUET}")
    if not tidy.empty:
        print(f"  date range: {tidy['trans_date'].min().date()} .. {tidy['trans_date'].max().date()}")
        print(f"  distinct tickers: {tidy['ticker'].nunique():,}")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quarters-dir", default=str(config.RAW / "sec_form4"),
                    help="dir containing per-quarter subdirectories of unzipped TSVs")
    return ap.parse_args()


if __name__ == "__main__":
    main(parse_args())
