#!/usr/bin/env python
"""
01_build_committee.py — build the committee membership table + name->ICPSR crosswalk.

Inputs (download first, see EXECUTION_GUIDE.md):
  * Stewart-Woon assignment file (e.g. data/raw/stewart_woon/allCongressDataPublishV2.tab)
  * (optional) committee-codes crosswalk if the assignment file has codes, not names
  * congress-legislators YAML (legislators-current.yaml + legislators-historical.yaml)

Outputs: data/processed/committee_membership.parquet, data/processed/name_to_icpsr.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from capitol_ingest.committee import StewartWoonLoader, normalize_politician_name


def parse_legislators(paths) -> dict[str, int]:
    import yaml
    name_to_icpsr: dict[str, int] = {}
    for p in paths:
        p = Path(p)
        if not p.exists():
            print(f"  (skip, not found) {p}")
            continue
        for leg in yaml.safe_load(p.read_text()):
            icpsr = leg.get("id", {}).get("icpsr")
            if not icpsr:
                continue
            nm = leg.get("name", {})
            variants = set()
            if nm.get("official_full"):
                variants.add(nm["official_full"])
            if nm.get("first") and nm.get("last"):
                variants.add(f"{nm['first']} {nm['last']}")
                variants.add(f"{nm['last']}, {nm['first']}")
            for v in variants:
                name_to_icpsr[normalize_politician_name(v)] = int(icpsr)
    return name_to_icpsr


def main(args):
    print("Building name -> ICPSR crosswalk from congress-legislators ...")
    name_to_icpsr = parse_legislators([args.legislators_current, args.legislators_historical])
    config.NAME_TO_ICPSR_JSON.write_text(json.dumps(name_to_icpsr))
    print(f"  {len(name_to_icpsr)} name variants -> {config.NAME_TO_ICPSR_JSON}")

    print("Loading Stewart-Woon committee assignments ...")
    membership = StewartWoonLoader.load(args.assignments, args.codes)
    membership.to_parquet(config.MEMBERSHIP_PARQUET, index=False)
    names = sorted(membership["committee_name"].astype(str).str.lower().unique())
    print(f"  {len(membership)} assignment rows -> {config.MEMBERSHIP_PARQUET}")
    print(f"  {len(names)} distinct committee names (verify the jurisdiction map matches these):")
    for n in names[:25]:
        print(f"     - {n}")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assignments", required=True, help="Stewart-Woon assignment file (.tab/.csv/.dta/.xlsx)")
    ap.add_argument("--codes", default=None, help="committee code->name crosswalk (optional)")
    ap.add_argument("--legislators-current", default=str(config.RAW / "legislators-current.yaml"))
    ap.add_argument("--legislators-historical", default=str(config.RAW / "legislators-historical.yaml"))
    return ap.parse_args()


if __name__ == "__main__":
    main(parse_args())
