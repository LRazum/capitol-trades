#!/usr/bin/env python
"""
00_download_raw_data.py — populate data/raw/ automatically where possible.

Automates:
  1. congress-legislators YAML (legislators-current.yaml + legislators-historical.yaml)
  2. SEC Form 4 quarterly datasets -> data/raw/sec_form4/YYYYqX/ (unzipped, zips removed)

The Stewart-Woon committee data lives on Harvard Dataverse, which can't be reliably scripted
(Terms of Use + a JS download UI), so this script only *checks* for it and points you to the
manual steps.

Examples:
  python scripts/00_download_raw_data.py --user-agent "Jane Doe jane@example.com"
  python scripts/00_download_raw_data.py --years-back 10 --user-agent "Jane Doe jane@x.com"
  SEC_USER_AGENT="Jane Doe jane@x.com" python scripts/00_download_raw_data.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import zipfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

import config

SEC_BASE = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets"
SEC_FILES_EXPECTED = ("SUBMISSION.tsv", "REPORTINGOWNER.tsv", "NONDERIV_TRANS.tsv")
SEC_EARLIEST_YEAR = 2006  # SEC insider data sets begin 2006 Q1

LEG_RAW = "https://raw.githubusercontent.com/unitedstates/congress-legislators/{branch}/{fname}"
LEG_BRANCHES = ("main", "master")
LEG_FILES = ("legislators-current.yaml", "legislators-historical.yaml")
GITHUB_UA = "capitol-trades-downloader"


def _download(url: str, dest: Path, headers: dict, *, attempts: int = 3, timeout: int = 120) -> int:
    """Stream a URL to dest. Returns HTTP-ish status: 200 ok, 404 missing, 0 failed."""
    tmp = dest.parent / (dest.name + ".part")
    for i in range(1, attempts + 1):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
                if r.status_code == 404:
                    return 404
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        if chunk:
                            f.write(chunk)
            tmp.replace(dest)
            return 200
        except requests.RequestException as e:
            if i == attempts:
                print(f"    ! failed after {attempts} attempts: {e}")
                tmp.unlink(missing_ok=True)
                return 0
            time.sleep(2 * i)
    return 0


def fetch_legislators(force: bool) -> None:
    print("== congress-legislators (name -> ICPSR source) ==")
    headers = {"User-Agent": GITHUB_UA}
    for fname in LEG_FILES:
        dest = config.RAW / fname
        if dest.exists() and not force:
            print(f"  [skip] {fname} already present")
            continue
        for branch in LEG_BRANCHES:
            code = _download(LEG_RAW.format(branch=branch, fname=fname), dest, headers)
            if code == 200:
                print(f"  [ok]   {fname}  (branch={branch}, {dest.stat().st_size // 1024} KB)")
                break
            if code == 404:
                continue
        else:
            print(f"  [FAIL] {fname} (tried branches {LEG_BRANCHES})")


def completed_quarters(n_years: int, today: date | None = None) -> list[tuple[int, int]]:
    """Quarters from the most recent COMPLETED quarter back n_years (newest first)."""
    today = today or date.today()
    cy, cq = today.year, (today.month - 1) // 3 + 1
    y, q = (cy - 1, 4) if cq == 1 else (cy, cq - 1)  # previous (completed) quarter
    out: list[tuple[int, int]] = []
    for _ in range(n_years * 4):
        if y < SEC_EARLIEST_YEAR:
            break
        out.append((y, q))
        q -= 1
        if q == 0:
            q, y = 4, y - 1
    return out


def fetch_sec(quarters: list[tuple[int, int]], user_agent: str, sleep: float, force: bool) -> None:
    print("== SEC Form 4 quarterly datasets ==")
    headers = {"User-Agent": user_agent}
    base = config.RAW / "sec_form4"
    base.mkdir(parents=True, exist_ok=True)
    for y, q in quarters:
        tag = f"{y}q{q}"
        dest_dir = base / tag
        if (dest_dir / "SUBMISSION.tsv").exists() and not force:
            print(f"  [skip] {tag} already unzipped")
            continue
        zpath = base / f"{tag}_form345.zip"
        code = _download(f"{SEC_BASE}/{tag}_form345.zip", zpath, headers)
        if code == 404:
            print(f"  [404]  {tag} not posted yet — skipping")
            continue
        if code != 200:
            print(f"  [FAIL] {tag}")
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(zpath) as z:
                z.extractall(dest_dir)
        except zipfile.BadZipFile:
            print(f"  [FAIL] {tag} (corrupt zip)")
            zpath.unlink(missing_ok=True)
            continue
        zpath.unlink(missing_ok=True)
        present = [f for f in SEC_FILES_EXPECTED if (dest_dir / f).exists()]
        print(f"  [ok]   {tag} -> sec_form4/{tag}/  ({', '.join(present) or 'no expected TSVs?!'})")
        time.sleep(sleep)


def check_stewart_woon() -> None:
    print("== Stewart-Woon committee assignments (MANUAL — see instructions) ==")
    sw = config.RAW / "stewart_woon"
    found = sorted(p.name for p in sw.glob("*assignment*")) if sw.exists() else []
    if found:
        print(f"  [ok]   found assignment file(s): {found}")
    else:
        print("  [todo] nothing in data/raw/stewart_woon/ yet.")
        print("         Harvard Dataverse can't be scripted (ToU + JS UI). Follow the manual")
        print("         click-by-click steps, then re-check with:")
        print("           python scripts/00_download_raw_data.py --skip-sec --skip-legislators")


def main(args) -> None:
    config.RAW.mkdir(parents=True, exist_ok=True)
    if not args.skip_legislators:
        fetch_legislators(args.force)
    if not args.skip_sec:
        ua = args.user_agent or os.getenv("SEC_USER_AGENT", "")
        if not ua or " " not in ua:
            raise SystemExit(
                "SEC blocks requests without a descriptive User-Agent (you'll get a 403).\n"
                "  Provide one: --user-agent 'Your Name your@email.com'  (or set SEC_USER_AGENT)."
            )
        qs = completed_quarters(args.years_back)
        if qs:
            print(f"== SEC range: {qs[-1][0]}q{qs[-1][1]} .. {qs[0][0]}q{qs[0][1]} "
                  f"({len(qs)} quarters) ==")
        fetch_sec(qs, ua, args.sleep, args.force)
    check_stewart_woon()
    print("\nDone. Next: scripts/01_build_committee.py, then scripts/02_build_insider.py")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user-agent", default=None,
                    help="SEC User-Agent, e.g. 'Jane Doe jane@example.com' (or env SEC_USER_AGENT)")
    ap.add_argument("--years-back", type=int, default=5, help="years of SEC quarters to fetch (default 5)")
    ap.add_argument("--sleep", type=float, default=0.5, help="seconds between SEC requests (politeness)")
    ap.add_argument("--skip-sec", action="store_true")
    ap.add_argument("--skip-legislators", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-download even if files already exist")
    return ap.parse_args()


if __name__ == "__main__":
    main(parse_args())
