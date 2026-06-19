"""
live_demo.py — wire the REAL committee + insider providers into the gold layer and screen.

Run from the directory containing `capitol_ingest`:  python live_demo.py

Uses synthetic data shaped like the real Stewart-Woon and SEC Form 4 files (so it runs
offline) but exercises the real provider logic, including the point-in-time filters.
"""
from __future__ import annotations

import hashlib
from datetime import date

import numpy as np
import pandas as pd

from capitol_ingest.committee import (
    StaticSectorProvider,
    StewartWoonCommitteeProvider,
    StewartWoonLoader,
    congress_for_date,
)
from capitol_ingest.cv import default_model
from capitol_ingest.daily import screen_signals
from capitol_ingest.gold import GoldConfig, GoldEventTableBuilder
from capitol_ingest.insider import Form4BulkLoader, Form4InsiderProvider
from capitol_ingest.prices import SyntheticPriceProvider

from gold_demo import synth_trades

# Shared fixtures ----------------------------------------------------------- #
NAME_TO_ICPSR = {"Rep. Alpha": 1, "Sen. Bravo": 2, "Rep. Charlie": 3, "Sen. Delta": 4}
SECTORS = {
    "AAA": ("Industrials", "Aerospace & Defense"), "BBB": ("Industrials", "Aerospace & Defense"),
    "CCC": ("Financials", "Banks"), "DDD": ("Financials", "Banks"),
    "EEE": ("Information Technology", "Software"), "FFF": ("Information Technology", "Software"),
    "GGG": ("Health Care", "Pharmaceuticals"), "HHH": ("Health Care", "Pharmaceuticals"),
}


def membership_df():
    rows = []
    for cong in (116, 117, 118):
        rows += [
            {"icpsr": 1, "cong": cong, "committee_code": "HSAS", "committee_name": "Armed Services"},
            {"icpsr": 3, "cong": cong, "committee_code": "HSBA", "committee_name": "Financial Services"},
            {"icpsr": 2, "cong": cong, "committee_code": "HSAG", "committee_name": "Agriculture"},
            {"icpsr": 4, "cong": cong, "committee_code": "HSPW", "committee_name": "Transportation and Infrastructure"},
        ]
    return pd.DataFrame(rows)


def insider_tidy(seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2020-10-01")
    for tkr in SECTORS:
        for _ in range(40):
            td = start + pd.Timedelta(days=int(rng.integers(0, 365 * 3 + 200)))
            code = "P" if rng.random() < 0.55 else "S"
            shares = float(rng.integers(500, 20000))
            price = float(rng.uniform(20, 300))
            rows.append({
                "ticker": tkr, "issuer_cik": f"CIK{tkr}",
                "trans_date": td, "filing_date": td + pd.Timedelta(days=2),  # Form 4 ~2 business days
                "trans_code": code, "shares": shares, "price": price,
                "value": shares * price,
                "signed_value": (shares * price) * (1 if code == "P" else -1),
                "signed_shares": shares * (1 if code == "P" else -1),
                "is_officer": bool(rng.random() < 0.7), "rptowner_cik": f"O{rng.integers(1, 6)}",
            })
    return pd.DataFrame(rows)


def part_a_committee():
    print("=== A) Stewart-Woon committee provider ===")
    # exercise the loader's column normalization on a raw-shaped frame
    raw = pd.DataFrame({"icpsr": [1, 3], "cong": [117, 117], "comcode": ["HSAS", "HSBA"]})
    codes = pd.DataFrame({"code": ["HSAS", "HSBA"], "name": ["Armed Services", "Financial Services"]})
    tidy = StewartWoonLoader.tidy(raw, codes)
    print(f"    loader normalized {len(tidy)} rows; committees: {list(tidy['committee_name'])}")

    prov = StewartWoonCommitteeProvider(membership_df(), NAME_TO_ICPSR, StaticSectorProvider(SECTORS))
    checks = [
        ("Rep. Alpha", "AAA", date(2021, 6, 1), True),    # Armed Services + Aerospace&Defense
        ("Rep. Alpha", "EEE", date(2021, 6, 1), False),   # Armed Services member, but a software name
        ("Sen. Bravo", "AAA", date(2021, 6, 1), False),   # not on a defense committee
        ("Rep. Charlie", "CCC", date(2022, 3, 1), True),  # Financial Services + Banks
        ("Rep. Charlie", "GGG", date(2022, 3, 1), False), # Financial Services member, pharma name
    ]
    for pol, tkr, d, expect in checks:
        got = prov.is_relevant(politician=pol, ticker=tkr, as_of=d)
        ok = "OK " if got == expect else "XX "
        print(f"    {ok}{pol:13} {tkr} on {d} (cong {congress_for_date(d)}) -> relevant={got}")


def part_b_insider():
    print("\n=== B) SEC Form 4 insider provider (point-in-time) ===")
    # tiny hand-built case proving the FILING_DATE <= as_of filter
    sub = pd.DataFrame({
        "ACCESSION_NUMBER": ["a1", "a2", "a3"],
        "FILING_DATE": ["2022-02-03", "2022-03-01", "2022-02-04"],  # a2 filed AFTER as_of
        "ISSUERTRADINGSYMBOL": ["ZZZ", "ZZZ", "ZZZ"],
        "ISSUERCIK": ["1", "1", "1"], "DOCUMENT_TYPE": ["4", "4", "4"],
    })
    ro = pd.DataFrame({
        "ACCESSION_NUMBER": ["a1", "a2", "a3"], "RPTOWNERCIK": ["o1", "o2", "o3"],
        "RPTOWNER_RELATIONSHIP": ["Officer", "Officer", "Director"],  # a3 is a director, not officer
    })
    nd = pd.DataFrame({
        "ACCESSION_NUMBER": ["a1", "a2", "a3"],
        "TRANS_DATE": ["2022-02-01", "2022-02-10", "2022-02-02"],
        "TRANS_CODE": ["P", "P", "P"], "TRANS_SHARES": ["1000", "5000", "2000"],
        "TRANS_PRICEPERSHARE": ["10", "10", "10"],
    })
    tidy = Form4BulkLoader.tidy_from_tables(sub, ro, nd)
    prov = Form4InsiderProvider(tidy, officers_only=True)
    res = prov.overlay(ticker="ZZZ", as_of=date(2022, 2, 15), lookback_days=30)
    print(f"    net_30 as-of 2022-02-15: ${res['insider_net_30']:,.0f}  buyers={res['insider_n_buyers']:.0f}")
    print("    expected $10,000: a1 counts; a2 excluded (filed 2022-03-01 > as_of); "
          "a3 excluded (director, not officer)")
    assert abs(res["insider_net_30"] - 10000.0) < 1e-6, "point-in-time filter failed"
    print("    OK point-in-time filing-date filter verified")


def part_c_screen():
    print("\n=== C) Real providers -> gold -> daily screen ===")
    trades, originals = synth_trades(n=160, seed=11)
    prices = SyntheticPriceProvider()
    committee = StewartWoonCommitteeProvider(membership_df(), NAME_TO_ICPSR, StaticSectorProvider(SECTORS))
    insider = Form4InsiderProvider(insider_tidy(), officers_only=True)
    cfg = GoldConfig(use_original_filing_date=False, benchmark_ticker="SPY")

    gold = GoldEventTableBuilder(
        prices, cfg, committee=committee, insider=insider, original_filing_dates=originals
    ).build(trades)
    et = gold.event_table
    print(f"    gold events: {len(et)}  committee_relevant: {int((et['committee_relevant']==1).sum())}  "
          f"insider_net_60>0: {int((et['insider_net_60']>0).sum())}")

    # train a model on the historical 70%, score the recent 30% (time-ordered)
    et = et.sort_values('t0').reset_index(drop=True)
    feat = [c for c in gold.feature_columns if et[c].notna().any()]
    split = int(len(et) * 0.7)
    train, recent = et.iloc[:split], et.iloc[split:]
    model = default_model().fit(train[feat].to_numpy(), (train["label"] == 1).astype(int).to_numpy())

    signals = screen_signals(
        recent, model=model, feature_columns=feat, proba_threshold=0.5,
        insider_col="insider_net_60",
    )
    print(f"    actionable signals (committee + insider + ML>=0.5): {len(signals)}")
    if not signals.empty:
        show = [c for c in ["as_of_date", "politician", "ticker", "transaction_type",
                            "committee_relevant", "insider_net_60", "ml_proba", "signal_score"] if c in signals]
        print(signals[show].head(8).to_string(index=False))
    else:
        print("    (none cleared all three gates this window — expected to be selective)")


if __name__ == "__main__":
    part_a_committee()
    part_b_insider()
    part_c_screen()
