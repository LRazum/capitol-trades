"""
gold_demo.py — end-to-end: silver trades -> gold event table -> purged-CV validation.

Run from the directory containing `capitol_ingest`:  python gold_demo.py

Offline + deterministic (synthetic prices, no network). On synthetic random data the
model AUC should hover near 0.5 — the point is that the framework reports *honest* OOS
performance and that leakage is provably zero, not that there is an edge.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from capitol_ingest.amounts import parse_amount
from capitol_ingest.cv import (
    CombinatorialPurgedKFold,
    PurgedKFold,
    evaluate_cv,
    leakage_report,
)
from capitol_ingest.gold import GoldConfig, GoldEventTableBuilder
from capitol_ingest.models import (
    AssetType,
    Chamber,
    NormalizedTrade,
    Owner,
    ResolutionMethod,
    SecurityRef,
    TransactionType,
)
from capitol_ingest.prices import SyntheticPriceProvider


# --------------------------------------------------------------------------- #
# Synthesize a silver-trade set (these would come from SilverDriver.build())
# --------------------------------------------------------------------------- #
def make_trade(i, politician, ticker, tx, filing, ttype, bracket, owner, chamber):
    return NormalizedTrade(
        trade_id=f"T{i:04d}",
        source="demo",
        source_record_id=f"SID{i:04d}",
        politician=politician,
        chamber=chamber,
        owner=owner,
        transaction_type=ttype,
        asset_type=AssetType.STOCK,
        security=SecurityRef(figi=f"BBG-{ticker}", ticker_as_of=ticker,
                             method=ResolutionMethod.TICKER, confidence=1.0),
        amount=parse_amount(bracket),
        transaction_date=tx,
        filing_date=filing,
    )


def synth_trades(n=80, seed=7):
    rng = np.random.default_rng(seed)
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]
    brackets = ["$1,001 - $15,000", "$15,001 - $50,000", "$50,001 - $100,000",
                "$100,001 - $250,000", "$250,001 - $500,000"]
    politicians = ["Rep. Alpha", "Sen. Bravo", "Rep. Charlie", "Sen. Delta"]
    start = pd.Timestamp("2021-01-04")
    trades, originals = [], {}
    for i in range(n):
        tx = (start + pd.Timedelta(days=int(rng.integers(0, 365 * 3)))).date()
        lag = int(rng.integers(5, 44))
        filing = (pd.Timestamp(tx) + pd.Timedelta(days=lag)).date()
        t = make_trade(
            i, politicians[i % len(politicians)], tickers[int(rng.integers(0, len(tickers)))],
            tx, filing,
            TransactionType.PURCHASE if rng.random() < 0.6 else TransactionType.SALE,
            brackets[int(rng.integers(0, len(brackets)))],
            Owner.SELF if rng.random() < 0.7 else Owner.SPOUSE,
            Chamber.HOUSE if rng.random() < 0.5 else Chamber.SENATE,
        )
        trades.append(t)
    # Mark two records as amended: silver carries the (later) amended filing date, while
    # the bronze-derived map remembers the earlier original disclosure date.
    for idx in (10, 25):
        amended_filing = (pd.Timestamp(trades[idx].filing_date) + pd.Timedelta(days=120)).date()
        originals[trades[idx].source_record_id] = trades[idx].filing_date  # original
        trades[idx] = trades[idx].model_copy(update={"filing_date": amended_filing})
    return trades, originals


def main():
    trades, originals = synth_trades()
    prices = SyntheticPriceProvider()
    cfg = GoldConfig(use_original_filing_date=False, benchmark_ticker="SPY")

    print("=== 1) Build gold event table (triple-barrier + point-in-time features) ===")
    builder = GoldEventTableBuilder(prices, cfg, original_filing_dates=originals)
    res = builder.build(trades)
    et = res.event_table
    print(f"    events: {res.n_events}   rejected: {res.n_rejected}")
    print(f"    label distribution: {et['label'].value_counts().to_dict()}")
    print(f"    feature columns ({len(res.feature_columns)}): {res.feature_columns}")
    print("\n    sample rows:")
    cols = ["politician", "ticker", "as_of_date", "t0", "t_end", "label", "barrier",
            "ret", "mom_21", "dist_200ma", "committee_relevant", "insider_available"]
    print(et[cols].head(6).to_string(index=False))

    print("\n=== 2) Entry-timing switch (original vs amendment) ===")
    res_orig = GoldEventTableBuilder(
        prices, GoldConfig(use_original_filing_date=True, benchmark_ticker="SPY"),
        original_filing_dates=originals,
    ).build(trades)
    amended_ids = list(originals.keys())
    a = res.event_table.set_index("source_record_id")
    b = res_orig.event_table.set_index("source_record_id")
    print("    for amended records, t0/label_interval shift when entering on the original date:")
    for sid in amended_ids:
        if sid in a.index and sid in b.index:
            print(f"      {sid}: amendment-date t0={a.loc[sid,'t0'].date()} "
                  f"(interval ends {a.loc[sid,'t_end'].date()})  |  "
                  f"original-date t0={b.loc[sid,'t0'].date()} "
                  f"(interval ends {b.loc[sid,'t_end'].date()})")

    print("\n=== 3) Purged K-Fold: prove zero label-interval leakage ===")
    # Drop all-NaN placeholder columns (e.g. the unwired Form 4 insider fields) from the
    # modeling matrix; they remain in the event table as schema, marked unavailable.
    usable = [c for c in res.feature_columns if et[c].notna().any()]
    dropped = [c for c in res.feature_columns if c not in usable]
    if dropped:
        print(f"    (excluded {len(dropped)} all-NaN placeholder feature(s) from X: {dropped})")
    X = et[usable]
    y = (et["label"] == 1).astype(int).to_numpy()
    t0, t1 = et["t0"], et["t_end"]
    pkf = PurgedKFold(n_splits=5, embargo_pct=0.02)
    for k, (tr, te) in enumerate(pkf.split(X, t0, t1), 1):
        print(f"    fold {k}: train={len(tr):3d} test={len(te):3d}  "
              f"leakage={leakage_report(t0, t1, tr, te)}")

    print("\n=== 4) Honest OOS evaluation (PurgedKFold) ===")
    rep = evaluate_cv(X, y, t0, t1, pkf)
    print(f"    per-fold AUC: {[round(a,3) if a==a else None for a in rep['fold_auc']]}")
    print(f"    mean AUC: {rep['mean_auc']:.3f}   max leakage across folds: {rep['max_leakage']}")
    print("    (≈0.5 expected on synthetic random data — framework reports honestly)")

    print("\n=== 5) Combinatorial Purged CV (multiple OOS paths) ===")
    cpcv = CombinatorialPurgedKFold(n_groups=6, n_test_groups=2, embargo_pct=0.02)
    rep2 = evaluate_cv(X, y, t0, t1, cpcv)
    print(f"    paths: {rep2['n_folds']}   mean AUC: {rep2['mean_auc']:.3f}   "
          f"max leakage: {rep2['max_leakage']}")


if __name__ == "__main__":
    main()
