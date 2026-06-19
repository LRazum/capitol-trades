#!/usr/bin/env python
"""
04_research.py — the reality check, then persist the model for live use.

Builds Gold from Bronze with the real providers, isolates the out-of-time hold-out, and on
the DEV block only: runs the event study, conditional-portfolio tests, purged-CV, and the
Deflated Sharpe / PBO deflation. Trains the screening model on dev and saves it.

The hold-out is NOT touched unless you pass --evaluate-holdout (do that exactly once, at the
very end, after you have stopped iterating).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from capitol_ingest import (
    BronzeStore,
    GoldEventTableBuilder,
    HoldoutProtocol,
    PurgedKFold,
    SilverDriver,
    compare_groups,
    cumulative_abnormal_returns,
    deflated_sharpe_ratio,
    default_model,
    earliest_filing_dates,
    evaluate_cv,
    forward_excess_returns,
    probability_of_backtest_overfitting,
)


def build_gold():
    store = BronzeStore(config.BRONZE)
    originals = earliest_filing_dates(store.read())
    silver = SilverDriver(store, config.get_security_master()).build()
    gold = GoldEventTableBuilder(
        config.get_price_provider(), config.GOLD,
        committee=config.get_committee_provider(),
        insider=config.get_insider_provider(),
        original_filing_dates=originals,
    ).build(silver.trades)
    return gold


def main(args):
    prices = config.get_price_provider()
    gold = build_gold()
    et = gold.event_table.sort_values("t0").reset_index(drop=True)
    feat = [c for c in gold.feature_columns if et[c].notna().any()]
    print(f"gold events: {len(et)}  features: {len(feat)}  "
          f"committee_relevant: {int((et['committee_relevant'] == 1).sum())}")

    hp = HoldoutProtocol(et, holdout_frac=config.HOLDOUT_FRAC, time_col="t0")
    dev = hp.dev
    print(f"hold-out split at {hp.split_date.date()}: dev={hp.n_dev}  holdout={hp.n_holdout} (sealed)\n")

    print("=== Event study (dev): CAR vs SPY ===")
    car = cumulative_abnormal_returns(dev, prices, "SPY", horizon=10)
    print(car.assign(mean_car_bps=(car["mean_car"] * 1e4).round(1),
                     t=car["t_stat"].round(2))[["day", "mean_car_bps", "t", "n"]].to_string(index=False))

    print("\n=== Conditional portfolios (dev): forward 21d excess ===")
    fwd = forward_excess_returns(dev, prices, "SPY", horizon=21)
    for title, mask in [("committee", dev["committee_relevant"] == 1.0),
                        ("insider_net_60>0", dev.get("insider_net_60", pd.Series(0, index=dev.index)) > 0)]:
        r = compare_groups(fwd, mask, label_a=title, label_b="other")
        if "error" in r:
            print(f"  {title}: {r}")
        else:
            print(f"  {title}: diff={r['mean_diff']*1e4:.1f}bps p(t)={r['p_ttest']:.3f} "
                  f"sig={r['ci_excludes_zero']}")

    print("\n=== Purged-CV + deflation (dev) ===")
    X, y = dev[feat], (dev["label"] == 1).astype(int).to_numpy()
    rep = evaluate_cv(X, y, dev["t0"], dev["t_end"], PurgedKFold(n_splits=5, embargo_pct=0.02))
    print(f"  purged-CV mean AUC={rep['mean_auc']:.3f}  max leakage={rep['max_leakage']}")

    strat = fwd.dropna().to_numpy()
    rng = np.random.default_rng(0)
    trial_srs = [s.mean() / s.std(ddof=1) for s in
                 (fwd[rng.random(len(fwd)) < 0.5].dropna().to_numpy() for _ in range(25))
                 if len(s) > 3 and s.std(ddof=1) > 0]
    dsr = deflated_sharpe_ratio(strat, sr_trials=np.array(trial_srs)) if trial_srs else {"dsr": float("nan"), "sr_hat": float("nan"), "sr0_expected_max": float("nan")}
    print(f"  DSR={dsr['dsr']:.3f} (SR_hat={dsr['sr_hat']:.3f}, E[max SR]={dsr['sr0_expected_max']:.3f})")

    d2 = dev.copy(); d2["fwd"] = fwd; d2["month"] = pd.to_datetime(d2["t0"]).dt.to_period("M").astype(str)
    months = sorted(d2["month"].unique())
    rfeats = [c for c in ["mom_21", "dist_200ma", "size_rank", "disclosure_lag_days"] if c in d2]
    cols = {}
    for j in range(20):
        f = rng.choice(rfeats); thr = float(d2[f].median()) * (0.5 + rng.random())
        cols[f"rule_{j}"] = d2[d2[f] > thr].groupby("month")["fwd"].mean().reindex(months).fillna(0.0).to_numpy()
    if len(months) >= 10:
        pbo = probability_of_backtest_overfitting(pd.DataFrame(cols, index=months), n_splits=10)
        print(f"  PBO={pbo['pbo']:.2f} over {pbo['n_combinations']} splits")

    print("\n=== Train + persist screening model (dev) ===")
    model = default_model().fit(X.to_numpy(), y)
    import joblib
    joblib.dump(model, config.MODEL_PATH)
    config.FEATURES_JSON.write_text(json.dumps(feat))
    print(f"  saved model -> {config.MODEL_PATH}\n  saved features -> {config.FEATURES_JSON}")

    if args.evaluate_holdout:
        print("\n=== ONE-SHOT hold-out evaluation (final) ===")
        res = hp.evaluate_once(lambda h: {
            "n": len(h),
            "holdout_mean_fwd_excess_bps": round(float(forward_excess_returns(h, prices, "SPY", 21).mean()) * 1e4, 1),
        })
        print(f"  {res}")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evaluate-holdout", action="store_true",
                    help="consume the sealed hold-out for a single final evaluation")
    return ap.parse_args()


if __name__ == "__main__":
    main(parse_args())
