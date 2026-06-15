# Copyright (c) 2026 Lovro Razum. All rights reserved.
# PROPRIETARY AND CONFIDENTIAL. Unauthorized use, copying, modification, or
# distribution of this file, in whole or in part, via any medium, is strictly
# prohibited without the prior written permission of Lovro Razum.
# See the accompanying LICENSE file for full terms.
"""
Equity curve — how the money moves through time
===============================================
Reads an exit-strategy tradelog and plots the cumulative profit/loss of the
strategy against SPY (same $100 per pick, same entry/exit dates), so you can
literally watch the booked money rise or fall as positions close.

Two charts per run:
  1. <rule> vs SPY        — one exit rule, strategy line vs the SPY benchmark,
                            with the gap shaded green (ahead) / red (behind).
  2. all rules vs SPY     — every exit rule's cumulative P&L overlaid, so you
                            see whether ANY variant stays above the index.

Money is "booked" on each trade's exit date (closed positions only); open
positions are not counted (they have no realized result yet).

Usage:
  python equity_curve.py                         # defaults to the MAX tradelog
  python equity_curve.py --rule time_6m          # pick a different exit rule
  python equity_curve.py --bankroll 10000        # show it as account value from $10k
  python equity_curve.py --log exit_strategy_tradelog.csv --outdir validation_out

Requires: numpy, pandas, matplotlib.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C_STRAT = "#2563EB"
C_SPY = "#6B7280"
C_POS = "#16A34A"
C_NEG = "#DC2626"


def load_tradelog(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Tradelog not found: {path}\n"
            "Run run_max_backtest.py (or backtest_capitol.py) first, or pass --log."
        )
    df = pd.read_csv(path)
    for c in ("buy_date", "exit_date"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    df["open"] = df["open"].astype(str).str.strip().str.lower().isin(("true", "1", "yes"))
    if "spy_pnl" not in df.columns:
        raise ValueError("Tradelog has no 'spy_pnl' — re-run the backtest so SPY is available.")
    return df


def cumulative(df: pd.DataFrame, rule: str):
    """Closed trades of one rule, sorted by exit date -> cumulative strat & SPY P&L."""
    sub = (df[(df["rule"] == rule) & (~df["open"])]
           .dropna(subset=["exit_date", "pnl", "spy_pnl"])
           .sort_values("exit_date"))
    if sub.empty:
        return None
    return (sub["exit_date"].to_numpy(),
            sub["pnl"].cumsum().to_numpy(),
            sub["spy_pnl"].cumsum().to_numpy(),
            len(sub))


def chart_rule_vs_spy(df, rule, outdir, bankroll):
    got = cumulative(df, rule)
    if got is None:
        print(f"  '{rule}': no closed positions — skipped.")
        return
    dates, strat, spy, n = got
    strat = strat + bankroll
    spy = spy + bankroll

    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.plot(dates, strat, color=C_STRAT, lw=2.2, drawstyle="steps-post",
            label=f"Strategy ({rule})")
    ax.plot(dates, spy, color=C_SPY, lw=2.0, ls="--", drawstyle="steps-post",
            label="SPY (same $/dates)")
    ax.fill_between(dates, strat, spy, where=(strat >= spy), step="post",
                    color=C_POS, alpha=0.15, interpolate=False)
    ax.fill_between(dates, strat, spy, where=(strat < spy), step="post",
                    color=C_NEG, alpha=0.15, interpolate=False)
    ax.axhline(bankroll, color="black", lw=0.9, ls=":")

    end_s, end_b = strat[-1] - bankroll, spy[-1] - bankroll
    unit = "account value (USD)" if bankroll else "cumulative profit/loss (USD)"
    ax.set_title(f"Money over time — '{rule}' vs SPY  ·  {n} closed positions\n"
                 f"final: strategy {end_s:+,.0f}  vs  SPY {end_b:+,.0f}  "
                 f"(difference {end_s - end_b:+,.0f})")
    ax.set_ylabel(f"Per-pick book — {unit}")
    ax.set_xlabel("Exit date (money is booked when a position closes)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", rotation=30)
    _save(fig, outdir, f"equity_{rule}_vs_spy.png")


def chart_all_rules(df, outdir, bankroll):
    rules = sorted(df["rule"].unique())
    cmap = plt.get_cmap("viridis")
    # SPY reference matched to the rule with the most closed trades (widest window).
    closed = df[~df["open"]].groupby("rule").size()
    ref_rule = str(closed.sort_values(ascending=False).index[0]) if not closed.empty else None

    fig, ax = plt.subplots(figsize=(12, 6.6))
    for i, rule in enumerate(rules):
        got = cumulative(df, rule)
        if got is None:
            continue
        dates, strat, _, _ = got
        ax.plot(dates, strat + bankroll, lw=1.4, drawstyle="steps-post",
                color=cmap(i / max(1, len(rules) - 1)), label=rule, alpha=0.9)
    if ref_rule:
        d, _, spy, _ = cumulative(df, ref_rule)
        ax.plot(d, spy + bankroll, color="black", lw=2.6, ls="--", drawstyle="steps-post",
                label=f"SPY (matched to {ref_rule})", zorder=5)
    ax.axhline(bankroll, color="grey", lw=0.9, ls=":")
    ax.set_title("Money over time — every exit rule vs SPY\n"
                 "(any line that stays above the black SPY dashes = a rule that beats the index)")
    ax.set_ylabel(f"Per-pick book — {'account value' if bankroll else 'cumulative P&L'} (USD)")
    ax.set_xlabel("Exit date")
    ax.legend(fontsize=8, ncol=2, loc="best")
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", rotation=30)
    _save(fig, outdir, "equity_all_rules_vs_spy.png")


def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  chart -> {path}")


def main(argv) -> int:
    p = argparse.ArgumentParser(description="Plot cumulative money over time, strategy vs SPY.")
    p.add_argument("--log", default="exit_strategy_tradelog_MAX.csv",
                   help="exit-strategy tradelog (default: the MAX run)")
    p.add_argument("--rule", default="time_1m", help="exit rule for the focused chart")
    p.add_argument("--outdir", default="validation_out_max", help="where charts are written")
    p.add_argument("--bankroll", type=float, default=0.0,
                   help="starting bankroll; 0 = plot pure cumulative P&L from zero")
    args = p.parse_args(argv)

    df = load_tradelog(args.log)
    n_entries = int((df["rule"] == df["rule"].iloc[0]).sum())
    print(f"Loaded {args.log}: {n_entries} entries × {df['rule'].nunique()} rules")
    if args.rule not in set(df["rule"]):
        print(f"  (rule '{args.rule}' not in log; using the one with most closed trades)")
        args.rule = str(df[~df["open"]].groupby("rule").size().sort_values(ascending=False).index[0])

    chart_rule_vs_spy(df, args.rule, args.outdir, args.bankroll)
    chart_all_rules(df, args.outdir, args.bankroll)
    print(f"\nDone. Charts in: {args.outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))