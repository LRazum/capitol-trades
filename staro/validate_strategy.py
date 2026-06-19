# Copyright (c) 2026 Lovro Razum. All rights reserved.
# PROPRIETARY AND CONFIDENTIAL. Unauthorized use, copying, modification, or
# distribution of this file, in whole or in part, via any medium, is strictly
# prohibited without the prior written permission of Lovro Razum.
# See the accompanying LICENSE file for full terms.
"""
Statistical validation harness for the Capitol-Trades strategy
===============================================================
`selftest_backtest.py` proves the engine is *mechanically* correct (P&L math,
exit triggers, look-ahead safety). This script answers the harder question:

    Is the backtested edge REAL, or is it luck / overfitting on ~30 trades?

It runs the full battery discussed, in two stages.

STAGE A  — offline, from an exit-strategy tradelog CSV (no network, instant):
  A1  Per-rule significance vs 0 + MULTIPLE-COMPARISONS correction (Bonferroni, BH)
  A2  BOOTSTRAP confidence intervals on per-trade excess return (forest + histogram)
  A3  WALK-FORWARD / out-of-sample split + IS-vs-OOS rank correlation (overfit check)
  A4  TRANSACTION-COST / slippage sensitivity sweep
  A5  STATISTICAL POWER / minimum-detectable-effect (the "n is too small" check)
  A6  CONCENTRATION: how much of the P&L rides on one or two names (fragility)

STAGE B  — live, needs `app` + `backtest_capitol` + network (yfinance):
  B1  PERMUTATION / NULL test: random ticker picks, same dates/exit, 100s of draws
  B2  COMPARATOR strategies: real vs random vs MOMENTUM-only vs size-only-no-MC-filter
      (isolates whether the edge is the politician signal, the MC filter, or just momentum)
  B3  SURVIVORSHIP diagnostic: price-history length of every selected name

Every analysis prints a summary AND writes a PNG chart to --outdir.

Usage:
  # Stage A on the MAX run (this is now the DEFAULT log/trades/outdir):
  python validate_strategy.py

  # validate the small S&P-only run instead:
  python validate_strategy.py --log exit_strategy_tradelog.csv \
      --trades capitol_trades_cache.csv --outdir validation_out

  # add the live comparator + permutation tests (needs the streamlit env + network):
  python validate_strategy.py --comparators --permutations 500 --pool-size 200

Requires: numpy, pandas, matplotlib, scipy. Stage B additionally needs the live
app stack (streamlit/yfinance) on the import path, exactly like backtest_capitol.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import textwrap
from datetime import date, timedelta

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Optional but strongly preferred — clean, standard tests. Graceful fallbacks below.
try:
    from scipy import stats as _sps
    _HAVE_SCIPY = True
except Exception:  # noqa: BLE001
    _sps = None
    _HAVE_SCIPY = False

# --------------------------------------------------------------------------- #
# Shared palette (matches the rest of the project)
# --------------------------------------------------------------------------- #
C_STRAT = "#2563EB"
C_SPY = "#6B7280"
C_POS = "#16A34A"
C_NEG = "#DC2626"
C_ACCENT = "#7C3AED"
C_AMBER = "#D97706"

# Standard normal quantiles (avoid a scipy dependency for the power maths).
Z_ALPHA_2 = 1.959963985   # two-sided alpha = 0.05
Z_POWER_80 = 0.841621234  # power = 0.80


# =========================================================================== #
# Statistics utilities (pure)
# =========================================================================== #
def bootstrap_stat(x: np.ndarray, stat=np.median, n_boot: int = 10000,
                   seed: int = 42) -> dict:
    """Percentile bootstrap of `stat` over a 1-D sample.

    Returns point estimate, 95% CI, and a two-sided bootstrap p-value for the
    null that the statistic is 0. `excludes_zero` is the headline: if the 95%
    CI does not straddle 0 there is bootstrap evidence of a non-zero effect.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return {"n": 0, "point": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "p": float("nan"), "excludes_zero": False}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = stat(x[idx], axis=1)
    point = float(stat(x))
    lo, hi = (float(v) for v in np.percentile(boot, [2.5, 97.5]))
    # two-sided bootstrap p: how often the resampled stat lands on the other
    # side of zero from the point estimate (x2, clipped).
    frac_le = float((boot <= 0).mean())
    frac_ge = float((boot >= 0).mean())
    p = min(1.0, 2.0 * min(frac_le, frac_ge))
    return {"n": n, "point": point, "lo": lo, "hi": hi, "p": p,
            "excludes_zero": not (lo <= 0.0 <= hi)}


def one_sample_pvalue(x: np.ndarray) -> dict:
    """p-value(s) for H0: location(excess) = 0.

    Primary = Wilcoxon signed-rank (non-parametric, robust to fat-tailed
    returns). Secondary = one-sample t. Falls back to a binomial SIGN test if
    scipy is unavailable. Returns NaN p when the sample is degenerate.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    out = {"n": len(x), "p_wilcoxon": float("nan"), "p_ttest": float("nan"),
           "p_sign": float("nan")}
    if len(x) < 3:
        return out
    # sign test (always available)
    pos = int((x > 0).sum())
    nonzero = int((x != 0).sum())
    if nonzero > 0:
        if _HAVE_SCIPY:
            out["p_sign"] = float(_sps.binomtest(pos, nonzero, 0.5).pvalue)
        else:
            # exact two-sided binomial via the normal approx (good enough at n>=10)
            mu, sd = 0.5 * nonzero, math.sqrt(0.25 * nonzero)
            z = (pos - mu) / sd if sd > 0 else 0.0
            out["p_sign"] = float(2 * (1 - _norm_cdf(abs(z))))
    if _HAVE_SCIPY:
        try:
            out["p_wilcoxon"] = float(_sps.wilcoxon(x, alternative="two-sided").pvalue)
        except Exception:  # noqa: BLE001 — all-zero / all-tied etc.
            out["p_wilcoxon"] = float("nan")
        try:
            out["p_ttest"] = float(_sps.ttest_1samp(x, 0.0).pvalue)
        except Exception:  # noqa: BLE001
            out["p_ttest"] = float("nan")
    else:
        out["p_wilcoxon"] = out["p_sign"]
        # normal-approx t
        sd = x.std(ddof=1)
        if sd > 0:
            t = x.mean() / (sd / math.sqrt(len(x)))
            out["p_ttest"] = float(2 * (1 - _norm_cdf(abs(t))))
    return out


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def bonferroni(pvals: list[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    return np.clip(p * np.sum(~np.isnan(p)), 0, 1)


def benjamini_hochberg(pvals: list[float]) -> np.ndarray:
    """BH-adjusted p-values (q-values); NaNs passed through untouched."""
    p = np.asarray(pvals, dtype=float)
    mask = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    pv = p[mask]
    m = len(pv)
    if m == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order]
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(m)
    out[order] = adj
    q[mask] = out
    return q


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return float("nan")
    if _HAVE_SCIPY:
        return float(_sps.spearmanr(a, b).correlation)
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


# =========================================================================== #
# IO
# =========================================================================== #
def load_tradelog(path: str) -> pd.DataFrame:
    """Load an exit-strategy tradelog, coercing dtypes we rely on."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Tradelog not found: {path}\n"
            "Run `python backtest_capitol.py` first to produce "
            "exit_strategy_tradelog.csv, or pass --log."
        )
    df = pd.read_csv(path)
    for c in ("buy_date", "exit_date", "rebalance_date"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ("open", "beat_spy"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower().isin(("true", "1", "yes"))
    if "excess_return" not in df.columns:
        raise ValueError(
            "Tradelog has no 'excess_return' column — the backtest was run "
            "without a SPY benchmark. Re-run so SPY is available; the whole "
            "point here is excess vs the index."
        )
    return df


def closed_excess(df: pd.DataFrame, rule: str) -> np.ndarray:
    """Per-trade excess returns for CLOSED positions of one rule."""
    sub = df[(df["rule"] == rule) & (~df["open"])]
    return sub["excess_return"].dropna().to_numpy()


# =========================================================================== #
# STAGE A — offline analyses (each prints + writes a chart)
# =========================================================================== #
def a1_significance(df: pd.DataFrame, outdir: str) -> pd.DataFrame:
    """Per-rule excess-vs-0 test with Bonferroni + BH multiple-comparisons control."""
    rules = sorted(df["rule"].unique())
    rows = []
    for r in rules:
        x = closed_excess(df, r)
        pv = one_sample_pvalue(x)
        rows.append({
            "rule": r,
            "n_closed": pv["n"],
            "median_excess_%": float(np.median(x)) * 100 if len(x) else float("nan"),
            "mean_excess_%": float(np.mean(x)) * 100 if len(x) else float("nan"),
            "p_raw": pv["p_wilcoxon"],
            "p_ttest": pv["p_ttest"],
        })
    tab = pd.DataFrame(rows)
    tab["p_bonferroni"] = bonferroni(tab["p_raw"].tolist())
    tab["q_bh"] = benjamini_hochberg(tab["p_raw"].tolist())
    tab["sig_raw"] = tab["p_raw"] < 0.05
    tab["sig_bonf"] = tab["p_bonferroni"] < 0.05
    tab["sig_bh"] = tab["q_bh"] < 0.05
    tab = tab.sort_values("p_raw", na_position="last").reset_index(drop=True)

    m = int((~tab["p_raw"].isna()).sum())
    print("\n" + "=" * 86)
    print(f"A1  PER-RULE SIGNIFICANCE vs 0  +  MULTIPLE-COMPARISONS CONTROL  (m={m} rules)")
    print("    H0: per-trade excess return over SPY has median/location 0.")
    print("=" * 86)
    with pd.option_context("display.float_format", lambda v: f"{v:0.4f}"):
        print(tab[["rule", "n_closed", "median_excess_%", "mean_excess_%",
                   "p_raw", "p_bonferroni", "q_bh",
                   "sig_raw", "sig_bonf", "sig_bh"]].to_string(index=False))
    n_raw, n_bonf, n_bh = tab["sig_raw"].sum(), tab["sig_bonf"].sum(), tab["sig_bh"].sum()
    print(f"\n  Significant at 0.05 — raw: {n_raw}/{m}   after Bonferroni: {n_bonf}/{m}   "
          f"after BH-FDR: {n_bh}/{m}")
    print("  Testing 11 rules and keeping the best inflates false positives; Bonferroni/BH "
          "is the honest bar.")

    # chart: -log10(p) per rule with raw + Bonferroni thresholds
    fig, ax = plt.subplots(figsize=(11, 5.5))
    t = tab.dropna(subset=["p_raw"]).copy()
    safe_p = t["p_raw"].clip(lower=1e-6)
    heights = -np.log10(safe_p)
    colors = [C_POS if s else C_NEG for s in t["sig_bonf"]]
    ax.bar(t["rule"], heights, color=colors, edgecolor="white")
    ax.axhline(-np.log10(0.05), color="grey", ls="--", lw=1,
               label="raw α = 0.05")
    ax.axhline(-np.log10(0.05 / m), color=C_ACCENT, ls="--", lw=1.2,
               label=f"Bonferroni α = 0.05/{m}")
    ax.set_ylabel("-log10(raw p-value)   (higher = stronger)")
    ax.set_title("A1 — Per-rule significance of excess vs SPY, with multiple-comparisons thresholds")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    ax.tick_params(axis="x", rotation=35)
    _save(fig, outdir, "A1_significance.png")
    return tab


def a2_bootstrap(df: pd.DataFrame, outdir: str, n_boot: int, headline: str,
                 seed: int) -> pd.DataFrame:
    """Bootstrap 95% CI of per-trade MEDIAN excess for every rule (forest plot)."""
    rules = sorted(df["rule"].unique())
    rows = []
    for r in rules:
        bs = bootstrap_stat(closed_excess(df, r), np.median, n_boot, seed)
        rows.append({"rule": r, **{k: bs[k] for k in
                                   ("n", "point", "lo", "hi", "p", "excludes_zero")}})
    tab = pd.DataFrame(rows).sort_values("point", ascending=False).reset_index(drop=True)
    for c in ("point", "lo", "hi"):
        tab[c + "_%"] = tab[c] * 100

    print("\n" + "=" * 86)
    print(f"A2  BOOTSTRAP 95% CI OF PER-TRADE MEDIAN EXCESS  ({n_boot:,} resamples)")
    print("=" * 86)
    with pd.option_context("display.float_format", lambda v: f"{v:0.3f}"):
        print(tab[["rule", "n", "point_%", "lo_%", "hi_%", "excludes_zero"]]
              .rename(columns={"point_%": "median%", "lo_%": "ci_lo%", "hi_%": "ci_hi%"})
              .to_string(index=False))
    any_excl = tab["excludes_zero"].any()
    print(f"\n  Rules whose 95% CI EXCLUDES zero: {int(tab['excludes_zero'].sum())}/{len(tab)}.")
    print("  A CI that straddles 0 = no bootstrap evidence the edge is real for that rule.")

    # forest / caterpillar plot
    fig, ax = plt.subplots(figsize=(11, 6))
    y = np.arange(len(tab))[::-1]
    for yi, (_, r) in zip(y, tab.iterrows()):
        col = C_POS if r["excludes_zero"] and r["point"] > 0 else (
            C_NEG if r["excludes_zero"] else C_SPY)
        ax.plot([r["lo_%"], r["hi_%"]], [yi, yi], color=col, lw=2.4, solid_capstyle="round")
        ax.plot(r["point_%"], yi, "o", color=col, ms=7)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(tab["rule"])
    ax.set_xlabel("Per-trade median excess vs SPY (%)  — bars = bootstrap 95% CI")
    ax.set_title("A2 — Bootstrap 95% CI of median excess by exit rule\n"
                 "(green = CI clears 0, grey = CI straddles 0 → not distinguishable from luck)")
    ax.grid(alpha=0.3, axis="x")
    _save(fig, outdir, "A2_bootstrap_forest.png")

    # headline-rule detail histogram
    x = closed_excess(df, headline)
    if len(x):
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(x), size=(n_boot, len(x)))
        boot = np.median(x[idx], axis=1) * 100
        bs = bootstrap_stat(x, np.median, n_boot, seed)
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.hist(boot, bins=40, color=C_STRAT, edgecolor="white", alpha=0.85)
        ax.axvline(0, color="black", lw=1.2, label="zero")
        ax.axvline(bs["point"] * 100, color=C_AMBER, lw=2, label=f"median {bs['point']*100:+.2f}%")
        ax.axvline(bs["lo"] * 100, color=C_NEG, ls="--", lw=1.4, label="95% CI")
        ax.axvline(bs["hi"] * 100, color=C_NEG, ls="--", lw=1.4)
        ax.set_title(f"A2 — Bootstrap distribution of median excess, headline rule '{headline}' "
                     f"(n={len(x)} closed)")
        ax.set_xlabel("Median per-trade excess vs SPY (%)")
        ax.set_ylabel("Resamples")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        _save(fig, outdir, "A2_bootstrap_headline.png")
    return tab


def a3_walk_forward(df: pd.DataFrame, outdir: str, split: float, n_boot: int,
                    seed: int) -> dict:
    """Pick the best rule IN-SAMPLE, then judge it OUT-OF-SAMPLE. Plus IS/OOS rank corr."""
    entries = (df[df["rule"] == df["rule"].iloc[0]]
               .sort_values("buy_date"))
    cut_dates = sorted(df["buy_date"].dropna().unique())
    if len(cut_dates) < 6:
        print("\nA3  WALK-FORWARD: too few distinct entry dates — skipped.")
        return {}
    k = max(1, int(len(cut_dates) * split))
    is_dates = set(cut_dates[:k])
    oos_dates = set(cut_dates[k:])
    is_cut = pd.Timestamp(sorted(is_dates)[-1])
    oos_start = pd.Timestamp(sorted(oos_dates)[0])

    rows = []
    for r in sorted(df["rule"].unique()):
        sub = df[(df["rule"] == r) & (~df["open"])]
        xi = sub[sub["buy_date"].isin(is_dates)]["excess_return"].dropna().to_numpy()
        xo = sub[sub["buy_date"].isin(oos_dates)]["excess_return"].dropna().to_numpy()
        rows.append({
            "rule": r,
            "is_n": len(xi), "oos_n": len(xo),
            "is_median_%": float(np.median(xi)) * 100 if len(xi) else float("nan"),
            "oos_median_%": float(np.median(xo)) * 100 if len(xo) else float("nan"),
        })
    wf = pd.DataFrame(rows)

    eligible = wf[wf["is_n"] >= 3].copy()
    chosen = (eligible.sort_values("is_median_%", ascending=False).iloc[0]["rule"]
              if not eligible.empty else None)

    rho = spearman(wf["is_median_%"].to_numpy(), wf["oos_median_%"].to_numpy())

    print("\n" + "=" * 86)
    print(f"A3  WALK-FORWARD  (split {split:.0%}: IS ≤ {is_cut.date()}, OOS ≥ {oos_start.date()})")
    print("=" * 86)
    with pd.option_context("display.float_format", lambda v: f"{v:0.3f}"):
        print(wf.to_string(index=False))
    oos_verdict = None
    if chosen is not None:
        xo = df[(df["rule"] == chosen) & (~df["open"]) &
                (df["buy_date"].isin(oos_dates))]["excess_return"].dropna().to_numpy()
        is_med = wf.loc[wf["rule"] == chosen, "is_median_%"].iloc[0]
        print(f"\n  Rule chosen ON IN-SAMPLE (best IS median excess): '{chosen}'  "
              f"(IS median {is_med:+.2f}%)")
        if len(xo) == 0:
            oos_verdict = "NO matured OOS trades"
            print("  It has NO matured out-of-sample trades — a long hold means its recent picks "
                  "are still open. Its IS number is UNPROVEN out-of-sample (a classic optimiser trap).")
        else:
            bs = bootstrap_stat(xo, np.median, n_boot, seed)
            holds = bs["excludes_zero"] and bs["point"] > 0
            oos_verdict = "HOLDS UP" if holds else "does NOT clear 0"
            print(f"  Its OUT-OF-SAMPLE median excess: {bs['point']*100:+.2f}%  "
                  f"95% CI [{bs['lo']*100:+.2f}%, {bs['hi']*100:+.2f}%]  (n={bs['n']}) -> {oos_verdict}")
    print(f"\n  Spearman rank corr (IS median vs OOS median across rules): rho = {rho:+.2f}")
    print("  rho near 0 or negative = the IS ranking does NOT predict OOS -> rule choice is overfit.")

    # chart 1: chosen-rule IS vs OOS
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    ax = axes[0]
    if chosen is not None:
        is_med = wf.loc[wf["rule"] == chosen, "is_median_%"].iloc[0]
        oos_med = wf.loc[wf["rule"] == chosen, "oos_median_%"].iloc[0]
        vals = [is_med, 0.0 if np.isnan(oos_med) else oos_med]
        ax.bar(["In-sample", "Out-of-sample"], vals,
               color=[C_STRAT, C_AMBER], edgecolor="white")
        if np.isnan(oos_med):
            ax.annotate("no matured\nOOS trades", (1, 0), ha="center", va="bottom",
                        fontsize=9, color=C_NEG, xytext=(0, 6), textcoords="offset points")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_title(f"A3 — Chosen rule '{chosen}': IS vs OOS median excess")
        ax.set_ylabel("Median excess vs SPY (%)")
        ax.grid(alpha=0.3, axis="y")
    # chart 2: IS vs OOS scatter across rules
    ax = axes[1]
    ax.scatter(wf["is_median_%"], wf["oos_median_%"], color=C_ACCENT,
               edgecolor="white", s=60, zorder=3)
    for _, r in wf.iterrows():
        ax.annotate(r["rule"], (r["is_median_%"], r["oos_median_%"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    lim = [np.nanmin(wf[["is_median_%", "oos_median_%"]].to_numpy()) - 1,
           np.nanmax(wf[["is_median_%", "oos_median_%"]].to_numpy()) + 1]
    ax.plot(lim, lim, color="grey", ls="--", lw=1, label="y = x")
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlabel("In-sample median excess (%)")
    ax.set_ylabel("Out-of-sample median excess (%)")
    ax.set_title(f"A3 — Does IS predict OOS?  Spearman rho = {rho:+.2f}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, outdir, "A3_walk_forward.png")
    return {"chosen": chosen, "rho": rho, "oos_verdict": oos_verdict}


def a4_costs(df: pd.DataFrame, outdir: str, cost_levels_bps: list[float],
             rules: list[str]) -> pd.DataFrame:
    """How fast does the excess edge erode once you charge slippage + commission?"""
    rows = []
    for r in rules:
        sub = df[(df["rule"] == r) & (~df["open"])].dropna(subset=["excess_return"])
        if sub.empty:
            continue
        bpx = sub["buy_price"].to_numpy()
        xpx = sub["exit_price"].to_numpy()
        spy = sub["spy_return_pct"].to_numpy()
        for bps in cost_levels_bps:
            c = bps / 10000.0
            strat_net = (xpx * (1 - c)) / (bpx * (1 + c)) - 1.0          # both legs
            spy_net = (1 + spy) * ((1 - c) / (1 + c)) - 1.0             # SPY pays the same
            exc = strat_net - spy_net
            rows.append({
                "rule": r, "cost_bps_per_side": bps,
                "median_net_excess_%": float(np.median(exc)) * 100,
                "mean_net_excess_%": float(np.mean(exc)) * 100,
                "beat_rate_%": float((strat_net > spy_net).mean()) * 100,
            })
    tab = pd.DataFrame(rows)

    print("\n" + "=" * 86)
    print("A4  TRANSACTION-COST / SLIPPAGE SENSITIVITY  (cost charged on BOTH legs, SPY too)")
    print("=" * 86)
    if tab.empty:
        print("  (no closed positions)")
        return tab
    piv = tab.pivot(index="rule", columns="cost_bps_per_side", values="median_net_excess_%")
    with pd.option_context("display.float_format", lambda v: f"{v:0.2f}"):
        print("  Median net excess (%) by cost (bps/side):")
        print(piv.to_string())
    print("  If the edge flips negative by ~10-25 bps/side, it was never real after costs.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    ax = axes[0]
    for r in rules:
        s = tab[tab["rule"] == r]
        if not s.empty:
            ax.plot(s["cost_bps_per_side"], s["median_net_excess_%"],
                    marker="o", label=r)
    ax.axhline(0, color="grey", ls="--", lw=1)
    ax.set_xlabel("Cost per side (bps)")
    ax.set_ylabel("Median net excess vs SPY (%)")
    ax.set_title("A4 — Edge vs transaction cost")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    for r in rules:
        s = tab[tab["rule"] == r]
        if not s.empty:
            ax.plot(s["cost_bps_per_side"], s["beat_rate_%"], marker="s", label=r)
    ax.axhline(50, color="grey", ls="--", lw=1, label="coin-flip (50%)")
    ax.set_xlabel("Cost per side (bps)")
    ax.set_ylabel("Beat-SPY rate (%)")
    ax.set_title("A4 — Beat-rate vs transaction cost")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, outdir, "A4_costs.png")
    return tab


def a5_power(df: pd.DataFrame, outdir: str, headline: str) -> dict:
    """Minimum detectable effect at 80% power, and the n you'd actually need."""
    x = closed_excess(df, headline)
    n = len(x)
    sd = float(np.std(x, ddof=1)) if n > 1 else float("nan")
    mean = float(np.mean(x)) if n else float("nan")
    factor = Z_ALPHA_2 + Z_POWER_80
    mde = factor * sd / math.sqrt(n) if n > 0 and not math.isnan(sd) else float("nan")
    n_needed = (factor * sd / mean) ** 2 if mean not in (0, float("nan")) and abs(mean) > 0 else float("nan")

    print("\n" + "=" * 86)
    print(f"A5  STATISTICAL POWER  (headline rule '{headline}', n={n} closed)")
    print("=" * 86)
    print(f"  Observed mean excess: {mean*100:+.2f}%   SD of per-trade excess: {sd*100:.2f}%")
    print(f"  Minimum detectable mean excess at 80% power, α=0.05: ±{mde*100:.2f}% "
          f"-> observed effect is {'ABOVE' if abs(mean) >= mde else 'BELOW'} that floor.")
    if not math.isnan(n_needed) and n_needed > 0:
        print(f"  To call the observed {mean*100:+.2f}% effect significant at 80% power you'd "
              f"need ~{math.ceil(n_needed)} trades (you have {n}).")
    print("  Small n + fat-tailed returns = wide error bars. This is the core caveat.")

    ns = np.arange(10, 205, 5)
    mdes = factor * sd / np.sqrt(ns) * 100 if not math.isnan(sd) else np.full_like(ns, np.nan, float)
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.plot(ns, mdes, color=C_STRAT, lw=2, label="Min detectable mean excess (80% power)")
    if not math.isnan(mean):
        ax.axhline(abs(mean) * 100, color=C_AMBER, ls="--", lw=1.5,
                   label=f"|observed mean| = {abs(mean)*100:.2f}%")
    ax.axvline(n, color=C_NEG, ls=":", lw=1.5, label=f"your n = {n}")
    ax.set_xlabel("Number of trades")
    ax.set_ylabel("Detectable mean excess (%)")
    ax.set_title(f"A5 — How many trades to detect the effect (rule '{headline}')")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, outdir, "A5_power.png")
    return {"n": n, "sd": sd, "mean": mean, "mde": mde, "n_needed": n_needed}


def a6_concentration(df: pd.DataFrame, outdir: str, headline: str) -> dict:
    """How much of total P&L (and how many winning trades) rides on the top names."""
    sub = df[(df["rule"] == headline) & (~df["open"])].copy()
    if sub.empty:
        print("\nA6  CONCENTRATION: no closed positions — skipped.")
        return {}
    by_t = sub.groupby("ticker")["pnl"].sum().sort_values(ascending=False)
    pnl = sub["pnl"].to_numpy()
    total_abs = float(np.abs(pnl).sum())              # well-defined even if net ~ 0
    net_total = float(pnl.sum())
    by_abs = by_t.abs().sort_values(ascending=False)
    top2_share = float(by_abs.head(2).sum() / total_abs) if total_abs > 0 else float("nan")
    n_names = by_t.shape[0]
    n_winners = int((by_t > 0).sum())

    print("\n" + "=" * 86)
    print(f"A6  CONCENTRATION / FRAGILITY  (rule '{headline}', {len(sub)} closed positions, "
          f"{n_names} names)")
    print("=" * 86)
    print("  Total realized P&L by ticker (top 8):")
    with pd.option_context("display.float_format", lambda v: f"{v:0.2f}"):
        print(by_t.head(8).to_string())
    print(f"\n  Net realized P&L: ${net_total:+.2f}   ·   {n_winners}/{n_names} names profitable")
    print(f"  Top-2 names = {top2_share:.0%} of total ABSOLUTE P&L (i.e. of all the up/down swings).")
    print("  High share = a couple of names drive the result; the 'edge' is a bet on them, "
          "not a general signal.")

    fig, ax = plt.subplots(figsize=(11, 5.6))
    colors = [C_POS if v >= 0 else C_NEG for v in by_t.values]
    ax.bar(by_t.index, by_t.values, color=colors, edgecolor="white")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_ylabel("Total realized P&L ($)")
    ax.set_title(f"A6 — P&L contribution by ticker (rule '{headline}'); "
                 f"top-2 = {top2_share:.0%} of absolute P&L")
    ax.tick_params(axis="x", rotation=40)
    ax.grid(alpha=0.3, axis="y")
    _save(fig, outdir, "A6_concentration.png")
    return {"top2_share": top2_share, "n_names": n_names, "net_total": net_total}


# =========================================================================== #
# STAGE B — live comparators + permutation null (needs app + network)
# =========================================================================== #
def _entry_dict(rebalance_date, ticker, buy_date, buy_px, size=np.nan,
                prob=np.nan, med=np.nan) -> dict:
    """Entry row in the exact schema backtest_capitol.build_entries emits."""
    return {
        "rebalance_date": pd.Timestamp(rebalance_date),
        "ticker": ticker,
        "trade_size": float(size),
        "signal_prob_up": float(prob),
        "signal_median_return": float(med),
        "buy_date": pd.Timestamp(buy_date),
        "buy_px": float(buy_px),
    }


def _eligible_pool(prices: dict, D, min_hist: int = 5) -> list[str]:
    D = pd.Timestamp(D)
    out = []
    for t, s in prices.items():
        if s is None or s.empty:
            continue
        if (s.index <= D).sum() >= min_hist and (s.index >= D).any():
            out.append(t)
    return out


def _real_entries_from_log(df: pd.DataFrame) -> list[dict]:
    """Reconstruct the strategy's actual entries from the tradelog (rule-independent)."""
    one = df[df["rule"] == df["rule"].iloc[0]]
    seen, out = set(), []
    for _, r in one.iterrows():
        key = (r["rebalance_date"], r["ticker"], r["buy_date"])
        if key in seen:
            continue
        seen.add(key)
        out.append(_entry_dict(r["rebalance_date"], r["ticker"], r["buy_date"],
                               r["buy_price"], prob=r.get("signal_prob_up", np.nan),
                               med=r.get("signal_median_return", np.nan)))
    return out


def _picks_per_date(real_entries: list[dict]) -> dict:
    out: dict = {}
    for e in real_entries:
        out[pd.Timestamp(e["rebalance_date"])] = out.get(pd.Timestamp(e["rebalance_date"]), 0) + 1
    return out


def _fetch_closes(tickers, start, end, cache_path: str) -> dict:
    """Batched yfinance close fetch, mirroring app semantics (auto_adjust, tz-naive). Cached."""
    import pickle
    cache = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as fh:
            cache = pickle.load(fh)
    want = [t for t in tickers if t not in cache]
    if want:
        import yfinance as yf  # noqa: PLC0415
        print(f"  fetching {len(want)} new tickers via yfinance (batched)…")
        data = yf.download(want, start=start, end=end, auto_adjust=True,
                           progress=False, group_by="ticker", threads=True)
        for t in want:
            try:
                s = data[t]["Close"] if len(want) > 1 else data["Close"]
                s = s.dropna()
                s.index = s.index.tz_localize(None)
                cache[t] = s.sort_index() if not s.empty else None
            except Exception:  # noqa: BLE001
                cache[t] = None
        if cache_path:
            with open(cache_path, "wb") as fh:
                pickle.dump(cache, fh)
    return {t: cache.get(t) for t in tickers if cache.get(t) is not None}


def _run_entries(entries, prices, spy_close, rule, cfg, bt) -> pd.DataFrame:
    return bt.run_with_exit(entries, prices, spy_close, rule, cfg, mc_fn=None)


def _median_excess(log: pd.DataFrame) -> float:
    if log.empty or "excess_return" not in log.columns:
        return float("nan")
    x = log[~log["open"]]["excess_return"].dropna()
    return float(np.median(x)) if len(x) else float("nan")


def stage_b(df: pd.DataFrame, args, outdir: str) -> None:
    """Permutation null + comparator strategies. Requires the live app stack + network."""
    print("\n" + "#" * 86)
    print("# STAGE B — LIVE COMPARATORS + PERMUTATION NULL (needs app + yfinance + network)")
    print("#" * 86)
    try:
        import backtest_capitol as bt  # safe: doesn't import app at module level
        import app  # noqa: PLC0415  — heavy deps; only needed here
    except Exception as exc:  # noqa: BLE001
        print(f"  Could not import the live stack ({exc}).")
        print("  Run Stage B inside the streamlit environment. Skipping.")
        return

    rule = {"name": args.exit_rule, "kind": "time",
            "max_hold_months": {"time_1m": 1, "time_3m": 3, "time_6m": 6}.get(args.exit_rule, 3)}
    if args.exit_rule not in ("time_1m", "time_3m", "time_6m"):
        print(f"  (comparators use a fixed neutral exit; '{args.exit_rule}' is not a plain "
              f"time rule, falling back to time_3m)")
        rule = {"name": "time_3m", "kind": "time", "max_hold_months": 3}

    # Reconstruct the universe of dates + per-date pick counts from the real log.
    real_entries = _real_entries_from_log(df)
    k_by_date = _picks_per_date(real_entries)
    rebalance_dates = sorted(k_by_date)
    selected_tickers = sorted({e["ticker"] for e in real_entries})

    # Load trades (for the size-only / no-MC-filter comparator).
    trades = None
    if os.path.exists(args.trades):
        trades = pd.read_csv(args.trades, parse_dates=["published", "traded"])

    # Build the price pool: always include selected names + (optionally) the
    # window's size-top-N candidates + a random slice of the full universe.
    pool_tickers = set(selected_tickers)
    cfg = bt.Config(rebalance_dates=[pd.Timestamp(d) for d in rebalance_dates])
    if trades is not None and not trades.empty:
        for D in rebalance_dates:
            D = pd.Timestamp(D)
            win = trades[(trades["published"] >= D - pd.Timedelta(days=cfg.lookback_days))
                         & (trades["published"] <= D)]
            pool_tickers.update(win.sort_values("size_num", ascending=False)
                                .head(cfg.top_n)["ticker"].tolist())
    universe = sorted(app.DEFAULT_TICKERS)
    rng = np.random.default_rng(args.seed)
    extra = [t for t in universe if t not in pool_tickers]
    rng.shuffle(extra)
    need = max(0, args.pool_size - len(pool_tickers))
    pool_tickers.update(extra[:need])
    pool_tickers = sorted(pool_tickers)

    hist_start = (min(rebalance_dates) - pd.Timedelta(days=260)).date()
    hist_end = (max(df["exit_date"].dropna()).date()
                if df["exit_date"].notna().any() else date.today()) + timedelta(days=2)
    print(f"  Pool: {len(pool_tickers)} tickers; prices {hist_start} -> {hist_end}")
    prices = _fetch_closes(pool_tickers + ["SPY"], hist_start, hist_end, args.price_cache)
    spy_close = prices.pop("SPY", None)
    if spy_close is None:
        print("  WARNING: SPY unavailable — Stage B excess numbers will be missing.")

    # ---- real entries under the neutral exit (the reference) ---------------- #
    real_log = _run_entries(real_entries, prices, spy_close, rule, cfg, bt)
    real_med = _median_excess(real_log)
    print(f"\n  Reference: real strategy under '{rule['name']}' -> "
          f"median excess {real_med*100:+.2f}% on {int((~real_log['open']).sum())} closed")

    # ---- B1: permutation / null (random picks) ------------------------------ #
    print(f"\n  B1  PERMUTATION NULL: {args.permutations} random-selection draws "
          f"(exposure-matched to the real strategy)…")
    null_meds = []
    for i in range(args.permutations):
        rng_i = np.random.default_rng(args.seed + 1 + i)
        ent = []
        for D in rebalance_dates:
            k = k_by_date[pd.Timestamp(D)]
            pool = _eligible_pool(prices, D)
            if not pool or k == 0:
                continue
            picks = rng_i.choice(pool, size=min(k, len(pool)), replace=False)
            for t in picks:
                buy = bt.price_on_or_after(prices[t], pd.Timestamp(D))
                if buy and buy[1] > 0:
                    ent.append(_entry_dict(D, t, buy[0], buy[1]))
        lg = _run_entries(ent, prices, spy_close, rule, cfg, bt)
        null_meds.append(_median_excess(lg))
    null_meds = np.array([m for m in null_meds if not math.isnan(m)])
    if len(null_meds):
        pct = float((null_meds >= real_med).mean())
        print(f"     Real median excess {real_med*100:+.2f}% sits at the "
              f"{100*(1-pct):.0f}th percentile of the null; permutation p = {pct:.3f}.")
        print("     p < ~0.05 = the real selection beats random politician-agnostic picks.")
        fig, ax = plt.subplots(figsize=(10, 5.6))
        ax.hist(null_meds * 100, bins=40, color=C_SPY, edgecolor="white", alpha=0.85,
                label="random picks (null)")
        ax.axvline(real_med * 100, color=C_POS, lw=2.5,
                   label=f"real strategy {real_med*100:+.2f}%")
        ax.set_xlabel("Median per-trade excess vs SPY (%)")
        ax.set_ylabel("Random draws")
        ax.set_title(f"B1 — Permutation null ({len(null_meds)} draws), exit '{rule['name']}'  "
                     f"| p = {pct:.3f}")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        _save(fig, outdir, "B1_permutation_null.png")

    # ---- B2: comparator strategies ------------------------------------------ #
    print("\n  B2  COMPARATORS: real vs momentum-only vs size-only (no MC filter)…")
    comp = {"real": real_log}

    # momentum-only: top-k by trailing W-session return at D, from the pool
    W = args.momentum_window
    mom_ent = []
    for D in rebalance_dates:
        k = k_by_date[pd.Timestamp(D)]
        if k == 0:
            continue
        scored = []
        for t in _eligible_pool(prices, D, min_hist=W + 2):
            h = prices[t].loc[:pd.Timestamp(D)]
            if len(h) > W:
                scored.append((t, float(h.iloc[-1] / h.iloc[-W] - 1.0)))
        scored.sort(key=lambda z: z[1], reverse=True)
        for t, _ in scored[:k]:
            buy = bt.price_on_or_after(prices[t], pd.Timestamp(D))
            if buy and buy[1] > 0:
                mom_ent.append(_entry_dict(D, t, buy[0], buy[1]))
    comp["momentum"] = _run_entries(mom_ent, prices, spy_close, rule, cfg, bt)

    # size-only (no MC filter): biggest politician buys in window, best k by size
    if trades is not None and not trades.empty:
        size_ent = []
        for D in rebalance_dates:
            k = k_by_date[pd.Timestamp(D)]
            if k == 0:
                continue
            D = pd.Timestamp(D)
            win = trades[(trades["published"] >= D - pd.Timedelta(days=cfg.lookback_days))
                         & (trades["published"] <= D)]
            top = win.sort_values("size_num", ascending=False)
            seen, picks = set(), []
            for _, r in top.iterrows():
                if r["ticker"] in seen:
                    continue
                seen.add(r["ticker"])
                picks.append((r["ticker"], float(r["size_num"])))
                if len(picks) >= k:
                    break
            for t, sz in picks:
                if prices.get(t) is None:
                    continue
                buy = bt.price_on_or_after(prices[t], D)
                if buy and buy[1] > 0:
                    size_ent.append(_entry_dict(D, t, buy[0], buy[1], size=sz))
        comp["size_only"] = _run_entries(size_ent, prices, spy_close, rule, cfg, bt)

    rows = []
    for name, lg in comp.items():
        x = lg[~lg["open"]]["excess_return"].dropna().to_numpy() if not lg.empty else np.array([])
        bs = bootstrap_stat(x, np.median, args.bootstrap, args.seed)
        rows.append({"strategy": name, "n": bs["n"],
                     "median_excess_%": bs["point"] * 100,
                     "ci_lo_%": bs["lo"] * 100, "ci_hi_%": bs["hi"] * 100,
                     "beat_rate_%": float((lg[~lg["open"]]["excess_return"] > 0).mean()) * 100
                     if not lg.empty else float("nan")})
    ctab = pd.DataFrame(rows)
    with pd.option_context("display.float_format", lambda v: f"{v:0.2f}"):
        print(ctab.to_string(index=False))
    print("  If 'momentum' or 'size_only' match 'real', the politician+MC signal adds little.")

    fig, ax = plt.subplots(figsize=(10, 5.8))
    order = ["real", "momentum", "size_only"]
    ctab2 = ctab.set_index("strategy").reindex([o for o in order if o in ctab["strategy"].values])
    xpos = np.arange(len(ctab2))
    err = np.vstack([ctab2["median_excess_%"] - ctab2["ci_lo_%"],
                     ctab2["ci_hi_%"] - ctab2["median_excess_%"]])
    bars = ax.bar(xpos, ctab2["median_excess_%"], yerr=err, capsize=6,
                  color=[C_POS, C_AMBER, C_STRAT][:len(ctab2)], edgecolor="white")
    ax.axhline(0, color="grey", lw=0.9)
    ax.set_xticks(xpos)
    ax.set_xticklabels(ctab2.index)
    ax.set_ylabel("Median excess vs SPY (%)  ± bootstrap 95% CI")
    ax.set_title(f"B2 — Real signal vs momentum vs size-only (exit '{rule['name']}')")
    ax.grid(alpha=0.3, axis="y")
    _save(fig, outdir, "B2_comparators.png")

    # ---- B3: survivorship — price-history length of selected names ---------- #
    lengths = []
    for t in selected_tickers:
        s = prices.get(t)
        if s is not None:
            lengths.append((t, int((s.index < pd.Timestamp(min(rebalance_dates))).sum())))
    if lengths:
        ldf = pd.DataFrame(lengths, columns=["ticker", "sessions_before_first_run"]) \
            .sort_values("sessions_before_first_run")
        print("\n  B3  SURVIVORSHIP: sessions of history before the first run "
              "(short = recently listed / possible survivorship gap):")
        print(ldf.head(8).to_string(index=False))
        fig, ax = plt.subplots(figsize=(11, 5.4))
        colors = [C_NEG if v < 60 else C_STRAT for v in ldf["sessions_before_first_run"]]
        ax.bar(ldf["ticker"], ldf["sessions_before_first_run"], color=colors, edgecolor="white")
        ax.axhline(60, color=C_AMBER, ls="--", lw=1.2, label="~3 months")
        ax.set_ylabel("Sessions of history before first run")
        ax.set_title("B3 — History depth of selected names (red = thin history)")
        ax.legend()
        ax.tick_params(axis="x", rotation=40)
        ax.grid(alpha=0.3, axis="y")
        _save(fig, outdir, "B3_survivorship.png")


# =========================================================================== #
# Plumbing
# =========================================================================== #
def _save(fig, outdir: str, name: str) -> None:
    path = os.path.join(outdir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    chart -> {path}")


def _pick_headline_rule(df: pd.DataFrame) -> str:
    """Rule with the MOST closed trades — most data for the single-rule charts."""
    closed = df[~df["open"]].groupby("rule").size()
    return str(closed.sort_values(ascending=False).index[0])


def final_verdict(a1, a2, a3, a5, a6) -> None:
    print("\n" + "=" * 86)
    print("VERDICT  (read this part)")
    print("=" * 86)
    m = len(a1) if a1 is not None else 0
    n_bonf = int(a1["sig_bonf"].sum()) if a1 is not None else 0
    # split by DIRECTION — "significant" can mean significantly WORSE than SPY
    sig = a1[a1["sig_bonf"]] if a1 is not None else None
    n_sig_pos = int((sig["median_excess_%"] > 0).sum()) if sig is not None else 0
    n_sig_neg = int((sig["median_excess_%"] < 0).sum()) if sig is not None else 0
    if a2 is not None:
        n_ci_pos = int(((a2["excludes_zero"]) & (a2["lo"] > 0)).sum())   # CI fully ABOVE 0
        n_ci_neg = int(((a2["excludes_zero"]) & (a2["hi"] < 0)).sum())   # CI fully BELOW 0
    else:
        n_ci_pos = n_ci_neg = 0
    lines = []
    lines.append(f"• Bonferroni-significant rules: {n_bonf} of {m}  "
                 f"({n_sig_pos} BEAT SPY, {n_sig_neg} TRAIL SPY).")
    lines.append(f"• Bootstrap 95% CI fully ABOVE zero (real positive edge): {n_ci_pos}.  "
                 f"Fully BELOW zero (reliably trails SPY): {n_ci_neg}.")
    if a3:
        lines.append(f"• Walk-forward: chosen rule '{a3.get('chosen')}', "
                     f"IS→OOS rank corr rho = {a3.get('rho', float('nan')):+.2f} "
                     f"({'ranking stable' if a3.get('rho', 0) > 0.3 else 'weak/overfit'}).")
    if a5:
        lines.append(f"• Power: n={a5['n']}, observed mean {a5['mean']*100:+.2f}% vs "
                     f"detectable floor ±{a5['mde']*100:.2f}% "
                     f"({'above' if abs(a5['mean']) >= a5['mde'] else 'below'} floor).")
    if a6:
        lines.append(f"• Concentration: top-2 names = {a6.get('top2_share', float('nan')):.0%} "
                     f"of P&L ({'fragile' if a6.get('top2_share', 0) > 0.5 else 'spread'}).")
    print("\n".join(lines))

    # honest one-line bottom line, sign-aware
    if n_ci_pos > 0:
        bottom = (f"OVERALL: {n_ci_pos} rule(s) show a positive edge vs SPY that survives the "
                  "tests — worth a closer look (and a forward test before trusting).")
    elif n_ci_neg > 0 or n_sig_neg > 0:
        bottom = ("OVERALL: NO rule beats SPY; several SIGNIFICANTLY TRAIL it. 'Significant' "
                  "and 'CI clears zero' here point the WRONG way — the picks underperform the "
                  "index. The strategy has no edge.")
    else:
        bottom = ("OVERALL: nothing clears zero either way — inconclusive. Collect more trades "
                  "(longer history / more picks) before drawing conclusions.")
    print("\n" + bottom)
    print("=" * 86)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Statistical validation of the Capitol-Trades strategy.
            Stage A runs offline from a tradelog; Stage B (--comparators) runs live."""))
    p.add_argument("--log", default="exit_strategy_tradelog_MAX.csv",
                   help="exit-strategy tradelog to validate (default: the MAX run; "
                        "pass exit_strategy_tradelog.csv for the small S&P-only run)")
    p.add_argument("--trades", default="capitol_trades_cache_max.csv",
                   help="scraped trades cache (for the size-only comparator in Stage B)")
    p.add_argument("--outdir", default="validation_out_max", help="where charts/CSVs are written")
    p.add_argument("--bootstrap", type=int, default=10000, help="bootstrap resamples")
    p.add_argument("--split", type=float, default=0.5, help="walk-forward in-sample fraction")
    p.add_argument("--costs-bps", default="0,5,10,25",
                   help="comma list of per-side cost levels (bps) for the cost sweep")
    p.add_argument("--headline-rule", default=None,
                   help="rule used for single-rule charts (default: most closed trades)")
    p.add_argument("--seed", type=int, default=42)
    # Stage B
    p.add_argument("--comparators", action="store_true",
                   help="run Stage B: permutation null + momentum/size comparators (needs network)")
    p.add_argument("--permutations", type=int, default=500, help="random draws for the null test")
    p.add_argument("--pool-size", type=int, default=200, help="ticker pool size for the null")
    p.add_argument("--momentum-window", type=int, default=60, help="trailing sessions for momentum")
    p.add_argument("--exit-rule", default="time_3m",
                   help="neutral fixed exit rule for the comparators")
    p.add_argument("--price-cache", default="validation_prices.pkl",
                   help="pickle cache for fetched prices (Stage B)")
    return p


def main(argv) -> int:
    args = build_parser().parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    df = load_tradelog(args.log)

    headline = args.headline_rule or _pick_headline_rule(df)
    cost_levels = [float(x) for x in str(args.costs_bps).split(",") if x.strip() != ""]
    # cost/concentration charts: headline + the two best-by-median rules, de-duped
    by_med = (df[~df["open"]].groupby("rule")["excess_return"].median()
              .sort_values(ascending=False))
    chart_rules = list(dict.fromkeys([headline, *by_med.head(2).index.tolist()]))

    n_entries = int((df["rule"] == df["rule"].iloc[0]).sum())
    print("#" * 86)
    print(f"# STRATEGY VALIDATION   log={args.log}")
    print(f"# {n_entries} entries  ·  {df['rule'].nunique()} exit rules  ·  "
          f"headline rule = '{headline}'  ·  scipy={'yes' if _HAVE_SCIPY else 'no (fallbacks)'}")
    print("#" * 86)

    a1 = a1_significance(df, args.outdir)
    a2 = a2_bootstrap(df, args.outdir, args.bootstrap, headline, args.seed)
    a3 = a3_walk_forward(df, args.outdir, args.split, args.bootstrap, args.seed)
    a4_costs(df, args.outdir, cost_levels, chart_rules)
    a5 = a5_power(df, args.outdir, headline)
    a6 = a6_concentration(df, args.outdir, headline)

    if args.comparators:
        stage_b(df, args, args.outdir)

    final_verdict(a1, a2, a3, a5, a6)

    # persist the headline tables
    a1.to_csv(os.path.join(args.outdir, "A1_significance.csv"), index=False)
    a2.to_csv(os.path.join(args.outdir, "A2_bootstrap.csv"), index=False)
    print(f"\nAll outputs in: {args.outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))