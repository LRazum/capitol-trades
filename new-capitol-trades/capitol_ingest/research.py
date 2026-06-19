"""
research.py — hypothesis testing on the gold event table.

* ``cumulative_abnormal_returns`` — event study relative to a benchmark (SPY): mean CAR
  per day after entry (t=0), with standard errors and t-stats. Answers "is there
  post-filing drift to ride?".
* ``forward_excess_returns`` — per-event excess return (stock minus benchmark) over a
  fixed horizon, the metric the conditional tests operate on.
* ``compare_groups`` — split events into two conditional portfolios (committee-relevant
  vs not, insider-corroborated vs not, ...) and compare with Welch t, Mann-Whitney, and a
  bootstrap CI on the mean difference.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import scipy.stats as ss

from .prices import PriceProvider


def _batch_prices(
    event_table: pd.DataFrame, prices: PriceProvider, benchmark_ticker: str, horizon: int
) -> dict[str, pd.Series]:
    """Fetch each ticker (and the benchmark) once over a window covering all events."""
    t0 = pd.to_datetime(event_table["t0"])
    lo = t0.min() - pd.Timedelta(days=15)
    hi = t0.max() + pd.Timedelta(days=int(horizon * 2) + 15)
    out: dict[str, pd.Series] = {}
    for tkr in event_table["ticker"].dropna().unique():
        out[tkr] = prices.get(tkr, lo.date(), hi.date())
    out[benchmark_ticker] = prices.get(benchmark_ticker, lo.date(), hi.date())
    return out


def cumulative_abnormal_returns(
    event_table: pd.DataFrame,
    prices: PriceProvider,
    benchmark_ticker: str = "SPY",
    horizon: int = 10,
) -> pd.DataFrame:
    """Mean CAR (stock minus benchmark, daily, summed) over days 1..horizon after entry."""
    pmap = _batch_prices(event_table, prices, benchmark_ticker, horizon)
    bench = pmap[benchmark_ticker]
    bench_ret = bench.pct_change(fill_method=None)

    ar = np.full((len(event_table), horizon), np.nan)
    for i, (_, row) in enumerate(event_table.iterrows()):
        s = pmap.get(row["ticker"])
        if s is None or s.empty:
            continue
        t0 = pd.Timestamp(row["t0"])
        if t0 not in s.index:
            continue
        pos = s.index.get_loc(t0)
        win = s.index[pos: pos + horizon + 1]
        s_ret = s.pct_change(fill_method=None).reindex(win).to_numpy()[1:]
        b_ret = bench_ret.reindex(win).to_numpy()[1:]
        d = min(len(s_ret), horizon)
        ar[i, :d] = (s_ret - b_ret)[:d]

    car = np.cumsum(ar, axis=1)  # NaN propagates over missing tails
    n = np.sum(~np.isnan(car), axis=0)
    mean = np.nanmean(car, axis=0)
    se = np.nanstd(car, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        t = mean / se
    return pd.DataFrame(
        {"day": np.arange(1, horizon + 1), "mean_car": mean, "se": se, "t_stat": t, "n": n}
    )


def forward_excess_returns(
    event_table: pd.DataFrame,
    prices: PriceProvider,
    benchmark_ticker: str = "SPY",
    horizon: int = 21,
) -> pd.Series:
    """Per-event excess return: stock t0->t0+horizon minus benchmark over the same dates."""
    pmap = _batch_prices(event_table, prices, benchmark_ticker, horizon)
    bench = pmap[benchmark_ticker]
    vals = np.full(len(event_table), np.nan)
    for i, (_, row) in enumerate(event_table.iterrows()):
        s = pmap.get(row["ticker"])
        if s is None or s.empty:
            continue
        t0 = pd.Timestamp(row["t0"])
        if t0 not in s.index:
            continue
        pos = s.index.get_loc(t0)
        end = s.index[min(pos + horizon, len(s) - 1)]
        b0, b1 = bench.asof(t0), bench.asof(end)
        if not (b0 and b1 and b0 > 0):
            continue
        vals[i] = (float(s.loc[end]) / float(s.loc[t0]) - 1) - (float(b1) / float(b0) - 1)
    return pd.Series(vals, index=event_table.index, name=f"fwd_excess_{horizon}")


def compare_groups(
    values: pd.Series,
    mask: pd.Series,
    *,
    label_a: str = "group_A",
    label_b: str = "group_B",
    n_boot: int = 5000,
    seed: int = 0,
) -> dict:
    """Compare a per-event metric across two conditional portfolios (mask True vs False)."""
    a = pd.Series(values)[mask.astype(bool)].dropna().to_numpy()
    b = pd.Series(values)[~mask.astype(bool)].dropna().to_numpy()
    if len(a) < 2 or len(b) < 2:
        return {"error": "insufficient samples", "n_a": len(a), "n_b": len(b)}

    diff = float(a.mean() - b.mean())
    t_stat, p_t = ss.ttest_ind(a, b, equal_var=False)         # Welch
    u_stat, p_u = ss.mannwhitneyu(a, b, alternative="two-sided")  # non-parametric

    rng = np.random.default_rng(seed)
    boot = np.array([
        rng.choice(a, len(a), replace=True).mean() - rng.choice(b, len(b), replace=True).mean()
        for _ in range(n_boot)
    ])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        label_a: {"n": len(a), "mean": float(a.mean()), "median": float(np.median(a))},
        label_b: {"n": len(b), "mean": float(b.mean()), "median": float(np.median(b))},
        "mean_diff": diff,
        "welch_t": float(t_stat),
        "p_ttest": float(p_t),
        "p_mannwhitney": float(p_u),
        "boot_ci95": (float(lo), float(hi)),
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
    }
