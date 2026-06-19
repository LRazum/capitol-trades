# Copyright (c) 2026 Lovro Razum. All rights reserved.
# PROPRIETARY AND CONFIDENTIAL. Unauthorized use, copying, modification, or
# distribution of this file, in whole or in part, via any medium, is strictly
# prohibited without the prior written permission of Lovro Razum.
# See the accompanying LICENSE file for full terms.
"""
Backtest of the Capitol-Trades strategy
=======================================
This does NOT test app.py itself. It tests a *trading strategy built on the
data app.py surfaces*: every run we look at the N largest politician "buy"
trades in a recent window, keep the ones the Monte-Carlo forecast says will
rise, buy a fixed dollar amount of the best K, and sell them X months later.

Strategy (all knobs in CONFIG below):
  * Run cadence:        1st and 15th of each month, last 12 months  (~24 runs)
  * Per run:            take TOP_N trades by estimated size (same ranking app.py uses)
  * Filter:             keep only tickers whose MC median forecast > today's price
  * Pick:               the best N_PICK of those (by SELECTION_METRIC)
  * Position:           buy DOLLARS_PER_TRADE of each at that day's close
  * Exit:               sell at the close HOLDING_MONTHS later

Look-ahead bias is the whole ballgame here. At a simulated run date D the
forecast is recomputed from prices truncated at D (`close.loc[:D]`), never from
the full series. Trades are filtered by `published <= D`, so nothing that was
disclosed after D can leak in. Get this wrong and the backtest is fiction.

Data layer (scrape + prices) is reused verbatim from app.py so the simulation
eats the exact same data the live app would. The engine is pure and is
exercised by the self-test (`python backtest_capitol.py --selftest`) with
synthetic data, so its correctness does not depend on the network.

Usage:
  python backtest_capitol.py            # scrape (or use cache) + run + report
  python backtest_capitol.py --refresh  # force re-scrape, ignore the trade cache
  python backtest_capitol.py --selftest # run the engine self-test only (no network)

Requires app.py next to this file, plus its deps (streamlit env already has them).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# CONFIG — everything tunable lives here
# --------------------------------------------------------------------------- #
REBALANCE_DAYS = (1, 15)        # twice a month: the 1st and the 15th
MONTHS_BACK = 12                # how far back the simulation runs

# How wide a window of "recent buys" each run looks at. app.py's UI defaults to
# 7, but the strategy runs every ~15 days, so 15 makes consecutive runs tile the
# year with no gaps and no double-counting. Change to 7 to mimic the UI exactly.
LOOKBACK_DAYS = 15

TOP_N = 10                      # consider the 10 largest trades in the window
N_PICK = 2                      # buy the best 2 of the survivors
DOLLARS_PER_TRADE = 100.0       # $100 per position (fractional shares)

HORIZON = 7                     # MC forecast horizon in trading days (app default).
                                # NOTE: this is the *selection* signal horizon; it is
                                # deliberately shorter than the holding period.

# Holding periods to compare. He said "x months, decide later" — so sweep several.
HOLDING_PERIODS_MONTHS = [1, 2, 3, 6]

# --- EXIT STRATEGIES ------------------------------------------------------- #
# The entry stays the same (top-N by size -> MC growth filter -> best N_PICK).
# Only the EXIT differs. Every rule has a max-hold backstop so nothing is held
# forever. The sweep below is intentionally small/round-numbered — optimising
# hard on ~30 trades just overfits. Judge rules by per-trade MEDIAN excess and
# beat-rate, not by whichever single number is biggest.
DEFAULT_MAX_HOLD_MONTHS = 6
EXIT_RULES = [
    # baselines: plain fixed holds
    {"name": "time_1m",  "kind": "time", "max_hold_months": 1},
    {"name": "time_3m",  "kind": "time", "max_hold_months": 3},
    {"name": "time_6m",  "kind": "time", "max_hold_months": 6},
    # ride winners, cut the giveback: trailing stop off the running peak
    {"name": "trail_10", "kind": "trailing", "trail": 0.10, "max_hold_months": 6},
    {"name": "trail_15", "kind": "trailing", "trail": 0.15, "max_hold_months": 6},
    {"name": "trail_20", "kind": "trailing", "trail": 0.20, "max_hold_months": 6},
    # bracket: hard stop-loss + take-profit
    {"name": "stop8_take20",  "kind": "stop_take", "stop": 0.08, "take": 0.20, "max_hold_months": 6},
    {"name": "stop10_take25", "kind": "stop_take", "stop": 0.10, "take": 0.25, "max_hold_months": 6},
    # volatility-adaptive trailing stop (k * 20-day return vol)
    {"name": "atr_k3", "kind": "atr_trailing", "k": 3.0, "max_hold_months": 6},
    {"name": "atr_k4", "kind": "atr_trailing", "k": 4.0, "max_hold_months": 6},
    # exit when the MC signal that got us IN flips bearish
    {"name": "mc_exit", "kind": "mc_exit", "prob_thr": 0.5, "check_every_days": 10, "max_hold_months": 6},
]

# How to pick the best N_PICK among the growth survivors:
#   "median_return" -> the 2 the MC predicts will grow the MOST   (default)
#   "prob_up"       -> the 2 with the highest P(up) over the horizon
#   "size"          -> the 2 largest trades (i.e. "best" = biggest politician buy)
SELECTION_METRIC = "median_return"

# Calendar days of price history to pull *before* the earliest run date, so the
# 60-session MC window is warm even for the oldest run. 60 sessions ~ 84 days;
# 220 is a safe cushion across weekends/holidays.
HISTORY_WARMUP_DAYS = 220

TRADES_CACHE = "capitol_trades_cache.csv"   # scraped trades cached here
OUTPUT_TRADELOG = "backtest_tradelog.csv"   # per-position log (all holding periods)
OUTPUT_CHARTS = "backtest_charts.png"       # dashboard image
OUTPUT_EXIT_CHARTS = "exit_strategy_charts.png"
OUTPUT_EXIT_LOG = "exit_strategy_tradelog.csv"


# --------------------------------------------------------------------------- #
# Config object passed to the engine
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    rebalance_dates: list  # list[pd.Timestamp]
    lookback_days: int = LOOKBACK_DAYS
    top_n: int = TOP_N
    n_pick: int = N_PICK
    dollars: float = DOLLARS_PER_TRADE
    horizon: int = HORIZON
    selection_metric: str = SELECTION_METRIC


# --------------------------------------------------------------------------- #
# Pure date / price helpers (no network, no app import) — unit-testable
# --------------------------------------------------------------------------- #
def generate_rebalance_dates(
    end: date, months_back: int = MONTHS_BACK, days=REBALANCE_DAYS
) -> list[pd.Timestamp]:
    """The 1st and 15th of each month within [end - months_back, end]."""
    end_ts = pd.Timestamp(end)
    start_ts = end_ts - pd.DateOffset(months=months_back)
    out: list[pd.Timestamp] = []
    cur = pd.Timestamp(start_ts.year, start_ts.month, 1)
    while cur <= end_ts:
        for d in days:
            cand = pd.Timestamp(cur.year, cur.month, d)
            if start_ts <= cand <= end_ts:
                out.append(cand)
        cur = cur + pd.DateOffset(months=1)
    return out


def price_on_or_after(close: pd.Series, target: pd.Timestamp):
    """First (date, price) on or after `target`. None if target is past the data.

    `close` must be sorted ascending by date. Models running the script on a
    weekend and transacting on the next trading day.
    """
    idx = close.index.searchsorted(pd.Timestamp(target), side="left")
    if idx >= len(close):
        return None
    return close.index[idx], float(close.iloc[idx])


# --------------------------------------------------------------------------- #
# Engine (pure: takes data + a forecast fn, returns a trade log)
# --------------------------------------------------------------------------- #
def select_positions_for_date(D, trades_df, price_histories, cfg: Config, mc_fn):
    """Replicate what the app would have shown on day D, then apply the strategy.

    Returns up to cfg.n_pick dicts: {ticker, size, prob_up, median_return}.
    Every forecast is computed from prices truncated at D (no look-ahead).
    """
    D = pd.Timestamp(D)
    window_start = D - pd.Timedelta(days=cfg.lookback_days)
    win = trades_df[(trades_df["published"] >= window_start)
                    & (trades_df["published"] <= D)]
    if win.empty:
        return []

    # Same ranking app.py uses: estimated trade size, descending.
    top = win.sort_values("size_num", ascending=False).head(cfg.top_n)

    # De-dupe tickers, keep best (first) appearance order.
    seen, candidates = set(), []
    for _, r in top.iterrows():
        t = r["ticker"]
        if t in seen:
            continue
        seen.add(t)
        candidates.append((t, float(r["size_num"])))

    scored = []
    for ticker, size in candidates:
        close = price_histories.get(ticker)
        if close is None or close.empty:
            continue
        hist = close.loc[:D]                 # <-- point-in-time slice
        if len(hist) < 5:
            continue
        fc = mc_fn(hist, horizon=cfg.horizon)
        if fc is None:                       # not enough data / zero vol
            continue
        s0 = fc["S0"]
        median_terminal = float(fc["p50"][-1])
        median_return = median_terminal / s0 - 1.0
        if median_return <= 0:               # MC must predict GROWTH
            continue
        scored.append({
            "ticker": ticker,
            "size": size,
            "prob_up": float(fc["prob_up"]),
            "median_return": median_return,
        })

    if not scored:
        return []

    key = {
        "median_return": "median_return",
        "prob_up": "prob_up",
        "size": "size",
    }[cfg.selection_metric]
    scored.sort(key=lambda d: d[key], reverse=True)
    return scored[:cfg.n_pick]


def build_entries(trades_df, price_histories, cfg: Config, mc_fn) -> list[dict]:
    """All positions the strategy OPENS (selection is independent of the exit rule).

    Computed once and reused across every exit rule, so a rule sweep is cheap.
    """
    entries = []
    for D in cfg.rebalance_dates:
        for pick in select_positions_for_date(D, trades_df, price_histories, cfg, mc_fn):
            close = price_histories[pick["ticker"]]
            buy = price_on_or_after(close, D)
            if buy is None or buy[1] <= 0:
                continue
            buy_date, buy_px = buy
            entries.append({
                "rebalance_date": pd.Timestamp(D),
                "ticker": pick["ticker"],
                "trade_size": pick["size"],
                "signal_prob_up": pick["prob_up"],
                "signal_median_return": pick["median_return"],
                "buy_date": buy_date,
                "buy_px": buy_px,
            })
    return entries


def resolve_exit(close: pd.Series, buy_date, buy_px, rule: dict, cfg: Config,
                 mc_fn=None, ticker=None, fc_cache=None):
    """Walk daily closes from entry to the max-hold backstop and apply `rule`.

    Returns (exit_date, exit_px, reason, still_open). Triggers are close-to-close
    (daily data, no intrabar fills). MC re-evaluation uses prices truncated at the
    check date, so it stays look-ahead-safe like the entry signal.

    Rule kinds:
      time          : hold until max_hold_months (the old behaviour, baseline)
      trailing      : exit when close falls `trail` below the running peak
      stop_take     : exit at first close <= -`stop` or >= +`take` vs entry
      atr_trailing  : trailing stop sized at `k` * 20d return-vol (vol-adaptive)
      mc_exit       : re-run MC every `check_every_days`; exit when prob_up < `prob_thr`
    All kinds honour `max_hold_months` as a backstop.
    """
    kind = rule["kind"]
    max_hold = rule.get("max_hold_months", DEFAULT_MAX_HOLD_MONTHS)
    end_target = pd.Timestamp(buy_date) + pd.DateOffset(months=max_hold)
    window = close.loc[pd.Timestamp(buy_date):end_target]
    last_data = close.index[-1]

    vol = close.pct_change().rolling(20).std() if kind == "atr_trailing" else None
    peak = buy_px
    dates = window.index
    for i in range(len(dates)):
        dt = dates[i]
        px = float(window.iloc[i])
        if i == 0:
            peak = max(peak, px)
            continue
        ret = px / buy_px - 1.0
        if kind == "trailing":
            peak = max(peak, px)
            if px <= peak * (1.0 - rule["trail"]):
                return dt, px, "trail", False
        elif kind == "stop_take":
            if ret <= -rule["stop"]:
                return dt, px, "stop", False
            if ret >= rule["take"]:
                return dt, px, "take", False
        elif kind == "atr_trailing":
            peak = max(peak, px)
            sig = vol.get(dt, np.nan)
            if not np.isnan(sig):
                tp = min(max(rule["k"] * float(sig), 0.03), 0.50)
                if px <= peak * (1.0 - tp):
                    return dt, px, "atr_trail", False
        elif kind == "mc_exit":
            if mc_fn is not None and i % rule.get("check_every_days", 10) == 0:
                hist = close.loc[:dt]
                if len(hist) >= 5:
                    key = (ticker, dt)
                    fc = fc_cache.get(key) if fc_cache is not None else None
                    if fc is None:
                        fc = mc_fn(hist, horizon=rule.get("horizon", HORIZON))
                        if fc_cache is not None:
                            fc_cache[key] = fc
                    if fc is not None and fc["prob_up"] < rule.get("prob_thr", 0.5):
                        return dt, px, "mc_flip", False
        # "time": no trigger, falls through to the backstop

    # No trigger fired within the window.
    if end_target <= last_data:
        # Matured: sell on the first trading day on/after the X-month mark.
        res = price_on_or_after(close, end_target)
        if res is not None:
            return res[0], res[1], "time", False
    # Not matured (or no price at/after the mark): still open, mark to last close.
    return last_data, float(close.iloc[-1]), "open", True


def run_with_exit(entries: list[dict], price_histories, benchmark_close, rule: dict,
                  cfg: Config, mc_fn=None, fc_cache=None) -> pd.DataFrame:
    """Apply one exit `rule` to a fixed set of entries; benchmark SPY over the SAME realized window."""
    rows = []
    for e in entries:
        close = price_histories[e["ticker"]]
        exit_date, exit_px, reason, still_open = resolve_exit(
            close, e["buy_date"], e["buy_px"], rule, cfg, mc_fn, e["ticker"], fc_cache)
        shares = cfg.dollars / e["buy_px"]
        value = shares * exit_px
        strat_ret = exit_px / e["buy_px"] - 1.0
        row = {
            "rule": rule["name"],
            "rebalance_date": e["rebalance_date"].date(),
            "ticker": e["ticker"],
            "signal_prob_up": e["signal_prob_up"],
            "signal_median_return": e["signal_median_return"],
            "buy_date": e["buy_date"].date(),
            "buy_price": round(e["buy_px"], 4),
            "exit_date": exit_date.date(),
            "exit_price": round(exit_px, 4),
            "exit_reason": reason,
            "hold_days": int((pd.Timestamp(exit_date) - pd.Timestamp(e["buy_date"])).days),
            "invested": cfg.dollars,
            "value": round(value, 4),
            "pnl": round(value - cfg.dollars, 4),
            "return_pct": round(strat_ret, 6),
            "open": still_open,
        }
        # SPY: same $100, entry on buy_date, exit on the strategy's ACTUAL exit date.
        if benchmark_close is not None and not benchmark_close.empty:
            b_buy = price_on_or_after(benchmark_close, e["buy_date"])
            if b_buy is not None and b_buy[1] > 0:
                b_sell = price_on_or_after(benchmark_close, exit_date)
                b_sell_px = float(benchmark_close.iloc[-1]) if b_sell is None else b_sell[1]
                spy_ret = b_sell_px / b_buy[1] - 1.0
                row.update({
                    "spy_value": round((cfg.dollars / b_buy[1]) * b_sell_px, 4),
                    "spy_pnl": round((cfg.dollars / b_buy[1]) * b_sell_px - cfg.dollars, 4),
                    "spy_return_pct": round(spy_ret, 6),
                    "excess_return": round(strat_ret - spy_ret, 6),
                    "beat_spy": bool(strat_ret > spy_ret),
                })
        rows.append(row)
    return pd.DataFrame(rows)


def run_backtest(trades_df, price_histories, cfg: Config, holding_months: int, mc_fn,
                 benchmark_close: pd.Series | None = None) -> pd.DataFrame:
    """Backward-compatible wrapper: fixed time exit at `holding_months`."""
    entries = build_entries(trades_df, price_histories, cfg, mc_fn)
    rule = {"name": f"time_{holding_months}m", "kind": "time", "max_hold_months": holding_months}
    log = run_with_exit(entries, price_histories, benchmark_close, rule, cfg, mc_fn)
    if not log.empty:
        log.insert(0, "holding_months", holding_months)
    return log


def summarize(log: pd.DataFrame, holding_months: int) -> dict:
    """Aggregate a trade log into headline stats (realized vs marked-to-market)."""
    if log.empty:
        return {"holding_months": holding_months, "n_positions": 0}

    closed = log[~log["open"]]
    invested_total = float(log["invested"].sum())
    value_total = float(log["value"].sum())          # incl. open positions @ last close

    out = {
        "holding_months": holding_months,
        "n_positions": int(len(log)),
        "n_closed": int(len(closed)),
        "n_open": int(log["open"].sum()),
        "invested_total": invested_total,
        # total return marking open positions to the latest available close
        "total_value_mtm": value_total,
        "total_pnl_mtm": value_total - invested_total,
        "total_return_pct_mtm": (value_total / invested_total - 1.0) if invested_total else float("nan"),
    }
    if len(closed):
        ci = float(closed["invested"].sum())
        cv = float(closed["value"].sum())
        out.update({
            "realized_invested": ci,
            "realized_value": cv,
            "realized_pnl": cv - ci,
            "realized_return_pct": cv / ci - 1.0,
            "win_rate": float((closed["pnl"] > 0).mean()),
            "avg_trade_return": float(closed["return_pct"].mean()),
            "median_trade_return": float(closed["return_pct"].median()),
            "best_trade": float(closed["return_pct"].max()),
            "worst_trade": float(closed["return_pct"].min()),
        })
        # SPY-matched benchmark on the same closed positions
        if "spy_value" in closed.columns and closed["spy_value"].notna().any():
            cb = closed.dropna(subset=["spy_value"])
            sci = float(cb["invested"].sum())
            scv = float(cb["spy_value"].sum())
            spy_ret = scv / sci - 1.0 if sci else float("nan")
            out.update({
                "spy_realized_value": scv,
                "spy_realized_return_pct": spy_ret,
                "excess_realized_return_pct": (cv / ci - 1.0) - spy_ret,
                "beat_spy_rate": float((cb["return_pct"] > cb["spy_return_pct"]).mean()),
            })
    return out


# --------------------------------------------------------------------------- #
# Data layer (live; reuses app.py verbatim). Imported lazily so the engine can
# be tested without app.py's heavy deps installed.
# --------------------------------------------------------------------------- #
def scrape_buy_trades_since(tickers, start_date: date, lookback_days: int,
                            max_pages: int = 120, page_size: int = 96,
                            empty_streak_stop: int = 4) -> pd.DataFrame:
    """Page back through capitoltrades.com buys until we've covered the window.

    Reuses app._fetch_page / app._parse_rows so parsing is identical to the app.
    The site is reverse-chronological, so once we pass the cutoff every page
    yields zero qualifying rows and the empty-streak stops us.
    """
    from app import _fetch_page, _parse_rows, norm  # noqa: PLC0415

    revolut_norm = {norm(t) for t in tickers}
    today = date.today()
    cutoff = start_date - timedelta(days=lookback_days)
    print(f"Scraping buys published since {cutoff} (universe={len(tickers)} tickers)…")

    rows, empty = [], 0
    for page in range(1, max_pages + 1):
        try:
            html = _fetch_page(page, page_size)
        except Exception as exc:  # noqa: BLE001
            print(f"  page {page}: fetch failed ({exc}); stopping.")
            break
        page_rows = _parse_rows(html, revolut_norm, today, cutoff)
        rows.extend(page_rows)
        if page % 5 == 0 or page_rows:
            print(f"  page {page:>3}: +{len(page_rows):>3} qualifying (total {len(rows)})")
        if not page_rows:
            empty += 1
            if empty >= empty_streak_stop:
                print(f"  {empty} empty pages in a row -> past the window, stopping.")
                break
        else:
            empty = 0

    df = pd.DataFrame(rows)
    if not df.empty:
        df["traded"] = pd.to_datetime(df["traded"])
        df["published"] = pd.to_datetime(df["published"])
        df = (df.sort_values(["published", "size_num"], ascending=[False, False])
                .reset_index(drop=True))
    print(f"Scraped {len(df)} qualifying buy trades.\n")
    return df


def fetch_price_histories(tickers, start: date, end: date) -> dict:
    """Per-ticker OHLCV via yfinance, mirroring app.fetch_history (auto_adjust, tz-naive)."""
    import yfinance as yf       # noqa: PLC0415
    from app import to_yahoo    # noqa: PLC0415

    out = {}
    print(f"Fetching prices for {len(tickers)} tickers {start} -> {end}…")
    for t in sorted(tickers):
        try:
            df = yf.Ticker(to_yahoo(t)).history(start=start, end=end, auto_adjust=True)
            if df is None or df.empty:
                print(f"  {t}: no data")
                continue
            close = df["Close"].copy()
            close.index = close.index.tz_localize(None)
            out[t] = close.sort_index()
        except Exception as exc:  # noqa: BLE001
            print(f"  {t}: fetch failed ({exc})")
    print(f"Got prices for {len(out)}/{len(tickers)} tickers.\n")
    return out


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_comparison(summaries: list[dict]) -> None:
    print("=" * 92)
    print("STRATEGY vs S&P 500 (SPY), BY HOLDING PERIOD")
    print(f"  pick {N_PICK} best of top {TOP_N} by '{SELECTION_METRIC}', "
          f"${DOLLARS_PER_TRADE:.0f}/position, MC horizon {HORIZON}d, lookback {LOOKBACK_DAYS}d")
    print("  Strat.% and SPY.% are realized returns on CLOSED positions, same $/dates.")
    print("=" * 92)
    hdr = (f"{'Hold':>5} {'Closed':>7} {'Open':>5} {'Strat.%':>9} {'SPY.%':>9} "
           f"{'Excess':>9} {'Beat%':>7} {'Win%':>6}")
    print(hdr)
    print("-" * len(hdr))
    for s in summaries:
        if s.get("n_positions", 0) == 0:
            print(f"{str(s['holding_months'])+'m':>5} {'0':>7}  (no positions)")
            continue
        strat = s.get("realized_return_pct", float("nan")) * 100
        spy = s.get("spy_realized_return_pct", float("nan")) * 100
        exc = s.get("excess_realized_return_pct", float("nan")) * 100
        beat = s.get("beat_spy_rate", float("nan")) * 100
        win = s.get("win_rate", float("nan")) * 100
        flag = "" if s.get("n_closed", 0) else "  (none closed yet)"
        print(f"{str(s['holding_months'])+'m':>5} {s['n_closed']:>7} {s['n_open']:>5} "
              f"{strat:>8.2f}% {spy:>8.2f}% {exc:>+8.2f}% {beat:>6.1f}% {win:>5.1f}%{flag}")
    print("-" * len(hdr))
    print("Excess = Strat.% − SPY.% (positive = beat the index).  "
          "Beat% = share of trades that beat SPY over the same window.")
    print("=" * 92)


def generate_charts(all_logs: list[pd.DataFrame], summaries: list[dict],
                    path: str = OUTPUT_CHARTS) -> None:
    """2x2 dashboard: equity curve, holding-period comparison, return spread, signal vs outcome."""
    try:
        import matplotlib       # noqa: PLC0415
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        print("\nmatplotlib not installed -> skipping charts. `pip install matplotlib` to enable.")
        return

    logs_by_hm = {int(lg["holding_months"].iloc[0]): lg for lg in all_logs if not lg.empty}
    if not logs_by_hm:
        print("No positions -> no charts.")
        return

    hms = sorted(logs_by_hm)
    cmap = plt.get_cmap("viridis")
    colors = {hm: cmap(i / max(1, len(hms) - 1)) for i, hm in enumerate(hms)}
    sm = {s["holding_months"]: s for s in summaries}

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Capitol-Trades strategy backtest", fontsize=14, fontweight="bold")

    has_spy = any("spy_return_pct" in lg.columns for lg in logs_by_hm.values())

    # Holding period with the most CLOSED trades drives the headline panels.
    best_hm = max(hms, key=lambda h: int((~logs_by_hm[h]["open"]).sum()))
    closed = logs_by_hm[best_hm][~logs_by_hm[best_hm]["open"]].copy()

    # (0,0) Cumulative realized P&L: strategy vs SPY (the "did we beat it?" panel)
    ax = axes[0, 0]
    if has_spy and not closed.empty:
        c = closed.copy()
        c["sell_date"] = pd.to_datetime(c["sell_date"])
        c = c.sort_values("sell_date")
        ax.plot(c["sell_date"], c["pnl"].cumsum(), marker="o", ms=3,
                color="#16A34A", label="Strategy")
        if "spy_pnl" in c.columns:
            ax.plot(c["sell_date"], c["spy_pnl"].cumsum(), marker="o", ms=3,
                    color="#6B7280", ls="--", label="SPY (same $/dates)")
        ax.set_title(f"Cumulative realized P&L — strategy vs SPY ({best_hm}m hold)")
        ax.legend()
    else:
        for hm in hms:
            cc = logs_by_hm[hm][~logs_by_hm[hm]["open"]].copy()
            if cc.empty:
                continue
            cc["sell_date"] = pd.to_datetime(cc["sell_date"])
            cc = cc.sort_values("sell_date")
            ax.plot(cc["sell_date"], cc["pnl"].cumsum(), marker="o", ms=3,
                    color=colors[hm], label=f"{hm}m")
        ax.set_title("Cumulative realized P&L (by close date)")
        ax.legend(title="hold")
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_ylabel("Cumulative P&L ($)")
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", rotation=30)

    # (0,1) Return by holding period: strategy vs SPY (or vs MtM if no benchmark)
    ax = axes[0, 1]
    xs = np.arange(len(hms))
    w = 0.38
    strat = [sm[hm].get("realized_return_pct", np.nan) * 100 for hm in hms]
    if has_spy:
        other = [sm[hm].get("spy_realized_return_pct", np.nan) * 100 for hm in hms]
        other_label, other_color = "SPY (matched) %", "#9CA3AF"
    else:
        other = [sm[hm].get("total_return_pct_mtm", np.nan) * 100 for hm in hms]
        other_label, other_color = "MtM %", "#93C5FD"
    ax.bar(xs - w / 2, strat, w, label="Strategy %", color="#2563EB")
    ax.bar(xs + w / 2, other, w, label=other_label, color=other_color)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{hm}m" for hm in hms])
    ax.set_title("Realized return by holding period")
    ax.set_ylabel("Return (%)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    # (1,0) Per-trade return distribution
    ax = axes[1, 0]
    if not closed.empty:
        rets = closed["return_pct"] * 100
        ax.hist(rets, bins=20, color="#10B981", edgecolor="white")
        ax.axvline(0, color="grey", lw=0.8, ls="--")
        ax.axvline(rets.mean(), color="#DC2626", lw=1.2, label=f"mean {rets.mean():.1f}%")
        ax.legend()
    ax.set_title(f"Per-trade returns ({best_hm}m hold, {len(closed)} closed)")
    ax.set_xlabel("Return (%)")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3, axis="y")

    # (1,1) Per-position: strategy vs SPY (points above the diagonal beat the index)
    ax = axes[1, 1]
    if has_spy and not closed.empty and "spy_return_pct" in closed.columns:
        x = closed["spy_return_pct"] * 100
        y = closed["return_pct"] * 100
        ax.scatter(x, y, alpha=0.6, color="#7C3AED", edgecolor="white")
        lim = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lim, lim, color="grey", lw=1.0, ls="--", label="y = x (tie)")
        ax.axhline(0, color="grey", lw=0.6)
        ax.axvline(0, color="grey", lw=0.6)
        ax.set_title(f"Each position: strategy vs SPY ({best_hm}m hold)")
        ax.set_xlabel("SPY return over same window (%)")
        ax.set_ylabel("Strategy return (%)")
        ax.legend()
    elif not closed.empty:
        ax.scatter(closed["signal_median_return"] * 100, closed["return_pct"] * 100,
                   alpha=0.6, color="#7C3AED", edgecolor="white")
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.axvline(0, color="grey", lw=0.8, ls="--")
        ax.set_title(f"MC signal vs realized return ({best_hm}m hold)")
        ax.set_xlabel("MC median forecast at entry (%, 7d)")
        ax.set_ylabel("Realized return (%)")
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Charts -> {path}")


def compare_exit_rules(entries, price_histories, benchmark_close, cfg: Config, mc_fn):
    """Run every EXIT_RULES rule on the same entries; print a leaderboard vs SPY.

    Returns (logs_by_rule, leaderboard_df). Ranked by per-trade MEDIAN excess vs
    SPY (robust to the handful of monster winners that dominate a tiny sample).
    """
    fc_cache: dict = {}
    logs_by_rule, rows = {}, []
    for rule in EXIT_RULES:
        log = run_with_exit(entries, price_histories, benchmark_close, rule, cfg, mc_fn, fc_cache)
        logs_by_rule[rule["name"]] = log
        if log.empty:
            continue
        closed = log[~log["open"]]
        has_spy = "excess_return" in log.columns
        cc = closed.dropna(subset=["excess_return"]) if has_spy else closed
        ci = float(closed["invested"].sum()) or float("nan")
        rec = {
            "rule": rule["name"],
            "kind": rule["kind"],
            "n_closed": int(len(closed)),
            "n_open": int(log["open"].sum()),
            "avg_hold_days": float(closed["hold_days"].mean()) if len(closed) else float("nan"),
            "realized_pct": (float(closed["value"].sum()) / ci - 1.0) * 100 if len(closed) else float("nan"),
            "win_pct": float((closed["pnl"] > 0).mean()) * 100 if len(closed) else float("nan"),
        }
        if has_spy and len(cc):
            rec["spy_pct"] = (float(cc["spy_value"].sum()) / float(cc["invested"].sum()) - 1.0) * 100
            rec["excess_pct"] = rec["realized_pct"] - rec["spy_pct"]
            rec["beat_pct"] = float((cc["return_pct"] > cc["spy_return_pct"]).mean()) * 100
            rec["med_trade_excess_pct"] = float(cc["excess_return"].median()) * 100
        rows.append(rec)

    lb = pd.DataFrame(rows)
    if not lb.empty:
        sort_key = "med_trade_excess_pct" if "med_trade_excess_pct" in lb.columns else "realized_pct"
        lb = lb.sort_values(sort_key, ascending=False).reset_index(drop=True)

    print("=" * 104)
    print("EXIT-STRATEGY LEADERBOARD  (same entries, ranked by per-trade MEDIAN excess vs SPY)")
    print(f"  entries: {len(entries)} positions, ${DOLLARS_PER_TRADE:.0f} each, "
          f"selection='{SELECTION_METRIC}'")
    print("=" * 104)
    if lb.empty:
        print("  (no closed positions)")
        print("=" * 104)
        return logs_by_rule, lb
    hdr = (f"{'Rule':>13} {'Closed':>7} {'Open':>5} {'AvgHold':>8} {'Strat%':>8} "
           f"{'SPY%':>7} {'Excess':>8} {'Beat%':>6} {'MedExc':>7} {'Win%':>6}")
    print(hdr)
    print("-" * len(hdr))
    for _, r in lb.iterrows():
        print(f"{r['rule']:>13} {r['n_closed']:>7} {r['n_open']:>5} "
              f"{r.get('avg_hold_days', float('nan')):>7.0f}d {r['realized_pct']:>7.2f}% "
              f"{r.get('spy_pct', float('nan')):>6.2f}% {r.get('excess_pct', float('nan')):>+7.2f}% "
              f"{r.get('beat_pct', float('nan')):>5.1f}% {r.get('med_trade_excess_pct', float('nan')):>+6.2f}% "
              f"{r['win_pct']:>5.1f}%")
    print("-" * len(hdr))
    print("MedExc = median per-trade (strategy − SPY). Positive + Beat% > 50 = real edge, "
          "not just one lucky name.")
    print("=" * 104)
    return logs_by_rule, lb


def generate_exit_charts(logs_by_rule: dict, leaderboard: pd.DataFrame,
                         path: str = OUTPUT_EXIT_CHARTS) -> None:
    """2x2: cumulative P&L of top rules vs SPY, excess by rule, avg hold, best-rule scatter."""
    try:
        import matplotlib       # noqa: PLC0415
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        print("\nmatplotlib not installed -> skipping charts. `pip install matplotlib`.")
        return
    if leaderboard.empty:
        print("No closed positions -> no exit charts.")
        return

    has_spy = "excess_pct" in leaderboard.columns
    order = leaderboard["rule"].tolist()
    best = order[0]
    top3 = order[:3]

    def cum(log, col):
        c = log[~log["open"]].copy()
        if c.empty:
            return None
        c["exit_date"] = pd.to_datetime(c["exit_date"])
        c = c.sort_values("exit_date")
        return c["exit_date"], c[col].cumsum()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Capitol-Trades — exit-strategy comparison", fontsize=14, fontweight="bold")

    # (0,0) cumulative realized P&L: top-3 rules + SPY (matched to the best rule)
    ax = axes[0, 0]
    cmap = plt.get_cmap("viridis")
    for i, name in enumerate(top3):
        got = cum(logs_by_rule[name], "pnl")
        if got is not None:
            ax.plot(got[0], got[1], marker="o", ms=3,
                    color=cmap(i / max(1, len(top3) - 1)), label=name)
    if has_spy:
        # Match SPY to the shown rule with the MOST closed trades, so the benchmark
        # line spans as wide a date range as it honestly can (a matched benchmark
        # can't be longer than the rule it tracks).
        spy_ref = max(top3, key=lambda n: int((~logs_by_rule[n]["open"]).sum()))
        sp = cum(logs_by_rule[spy_ref], "spy_pnl")
        if sp is not None:
            ax.plot(sp[0], sp[1], color="#6B7280", ls="--", marker="o", ms=3,
                    label=f"SPY (matched to {spy_ref})")
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_title("Cumulative realized P&L — top exit rules vs SPY")
    ax.set_ylabel("Cumulative P&L ($)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", rotation=30)

    # (0,1) excess (or realized) % by rule, sorted
    ax = axes[0, 1]
    metric = "excess_pct" if has_spy else "realized_pct"
    lb2 = leaderboard.sort_values(metric)
    colors = ["#16A34A" if v >= 0 else "#DC2626" for v in lb2[metric]]
    ax.barh(lb2["rule"], lb2[metric], color=colors)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_title("Excess return vs SPY by rule (realized)" if has_spy
                 else "Realized return by rule")
    ax.set_xlabel("Excess %" if has_spy else "Realized %")
    ax.grid(alpha=0.3, axis="x")

    # (1,0) average holding days by rule (shows how much each shortens the hold)
    ax = axes[1, 0]
    lb3 = leaderboard.sort_values("avg_hold_days")
    ax.barh(lb3["rule"], lb3["avg_hold_days"], color="#2563EB")
    ax.set_title("Average holding period by rule")
    ax.set_xlabel("Days held")
    ax.grid(alpha=0.3, axis="x")

    # (1,1) best rule: per-trade strategy vs SPY (above y=x beats the index)
    ax = axes[1, 1]
    bl = logs_by_rule[best]
    bc = bl[~bl["open"]]
    if has_spy and "spy_return_pct" in bc.columns and not bc.empty:
        x = bc["spy_return_pct"] * 100
        y = bc["return_pct"] * 100
        ax.scatter(x, y, alpha=0.6, color="#7C3AED", edgecolor="white")
        lim = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lim, lim, color="grey", lw=1.0, ls="--", label="y = x (tie)")
        ax.axhline(0, color="grey", lw=0.6)
        ax.axvline(0, color="grey", lw=0.6)
        ax.legend()
    ax.set_title(f"Best rule '{best}': each position vs SPY")
    ax.set_xlabel("SPY return over same window (%)")
    ax.set_ylabel("Strategy return (%)")
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Exit charts -> {path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def load_or_scrape_trades(universe, earliest_rebalance: date, refresh: bool) -> pd.DataFrame:
    import os
    if not refresh and os.path.exists(TRADES_CACHE):
        df = pd.read_csv(TRADES_CACHE, parse_dates=["published", "traded"])
        print(f"Loaded {len(df)} cached trades from {TRADES_CACHE} "
              f"(use --refresh to re-scrape).\n")
        return df
    df = scrape_buy_trades_since(universe, earliest_rebalance, LOOKBACK_DAYS)
    if not df.empty:
        df.to_csv(TRADES_CACHE, index=False)
        print(f"Cached scraped trades -> {TRADES_CACHE}\n")
    return df


def main(argv) -> int:
    import app  # noqa: PLC0415  (the live data path needs the full app + its deps)

    refresh = "--refresh" in argv
    today = date.today()

    rebalance_dates = generate_rebalance_dates(today)
    earliest = min(rebalance_dates).date()
    print(f"Rebalance dates: {len(rebalance_dates)} runs "
          f"({rebalance_dates[0].date()} … {rebalance_dates[-1].date()})\n")

    universe = sorted(app.DEFAULT_TICKERS)
    trades = load_or_scrape_trades(universe, earliest, refresh)
    if trades.empty:
        print("No trades scraped — nothing to simulate.")
        return 1

    cfg = Config(rebalance_dates=rebalance_dates)

    # Only fetch prices for tickers that can ever be selected: the union of each
    # run's top-N-by-size. Much smaller than the full universe.
    candidate_tickers = set()
    for D in rebalance_dates:
        D = pd.Timestamp(D)
        win = trades[(trades["published"] >= D - pd.Timedelta(days=cfg.lookback_days))
                     & (trades["published"] <= D)]
        candidate_tickers.update(
            win.sort_values("size_num", ascending=False).head(cfg.top_n)["ticker"].tolist()
        )
    print(f"{len(candidate_tickers)} candidate tickers across all runs.\n")

    hist_start = earliest - timedelta(days=HISTORY_WARMUP_DAYS)
    hist_end = today + timedelta(days=1)
    price_histories = fetch_price_histories(candidate_tickers, hist_start, hist_end)

    # S&P 500 benchmark (SPY): same dollars, same entry/exit dates, per position.
    spy_hist = fetch_price_histories(["SPY"], hist_start, hist_end)
    spy_close = spy_hist.get("SPY")
    if spy_close is None:
        print("WARNING: SPY benchmark unavailable — skipping the S&P 500 comparison.\n")

    # Entries are computed ONCE (selection doesn't depend on the exit rule); then
    # every exit rule is applied to the same set and ranked against SPY.
    entries = build_entries(trades, price_histories, cfg, app.monte_carlo_forecast)
    print(f"Opened {len(entries)} positions across all runs.\n")

    logs_by_rule, leaderboard = compare_exit_rules(
        entries, price_histories, spy_close, cfg, app.monte_carlo_forecast)

    non_empty = [lg.assign(rule=name) if "rule" not in lg.columns else lg
                 for name, lg in logs_by_rule.items() if not lg.empty]
    if non_empty:
        combined = pd.concat(non_empty, ignore_index=True)
        combined.to_csv(OUTPUT_EXIT_LOG, index=False)
        print(f"\nPer-position log for all rules ({len(combined)} rows) -> {OUTPUT_EXIT_LOG}")

    generate_exit_charts(logs_by_rule, leaderboard)
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        import selftest_backtest  # noqa: PLC0415
        raise SystemExit(selftest_backtest.run())
    raise SystemExit(main(sys.argv[1:]))