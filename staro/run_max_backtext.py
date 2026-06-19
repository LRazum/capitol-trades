# Copyright (c) 2026 Lovro Razum. All rights reserved.
# PROPRIETARY AND CONFIDENTIAL. Unauthorized use, copying, modification, or
# distribution of this file, in whole or in part, via any medium, is strictly
# prohibited without the prior written permission of Lovro Razum.
# See the accompanying LICENSE file for full terms.
"""
Maximum-sample backtest runner
==============================
Produces the LARGEST honest test set the data can support, then (optionally)
feeds it straight into validate_strategy.py.

The sample size of the Capitol-Trades backtest is driven by three levers:
  1. TIME SPAN      — how far back we scrape & simulate  (the big one)
  2. BREADTH        — how many trades per window enter the MC filter (TOP_N)
  3. ALL SURVIVORS  — buy EVERY growth-filtered name, not just the best 2

Cadence (twice-monthly vs weekly) barely changes the *count* as long as the
look-back tiles the gap (each disclosure is then counted exactly once); it only
changes the point-in-time at which each trade is evaluated. So this script
maximises 1–3 and keeps the methodology otherwise identical to backtest_capitol.

IMPORTANT honesty caveat: the same megacaps (GOOGL/AAPL) get bought over and
over, and overlapping windows produce correlated trades. More rows is NOT the
same as more *independent* evidence — the real win is a longer time span and a
wider set of distinct names (which ALL-SURVIVORS + a big TOP_N pulls in). The
validator's bootstrap/permutation tests already account for this.

It is NON-DESTRUCTIVE: it writes to its own cache/log files (…_max / _MAX)
and never touches your existing 8-month capitol_trades_cache.csv.

Usage (run inside the streamlit env, with network):
  python run_max_backtest.py                      # ALL US buys + scrape max + biggest log
  python run_max_backtest.py --validate           # …then run Stage A validation
  python run_max_backtest.py --validate --validate-comparators   # + Stage B
  python run_max_backtest.py --tickers sp500       # restrict to the app's ~400 names
  python run_max_backtest.py --months-back 36 --cadence weekly   # tune the levers
  python run_max_backtest.py --refresh            # ignore the cache, re-scrape

Requires: app.py + backtest_capitol.py next to this file, their deps
(numpy/pandas/yfinance), and validate_strategy.py for --validate.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Rebalance dates for any cadence
# --------------------------------------------------------------------------- #
def build_rebalance_dates(today: date, months_back: int, cadence: str,
                          every_days: int) -> list[pd.Timestamp]:
    """Twice-monthly (1st/15th), weekly (Mondays), or every-N-days over the span."""
    import backtest_capitol as bt  # noqa: PLC0415
    if cadence == "twicemonthly":
        return bt.generate_rebalance_dates(today, months_back, (1, 15))

    end = pd.Timestamp(today)
    start = end - pd.DateOffset(months=months_back)
    step = 7 if cadence == "weekly" else max(1, every_days)
    cur = pd.Timestamp(start.date())
    if cadence == "weekly":  # align to the next Monday
        cur = cur + pd.Timedelta(days=(7 - cur.weekday()) % 7)
    out: list[pd.Timestamp] = []
    while cur <= end:
        out.append(pd.Timestamp(cur.date()))
        cur = cur + pd.Timedelta(days=step)
    return out


def default_lookback(cadence: str, every_days: int) -> int:
    """Look-back that tiles the cadence gap so each disclosure is counted once."""
    if cadence == "twicemonthly":
        return 15
    if cadence == "weekly":
        return 7
    return max(1, every_days)


# --------------------------------------------------------------------------- #
# Scrape as far back as the site allows (incremental, polite, cached)
# --------------------------------------------------------------------------- #
def _parse_rows_all(html: str, today: date, cutoff: date) -> list:
    """Like app._parse_rows, but WITHOUT the ticker-universe filter.

    app._parse_rows drops any buy whose ticker is not in DEFAULT_TICKERS (~400
    S&P names). Politicians trade far more than that — small/mid caps, ETFs —
    so this keeps EVERY US buy disclosure (any 'XYZ:US' symbol) to maximise the
    data pulled. Parsing is otherwise identical (same cells, same helpers).
    """
    from bs4 import BeautifulSoup  # noqa: PLC0415
    from app import (parse_published, parse_size, parse_date_token,  # noqa: PLC0415
                     TICKER_RE)
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for tr in soup.select("table tbody tr"):
        cells = tr.find_all("td")
        if len(cells) < 9:
            continue
        if cells[6].get_text(" ", strip=True).lower() != "buy":
            continue
        published = parse_published(cells[2].get_text(" ", strip=True), today)
        if published is None or published < cutoff:
            continue
        issuer = cells[1].get_text(" ", strip=True)
        match = TICKER_RE.search(issuer)
        if not match:                       # no parseable US ticker -> can't price it
            continue
        size_str = cells[7].get_text(" ", strip=True)
        rows.append({
            "politician": cells[0].get_text(" ", strip=True),
            "issuer": issuer,
            "ticker": match.group(1),
            "published": published,
            "traded": parse_date_token(cells[3].get_text(" ", strip=True)),
            "owner": cells[5].get_text(" ", strip=True),
            "size_str": size_str,
            "size_num": parse_size(size_str),
            "price": cells[8].get_text(" ", strip=True),
        })
    return rows


def scrape_max(universe, cutoff: date, max_pages: int, page_size: int,
               cache_path: str, refresh: bool, sleep_s: float,
               all_tickers: bool = True, empty_streak_stop: int = 8) -> pd.DataFrame:
    """Page back through capitoltrades buys until past `cutoff` (or pages run out).

    all_tickers=True  -> keep every US buy (max data; uses _parse_rows_all).
    all_tickers=False -> only the app's DEFAULT_TICKERS universe (app._parse_rows).

    Saves a partial cache every 50 pages so a rate-limit mid-run does not throw
    away progress.
    """
    if cache_path and os.path.exists(cache_path) and not refresh:
        df = pd.read_csv(cache_path, parse_dates=["published", "traded"])
        print(f"Loaded {len(df)} cached trades from {cache_path} "
              f"(use --refresh to re-scrape).\n")
        return df

    from app import _fetch_page, _parse_rows, norm  # noqa: PLC0415
    revolut_norm = {norm(t) for t in universe} if not all_tickers else None
    today = date.today()
    mode = "ALL US tickers" if all_tickers else f"{len(universe)}-ticker universe"
    print(f"Scraping buys published since {cutoff} "
          f"({mode}, up to {max_pages} pages)…")

    rows, empty = [], 0
    for page in range(1, max_pages + 1):
        try:
            html = _fetch_page(page, page_size)
        except Exception as exc:  # noqa: BLE001 — transient HTTP / rate limit
            print(f"  page {page}: fetch failed ({exc}); waiting then retrying once…")
            time.sleep(max(2.0, sleep_s * 4))
            try:
                html = _fetch_page(page, page_size)
            except Exception as exc2:  # noqa: BLE001
                print(f"  page {page}: failed again ({exc2}); stopping with what we have.")
                break

        page_rows = (_parse_rows_all(html, today, cutoff) if all_tickers
                     else _parse_rows(html, revolut_norm, today, cutoff))
        rows.extend(page_rows)
        if page % 10 == 0 or page_rows:
            print(f"  page {page:>4}: +{len(page_rows):>3} qualifying (total {len(rows)})")

        if not page_rows:
            empty += 1
            if empty >= empty_streak_stop:
                print(f"  {empty} empty pages in a row -> past the window, stopping.")
                break
        else:
            empty = 0

        if cache_path and page % 50 == 0 and rows:  # periodic partial save
            _to_cache(pd.DataFrame(rows), cache_path, announce=False)
        time.sleep(sleep_s)

    df = _finalize_trades(pd.DataFrame(rows))
    if cache_path and not df.empty:
        _to_cache(df, cache_path)
    print(f"Scraped {len(df)} qualifying buy trades.\n")
    return df


def _finalize_trades(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.drop_duplicates().copy()
    df["traded"] = pd.to_datetime(df["traded"])
    df["published"] = pd.to_datetime(df["published"])
    return (df.sort_values(["published", "size_num"], ascending=[False, False])
              .reset_index(drop=True))


def _to_cache(df: pd.DataFrame, path: str, announce: bool = True) -> None:
    _finalize_trades(df).to_csv(path, index=False)
    if announce:
        print(f"Cached scraped trades -> {path}")


# --------------------------------------------------------------------------- #
# Prices (batched + pickle-cached), mirroring app semantics
# --------------------------------------------------------------------------- #
def fetch_closes_cached(tickers, start: date, end: date, cache_path: str,
                        chunk: int = 40) -> dict:
    """Per-ticker adjusted Close via a batched yfinance download. Cached to pickle.

    Keyed by the *Capitol* ticker (e.g. 'BRK.B') so it drops straight into
    backtest_capitol.build_entries; symbol mapping to Yahoo ('BRK-B') is internal.
    """
    from app import to_yahoo  # noqa: PLC0415
    cache: dict = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as fh:
            cache = pickle.load(fh)

    want = [t for t in tickers if t not in cache]
    if want:
        import yfinance as yf  # noqa: PLC0415
        ymap = {to_yahoo(t).replace("/", "-"): t for t in want}   # yahoo symbol -> capitol ticker
        ysyms = list(ymap)
        print(f"Fetching prices for {len(want)} tickers {start} -> {end} "
              f"(batched, chunks of {chunk})…")
        for i in range(0, len(ysyms), chunk):
            batch = ysyms[i:i + chunk]
            try:
                data = yf.download(batch, start=start, end=end, auto_adjust=True,
                                   progress=False, group_by="ticker", threads=True)
            except Exception as exc:  # noqa: BLE001
                print(f"  chunk {i // chunk + 1}: download failed ({exc}); skipping.")
                for ys in batch:
                    cache[ymap[ys]] = None
                continue
            for ys in batch:
                t = ymap[ys]
                try:
                    s = data[ys]["Close"] if len(batch) > 1 else data["Close"]
                    s = s.dropna()
                    if s.empty:
                        cache[t] = None
                        continue
                    s.index = s.index.tz_localize(None)
                    cache[t] = s.sort_index()
                except Exception:  # noqa: BLE001
                    cache[t] = None
            if cache_path:                              # persist after each chunk
                with open(cache_path, "wb") as fh:
                    pickle.dump(cache, fh)
            print(f"  chunk {i // chunk + 1}/{(len(ysyms) + chunk - 1) // chunk}: "
                  f"{sum(cache.get(ymap[ys]) is not None for ys in batch)}/{len(batch)} ok")
        got = sum(cache.get(t) is not None for t in want)
        print(f"  got {got}/{len(want)} new tickers.\n")
    return {t: cache.get(t) for t in tickers if cache.get(t) is not None}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Maximise the Capitol-Trades backtest sample, then validate it.")
    p.add_argument("--months-back", type=int, default=60,
                   help="how far back to scrape & simulate (bounded by the site's depth)")
    p.add_argument("--cadence", choices=["twicemonthly", "weekly", "every-days"],
                   default="twicemonthly", help="rebalance cadence")
    p.add_argument("--every-days", type=int, default=10,
                   help="step in days when --cadence every-days")
    p.add_argument("--tickers", choices=["all", "sp500"], default="all",
                   help="'all' = every US buy disclosure (max data); "
                        "'sp500' = the app's ~400-name DEFAULT_TICKERS universe")
    p.add_argument("--top-n", type=int, default=60,
                   help="how many of the largest trades per window enter the MC filter")
    p.add_argument("--n-pick", type=int, default=0,
                   help="positions to open per run; 0 = ALL growth-survivors (max sample)")
    p.add_argument("--lookback", type=int, default=0,
                   help="window in days; 0 = auto (tiles the cadence so no double-count)")
    p.add_argument("--dollars", type=float, default=100.0, help="$ per position")
    p.add_argument("--horizon", type=int, default=7, help="MC forecast horizon (trading days)")
    p.add_argument("--selection-metric", default="median_return",
                   choices=["median_return", "prob_up", "size"])
    p.add_argument("--warmup-days", type=int, default=260,
                   help="calendar days of price history before the earliest run")
    p.add_argument("--max-pages", type=int, default=1500, help="scrape page cap")
    p.add_argument("--page-size", type=int, default=96)
    p.add_argument("--sleep", type=float, default=0.4, help="seconds between page fetches (be polite)")
    p.add_argument("--trades-cache", default="capitol_trades_cache_max.csv")
    p.add_argument("--price-cache", default="prices_cache_max.pkl")
    p.add_argument("--out", default="exit_strategy_tradelog_MAX.csv",
                   help="the big exit tradelog (feed this to validate_strategy.py)")
    p.add_argument("--refresh", action="store_true", help="ignore the trade cache and re-scrape")
    # validation handoff
    p.add_argument("--validate", action="store_true",
                   help="run validate_strategy.py Stage A on the produced log")
    p.add_argument("--validate-comparators", action="store_true",
                   help="also run validate_strategy Stage B (permutation + comparators)")
    p.add_argument("--outdir", default="validation_out_max", help="validator output dir")
    return p


def main(argv) -> int:
    args = build_parser().parse_args(argv)

    # Lazy heavy imports (need streamlit/yfinance on the path).
    try:
        import app  # noqa: PLC0415
        import backtest_capitol as bt  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        print(f"Could not import the live stack ({exc}). "
              "Run this inside the streamlit environment.")
        return 2

    today = date.today()
    lookback = args.lookback or default_lookback(args.cadence, args.every_days)
    n_pick = args.n_pick if args.n_pick > 0 else args.top_n  # 0 -> all survivors
    all_survivors = args.n_pick == 0

    rebalance_dates = build_rebalance_dates(today, args.months_back, args.cadence, args.every_days)
    if not rebalance_dates:
        print("No rebalance dates generated — check --months-back / --cadence.")
        return 1

    all_tickers = args.tickers == "all"
    print("=" * 80)
    print("MAX-SAMPLE BACKTEST")
    print(f"  tickers={'ALL US buys' if all_tickers else 'S&P universe'}  "
          f"cadence={args.cadence}  months_back={args.months_back}  lookback={lookback}d")
    print(f"  top_n={args.top_n}  n_pick={'ALL survivors' if all_survivors else n_pick}  "
          f"selection='{args.selection_metric}'  ${args.dollars:.0f}/pos  horizon={args.horizon}d")
    print(f"  {len(rebalance_dates)} candidate runs "
          f"({rebalance_dates[0].date()} … {rebalance_dates[-1].date()})")
    print("=" * 80 + "\n")

    universe = sorted(app.DEFAULT_TICKERS)
    earliest = min(rebalance_dates).date()
    cutoff = earliest - timedelta(days=lookback)

    # ---- scrape (or load) the deepest set of buys we can ------------------- #
    trades = scrape_max(universe, cutoff, args.max_pages, args.page_size,
                        args.trades_cache, args.refresh, args.sleep,
                        all_tickers=all_tickers)
    if trades.empty:
        print("No trades scraped — nothing to simulate.")
        return 1
    print(f"Captured {len(trades)} buys across {trades['ticker'].nunique()} distinct "
          f"tickers ({trades['published'].min().date()} … {trades['published'].max().date()}).\n")

    # Trim dead early runs: drop rebalance dates before any disclosure could
    # fall in-window (saves a pointless multi-year price fetch when the site is
    # shallower than --months-back).
    min_pub = pd.Timestamp(trades["published"].min())
    floor = min_pub - pd.Timedelta(days=lookback)
    kept = [d for d in rebalance_dates if d >= floor]
    if len(kept) < len(rebalance_dates):
        print(f"Trimmed {len(rebalance_dates) - len(kept)} runs before the data starts "
              f"({min_pub.date()}). {len(kept)} runs remain "
              f"({kept[0].date()} … {kept[-1].date()}).\n")
    rebalance_dates = kept

    cfg = bt.Config(
        rebalance_dates=rebalance_dates, lookback_days=lookback, top_n=args.top_n,
        n_pick=n_pick, dollars=args.dollars, horizon=args.horizon,
        selection_metric=args.selection_metric,
    )

    # ---- only fetch prices for tickers that can ever be selected ----------- #
    candidates: set = set()
    for D in rebalance_dates:
        D = pd.Timestamp(D)
        win = trades[(trades["published"] >= D - pd.Timedelta(days=lookback))
                     & (trades["published"] <= D)]
        candidates.update(win.sort_values("size_num", ascending=False)
                          .head(args.top_n)["ticker"].tolist())
    print(f"{len(candidates)} candidate tickers across all runs.\n")

    hist_start = earliest - timedelta(days=args.warmup_days)
    hist_end = today + timedelta(days=1)
    prices = fetch_closes_cached(sorted(candidates) + ["SPY"], hist_start, hist_end,
                                 args.price_cache)
    spy_close = prices.pop("SPY", None)
    if spy_close is None:
        print("WARNING: SPY unavailable — excess-vs-index columns will be missing.\n")

    # ---- entries (selection is independent of the exit rule) --------------- #
    entries = bt.build_entries(trades, prices, cfg, app.monte_carlo_forecast)
    print(f"Opened {len(entries)} positions across {len(rebalance_dates)} runs.")
    if entries:
        n_names = len({e["ticker"] for e in entries})
        print(f"  spanning {n_names} distinct tickers "
              f"({entries[0]['buy_date'].date()} … {max(e['buy_date'] for e in entries).date()}).")
    print()
    if not entries:
        print("No positions opened (the MC growth filter rejected everything). "
              "Try a larger --top-n or a different --selection-metric.")
        return 1

    # ---- every exit rule on the same entries, benchmarked vs SPY ----------- #
    logs_by_rule, _ = bt.compare_exit_rules(entries, prices, spy_close, cfg,
                                            app.monte_carlo_forecast)

    non_empty = [lg for lg in logs_by_rule.values() if not lg.empty]
    if not non_empty:
        print("No closed/realized positions — nothing to write.")
        return 1
    combined = pd.concat(non_empty, ignore_index=True)
    combined.to_csv(args.out, index=False)

    print("\n" + "=" * 80)
    print(f"BIGGEST TEST SET WRITTEN -> {args.out}")
    print(f"  {len(entries)} entries × {len(non_empty)} exit rules = {len(combined)} rows")
    closed = combined[~combined["open"].astype(bool)]
    print(f"  closed (realized) positions: {len(closed)}   open (still held): "
          f"{len(combined) - len(closed)}")
    print("  closed per rule:")
    for rule, n in closed.groupby("rule").size().sort_values(ascending=False).items():
        print(f"    {rule:>14}: {n}")
    print("=" * 80)

    # ---- optional: straight into the validator ----------------------------- #
    if args.validate or args.validate_comparators:
        try:
            import validate_strategy as vs  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            print(f"\nCould not import validate_strategy ({exc}); skipping validation. "
                  f"Run it manually:\n  python validate_strategy.py --log {args.out}")
            return 0
        vargs = ["--log", args.out, "--trades", args.trades_cache, "--outdir", args.outdir]
        if args.validate_comparators:
            vargs += ["--comparators"]
        print(f"\n>>> handing off to validate_strategy.py "
              f"({'Stage A+B' if args.validate_comparators else 'Stage A'})…\n")
        vs.main(vargs)
    else:
        print(f"\nNext: python validate_strategy.py --log {args.out} "
              f"--trades {args.trades_cache} --outdir {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))