"""
Self-test for backtest_capitol's engine.

Stubs streamlit / yfinance / plotly so we can import the *real* app.monte_carlo_forecast
(pure numpy/pandas), then drives the engine with synthetic price series whose drift
we control. No network. Run: python backtest_capitol.py --selftest
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd


def _install_stubs():
    """Fake the heavy modules app.py imports at top, so `import app` works headless."""
    # streamlit: only need st.cache_data to be a no-op decorator factory; everything
    # else app touches at import time is just attribute access.
    st = types.ModuleType("streamlit")

    def cache_data(*a, **k):
        def deco(fn):
            return fn
        return deco
    st.cache_data = cache_data

    class _Anything:
        def __call__(self, *a, **k):
            return self
        def __getattr__(self, _):
            return _Anything()
    # app only *calls* st.* inside functions we never invoke here, but give it a
    # forgiving fallback just in case.
    st.__getattr__ = lambda name: _Anything()  # type: ignore[attr-defined]
    sys.modules["streamlit"] = st

    yf = types.ModuleType("yfinance")
    sys.modules["yfinance"] = yf

    plotly = types.ModuleType("plotly")
    go = types.ModuleType("plotly.graph_objects")
    subplots = types.ModuleType("plotly.subplots")
    subplots.make_subplots = lambda *a, **k: None
    plotly.graph_objects = go
    plotly.subplots = subplots
    sys.modules["plotly"] = plotly
    sys.modules["plotly.graph_objects"] = go
    sys.modules["plotly.subplots"] = subplots


def _geom_series(drift, vol, n=400, start_price=100.0, seed=0, end=None):
    """Deterministic geometric random walk of ~n business days ending at `end`."""
    end = pd.Timestamp(end or pd.Timestamp.today().normalize())
    idx = pd.bdate_range(end=end, periods=n)
    rng = np.random.default_rng(seed)
    rets = drift + vol * rng.standard_normal(len(idx))
    px = start_price * np.exp(np.cumsum(rets))
    return pd.Series(px, index=idx)


def run() -> int:
    _install_stubs()
    import app  # real monte_carlo_forecast / helpers
    import backtest_capitol as bt

    mc = app.monte_carlo_forecast
    failures = []

    def check(name, cond):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            failures.append(name)

    # ----- build synthetic world ------------------------------------------- #
    # Anchor "today" so dates are stable within the test.
    today = pd.Timestamp("2025-06-02").normalize()  # a Monday
    # Low vol so the *trailing-window* sample drift reliably matches the sign of
    # the true drift (the growth filter reads the estimated drift, not the true one).
    up   = _geom_series(drift=0.003,   vol=0.003, seed=1, end=today)  # strong up
    up2  = _geom_series(drift=0.0015,  vol=0.003, seed=2, end=today)  # mild up
    dn   = _geom_series(drift=-0.003,  vol=0.003, seed=3, end=today)  # down
    prices = {"UP": up, "UP2": up2, "DN": dn}

    # A run date well inside the series.
    D = pd.Timestamp("2025-03-03")

    # Trades visible in the window before D. Sizes set so DN is largest, then UP, then UP2,
    # so that size-ranking and forecast-ranking disagree (a good test).
    trades = pd.DataFrame([
        {"ticker": "DN",  "size_num": 5_000_000, "published": pd.Timestamp("2025-02-25")},
        {"ticker": "UP",  "size_num": 3_000_000, "published": pd.Timestamp("2025-02-26")},
        {"ticker": "UP2", "size_num": 1_000_000, "published": pd.Timestamp("2025-02-27")},
        # noise: outside window / wrong ticker — must be ignored
        {"ticker": "UP",  "size_num": 9_000_000, "published": pd.Timestamp("2024-12-01")},
    ])

    print("\n0) sanity: MC growth sign matches drift")
    f_up, f_dn = mc(up, horizon=bt.HORIZON), mc(dn, horizon=bt.HORIZON)
    check("UP forecast exists", f_up is not None)
    check("DN forecast exists", f_dn is not None)
    check("UP median_return > 0 and prob_up > 0.5",
          f_up["p50"][-1] > f_up["S0"] and f_up["prob_up"] > 0.5)
    check("DN median_return < 0 and prob_up < 0.5",
          f_dn["p50"][-1] < f_dn["S0"] and f_dn["prob_up"] < 0.5)
    # the equivalence p50>S0  <=>  prob_up>0.5 that the filter relies on
    check("growth-sign equivalence holds for UP/DN",
          ((f_up["p50"][-1] > f_up["S0"]) == (f_up["prob_up"] > 0.5)) and
          ((f_dn["p50"][-1] > f_dn["S0"]) == (f_dn["prob_up"] > 0.5)))

    print("\n1) selection: DN filtered out; ranked by median_return -> [UP, UP2]")
    cfg = bt.Config(rebalance_dates=[D], lookback_days=15, top_n=10, n_pick=2,
                    dollars=100.0, horizon=bt.HORIZON, selection_metric="median_return")
    picks = bt.select_positions_for_date(D, trades, prices, cfg, mc)
    tickers = [p["ticker"] for p in picks]
    check("exactly 2 picks", len(picks) == 2)
    check("DN excluded (negative forecast)", "DN" not in tickers)
    check("ordered by median_return: UP before UP2",
          tickers == ["UP", "UP2"])

    print("\n2) selection by 'size' re-orders survivors (UP has bigger size than UP2)")
    cfg_sz = bt.Config(rebalance_dates=[D], lookback_days=15, top_n=10, n_pick=2,
                       dollars=100.0, horizon=bt.HORIZON, selection_metric="size")
    picks_sz = [p["ticker"] for p in bt.select_positions_for_date(D, trades, prices, cfg_sz, mc)]
    check("size-ranked still excludes DN", "DN" not in picks_sz)
    check("size-ranked = [UP, UP2] (UP size 3M > UP2 1M)", picks_sz == ["UP", "UP2"])

    print("\n2b) window filter: a tiny lookback drops trades published before it")
    cfg_narrow = bt.Config(rebalance_dates=[D], lookback_days=1, top_n=10, n_pick=2,
                           dollars=100.0, horizon=bt.HORIZON, selection_metric="size")
    # window = [D-1, D] = [Mar 2, Mar 3]; all sample trades are Feb -> none qualify
    check("lookback=1 -> no positions", bt.select_positions_for_date(D, trades, prices, cfg_narrow, mc) == [])

    print("\n3) P&L math matches 100*(sell/buy - 1) with point-in-time prices")
    log = bt.run_backtest(trades, prices, cfg, holding_months=1, mc_fn=mc)
    check("2 positions opened", len(log) == 2)
    ok_math = True
    for _, row in log.iterrows():
        close = prices[row["ticker"]]
        bdate, bpx = bt.price_on_or_after(close, D)
        sdate, spx = bt.price_on_or_after(close, pd.Timestamp(bdate) + pd.DateOffset(months=1))
        expect_pnl = 100.0 * (spx / bpx - 1.0)
        ok_math &= abs(row["pnl"] - expect_pnl) < 1e-3       # log rounds to 4dp
        ok_math &= abs(row["return_pct"] - (spx / bpx - 1.0)) < 1e-4
        ok_math &= row["buy_date"] == bdate.date()
        ok_math &= row["exit_date"] == sdate.date()
    check("pnl / return / dates correct for every position", ok_math)
    # UP rose, so its 1-month position should be profitable
    up_row = log[log["ticker"] == "UP"].iloc[0]
    check("UP 1-month position is profitable", up_row["pnl"] > 0)

    print("\n4) open-position handling: huge holding period -> marked open at last close")
    log_open = bt.run_backtest(trades, prices, cfg, holding_months=60, mc_fn=mc)
    all_open = bool(log_open["open"].all())
    last_close_ok = all(
        abs(r["exit_price"] - float(prices[r["ticker"]].iloc[-1])) < 1e-3
        for _, r in log_open.iterrows()
    )
    check("all positions flagged open", all_open)
    check("open positions priced at last available close", last_close_ok)

    print("\n5) price_on_or_after edge cases")
    s = prices["UP"]
    check("target before series start -> first row",
          bt.price_on_or_after(s, s.index[0] - pd.Timedelta(days=10))[0] == s.index[0])
    check("target after series end -> None",
          bt.price_on_or_after(s, s.index[-1] + pd.Timedelta(days=10)) is None)

    print("\n6) rebalance-date generation ~ twice a month for 12 months")
    rds = bt.generate_rebalance_dates(pd.Timestamp("2025-06-14").date())
    check(f"23-25 rebalance dates (got {len(rds)})", 23 <= len(rds) <= 25)
    check("all within last 12 months",
          all(pd.Timestamp("2024-06-14") <= d <= pd.Timestamp("2025-06-14") for d in rds))

    print("\n7) summarize() splits realized vs marked-to-market")
    summ = bt.summarize(log_open, 60)
    check("summarize counts all open", summ["n_open"] == len(log_open) and summ["n_closed"] == 0)
    summ_c = bt.summarize(log, 1)
    check("closed summary has win_rate", "win_rate" in summ_c and 0 <= summ_c["win_rate"] <= 1)

    print("\n8) benchmark: SPY-matched columns + excess return are correct")
    # Use DN (down series) as a stand-in benchmark to exercise the path.
    log_bench = bt.run_backtest(trades, prices, cfg, holding_months=1, mc_fn=mc,
                                benchmark_close=dn)
    has_cols = all(c in log_bench.columns
                   for c in ["spy_return_pct", "excess_return", "beat_spy"])
    check("benchmark columns present", has_cols)
    if has_cols:
        ok_b = True
        for _, r in log_bench.iterrows():
            ok_b &= abs(r["excess_return"] - (r["return_pct"] - r["spy_return_pct"])) < 1e-4
            ok_b &= r["beat_spy"] == (r["return_pct"] > r["spy_return_pct"])
        check("excess_return = strat - spy and beat_spy flag correct", ok_b)
        summ_b = bt.summarize(log_bench, 1)
        check("summary has spy_realized_return_pct and beat_spy_rate",
              "spy_realized_return_pct" in summ_b and "beat_spy_rate" in summ_b)
        # picks are up-trending vs a down benchmark -> should beat it
        check("strategy beats the down benchmark", summ_b.get("excess_realized_return_pct", -1) > 0)

    print("\n9) exit rules trigger correctly on deterministic paths")

    def _path(values, end="2025-06-13"):
        idx = pd.bdate_range(end=pd.Timestamp(end), periods=len(values))
        return pd.Series([float(v) for v in values], index=idx)

    cfg9 = bt.Config(rebalance_dates=[], horizon=bt.HORIZON)

    # trailing 10%: peak 121 then 108 (<= 121*0.9=108.9) -> trail @108
    p = _path([100, 110, 121, 108])
    d, px, reason, _ = bt.resolve_exit(p, p.index[0], 100.0,
                                       {"kind": "trailing", "trail": 0.10, "max_hold_months": 6}, cfg9)
    check("trailing stop fires at the right bar", reason == "trail" and abs(px - 108) < 1e-9)

    # take-profit 20%: 125 -> +25% -> take
    p = _path([100, 125])
    _, px, reason, _ = bt.resolve_exit(p, p.index[0], 100.0,
                                       {"kind": "stop_take", "stop": 0.10, "take": 0.20, "max_hold_months": 6}, cfg9)
    check("take-profit fires", reason == "take" and abs(px - 125) < 1e-9)

    # stop-loss 10%: 88 -> -12% -> stop
    p = _path([100, 88])
    _, px, reason, _ = bt.resolve_exit(p, p.index[0], 100.0,
                                       {"kind": "stop_take", "stop": 0.10, "take": 0.20, "max_hold_months": 6}, cfg9)
    check("stop-loss fires", reason == "stop" and abs(px - 88) < 1e-9)

    # time exit: ~40 sessions, 1-month backstop reached inside the data -> "time"
    p = _path([100 + 0.01 * i for i in range(40)])
    _, _, reason, op = bt.resolve_exit(p, p.index[0], 100.0,
                                       {"kind": "time", "max_hold_months": 1}, cfg9)
    check("time backstop closes when reached", reason == "time" and op is False)

    # open: short path, 6-month backstop in the future -> "open" at last close
    p = _path([100, 101, 102])
    _, px, reason, op = bt.resolve_exit(p, p.index[0], 100.0,
                                        {"kind": "time", "max_hold_months": 6}, cfg9)
    check("unmatured position stays open at last close", reason == "open" and op and abs(px - 102) < 1e-9)

    # atr trailing: long flat (low vol) then a sustained bleed -> atr_trail before -30%
    bleed = [100.0] * 60 + [100 * (0.983 ** k) for k in range(1, 26)]
    p = _path(bleed)
    _, px, reason, _ = bt.resolve_exit(p, p.index[0], 100.0,
                                       {"kind": "atr_trailing", "k": 3.0, "max_hold_months": 12}, cfg9)
    check("atr trailing fires on the bleed", reason == "atr_trail" and 70 < px < 97)

    # mc_exit: long up-drift then long down-drift -> MC flips bearish -> mc_flip
    up_then_down = pd.concat([
        _geom_series(drift=0.004, vol=0.004, n=120, seed=21, end=pd.Timestamp("2025-01-01")),
        _geom_series(drift=-0.004, vol=0.004, n=80, seed=22, end=pd.Timestamp("2025-05-01")),
    ])
    up_then_down = up_then_down[~up_then_down.index.duplicated()].sort_index()
    _, _, reason, _ = bt.resolve_exit(up_then_down, up_then_down.index[0], float(up_then_down.iloc[0]),
                                      {"kind": "mc_exit", "prob_thr": 0.5, "check_every_days": 10,
                                       "max_hold_months": 12}, cfg9, mc_fn=mc, ticker="X", fc_cache={})
    check("mc_exit fires when the signal flips bearish", reason == "mc_flip")

    print("\n10) run_with_exit produces rule/hold_days/excess columns")
    entries10 = bt.build_entries(trades, prices, cfg, mc)
    log10 = bt.run_with_exit(entries10, prices, dn,
                             {"name": "trail_15", "kind": "trailing", "trail": 0.15, "max_hold_months": 6},
                             cfg, mc)
    cols_ok = all(c in log10.columns for c in ["rule", "hold_days", "exit_reason", "excess_return"])
    check("exit log has rule/hold_days/exit_reason/excess_return", cols_ok)
    check("hold_days non-negative", bool((log10["hold_days"] >= 0).all()))

    print("\n" + ("ALL TESTS PASSED" if not failures else f"FAILURES: {failures}"))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(run())