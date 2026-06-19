"""
research_demo.py — end-to-end research + execution harness over the gold event table.

Run from the directory containing `capitol_ingest`:  python research_demo.py

Offline + deterministic. On synthetic random data every test should come back a NULL
(flat CARs, non-significant group differences, low DSR, high PBO, negative net edge) — the
point is that the harness reports honestly and won't manufacture an edge.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from capitol_ingest.deflation import (
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from capitol_ingest.execution import CostModel, apply_costs, break_even_cost_bps, cost_sweep
from capitol_ingest.gold import GoldConfig, GoldEventTableBuilder, InsiderProvider
from capitol_ingest.gold import CommitteeProvider
from capitol_ingest.holdout import HoldoutProtocol
from capitol_ingest.prices import SyntheticPriceProvider
from capitol_ingest.research import (
    compare_groups,
    cumulative_abnormal_returns,
    forward_excess_returns,
)

from gold_demo import synth_trades  # reuse the synthetic silver-trade generator


class DemoCommitteeProvider(CommitteeProvider):
    def __init__(self, relevant_tickers):
        self.relevant = set(relevant_tickers)

    def is_relevant(self, *, politician, ticker, as_of):
        return ticker in self.relevant


class DemoInsiderProvider(InsiderProvider):
    """Deterministic pseudo Form 4 overlay so the insider-corroboration split has data."""
    def overlay(self, *, ticker, as_of, lookback_days):
        seed = int(hashlib.md5(f"{ticker}{as_of}".encode()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        net = float(rng.normal(0, 1))
        return {
            "insider_net_buys": net,
            "insider_n_buyers": float(rng.integers(0, 5)),
            "insider_buy_value": float(abs(net) * 1e5),
            "insider_available": True,
        }


def main():
    trades, originals = synth_trades(n=140, seed=11)
    prices = SyntheticPriceProvider()
    cfg = GoldConfig(use_original_filing_date=False, benchmark_ticker="SPY")
    builder = GoldEventTableBuilder(
        prices, cfg,
        committee=DemoCommitteeProvider({"AAA", "BBB", "CCC"}),
        insider=DemoInsiderProvider(),
        original_filing_dates=originals,
    )
    et = builder.build(trades).event_table
    print(f"gold event table: {len(et)} events\n")

    # 1) Event study -------------------------------------------------------- #
    print("=== 1) Event study: cumulative abnormal returns vs SPY (t=0 is entry) ===")
    car = cumulative_abnormal_returns(et, prices, "SPY", horizon=10)
    car_fmt = car.copy()
    car_fmt["mean_car"] = (car_fmt["mean_car"] * 1e4).round(1)   # bps
    car_fmt["se"] = (car_fmt["se"] * 1e4).round(1)
    car_fmt["t_stat"] = car_fmt["t_stat"].round(2)
    print(car_fmt.rename(columns={"mean_car": "mean_car_bps", "se": "se_bps"}).to_string(index=False))

    # 2) Conditional portfolios -------------------------------------------- #
    print("\n=== 2) Conditional portfolios: forward 21d excess return ===")
    fwd = forward_excess_returns(et, prices, "SPY", horizon=21)

    def show(title, r):
        if "error" in r:
            print(f"  {title}: {r}")
            return
        ka, kb = [k for k in r if isinstance(r[k], dict)]
        print(f"  {title}")
        print(f"    {ka}: n={r[ka]['n']:3d} mean={r[ka]['mean']*1e4:7.1f}bps | "
              f"{kb}: n={r[kb]['n']:3d} mean={r[kb]['mean']*1e4:7.1f}bps")
        print(f"    diff={r['mean_diff']*1e4:7.1f}bps  p(t)={r['p_ttest']:.3f}  "
              f"p(MWU)={r['p_mannwhitney']:.3f}  "
              f"boot95=({r['boot_ci95'][0]*1e4:.0f},{r['boot_ci95'][1]*1e4:.0f})bps  "
              f"sig={r['ci_excludes_zero']}")

    show("Committee-relevant vs not",
         compare_groups(fwd, et["committee_relevant"] == 1.0,
                        label_a="committee_relevant", label_b="not_relevant"))
    show("Insider-corroborated vs not",
         compare_groups(fwd, et["insider_net_buys"] > 0,
                        label_a="insider_corroborated", label_b="not_corroborated"))

    # 3) Deflation ---------------------------------------------------------- #
    print("\n=== 3) Deflation (reality check) ===")
    strat = fwd.dropna().to_numpy()
    rng = np.random.default_rng(0)
    trial_srs = []
    for _ in range(25):  # 25 candidate feature/threshold rules we 'tried'
        sub = fwd[rng.random(len(fwd)) < 0.5].dropna().to_numpy()
        if len(sub) > 3 and sub.std(ddof=1) > 0:
            trial_srs.append(sub.mean() / sub.std(ddof=1))
    dsr = deflated_sharpe_ratio(strat, sr_trials=np.array(trial_srs))
    print(f"  DSR: SR_hat={dsr['sr_hat']:.3f}  E[max SR| {dsr['n_trials']} trials]="
          f"{dsr['sr0_expected_max']:.3f}  ->  DSR={dsr['dsr']:.3f}  "
          f"credible@95%={dsr['credible_at_95']}")

    # PBO via CSCV: columns = candidate rules' per-month returns over the real events
    et2 = et.copy()
    et2["fwd"] = fwd
    et2["month"] = pd.to_datetime(et2["t0"]).dt.to_period("M").astype(str)
    months = sorted(et2["month"].unique())
    rule_feats = ["mom_21", "dist_200ma", "size_rank", "disclosure_lag_days", "is_purchase"]
    cols = {}
    for j in range(20):
        f = rng.choice(rule_feats)
        thr = float(et2[f].median()) * (0.5 + rng.random())
        sel = et2[et2[f] > thr]
        cols[f"rule_{j}"] = sel.groupby("month")["fwd"].mean().reindex(months).fillna(0.0).to_numpy()
    M = pd.DataFrame(cols, index=months)
    pbo = probability_of_backtest_overfitting(M, n_splits=10)
    print(f"  PBO (CSCV): {pbo['pbo']:.2f} over {pbo['n_combinations']} splits, "
          f"{M.shape[1]} candidate rules  (high -> selection overfits noise)")

    # 4) Execution & break-even -------------------------------------------- #
    print("\n=== 4) Execution simulation & break-even cost ===")
    gross = fwd.dropna()
    sigma = et.loc[gross.index, "realized_vol_daily"].to_numpy()
    adv = rng.uniform(5e6, 5e8, len(gross))   # $5M-$500M daily volume
    order = 50_000.0                          # $50k retail copy order
    model = CostModel(half_spread_bps=2.0, Y=0.5, extra_bps=1.0)
    per_side = model.per_side_bps(sigma, order, adv)
    net = apply_costs(gross.to_numpy(), per_side)
    be = break_even_cost_bps(gross.to_numpy())
    print(f"  gross edge: {be['mean_gross_bps']:.1f} bps/trade")
    print(f"  modeled cost: per-side≈{np.mean(per_side):.1f} bps (impact tiny vs $50k/ADV; "
          f"spread dominates), round-trip≈{2*np.mean(per_side):.1f} bps")
    print(f"  net edge after costs: {np.nanmean(net)*1e4:.1f} bps/trade")
    print(f"  break-even per-side cost: {be['break_even_per_side_bps']:.1f} bps  "
          f"(edge_positive={be['edge_positive']})")
    sweep = cost_sweep(gross.to_numpy(), [0, 2, 5, 10, 20])
    sweep["net_mean_bps"] = (sweep["net_mean_return"] * 1e4).round(1)
    print("  cost sweep:\n" + sweep[["per_side_bps", "net_mean_bps"]].to_string(index=False))

    # 5) Out-of-time hold-out ---------------------------------------------- #
    print("\n=== 5) Out-of-time hold-out protocol ===")
    hp = HoldoutProtocol(et, holdout_frac=0.2, time_col="t0")
    print(f"  split at {hp.split_date.date()}: dev={hp.n_dev} events, holdout={hp.n_holdout} events (untouched)")
    final = hp.evaluate_once(
        lambda h: {"n": len(h),
                   "holdout_mean_fwd_excess_bps": round(float(forward_excess_returns(h, prices, "SPY", 21).mean()) * 1e4, 1)}
    )
    print(f"  one-shot hold-out evaluation: {final}")
    try:
        hp.evaluate_once(lambda h: None)
    except RuntimeError as e:
        print(f"  guard works -> {str(e).splitlines()[0]}")


if __name__ == "__main__":
    main()
