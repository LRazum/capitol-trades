"""
execution.py — realistic cost model + break-even analysis.

Costs decompose into spread, market impact, and a buffer. Impact uses the square-root law
``impact_bps = 1e4 * Y * sigma_daily * sqrt(order_qty / ADV)`` (sigma_daily and order
sizes are fractions / share-or-dollar consistent). ``break_even_cost_bps`` returns the
cost level at which the strategy's mean edge hits zero — if that level is below realistic
costs, the edge does not survive.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def square_root_impact_bps(sigma_daily, order_qty, adv, Y: float = 0.5):
    """Temporary+permanent impact approximation, in basis points. Array-friendly."""
    sigma_daily = np.asarray(sigma_daily, float)
    order_qty = np.asarray(order_qty, float)
    adv = np.asarray(adv, float)
    ratio = np.divide(order_qty, adv, out=np.zeros_like(adv, dtype=float), where=adv > 0)
    return 1e4 * Y * sigma_daily * np.sqrt(ratio)


@dataclass
class CostModel:
    half_spread_bps: float = 2.0   # one side of the quoted spread
    Y: float = 0.5                 # square-root-law coefficient (~0.3-1.0)
    extra_bps: float = 0.0         # delay/slippage buffer

    def per_side_bps(self, sigma_daily, order_qty, adv):
        return self.half_spread_bps + square_root_impact_bps(sigma_daily, order_qty, adv, self.Y) + self.extra_bps

    def round_trip_bps(self, sigma_daily, order_qty, adv):
        return 2.0 * self.per_side_bps(sigma_daily, order_qty, adv)


def apply_costs(gross_returns, per_side_bps) -> np.ndarray:
    """Net per-trade return after a round trip (entry + exit). ``per_side_bps`` may be a
    scalar or a per-trade array."""
    gross = np.asarray(gross_returns, float)
    return gross - 2.0 * np.asarray(per_side_bps, float) / 1e4


def cost_sweep(gross_returns, per_side_bps_grid) -> pd.DataFrame:
    """Mean net return across a grid of per-side cost levels (bps)."""
    gross = np.asarray(gross_returns, float)
    rows = [
        {"per_side_bps": c, "round_trip_bps": 2 * c,
         "net_mean_return": float(np.nanmean(apply_costs(gross, c)))}
        for c in per_side_bps_grid
    ]
    return pd.DataFrame(rows)


def break_even_cost_bps(gross_returns) -> dict:
    """The cost at which the mean edge -> 0. mean_gross - 2*(c/1e4) = 0  =>  c = mean*1e4/2.
    A non-positive break-even means there is no positive edge for costs to erode."""
    m = float(np.nanmean(gross_returns))
    return {
        "mean_gross_return": m,
        "mean_gross_bps": m * 1e4,
        "break_even_per_side_bps": m * 1e4 / 2.0,
        "break_even_round_trip_bps": m * 1e4,
        "edge_positive": m > 0,
    }
