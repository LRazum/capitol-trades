"""
holdout.py — strict out-of-time hold-out protocol.

The final ``holdout_frac`` of the timeline is isolated as an absolute, untouched block.
All research, CV, feature selection, and threshold tuning happen on ``dev``; the hold-out
is evaluated exactly once, at the very end. ``HoldoutProtocol.evaluate_once`` hard-enforces
the "once" rule by refusing a second call — re-using the hold-out is how overfitting sneaks
back in after you thought you were done.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd


def time_holdout_split(
    event_table: pd.DataFrame, holdout_frac: float = 0.2, time_col: str = "t0"
):
    """Return (dev_index, holdout_index, split_date). Hold-out = most-recent events."""
    order = pd.to_datetime(event_table[time_col]).sort_values().index
    n = len(order)
    n_hold = max(1, int(round(n * holdout_frac)))
    holdout_index = order[-n_hold:]
    dev_index = order[:-n_hold]
    split_date = pd.to_datetime(event_table.loc[holdout_index, time_col]).min()
    return dev_index, holdout_index, split_date


class HoldoutProtocol:
    def __init__(self, event_table: pd.DataFrame, holdout_frac: float = 0.2, time_col: str = "t0"):
        self._table = event_table
        self.time_col = time_col
        self.dev_index, self.holdout_index, self.split_date = time_holdout_split(
            event_table, holdout_frac, time_col
        )
        self._consumed = False

    @property
    def dev(self) -> pd.DataFrame:
        return self._table.loc[self.dev_index]

    @property
    def n_dev(self) -> int:
        return len(self.dev_index)

    @property
    def n_holdout(self) -> int:
        return len(self.holdout_index)

    def evaluate_once(self, fn: Callable[[pd.DataFrame], object]):
        """Run ``fn`` on the hold-out block exactly once. Raises on any second call."""
        if self._consumed:
            raise RuntimeError(
                "Hold-out already consumed. Evaluating it again defeats its purpose — "
                "any further iteration must use a fresh out-of-time block."
            )
        self._consumed = True
        return fn(self._table.loc[self.holdout_index])
