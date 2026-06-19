"""
deflation.py — discount apparent performance for the number of trials run.

* ``deflated_sharpe_ratio`` (Bailey & López de Prado): the probability the observed Sharpe
  is real once you account for (a) the number of configurations tried and (b) non-normal
  returns. DSR < ~0.95 means the result is not credible after multiple testing.
* ``probability_of_backtest_overfitting`` (CSCV): given a matrix of per-period returns for
  all candidate strategies, estimates how often the in-sample best underperforms the
  out-of-sample median. High PBO (-> 0.5) means your selection process overfits noise.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy.stats import norm

EULER = 0.5772156649015329


def expected_max_sharpe(sr_variance: float, n_trials: int) -> float:
    """E[max Sharpe] under the null of no skill across ``n_trials`` independent trials."""
    if n_trials < 2 or sr_variance <= 0:
        return 0.0
    z1 = norm.ppf(1 - 1.0 / n_trials)
    z2 = norm.ppf(1 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(sr_variance) * ((1 - EULER) * z1 + EULER * z2))


def probabilistic_sharpe_ratio(returns, sr_benchmark: float = 0.0) -> float:
    """P(true SR > sr_benchmark), correcting for skew/kurtosis. Per-period SR basis."""
    r = np.asarray(returns, float)
    r = r[~np.isnan(r)]
    T = len(r)
    if T < 3 or r.std(ddof=1) == 0:
        return float("nan")
    sr = r.mean() / r.std(ddof=1)
    skew = float(ss.skew(r))
    kurt = float(ss.kurtosis(r, fisher=False))  # Pearson (normal == 3)
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4.0 * sr**2, 1e-12))
    return float(norm.cdf((sr - sr_benchmark) * np.sqrt(T - 1) / denom))


def deflated_sharpe_ratio(
    returns,
    *,
    sr_trials=None,
    n_trials: int | None = None,
    sr_variance: float | None = None,
) -> dict:
    """DSR for a strategy's per-period returns. Provide either ``sr_trials`` (the Sharpe of
    every config you tried) or both ``n_trials`` and ``sr_variance``."""
    r = np.asarray(returns, float)
    r = r[~np.isnan(r)]
    if sr_trials is not None:
        sr_trials = np.asarray(sr_trials, float)
        V, N = float(np.var(sr_trials, ddof=1)), len(sr_trials)
    elif n_trials is not None and sr_variance is not None:
        V, N = float(sr_variance), int(n_trials)
    else:
        raise ValueError("provide sr_trials, or both n_trials and sr_variance")

    sr0 = expected_max_sharpe(V, N)
    dsr = probabilistic_sharpe_ratio(r, sr_benchmark=sr0)
    sr_hat = r.mean() / r.std(ddof=1) if r.std(ddof=1) > 0 else float("nan")
    return {
        "sr_hat": float(sr_hat),
        "sr0_expected_max": sr0,
        "dsr": dsr,
        "n_trials": N,
        "sr_variance": V,
        "T": len(r),
        "credible_at_95": bool(dsr is not None and dsr >= 0.95),
    }


def probability_of_backtest_overfitting(
    perf_matrix: pd.DataFrame, n_splits: int = 10, metric: str = "sharpe"
) -> dict:
    """CSCV PBO. ``perf_matrix``: rows = time periods, columns = candidate strategies,
    values = per-period returns. Returns PBO and the logit distribution."""
    M = np.asarray(perf_matrix, float)
    T, N = M.shape
    if n_splits % 2 != 0:
        n_splits += 1
    blocks = np.array_split(np.arange(T), n_splits)

    def score(block_rows):
        b = M[block_rows]
        mu, sd = b.mean(0), b.std(0, ddof=1)
        return mu / np.where(sd == 0, np.nan, sd) if metric == "sharpe" else mu

    logits = []
    for combo in combinations(range(n_splits), n_splits // 2):
        is_rows = np.concatenate([blocks[i] for i in combo])
        oos_rows = np.concatenate([blocks[i] for i in range(n_splits) if i not in combo])
        is_perf, oos_perf = score(is_rows), score(oos_rows)
        best = int(np.nanargmax(is_perf))
        # relative OOS rank of the IS-best strategy, in (0, 1)
        w = (np.sum(oos_perf <= oos_perf[best])) / (N + 1)
        w = min(max(w, 1.0 / (N + 1)), 1 - 1.0 / (N + 1))
        logits.append(np.log(w / (1 - w)))

    logits = np.array(logits)
    return {
        "pbo": float(np.mean(logits <= 0)),  # P(IS-best below OOS median)
        "n_combinations": int(len(logits)),
        "median_logit": float(np.median(logits)),
        "logits": logits,
    }
