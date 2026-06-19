"""
cv.py — purged-and-embargoed cross-validation (López de Prado).

Standard K-Fold leaks here because event labels span overlapping time intervals
[t0, t_end] and adjacent observations are serially correlated. These splitters fix that:

* test folds are contiguous blocks in *event-start order*;
* PURGE: any training observation whose label interval overlaps the test block's interval
  is dropped;
* EMBARGO: a buffer of observations immediately after the test block is also dropped.

``leakage_report`` proves, post-split, that no surviving training interval overlaps the
test span. ``evaluate_cv`` wires it to a model and returns honest per-fold OOS scores.
"""
from __future__ import annotations

from itertools import combinations
from typing import Callable, Iterator, Optional

import numpy as np
import pandas as pd


def _as_dt64(x) -> np.ndarray:
    return pd.to_datetime(pd.Series(x)).to_numpy()


class PurgedKFold:
    """K contiguous time folds with purging + embargo. ``split`` yields ORIGINAL-index
    arrays (it sorts internally by event start, so the caller's order is preserved)."""

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, X, t0, t1) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        t0a, t1a = _as_dt64(t0), _as_dt64(t1)
        n = len(t0a)
        order = np.argsort(t0a, kind="mergesort")
        s_t0, s_t1 = t0a[order], t1a[order]
        embargo = int(n * self.embargo_pct)

        for test_block in np.array_split(np.arange(n), self.n_splits):
            if len(test_block) == 0:
                continue
            test_lo, test_hi = s_t0[test_block].min(), s_t1[test_block].max()
            train = np.ones(n, dtype=bool)
            train[test_block] = False
            # purge: drop train whose [t0,t1] overlaps the test interval
            overlap = (s_t0 <= test_hi) & (s_t1 >= test_lo)
            train &= ~overlap
            # embargo: drop a buffer immediately after the test block (in sorted order)
            hi_pos = int(test_block.max())
            train[hi_pos + 1: min(hi_pos + 1 + embargo, n)] = False
            yield order[np.where(train)[0]], order[test_block]


class CombinatorialPurgedKFold:
    """Simplified CPCV: split into ``n_groups`` contiguous time groups and test on every
    combination of ``n_test_groups`` of them, purging/embargoing around each test group.
    Yields C(n_groups, n_test_groups) train/test paths for a distribution of OOS scores."""

    def __init__(self, n_groups: int = 6, n_test_groups: int = 2, embargo_pct: float = 0.01):
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.embargo_pct = embargo_pct

    def split(self, X, t0, t1) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        t0a, t1a = _as_dt64(t0), _as_dt64(t1)
        n = len(t0a)
        order = np.argsort(t0a, kind="mergesort")
        s_t0, s_t1 = t0a[order], t1a[order]
        groups = np.array_split(np.arange(n), self.n_groups)
        embargo = int(n * self.embargo_pct)

        for combo in combinations(range(self.n_groups), self.n_test_groups):
            test_block = np.concatenate([groups[g] for g in combo])
            train = np.ones(n, dtype=bool)
            train[test_block] = False
            for g in combo:
                gp = groups[g]
                if len(gp) == 0:
                    continue
                lo, hi = s_t0[gp].min(), s_t1[gp].max()
                train &= ~((s_t0 <= hi) & (s_t1 >= lo))         # purge
                gmax = int(gp.max())
                train[gmax + 1: min(gmax + 1 + embargo, n)] = False  # embargo
            yield order[np.where(train)[0]], order[test_block]


def leakage_report(t0, t1, train_idx, test_idx) -> int:
    """Count training observations whose label interval overlaps *any individual* test
    observation's interval. Exact (not span-based), so it is valid for the non-contiguous
    test sets produced by CPCV as well as contiguous folds. Should be 0 after purging."""
    t0a, t1a = _as_dt64(t0), _as_dt64(t1)
    if len(test_idx) == 0 or len(train_idx) == 0:
        return 0
    tr0 = t0a[train_idx][:, None]
    tr1 = t1a[train_idx][:, None]
    te0 = t0a[test_idx][None, :]
    te1 = t1a[test_idx][None, :]
    overlap = (tr0 <= te1) & (tr1 >= te0)  # (n_train, n_test)
    return int(np.any(overlap, axis=1).sum())


def default_model():
    """Impute -> scale -> L2 logistic regression. Linear + regularized is the right
    starting capacity given few truly-independent (overlapping) observations."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000)),
    ])


def evaluate_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    t0,
    t1,
    splitter,
    *,
    make_model: Callable = default_model,
) -> dict:
    """Fit per fold and return per-fold OOS ROC-AUC plus a leakage count per fold. Folds
    whose test or train set is single-class are skipped (AUC undefined)."""
    from sklearn.metrics import roc_auc_score

    Xv = X.to_numpy()
    y = np.asarray(y)
    aucs, leaks, sizes = [], [], []
    for train_idx, test_idx in splitter.split(X, t0, t1):
        leaks.append(leakage_report(t0, t1, train_idx, test_idx))
        sizes.append((len(train_idx), len(test_idx)))
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            aucs.append(float("nan"))
            continue
        model = make_model()
        model.fit(Xv[train_idx], y[train_idx])
        proba = model.predict_proba(Xv[test_idx])[:, 1]
        aucs.append(float(roc_auc_score(y[test_idx], proba)))
    valid = [a for a in aucs if a == a]  # drop NaN
    return {
        "fold_auc": aucs,
        "mean_auc": float(np.mean(valid)) if valid else float("nan"),
        "leakage_per_fold": leaks,
        "max_leakage": int(max(leaks)) if leaks else 0,
        "fold_sizes": sizes,
        "n_folds": len(aucs),
    }
