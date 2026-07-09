"""
ml.py — the model, evaluated honestly.

"Advanced ML stock selection" here means a gradient-boosted (or RF / logistic) classifier
that ranks tickers by P(outperform benchmark over H days) — but the numbers it reports are
*out-of-sample*. Evaluation uses purged, expanding-window time splits (no future leakage,
with an embargo equal to the label horizon), and every run is compared against a
shuffled-label null so you can see whether the model beats chance rather than just
producing a confident-looking score.

This is the difference between a tool that looks data-driven and one that is: the headline
isn't the prediction, it's whether the prediction generalizes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODELS = ("Gradient Boosting", "Random Forest", "Logistic Regression")


def build_model(name: str):
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=20,
                                      n_jobs=-1, random_state=0, class_weight="balanced")
    if name == "Logistic Regression":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, class_weight="balanced"))
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.06,
                                          max_iter=300, l2_regularization=1.0,
                                          random_state=0)


@dataclass
class ModelResult:
    model_name: str
    horizon: int
    n_samples: int
    class_balance: float
    feature_names: list
    # out-of-sample
    oos_auc: float = float("nan")
    oos_pr_auc: float = float("nan")
    oos_acc: float = float("nan")
    oos_ic: float = float("nan")           # Spearman(score, fwd_excess)
    brier: float = float("nan")
    cv_aucs: list = field(default_factory=list)
    null_aucs: list = field(default_factory=list)
    null_p_value: float = float("nan")
    roc: dict = field(default_factory=dict)            # fpr, tpr
    calibration: dict = field(default_factory=dict)    # prob_pred, prob_true
    confusion: list = field(default_factory=list)
    threshold: float = 0.5
    importances: pd.DataFrame = field(default_factory=pd.DataFrame)
    oos_frame: pd.DataFrame = field(default_factory=pd.DataFrame)  # asof,ticker,score,y,fwd
    model: object = None                    # final model fit on ALL data (for selection)
    warnings: list = field(default_factory=list)
    trained_at: str = field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds"))

    @property
    def beats_chance(self) -> Optional[bool]:
        if np.isnan(self.null_p_value):
            return None
        return self.null_p_value < 0.05


def purged_splits(asof: np.ndarray, n_splits: int, embargo_days: int,
                  min_train: int = 50):
    """Expanding-window splits over time-sorted samples; training is purged to end
    `embargo_days` before each test block begins (the label-horizon embargo)."""
    order = np.argsort(asof, kind="stable")
    n = len(order)
    bounds = np.linspace(0, n, n_splits + 2, dtype=int)
    embargo = np.timedelta64(int(embargo_days), "D")
    for k in range(1, n_splits + 1):
        test_pos = order[bounds[k]:bounds[k + 1]]
        if len(test_pos) == 0:
            continue
        test_start = asof[test_pos].min()
        train_pos = order[:bounds[k]]
        train_pos = train_pos[asof[train_pos] <= (test_start - embargo)]
        if len(train_pos) >= min_train:
            yield train_pos, test_pos


def _oos_predictions(X, y, asof, model_name, n_splits, embargo_days):
    """Aggregate held-out predictions across all purged folds."""
    Xv, yv = X.values, y.values
    rows = []
    aucs = []
    for tr, te in purged_splits(asof, n_splits, embargo_days):
        if len(np.unique(yv[tr])) < 2:
            continue
        m = build_model(model_name)
        m.fit(Xv[tr], yv[tr])
        p = m.predict_proba(Xv[te])[:, 1]
        rows.append(pd.DataFrame({"idx": te, "score": p, "y": yv[te]}))
        if len(np.unique(yv[te])) == 2:
            aucs.append(roc_auc_score(yv[te], p))
    if not rows:
        return pd.DataFrame(), []
    return pd.concat(rows, ignore_index=True), aucs


def evaluate(dataset, *, model_name: str = "Gradient Boosting", horizon: int = 21,
             n_splits: int = 5, embargo_days: Optional[int] = None,
             threshold: float = 0.5, n_null: int = 20) -> ModelResult:
    X, y, meta = dataset.X, dataset.y, dataset.meta
    embargo_days = embargo_days if embargo_days is not None else horizon + 2
    res = ModelResult(model_name=model_name, horizon=horizon, n_samples=int(len(X)),
                      class_balance=float(y.mean()) if len(y) else float("nan"),
                      feature_names=list(X.columns), threshold=threshold)

    if len(X) < 80 or y.nunique() < 2:
        res.warnings.append(
            f"Insufficient data for reliable evaluation (n={len(X)}, classes={y.nunique()}). "
            "Metrics are omitted rather than reported on too few samples.")
        if len(X) and y.nunique() == 2:           # still fit a final model for selection
            res.model = build_model(model_name).fit(X.values, y.values)
        return res

    asof = pd.to_datetime(meta["asof"]).values.astype("datetime64[D]")

    oos, cv_aucs = _oos_predictions(X, y, asof, model_name, n_splits, embargo_days)
    if oos.empty:
        res.warnings.append("Purged CV produced no usable folds (timeline too short for the "
                            "chosen horizon/splits). Try a shorter horizon or fewer splits.")
        res.model = build_model(model_name).fit(X.values, y.values)
        return res

    yt, sc = oos["y"].values, oos["score"].values
    res.cv_aucs = [float(a) for a in cv_aucs]
    res.oos_auc = float(roc_auc_score(yt, sc)) if len(np.unique(yt)) == 2 else float("nan")
    res.oos_pr_auc = float(average_precision_score(yt, sc)) if len(np.unique(yt)) == 2 else float("nan")
    res.oos_acc = float(((sc >= threshold).astype(int) == yt).mean())
    res.brier = float(brier_score_loss(yt, sc)) if len(np.unique(yt)) == 2 else float("nan")

    # attach meta + forward returns for the OOS scatter / IC
    oos_meta = meta.iloc[oos["idx"].values].reset_index(drop=True)
    res.oos_frame = pd.DataFrame({
        "asof": oos_meta["asof"].values, "ticker": oos_meta["ticker"].values,
        "score": sc, "y": yt, "fwd_excess": oos_meta["fwd_excess"].values})
    res.oos_ic = float(pd.Series(sc).corr(pd.Series(res.oos_frame["fwd_excess"]), method="spearman"))

    if len(np.unique(yt)) == 2:
        fpr, tpr, _ = roc_curve(yt, sc)
        res.roc = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        try:
            pt, pp = calibration_curve(yt, sc, n_bins=8, strategy="quantile")
            res.calibration = {"prob_true": pt.tolist(), "prob_pred": pp.tolist()}
        except Exception:
            pass
        res.confusion = confusion_matrix(yt, (sc >= threshold).astype(int)).tolist()

    # ---- shuffled-label null band (the honesty centerpiece) ---------------- #
    if n_null and len(np.unique(yt)) == 2:
        rng = np.random.default_rng(0)
        nulls = []
        for _ in range(n_null):
            yp = pd.Series(rng.permutation(y.values), index=y.index)
            o, _ = _oos_predictions(X, yp, asof, model_name, n_splits, embargo_days)
            if not o.empty and o["y"].nunique() == 2:
                nulls.append(roc_auc_score(o["y"].values, o["score"].values))
        res.null_aucs = [float(a) for a in nulls]
        if nulls and not np.isnan(res.oos_auc):
            res.null_p_value = float((np.sum(np.array(nulls) >= res.oos_auc) + 1) / (len(nulls) + 1))

    # ---- permutation importance on the most recent held-out block ---------- #
    splits = list(purged_splits(asof, n_splits, embargo_days))
    if splits:
        tr, te = splits[-1]
        if len(np.unique(y.values[tr])) == 2:
            m = build_model(model_name).fit(X.values[tr], y.values[tr])
            try:
                pi = permutation_importance(m, X.values[te], y.values[te],
                                            n_repeats=10, random_state=0, scoring="roc_auc")
                res.importances = (pd.DataFrame({"feature": X.columns,
                                                 "importance": pi.importances_mean,
                                                 "std": pi.importances_std})
                                   .sort_values("importance", ascending=False)
                                   .reset_index(drop=True))
            except Exception as exc:
                res.warnings.append(f"permutation importance unavailable: {exc}")

    # final model on ALL data for live selection
    res.model = build_model(model_name).fit(X.values, y.values)

    if not np.isnan(res.oos_auc) and res.oos_auc < 0.55 and res.beats_chance is not True:
        res.warnings.append(
            "Out-of-sample AUC is near 0.5 and not separable from the shuffled-label null — "
            "on this data the model is not finding a generalizable edge. Treat the rankings as "
            "exploratory, not as validated signals.")
    return res


def select_stocks(result: ModelResult, current: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Score the current cross-section and return a ranked selection."""
    if result.model is None or current is None or current.empty:
        return pd.DataFrame(columns=["ticker", "score"])
    feats = current[result.feature_names].astype(float).fillna(0.0).values
    score = result.model.predict_proba(feats)[:, 1]
    out = current[["ticker"]].copy()
    out["score"] = score
    # surface the strongest raw drivers per pick for interpretability
    for c in ["cong_net_buys_90d", "cong_dollar_30d", "insider_net_30d",
              "wsb_mentions_chg", "ret_63d"]:
        if c in current:
            out[c] = current[c].values
    return out.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)
