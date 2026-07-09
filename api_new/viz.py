"""
viz.py — interactive Plotly figures on a single, deliberate visual identity.

Palette (ink-navy / amber / muted-teal — chosen to avoid the default dark-dashboard-with-
neon-accent cliche and to read like a financial print/terminal):
    ink      #0B1220   surface #16202E   line #243244
    text     #E6EDF3   muted   #8FA3B8
    amber    #E0A23A   (signal / picks)   teal #4FB0A5 (model / OOS)
    pos      #3FB17F   neg     #D9534F     null #5B6B7E

The signature figure is `perf_vs_null` — observed OOS AUC against the shuffled-label null —
because it answers the only question that matters for a screener: is this edge real?
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

INK = "#0B1220"; SURFACE = "#16202E"; LINE = "#243244"
TEXT = "#E6EDF3"; MUTED = "#8FA3B8"
AMBER = "#E0A23A"; TEAL = "#4FB0A5"; POS = "#3FB17F"; NEG = "#D9534F"; NULL = "#5B6B7E"

_tmpl = go.layout.Template()
_tmpl.layout = go.Layout(
    paper_bgcolor=INK, plot_bgcolor=INK,
    font=dict(family="Inter, system-ui, sans-serif", color=TEXT, size=13),
    title=dict(font=dict(family="Space Grotesk, Inter, sans-serif", size=17, color=TEXT)),
    colorway=[AMBER, TEAL, POS, NEG, MUTED, "#C77DFF"],
    xaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE, color=MUTED),
    yaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE, color=MUTED),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
    margin=dict(l=56, r=24, t=56, b=48),
)
pio.templates["quiver"] = _tmpl
pio.templates.default = "quiver"


def _empty(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(color=MUTED, size=14))
    fig.update_layout(template="quiver", xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def perf_vs_null(result) -> go.Figure:
    """SIGNATURE: observed out-of-sample AUC vs the shuffled-label null distribution."""
    if not result.null_aucs or np.isnan(result.oos_auc):
        return _empty("Null band needs a completed CV run with both classes present.")
    nulls = np.array(result.null_aucs)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=nulls, nbinsx=18, marker_color=NULL, opacity=0.85,
                               name="Shuffled-label null"))
    fig.add_vline(x=0.5, line=dict(color=MUTED, dash="dot"),
                  annotation_text="chance", annotation_position="top")
    fig.add_vline(x=float(result.oos_auc), line=dict(color=AMBER, width=3),
                  annotation_text=f"observed {result.oos_auc:.3f}",
                  annotation_position="top right", annotation_font_color=AMBER)
    verdict = ("beats the null (p<0.05)" if result.beats_chance
               else "not separable from chance")
    fig.update_layout(title=f"Does the edge survive? — {verdict}  ·  p = {result.null_p_value:.3f}",
                      xaxis_title="ROC-AUC", yaxis_title="null runs", bargap=0.05,
                      showlegend=False, height=360)
    return fig


def roc(result) -> go.Figure:
    if not result.roc:
        return _empty("No ROC — needs both classes in the OOS set.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.roc["fpr"], y=result.roc["tpr"], mode="lines",
                             line=dict(color=TEAL, width=2.5),
                             name=f"OOS ROC (AUC {result.oos_auc:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(color=MUTED, dash="dash"), name="chance"))
    fig.update_layout(title="Out-of-sample ROC", xaxis_title="false positive rate",
                      yaxis_title="true positive rate", height=340,
                      xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]))
    return fig


def calibration(result) -> go.Figure:
    c = result.calibration
    if not c:
        return _empty("No calibration curve available.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(color=MUTED, dash="dash"), name="perfect"))
    fig.add_trace(go.Scatter(x=c["prob_pred"], y=c["prob_true"], mode="lines+markers",
                             line=dict(color=AMBER, width=2.5), name="model"))
    fig.update_layout(title="Calibration (reliability)", xaxis_title="predicted P(outperform)",
                      yaxis_title="observed frequency", height=340,
                      xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]))
    return fig


def cv_spread(result) -> go.Figure:
    if not result.cv_aucs:
        return _empty("No per-fold scores.")
    folds = list(range(1, len(result.cv_aucs) + 1))
    fig = go.Figure(go.Bar(x=folds, y=result.cv_aucs, marker_color=TEAL,
                           text=[f"{a:.2f}" for a in result.cv_aucs], textposition="outside"))
    fig.add_hline(y=0.5, line=dict(color=MUTED, dash="dot"))
    if result.null_aucs:
        lo, hi = np.percentile(result.null_aucs, [5, 95])
        fig.add_hrect(y0=lo, y1=hi, fillcolor=NULL, opacity=0.18, line_width=0,
                      annotation_text="null 5–95%", annotation_position="top left")
    fig.update_layout(title="Per-fold OOS AUC (walk-forward)", xaxis_title="fold",
                      yaxis_title="AUC", height=340, yaxis=dict(range=[0.3, 1.0]))
    return fig


def importance(result) -> go.Figure:
    imp = result.importances
    if imp is None or imp.empty:
        return _empty("No importance scores.")
    d = imp.iloc[::-1]
    fig = go.Figure(go.Bar(y=d["feature"], x=d["importance"], orientation="h",
                           marker_color=AMBER,
                           error_x=dict(type="data", array=d["std"], color=MUTED)))
    fig.update_layout(title="Permutation importance (OOS AUC drop)",
                      xaxis_title="mean importance", height=460)
    return fig


def score_vs_forward(result) -> go.Figure:
    df = result.oos_frame
    if df is None or df.empty:
        return _empty("No OOS predictions to plot.")
    colors = np.where(df["fwd_excess"] > 0, POS, NEG)
    fig = go.Figure(go.Scatter(x=df["score"], y=df["fwd_excess"], mode="markers",
                               marker=dict(color=colors, size=6, opacity=0.55,
                                           line=dict(width=0)),
                               text=df["ticker"]))
    if len(df) > 3:
        z = np.polyfit(df["score"], df["fwd_excess"], 1)
        xs = np.linspace(df["score"].min(), df["score"].max(), 50)
        fig.add_trace(go.Scatter(x=xs, y=np.polyval(z, xs), mode="lines",
                                 line=dict(color=AMBER, width=2), name="fit"))
    fig.add_hline(y=0, line=dict(color=MUTED, dash="dot"))
    fig.update_layout(title=f"OOS score vs realized forward excess return  ·  IC {result.oos_ic:+.3f}",
                      xaxis_title="predicted score", yaxis_title="forward excess return",
                      height=380, showlegend=False)
    return fig


def prob_hist(result) -> go.Figure:
    df = result.oos_frame
    if df is None or df.empty:
        return _empty("No predictions yet.")
    fig = go.Figure()
    for label, color, name in [(1, POS, "outperformed"), (0, NEG, "underperformed")]:
        s = df[df["y"] == label]["score"]
        if not s.empty:
            fig.add_trace(go.Histogram(x=s, nbinsx=20, opacity=0.6,
                                       marker_color=color, name=name))
    fig.update_layout(title="OOS predicted-score distribution by outcome", barmode="overlay",
                      xaxis_title="predicted P(outperform)", yaxis_title="count", height=340)
    return fig


def top_picks(sel: pd.DataFrame) -> go.Figure:
    if sel is None or sel.empty:
        return _empty("No selection yet — run the screen.")
    d = sel.iloc[::-1]
    fig = go.Figure(go.Bar(y=d["ticker"], x=d["score"], orientation="h",
                           marker=dict(color=d["score"], colorscale=[[0, NULL], [1, AMBER]],
                                       showscale=False),
                           text=[f"{s:.2f}" for s in d["score"]], textposition="outside"))
    fig.update_layout(title="Current selection — score = P(outperform)",
                      xaxis_title="score", height=max(320, 26 * len(d)),
                      xaxis=dict(range=[0, 1]))
    return fig


def confusion(result) -> go.Figure:
    cm = result.confusion
    if not cm:
        return _empty("No confusion matrix.")
    cm = np.array(cm)
    fig = go.Figure(go.Heatmap(
        z=cm, x=["pred under", "pred out"], y=["under", "out"],
        text=cm, texttemplate="%{text}", colorscale=[[0, SURFACE], [1, TEAL]],
        showscale=False))
    fig.update_layout(title=f"Confusion @ threshold {result.threshold:.2f}", height=320)
    return fig


def congress_flow(cong: pd.DataFrame, ticker: Optional[str] = None) -> go.Figure:
    if cong is None or cong.empty:
        return _empty("No congressional rows.")
    d = cong if not ticker else cong[cong["ticker"] == ticker]
    if d.empty:
        return _empty(f"No congressional trades for {ticker}.")
    g = (d.assign(week=pd.to_datetime(d["date"]).dt.to_period("W").dt.start_time)
           .groupby(["week", "side"])["amount"].sum().reset_index())
    fig = go.Figure()
    for side, color, name in [(1, POS, "buys"), (-1, NEG, "sells")]:
        s = g[g["side"] == side]
        fig.add_trace(go.Bar(x=s["week"], y=s["amount"] * side, marker_color=color, name=name))
    fig.update_layout(title=f"Congressional flow{' · ' + ticker if ticker else ''}",
                      barmode="relative", xaxis_title="week",
                      yaxis_title="signed $ disclosed", height=320)
    return fig
