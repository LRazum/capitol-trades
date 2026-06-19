"""
app_signals.py — Streamlit view over the daily actionable-signals output.

Run with:  streamlit run app_signals.py
Reads the parquet written by DailyScreener.run() (signals/actionable_latest.parquet) and
presents the high-conviction names — no heavy computation in the app; the pipeline does the
work on a cron and the dashboard just renders the result.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

SIGNALS_PATH = Path("signals/actionable_latest.parquet")

st.set_page_config(page_title="Capitol Trades — Actionable Signals", layout="wide")
st.title("Capitol Trades — Daily Actionable Signals")
st.caption(
    "Committee-relevant + insider-corroborated disclosures that clear the ML threshold. "
    "Research tool, not investment advice."
)


@st.cache_data(ttl=900)
def load_signals(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_parquet(path)


if not SIGNALS_PATH.exists():
    st.warning("No signals file yet. Run the daily screener (DailyScreener.run()) first.")
    st.stop()

mtime = SIGNALS_PATH.stat().st_mtime
df = load_signals(str(SIGNALS_PATH), mtime)
st.caption(f"Last updated: {datetime.fromtimestamp(mtime):%Y-%m-%d %H:%M}")

# --- sidebar filters ------------------------------------------------------- #
with st.sidebar:
    st.header("Filters")
    min_proba = st.slider("Min ML probability", 0.0, 1.0, 0.5, 0.05)
    politicians = sorted(df["politician"].dropna().unique()) if "politician" in df else []
    pick = st.multiselect("Politician", politicians, default=politicians)

view = df.copy()
if "ml_proba" in view:
    view = view[(view["ml_proba"].isna()) | (view["ml_proba"] >= min_proba)]
if pick and "politician" in view:
    view = view[view["politician"].isin(pick)]

# --- summary --------------------------------------------------------------- #
c1, c2, c3 = st.columns(3)
c1.metric("Actionable signals", len(view))
c2.metric("Unique tickers", view["ticker"].nunique() if "ticker" in view else 0)
if "ml_proba" in view and view["ml_proba"].notna().any():
    c3.metric("Median ML probability", f"{view['ml_proba'].median():.2f}")

# --- table ----------------------------------------------------------------- #
fmt = {}
if "ml_proba" in view:
    fmt["ml_proba"] = "{:.2%}"
if "signal_score" in view:
    fmt["signal_score"] = "{:.3f}"
if "amount_point" in view:
    fmt["amount_point"] = "${:,.0f}"

st.dataframe(
    view.style.format(fmt).background_gradient(
        subset=[c for c in ["ml_proba", "signal_score"] if c in view], cmap="Greens"
    ),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Download signals (CSV)",
    view.to_csv(index=False).encode(),
    file_name=f"actionable_signals_{datetime.now():%Y%m%d}.csv",
    mime="text/csv",
)
