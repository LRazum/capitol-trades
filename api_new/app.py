"""
app.py — Quiver Signals (investor view, hardened for shipping).

Finance-focused front end over the validated ML engine. Model selection, purged
walk-forward validation, the shuffle test and out-of-sample scoring all run silently
under the hood; the interface speaks in ranked names, plain-English reasons, and an
honest Signal Quality rating.

Ship-readiness in this build:
  • Input validation (ticker parsing/dedupe/caps) with clear feedback
  • Specific, actionable errors (auth expired -> back to key screen; network/yfinance/
    empty-universe each get their own message + fix hint)
  • Results only recompute when you press "Find signals" (widget changes just show a hint)
  • Sample mode auto-runs on first load — instant working demo
  • Staged progress (“pulling data → prices → features → training & stress-testing”)
  • Per-tab error boundaries: one broken view can't take down the app
  • Source availability + dropped-ticker reporting, “data as of …” caption
  • Versioned cache keys so old cache files can never crash a new build

    ./run.sh            (creates venv, installs, health-checks, launches)
    # or: python -m pip install -U -r requirements.txt && streamlit run app.py

Research tool — not investment advice.
"""
from __future__ import annotations

import re
import traceback
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import features as F
import ml as ML
import viz as V
from cache import JoblibCache, fingerprint
from quiver_client import QuiverClient, QuiverError

st.set_page_config(page_title="Quiver Signals", layout="wide", page_icon="📊",
                   initial_sidebar_state="expanded")

# ---- fixed engine configuration (never user-facing) ------------------------ #
MODEL_NAME = "Gradient Boosting"
N_SPLITS = 5
N_NULL = 20
THRESHOLD = 0.5
LOOKBACK_DAYS = 500
HORIZONS = {"Short-term — next ~2 weeks": 10,
            "Medium-term — next ~1 month": 21,
            "Long-term — next ~3 months": 63}
TOP_N = 10
MAX_TICKERS = 60
CACHE_VERSION = "v2"          # bump when cached object shapes change
FEEDS_TTL = 12 * 3600

_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


# --------------------------------------------------------------------------- #
# Errors with a user message and a fix hint
# --------------------------------------------------------------------------- #
class AppError(Exception):
    def __init__(self, msg: str, hint: str = ""):
        super().__init__(msg)
        self.hint = hint


class AuthExpired(AppError):
    pass


# --------------------------------------------------------------------------- #
# Visual identity (ink-navy / amber / muted-teal; native theme in config.toml)
# --------------------------------------------------------------------------- #
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root{--ink:#0B1220;--surface:#16202E;--line:#243244;--text:#E6EDF3;--muted:#8FA3B8;--amber:#E0A23A;--teal:#4FB0A5;--pos:#3FB17F;--neg:#D9534F;}
h1,h2,h3{font-family:'Space Grotesk',Inter,sans-serif!important;letter-spacing:-0.01em;}
code,[data-testid="stMetricValue"]{font-family:'IBM Plex Mono',monospace!important;}
[data-testid="stMetric"]{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:14px 16px;}
[data-testid="stMetricValue"]{color:var(--amber);}
.stTabs [data-baseweb="tab-list"]{gap:6px;border-bottom:1px solid var(--line);}
.stTabs [data-baseweb="tab"]{color:var(--muted);font-weight:500;}
.stTabs [aria-selected="true"]{color:var(--amber)!important;}
.gate{max-width:520px;margin:6vh auto 0;background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:34px;}
.tag{display:inline-block;font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--ink);background:var(--amber);border-radius:6px;padding:2px 8px;margin-bottom:14px;}
.small,.muted{color:var(--muted);font-size:13px;}
.badge{display:inline-block;padding:6px 16px;border-radius:999px;font-weight:600;font-family:'IBM Plex Mono',monospace;font-size:14px;}
.rel-high{background:rgba(63,177,127,.15);color:var(--pos);border:1px solid var(--pos);}
.rel-med{background:rgba(224,162,58,.15);color:var(--amber);border:1px solid var(--amber);}
.rel-low{background:rgba(217,83,79,.15);color:var(--neg);border:1px solid var(--neg);}
.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:14px 18px;margin-bottom:10px;}
.pick{display:flex;justify-content:space-between;align-items:center;}
.tk{font-size:18px;font-weight:600;}
.score{font-family:'IBM Plex Mono',monospace;font-size:22px;font-weight:600;}
.banner{border-radius:12px;padding:14px 18px;margin:4px 0 16px;}
.banner-low{background:rgba(217,83,79,.10);border:1px solid rgba(217,83,79,.45);}
.banner-high{background:rgba(63,177,127,.09);border:1px solid rgba(63,177,127,.4);}
.banner-med{background:rgba(224,162,58,.09);border:1px solid rgba(224,162,58,.4);}
hr{border-color:var(--line);}
</style>
""", unsafe_allow_html=True)

CACHE = JoblibCache(".cache")
for k, v in {"authed": False, "api_key": "", "auth_scheme": "Token",
             "mode": None, "req": None}.items():
    st.session_state.setdefault(k, v)


# --------------------------------------------------------------------------- #
# Pure helpers (no Streamlit calls — unit-testable)
# --------------------------------------------------------------------------- #
def parse_tickers(raw: str) -> tuple[list[str], list[str], bool]:
    """'$nvda, aapl; msft msft' -> (['NVDA','AAPL','MSFT'], rejected, truncated)."""
    if not raw or not raw.strip():
        return [], [], False
    parts = re.split(r"[,\s;]+", raw.strip())
    ok, rejected, seen = [], [], set()
    for p in parts:
        t = p.strip().lstrip("$").upper()
        if not t:
            continue
        if _TICKER_RE.match(t):
            if t not in seen:
                seen.add(t); ok.append(t)
        else:
            rejected.append(p.strip())
    truncated = len(ok) > MAX_TICKERS
    return ok[:MAX_TICKERS], rejected, truncated


def _money(x: float) -> str:
    x = float(x)
    if abs(x) >= 1e6:
        return f"${x/1e6:.1f}M"
    if abs(x) >= 1e3:
        return f"${x/1e3:.0f}K"
    return f"${x:.0f}"


def reliability_rating(r) -> tuple[str, str, str, str]:
    """(label, css_class, headline, plain-English detail) from the real OOS metrics."""
    if r is None or r.n_samples == 0 or r.oos_auc != r.oos_auc:
        return ("Insufficient data", "rel-low", "Not enough history to judge",
                "There isn't enough disclosure history yet to tell whether these signals are "
                "reliable. Treat everything here as exploratory only.")
    auc, p = r.oos_auc, r.null_p_value
    beats = (p == p) and p < 0.05
    if beats and auc >= 0.57:
        return ("High", "rel-high", "Signals look dependable on recent data",
                "Tested only on data the model never trained on, the rankings clearly beat a "
                "random baseline. Higher-conviction framing is warranted — though no signal is "
                "ever a guarantee.")
    if (beats and auc >= 0.53) or ((p == p) and p < 0.20 and auc >= 0.54):
        return ("Medium", "rel-med", "Signals are suggestive, not conclusive",
                "The model shows a modest edge over random selection on unseen data, but not a "
                "strong one. Use these as one input among several, not a standalone trigger.")
    return ("Low", "rel-low", "Signals are weak on current data",
            "On unseen data, these rankings are not clearly better than picking names at "
            "random. Use them as research leads — not as buy or sell decisions.")


def framing_for(label: str) -> str:
    return {"High": "actionable", "Medium": "soft"}.get(label, "research")


def rank_all(result, current: pd.DataFrame) -> pd.DataFrame:
    if result is None or result.model is None or current is None or current.empty:
        return pd.DataFrame()
    feats = current[result.feature_names].astype(float).fillna(0.0).values
    score = result.model.predict_proba(feats)[:, 1]
    df = current.copy()
    df["score"] = score
    df["signal"] = (score * 100).round().astype(int)
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def _drivers(row: pd.Series) -> list[str]:
    g = lambda c: float(row.get(c, 0) or 0)
    out = []
    cb90 = g("cong_net_buys_90d"); npol = int(g("cong_n_politicians_90d"))
    cd30 = g("cong_dollar_30d"); dsc = g("days_since_cong")
    if cb90 >= 2:
        who = f" across {npol} member{'s' if npol != 1 else ''}" if npol else ""
        out.append(f"Strong congressional buying — net {int(cb90)} purchases{who} in the last 90 days.")
    elif cb90 <= -2:
        out.append(f"Congressional selling pressure — net {int(abs(cb90))} sales in the last 90 days.")
    if cd30 >= 50_000:
        out.append(f"Sizable disclosed congressional purchases (~{_money(cd30)}) in the last 30 days.")
    elif cd30 <= -50_000:
        out.append(f"Notable disclosed congressional sales (~{_money(abs(cd30))}) in the last 30 days.")
    if 0 < dsc <= 14:
        out.append(f"Very recent congressional activity ({int(dsc)} days ago).")
    ins = g("insider_net_30d"); insd = g("insider_dollar_90d")
    if ins > 0 or insd > 0:
        out.append("Insider accumulation — net company-insider buying detected.")
    elif ins < 0 or insd < 0:
        out.append("Insider distribution — net company-insider selling detected.")
    if g("govcon_dollar_90d") > 0:
        out.append(f"Recent federal contract awards (~{_money(g('govcon_dollar_90d'))}).")
    if g("lobby_dollar_90d") > 0:
        out.append(f"Active lobbying spend (~{_money(g('lobby_dollar_90d'))}).")
    if g("wsb_mentions_chg") > 5:
        out.append("Rising retail attention — Reddit/WSB mentions up week-over-week.")
    elif g("wsb_mentions_chg") < -5:
        out.append("Cooling retail attention — Reddit/WSB mentions down week-over-week.")
    if g("ret_63d") > 0.03:
        out.append("Positive 3-month price trend.")
    elif g("ret_63d") < -0.03:
        out.append("Negative 3-month price trend.")
    if g("dist_ma200") > 0:
        out.append("Trading above its 200-day average (longer-term uptrend).")
    elif g("dist_ma200") < -0.05:
        out.append("Trading well below its 200-day average (longer-term downtrend).")
    if not out:
        out.append("No standout single driver — ranked on a blend of weaker factors.")
    return out


def top_driver(row: pd.Series) -> str:
    return _drivers(row)[0]


def accuracy_bar(result) -> go.Figure:
    acc = result.oos_acc * 100 if (result and result.oos_acc == result.oos_acc) else 50.0
    fig = go.Figure()
    fig.add_trace(go.Bar(y=["Random (coin flip)", "This model (unseen data)"], x=[50, acc],
                         orientation="h",
                         marker_color=[V.NULL, V.POS if acc >= 53 else V.AMBER if acc >= 51 else V.NEG],
                         text=[f"{50:.0f}%", f"{acc:.0f}%"], textposition="outside"))
    fig.add_vline(x=50, line=dict(color=V.MUTED, dash="dot"))
    fig.update_layout(title="How often the model was right on data it had never seen",
                      xaxis_title="directional accuracy", xaxis=dict(range=[0, 100]),
                      height=240, showlegend=False, margin=dict(l=10, r=20, t=50, b=40))
    return fig


def hist_months(dataset) -> int:
    if dataset is None or dataset.meta is None or dataset.meta.empty:
        return 0
    a = pd.to_datetime(dataset.meta["asof"])
    return max(1, round((a.max() - a.min()).days / 30))


def classify_exception(exc: Exception) -> AppError:
    """Map arbitrary failures to a user-facing message + fix hint."""
    if isinstance(exc, AppError):
        return exc
    s = str(exc)
    if isinstance(exc, ImportError) and "yfinance" in s:
        return AppError("Price data library (yfinance) isn't installed.",
                        "Run ./run.sh, or: python -m pip install -U yfinance")
    if "401" in s or "403" in s:
        return AuthExpired("Your Quiver API key was rejected.",
                           "It may have expired — you'll be taken back to the connect screen.")
    if any(w in s.lower() for w in ("connection", "timeout", "network", "resolve", "ssl")):
        return AppError("Couldn't reach the data services (network problem).",
                        "Check your internet connection and press Find signals again. "
                        "Sample mode works fully offline.")
    return AppError(f"Unexpected error: {s}",
                    "Press Find signals to retry, tick 'Refresh latest data' in Settings, "
                    "or see details below.")


# --------------------------------------------------------------------------- #
# Startup gate — API key required first
# --------------------------------------------------------------------------- #
def gate() -> None:
    st.markdown('<div class="gate">', unsafe_allow_html=True)
    st.markdown('<span class="tag">QUIVER SIGNALS</span>', unsafe_allow_html=True)
    st.markdown("# Connect your Quiver account")
    st.markdown('<p class="small">Enter your Quiver API key to load live political, insider and '
                'market-flow data. The key is checked once and kept only for this session.</p>',
                unsafe_allow_html=True)
    if st.session_state.pop("auth_expired_msg", None):
        st.warning("Your session's API key stopped working (expired or revoked). "
                   "Please reconnect.")
    key = st.text_input("API key", type="password", label_visibility="collapsed",
                        placeholder="paste your Quiver token")
    scheme = st.radio("Connection type", ["Token", "Bearer"], horizontal=True,
                      help="Most accounts use 'Token'. Switch to 'Bearer' if the key is rejected.")
    c1, c2 = st.columns(2)
    if c1.button("Connect", type="primary", use_container_width=True):
        if not key.strip():
            st.error("Enter a key, or try the sample data below.")
        else:
            with st.spinner("Checking your key…"):
                try:
                    ok, msg = QuiverClient(key.strip(), auth_scheme=scheme).validate_key()
                except QuiverError as exc:
                    ok, msg = False, str(exc)
            if ok:
                st.session_state.update(authed=True, api_key=key.strip(),
                                        auth_scheme=scheme, mode="live", req=None)
                st.rerun()
            else:
                st.error(msg)
    if c2.button("Explore with sample data", use_container_width=True):
        st.session_state.update(authed=True, mode="demo", req=None)
        st.rerun()
    st.markdown('<p class="small">Sample mode uses illustrative data to preview the dashboard — '
                'no key or internet needed.</p></div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Pipeline (cached, versioned; ML config fixed and hidden)
# --------------------------------------------------------------------------- #
def _cached_stage(name: str, producer, progress, label: str, *, ttl=None, force=False):
    age = CACHE.age_seconds(name)
    fresh = (age is not None) and (ttl is None or age < ttl) and not force
    progress(f"{label} — loaded from saved analysis ({age/60:.0f} min old)." if fresh
             else f"{label}…")
    return CACHE.memoize(name, producer, ttl=ttl, force=force)


def _pull_live(tickers: list[str]) -> tuple[dict, dict, dict]:
    cl = QuiverClient(st.session_state.api_key, auth_scheme=st.session_state.auth_scheme)
    raw, ok_src, failed_src, auth_fails = {}, [], [], 0
    for key in ("congress", "insiders", "gov_contracts", "lobbying", "wsb", "offexchange"):
        try:
            raw[key] = cl.fetch(key)
            ok_src.append(key)
        except QuiverError as exc:
            raw[key] = pd.DataFrame()
            failed_src.append(key)
            if "401" in str(exc) or "403" in str(exc):
                auth_fails += 1
    if failed_src and auth_fails == len(failed_src) and not ok_src:
        raise AuthExpired("Your Quiver API key was rejected on every data source.",
                          "It may have expired — you'll be taken back to the connect screen.")

    cong = raw.get("congress", pd.DataFrame())
    if tickers:
        uni = tickers
    elif not cong.empty and "Ticker" in cong:
        uni = cong["Ticker"].astype(str).str.upper().value_counts().head(40).index.tolist()
    else:
        raise AppError("Couldn't build a universe automatically (congressional feed was empty).",
                       "Type the tickers you're interested in and press Find signals.")

    prices = F.YFPriceProvider().get(uni, date.today() - timedelta(days=LOOKBACK_DAYS),
                                     date.today())
    priced = [t for t in uni if t in prices and len(prices[t].dropna()) >= 60]
    if F.BENCHMARK not in prices or prices[F.BENCHMARK].dropna().empty:
        raise AppError("Couldn't download market benchmark prices.",
                       "Price service may be blocked or offline — check connectivity and retry.")
    if not priced:
        raise AppError("None of the requested tickers returned usable price history.",
                       "Check the symbols (US-listed tickers work best) and try again.")
    dropped = sorted(set(uni) - set(priced))
    prices = {t: s for t, s in prices.items() if t in priced or t == F.BENCHMARK}
    info = {"sources_ok": ok_src, "sources_failed": failed_src,
            "requested": uni, "priced": priced, "dropped": dropped}
    return raw, prices, info


def analyze(mode: str, tickers: list[str], horizon: int, force: bool, progress):
    if mode == "demo":
        def demo_pull():
            prov = F.SyntheticProvider()
            return prov.feeds(), prov.prices(), {"sources_ok": ["sample data"],
                                                 "sources_failed": [], "requested": [],
                                                 "priced": [], "dropped": []}
        raw, prices, info = _cached_stage(f"feeds_demo_{CACHE_VERSION}", demo_pull, progress,
                                          "Loading sample data", force=force)
    else:
        fp = fingerprint(CACHE_VERSION, "feeds_live", date.today().isoformat(), tickers)
        raw, prices, info = _cached_stage(f"feeds_{fp}", lambda: _pull_live(tickers), progress,
                                          "Pulling political, insider & market-flow data",
                                          ttl=FEEDS_TTL, force=force)

    progress("Reading the disclosures…")
    feeds = F.normalize_feeds(raw)

    ds_fp = fingerprint(CACHE_VERSION, "ds", mode, sorted(prices.keys()), horizon,
                        {k: len(v) for k, v in feeds.items()})
    dataset = _cached_stage(f"dataset_{ds_fp}",
                            lambda: F.build_dataset(feeds, prices, horizon=horizon),
                            progress, "Building point-in-time features", force=force)

    res_fp = fingerprint(CACHE_VERSION, "res", ds_fp, MODEL_NAME, horizon,
                         N_SPLITS, N_NULL, THRESHOLD)
    result = _cached_stage(
        f"result_{res_fp}",
        lambda: ML.evaluate(dataset, model_name=MODEL_NAME, horizon=horizon,
                            n_splits=N_SPLITS, n_null=N_NULL, threshold=THRESHOLD),
        progress, "Training the model & stress-testing it on unseen periods", force=force)

    progress("Ranking today's names…")
    return dataset, result, info


# --------------------------------------------------------------------------- #
# Tab renderers (each wrapped in an error boundary at the call site)
# --------------------------------------------------------------------------- #
def render_pick_card(row, color):
    chips = " &nbsp;·&nbsp; ".join(_drivers(row)[:2])
    st.markdown(
        f'<div class="card"><div class="pick">'
        f'<div><span class="tk">{row["ticker"]}</span>'
        f'<div class="muted">{chips}</div></div>'
        f'<div class="score" style="color:{color}">{int(row["signal"])}</div>'
        f'</div></div>', unsafe_allow_html=True)


def tab_top_picks(ranked, label, framing):
    if ranked.empty:
        st.info("No names could be ranked. Add tickers in the sidebar and press **Find signals**.")
        return
    buys = ranked.head(TOP_N)
    avoid = ranked.tail(min(TOP_N, max(0, len(ranked) - TOP_N))).iloc[::-1].reset_index(drop=True)

    if framing == "research":
        st.markdown('<div class="banner banner-low"><b>Signal quality is currently Low.</b> '
                    'The names below are the model\'s highest- and lowest-ranked on the latest '
                    'data, but they are <b>not statistically distinguishable from random</b> right '
                    'now — so they are research leads, <b>not buy or sell calls</b>.</div>',
                    unsafe_allow_html=True)
        st.subheader("Highest-ranked by the model")
        for _, r in buys.iterrows():
            render_pick_card(r, "var(--amber)")
        if not avoid.empty:
            with st.expander("Lowest-ranked names"):
                for _, r in avoid.iterrows():
                    render_pick_card(r, "var(--muted)")
    else:
        if framing == "actionable":
            st.markdown('<div class="banner banner-high"><b>Signal quality is High.</b> '
                        'The model\'s rankings are beating a random baseline on unseen data, so '
                        'these are presented as higher-conviction ideas. Still your call, never a '
                        'guarantee.</div>', unsafe_allow_html=True)
            buy_h, avoid_h = "✅ Higher-conviction candidates", "⛔ Weakest signals — consider avoiding"
        else:
            st.markdown('<div class="banner banner-med"><b>Signal quality is Medium.</b> '
                        'A modest edge over random — treat these as leans, not triggers.</div>',
                        unsafe_allow_html=True)
            buy_h, avoid_h = "Model leans toward", "Model leans away from"
        c1, c2 = st.columns(2)
        with c1:
            st.subheader(buy_h)
            for _, r in buys.iterrows():
                render_pick_card(r, "var(--pos)")
        with c2:
            st.subheader(avoid_h)
            for _, r in avoid.iterrows():
                render_pick_card(r, "var(--neg)")

    with st.expander("Full ranked list"):
        disp = ranked[["ticker", "signal"]].copy()
        disp.insert(0, "rank", range(1, len(disp) + 1))
        disp["top reason"] = [top_driver(r) for _, r in ranked.iterrows()]
        disp = disp.rename(columns={"ticker": "Ticker", "signal": "Signal Score"})
        st.dataframe(disp, use_container_width=True, hide_index=True)
        st.download_button("Download as CSV", disp.to_csv(index=False).encode(),
                           "quiver_signals.csv", "text/csv")


def tab_deep_dive(ranked, dataset):
    if ranked.empty:
        st.info("Run a screen first.")
        return
    st.markdown('<p class="small">Why the model ranks each name where it does — read straight '
                'from the underlying Quiver data, no math required.</p>', unsafe_allow_html=True)
    names = ranked["ticker"].tolist()
    pick = st.selectbox("Choose a name", names, index=0)
    row = ranked[ranked["ticker"] == pick].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Signal Score", int(row["signal"]))
    c2.metric("Rank", f"{names.index(pick) + 1} of {len(names)}")
    c3.metric("Congressional net (90d)", int(float(row.get("cong_net_buys_90d", 0) or 0)))
    st.subheader("What's driving it")
    for d in _drivers(row):
        st.markdown(f"- {d}")
    cong = dataset.feeds.get("congress", pd.DataFrame())
    if not cong.empty and pick in set(cong["ticker"]):
        st.subheader("Congressional buying vs selling")
        st.plotly_chart(V.congress_flow(cong, pick), use_container_width=True)


def tab_reliability(result, dataset, label, css, headline, detail):
    st.markdown(f'<span class="badge {css}">Signal Quality: {label}</span>', unsafe_allow_html=True)
    st.markdown(f"### {headline}")
    st.markdown(f'<p class="muted">{detail}</p>', unsafe_allow_html=True)
    st.plotly_chart(accuracy_bar(result), use_container_width=True)

    beats = (result.null_p_value == result.null_p_value) and result.null_p_value < 0.05
    rnd = ("**did clearly beat** that random baseline" if beats
           else "**did not clearly beat** that random baseline")
    n_names = len(set(dataset.meta['ticker'])) if not dataset.meta.empty else 0
    st.markdown(f"""
**Tested the honest way.** Every result above is measured only on time periods the model was
*not* trained on — so it reflects how the signals would have behaved in real time, not a
pattern fitted to the past after the fact.

**Checked against luck.** We re-ran the whole analysis many times on deliberately scrambled
history, to see how often the model "finds" a pattern when there is none. On the current data,
the real signal {rnd}.

**How much history we have.** These signals are built on about **{hist_months(dataset)} months**
of disclosure data across **{n_names}** names. Short histories make any edge genuinely hard to
confirm — a Low rating often just means "not enough evidence yet," not "definitely worthless."

**What the rating means for you.**
- **High** — reasonable to treat top names as higher-conviction ideas.
- **Medium** — useful as one input; pair with your own research.
- **Low** — use as a watchlist/idea generator only; don't trade on it alone.
""")


def tab_methodology():
    st.markdown("""
### How Quiver Signals works

**The data.** Each name is scored from signals that often move *before* the broader market
notices them: trades disclosed by members of Congress, company-insider (Form 4) buying and
selling, federal contract awards, lobbying activity, retail attention on Reddit, and the
stock's own recent price trend.

**The model.** A machine-learning model studies how those signals lined up with what stocks
did *next*, historically, and uses that to rank today's names by how likely each is to
outperform the market over your chosen horizon. The result is the **Signal Score** (0–100) —
a *ranking of relative attractiveness*, not a price target or a promise.

**Why you can trust the rating (and when you shouldn't).** The model is only ever graded on
periods it never saw while learning, and its results are compared against deliberately
scrambled data — so it can't fool us by memorizing the past. When that grading shows a real
edge, the **Signal Quality** badge reads High; when it doesn't, it reads Low and the dashboard
deliberately holds back buy/sell language. The honesty is built in: the tool tells you when
*not* to rely on it.

**Limitations worth knowing.** Disclosure data is reported with delays and only reaches back a
limited time, so edges are often weak or unconfirmable — a Low rating is common and expected.
Markets also change; a signal that worked last quarter may not work next quarter.

---
*Quiver Signals is a research and idea-generation tool. It is not investment advice and not a
recommendation to buy or sell any security. Signal Scores are model rankings, not forecasts or
guarantees. Do your own research and consider consulting a licensed financial professional.*
""")


def safe_render(fn, *args) -> None:
    """Per-tab error boundary: a broken view degrades gracefully."""
    try:
        fn(*args)
    except Exception:
        st.error("This view hit a snag — the rest of the app is unaffected.")
        with st.expander("Technical details (for bug reports)"):
            st.code(traceback.format_exc())


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    mode = st.session_state.mode
    st.title("📊 Quiver Signals")
    st.markdown('<p class="small">Where the smart-money and political-flow data points next — '
                'in plain English. Research tool, not investment advice.</p>',
                unsafe_allow_html=True)

    # ---- sidebar: two inputs + a run button ---- #
    with st.sidebar:
        st.header("Your screen")
        tickers_raw = st.text_input(
            "Stocks to analyze",
            value="", disabled=(mode == "demo"),
            help="Comma-separated tickers, e.g. NVDA, AAPL, MSFT. Leave blank to scan the most "
                 "politically-active names automatically.",
            placeholder="sample mode uses a fixed universe" if mode == "demo"
                        else "e.g. NVDA, AAPL, MSFT — or leave blank")
        tickers, rejected, truncated = parse_tickers(tickers_raw)
        if rejected:
            st.warning(f"Ignored (not valid tickers): {', '.join(rejected[:6])}"
                       + ("…" if len(rejected) > 6 else ""))
        if truncated:
            st.warning(f"Using the first {MAX_TICKERS} tickers to keep the analysis fast.")
        horizon_label = st.radio("Investment horizon", list(HORIZONS.keys()), index=1)
        with st.expander("Settings"):
            force = st.checkbox("Refresh latest data",
                                help="Ignore saved analyses and pull everything fresh.")
        run = st.button("🔎 Find signals", type="primary", use_container_width=True)
        st.divider()
        if st.button("Sign out", use_container_width=True):
            for kk, vv in {"authed": False, "api_key": "", "mode": None, "req": None}.items():
                st.session_state[kk] = vv
            st.rerun()
        st.caption(("Sample data — illustrative only" if mode == "demo" else "Live Quiver data")
                   + f" · cache: {len(CACHE.entries())} saved analyses")

    # ---- capture the request ONLY on button press (widget changes don't recompute) ---- #
    if run:
        st.session_state.req = {"tickers": tickers, "horizon_label": horizon_label,
                                "force": force, "mode": mode}
    elif st.session_state.req is None and mode == "demo":
        # instant gratification: sample mode auto-runs with defaults
        st.session_state.req = {"tickers": [], "horizon_label": list(HORIZONS)[1],
                                "force": False, "mode": mode}

    req = st.session_state.req
    if req is None:
        st.info("👈 Enter the stocks you're interested in and a horizon, then press "
                "**Find signals**.")
        return
    if (req["tickers"] != tickers or req["horizon_label"] != horizon_label) and not run:
        st.caption("⚙️ Settings changed — press **Find signals** to update the results below.")

    horizon = HORIZONS[req["horizon_label"]]

    # ---- run the (cached) pipeline with staged progress and typed errors ---- #
    try:
        with st.status("Analyzing…", expanded=True) as status:
            dataset, result, info = analyze(req["mode"], req["tickers"], horizon,
                                            req["force"], progress=status.write)
            status.update(label="Analysis complete", state="complete", expanded=False)
        if req["force"]:
            st.session_state.req = {**req, "force": False}   # force applies once
        ranked = rank_all(result, dataset.current)
    except Exception as exc:
        err = classify_exception(exc)
        if isinstance(err, AuthExpired):
            st.session_state.update(authed=False, api_key="", req=None,
                                    auth_expired_msg=True)
            st.rerun()
        st.error(str(err))
        if err.hint:
            st.info(err.hint)
        if not isinstance(exc, AppError):
            with st.expander("Technical details (for bug reports)"):
                st.code(traceback.format_exc())
        return

    label, css, headline, detail = reliability_rating(result)
    framing = framing_for(label)

    # ---- headline strip + data provenance ---- #
    m = st.columns(4)
    m[0].markdown(f'<div style="padding-top:6px"><span class="badge {css}">Signal Quality: '
                  f'{label}</span></div>', unsafe_allow_html=True)
    m[1].metric("Names analyzed", len(ranked))
    if not ranked.empty:
        m[2].metric("Top name", ranked.iloc[0]["ticker"], f'score {int(ranked.iloc[0]["signal"])}')
    m[3].metric("Horizon", req["horizon_label"].split(" — ")[0])

    asof = (pd.to_datetime(dataset.meta["asof"]).max().date()
            if not dataset.meta.empty else date.today())
    prov_bits = [f"Data as of {asof}"]
    if info.get("sources_ok"):
        prov_bits.append("sources: " + ", ".join(info["sources_ok"]))
    if info.get("sources_failed"):
        prov_bits.append("unavailable: " + ", ".join(info["sources_failed"]))
    st.caption(" · ".join(prov_bits))
    if info.get("dropped"):
        st.caption(f"⚠️ No usable price history for: {', '.join(info['dropped'])} — "
                   "analyzed without them.")
    for w in getattr(result, "warnings", []) or []:
        if "Insufficient" in w:
            st.caption("ℹ️ Limited history for this universe/horizon — ratings default to "
                       "cautious until more data accumulates.")

    if label in ("Low", "Insufficient data"):
        st.markdown('<div class="banner banner-low" style="margin-top:8px">Heads-up: on the '
                    'current data these signals aren\'t beating random selection. The dashboard '
                    'below stays in research mode — ideas to investigate, not trades to place.'
                    '</div>', unsafe_allow_html=True)

    t1, t2, t3, t4 = st.tabs(["📊 Top Picks & Signals", "🔍 Deep Dive",
                              "🛡️ Risk & Reliability", "📓 Methodology"])
    with t1:
        safe_render(tab_top_picks, ranked, label, framing)
    with t2:
        safe_render(tab_deep_dive, ranked, dataset)
    with t3:
        safe_render(tab_reliability, result, dataset, label, css, headline, detail)
    with t4:
        safe_render(tab_methodology)


# --------------------------------------------------------------------------- #
def _in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if _in_streamlit():
    if not st.session_state.authed:
        gate()
    else:
        main()
