# Copyright (c) 2026 Lovro Razum. All rights reserved.
# PROPRIETARY AND CONFIDENTIAL. Unauthorized use, copying, modification, or
# distribution of this file, in whole or in part, via any medium, is strictly
# prohibited without the prior written permission of Lovro Razum.
# See the accompanying LICENSE file for full terms.
"""
Capitol Trades Dashboard — Streamlit web app
=============================================
Interactive dashboard that scrapes recent "buy" trades by US politicians from
capitoltrades.com, filters them and layers on
technical analysis, interactive Plotly charts, and two scikit-learn ML insights
(a short-term price-trend forecast and a volume-anomaly detector) powered by
yfinance historical data.

Run with:
    streamlit run app.py

Requirements (install once):
    pip install streamlit requests beautifulsoup4 pandas numpy yfinance plotly scikit-learn
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
BASE_URL = "https://www.capitoltrades.com/trades"
HEADERS = {"User-Agent": "Mozilla/5.0 (CapitolTradesDashboard/1.0)"}

# A coherent, intentional palette reused across every chart so the whole app
# reads as one design rather than Plotly defaults.
COLOR_PRICE = "#1E40AF"     # deep indigo  — close price
COLOR_SMA = "#D97706"       # amber        — moving average
COLOR_FORECAST = "#059669"  # emerald      — ML trend forecast
COLOR_ANOMALY = "#DC2626"   # red          — flagged anomalies
COLOR_VOLUME = "#94A3B8"    # slate        — volume bars
COLOR_RSI = "#7C3AED"       # violet       — RSI line

# Default ticker universe (approx. ~400 names). Editable at runtime in
# the sidebar; the working copy lives in st.session_state.
DEFAULT_TICKERS = {
    "A", "AAL", "AAP", "AAPL", "ABBV", "ABNB", "ABT", "ACN", "ADBE", "ADI", "ADM", "ADP",
    "ADSK", "AEE", "AEP", "AES", "AFL", "AIG", "AIZ", "AJG", "AKAM", "ALB", "ALGN", "ALL",
    "ALLE", "ALLY", "AMAT", "AMC", "AMCR", "AMD", "AME", "AMGN", "AMP", "AMT", "AMZN",
    "ANET", "ANSS", "AON", "AOS", "APA", "APD", "APH", "APTV", "ARE", "ARM", "ATO", "AVB",
    "AVGO", "AVY", "AWK", "AXON", "AXP", "AZO", "BA", "BABA", "BAC", "BALL", "BAX", "BBWI",
    "BBY", "BDX", "BEN", "BG", "BIIB", "BIO", "BK", "BKNG", "BKR", "BLK", "BMY", "BR",
    "BRK.B", "BRO", "BSX", "BWA", "BX", "BXP", "C", "CAG", "CAH", "CARR", "CAT", "CB",
    "CBOE", "CBRE", "CCI", "CCL", "CDNS", "CDW", "CE", "CEG", "CF", "CFG", "CHD", "CHRW",
    "CHTR", "CI", "CINF", "CL", "CLX", "CMA", "CMCSA", "CME", "CMG", "CMI", "CMS", "CNC",
    "CNP", "COF", "COIN", "COO", "COP", "COR", "COST", "CPB", "CPRT", "CRL", "CRM", "CRWD",
    "CSCO", "CSGP", "CSX", "CTAS", "CTRA", "CTSH", "CTVA", "CVS", "CVX", "CZR", "D", "DAL",
    "DD", "DE", "DELL", "DFS", "DG", "DGX", "DHI", "DHR", "DIS", "DLR", "DLTR", "DOC",
    "DOV", "DOW", "DPZ", "DRI", "DTE", "DUK", "DVA", "DVN", "DXCM", "EA", "EBAY", "ECL",
    "ED", "EFX", "EG", "EIX", "EL", "ELV", "EMN", "EMR", "ENPH", "EOG", "EPAM", "EQIX",
    "EQR", "EQT", "ES", "ESS", "ETN", "ETR", "ETSY", "EVRG", "EW", "EXC", "EXPD", "EXPE",
    "EXR", "F", "FANG", "FAST", "FCX", "FDS", "FDX", "FE", "FFIV", "FI", "FICO", "FIS",
    "FITB", "FMC", "FOX", "FOXA", "FRT", "FSLR", "FTNT", "FTV", "GD", "GE", "GEHC", "GEN",
    "GEV", "GILD", "GIS", "GL", "GLW", "GM", "GNRC", "GOOG", "GOOGL", "GPC", "GPN", "GRMN",
    "GS", "GWW", "HAL", "HAS", "HBAN", "HCA", "HD", "HES", "HIG", "HII", "HLT", "HOLX",
    "HON", "HPE", "HPQ", "HRL", "HSIC", "HST", "HSY", "HUBB", "HUM", "HWM", "IBM", "ICE",
    "IDXX", "IEX", "IFF", "ILMN", "INCY", "INTC", "INTU", "INVH", "IP", "IPG", "IQV", "IR",
    "IRM", "ISRG", "IT", "ITW", "IVZ", "J", "JBHT", "JBL", "JCI", "JKHY", "JNJ", "JNPR",
    "JPM", "K", "KDP", "KEY", "KEYS", "KHC", "KIM", "KKR", "KLAC", "KMB", "KMI", "KMX",
    "KO", "KR", "KVUE", "L", "LDOS", "LEN", "LH", "LHX", "LIN", "LKQ", "LLY", "LMT", "LNT",
    "LOW", "LRCX", "LULU", "LUV", "LVS", "LW", "LYB", "LYV", "MA", "MAA", "MAR", "MAS",
    "MCD", "MCHP", "MCK", "MCO", "MDLZ", "MDT", "MET", "META", "MGM", "MHK", "MKC", "MKTX",
    "MLM", "MMC", "MMM", "MNST", "MO", "MOH", "MOS", "MPC", "MPWR", "MRK", "MRNA", "MRO",
    "MS", "MSCI", "MSFT", "MSI", "MTB", "MTCH", "MTD", "MU", "NCLH", "NDAQ", "NDSN", "NEE",
    "NEM", "NFLX", "NI", "NKE", "NOC", "NOW", "NRG", "NSC", "NTAP", "NTRS", "NUE", "NVDA",
    "NVR", "NWS", "NWSA", "NXPI", "O", "ODFL", "OKE", "OMC", "ON", "ORCL", "ORLY", "OTIS",
    "OXY", "PANW", "PARA", "PAYC", "PAYX", "PCAR", "PCG", "PEG", "PEP", "PFE", "PFG", "PG",
    "PGR", "PH", "PHM", "PKG", "PLD", "PLTR", "PM", "PNC", "PNR", "PNW", "POOL", "PPG",
    "PPL", "PRU", "PSA", "PSX", "PTC", "PWR", "PYPL", "QCOM", "QRVO", "RCL", "REG", "REGN",
    "RF", "RJF", "RL", "RMD", "ROK", "ROL", "ROP", "ROST", "RSG", "RTX", "RVTY", "SBAC",
    "SBUX", "SCHW", "SHW", "SJM", "SLB", "SMCI", "SNA", "SNPS", "SO", "SOLV", "SPG", "SPGI",
    "SQ", "SRE", "STE", "STLD", "STT", "STX", "STZ", "SWK", "SWKS", "SYF", "SYK", "SYY",
    "T", "TAP", "TDG", "TDY", "TECH", "TEL", "TER", "TFC", "TFX", "TGT", "TJX", "TMO",
    "TMUS", "TPR", "TRGP", "TRMB", "TROW", "TRV", "TSCO", "TSLA", "TSN", "TT", "TTWO",
    "TXN", "TXT", "TYL", "UAL", "UBER", "UDR", "UHS", "ULTA", "UNH", "UNP", "UPS", "URI",
    "USB", "V", "VEEV", "VFC", "VICI", "VLO", "VLTO", "VMC", "VRSK", "VRSN", "VRTX", "VST",
    "VTR", "VTRS", "VZ", "WAB", "WAT", "WBA", "WBD", "WDC", "WEC", "WELL", "WFC", "WM",
    "WMB", "WMT", "WRB", "WST", "WTW", "WY", "WYNN", "XEL", "XOM", "XRAY", "XYL", "YUM",
    "ZBH", "ZBRA", "ZTS",
}

# Matches tickers in an issuer cell, e.g. "Apple Inc AAPL:US" -> "AAPL".
TICKER_RE = re.compile(r"\b([A-Z][A-Z0-9./\-]{0,9}):US\b")


# --------------------------------------------------------------------------- #
# Pure helpers (no Streamlit, no network) — kept independent so they are easy
# to reason about and unit-test in isolation.
# --------------------------------------------------------------------------- #
def norm(ticker: str) -> str:
    """Normalize a ticker for set membership: uppercase, strip punctuation."""
    return re.sub(r"[^A-Z0-9]", "", ticker.upper())


def to_yahoo(ticker: str) -> str:
    """Convert a Capitol ticker to Yahoo Finance format (BRK.B -> BRK-B)."""
    return ticker.strip().upper().replace(".", "-")


def parse_size(s: str) -> float:
    """Convert a trade-size range to a midpoint estimate.

    '15K-50K' -> 32500 ; '1M-5M' -> 3_000_000 ; '50M+' -> 50_000_000
    """
    s = s.replace("\u2013", "-").replace("\u2014", "-").replace(",", "").strip()

    def token(x: str) -> float:
        x = x.strip().upper().replace("+", "")
        if not x:
            return 0.0
        mult = 1.0
        if x.endswith("K"):
            mult, x = 1_000, x[:-1]
        elif x.endswith("M"):
            mult, x = 1_000_000, x[:-1]
        elif x.endswith("B"):
            mult, x = 1_000_000_000, x[:-1]
        try:
            return float(x) * mult
        except ValueError:
            return 0.0

    if "-" in s:
        low, high = s.split("-", 1)
        low, high = token(low), token(high)
        return (low + high) / 2 if high else low
    return token(s)


def parse_published(text: str, today: date):
    """Parse a 'published' cell ('Today', 'Yesterday', 'N days ago', or a date)."""
    t = text.strip().lower()
    if "today" in t:
        return today
    if "yesterday" in t:
        return today - timedelta(days=1)
    m = re.search(r"(\d+)\s*days?\s*ago", t)
    if m:
        return today - timedelta(days=int(m.group(1)))
    m = re.search(r"(\d{1,2}\s+[A-Za-z]+\s+\d{4})", text)
    if m:
        for fmt in ("%d %b %Y", "%d %B %Y"):
            try:
                return datetime.strptime(m.group(1), fmt).date()
            except ValueError:
                pass
    return None


def parse_date_token(text: str):
    """Parse an explicit 'DD Mon YYYY' date token, returning a date or None."""
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", text)
    if not m:
        return None
    for fmt in ("%d %b %Y", "%d %B %Y"):
        try:
            return datetime.strptime(" ".join(m.groups()), fmt).date()
        except ValueError:
            pass
    return None


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's exponential smoothing."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_indicators(hist: pd.DataFrame, sma_window: int, rsi_period: int) -> dict:
    """Compute SMA, RSI, volatility and period-return metrics from price history."""
    close = hist["Close"]
    returns = close.pct_change()
    sma_series = close.rolling(sma_window).mean()
    rsi_series = compute_rsi(close, rsi_period)

    daily_vol = float(returns.std()) if returns.notna().sum() > 1 else float("nan")
    latest_sma = float(sma_series.iloc[-1]) if not np.isnan(sma_series.iloc[-1]) else float("nan")
    latest_rsi = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else float("nan")

    return {
        "latest_close": float(close.iloc[-1]),
        "latest_sma": latest_sma,
        "latest_rsi": latest_rsi,
        "daily_vol": daily_vol,
        "ann_vol": daily_vol * np.sqrt(252) if not np.isnan(daily_vol) else float("nan"),
        "period_return": float(close.iloc[-1] / close.iloc[0] - 1),
        "sma_series": sma_series,
        "rsi_series": rsi_series,
    }


def monte_carlo_forecast(
    prices: pd.Series, horizon: int = 7, n_sims: int = 1000, window: int = 60
) -> dict | None:
    """Monte Carlo price simulation using Geometric Brownian Motion.

    Estimates drift (μ) and volatility (σ) from the last `window` sessions of
    log-returns, then simulates `n_sims` independent price paths forward.
    Returns p10/p50/p90 percentile bands, prob_up, and the anchor price S0.
    Returns None if there is too little data.
    """
    y = prices.dropna()
    if len(y) < 5:
        return None
    y_window = y.iloc[-min(window, len(y)):]
    log_returns = np.log(y_window / y_window.shift(1)).dropna()
    if len(log_returns) < 5:
        return None
    mu = float(log_returns.mean())
    sigma = float(log_returns.std())
    if sigma == 0:
        return None
    S0 = float(y.iloc[-1])

    rng = np.random.default_rng(42)
    # GBM increments: log(S_{t+1}/S_t) ~ N((μ - σ²/2)·dt, σ·√dt), dt=1 day
    shocks = rng.normal(mu - 0.5 * sigma ** 2, sigma, size=(n_sims, horizon))
    paths = S0 * np.exp(np.cumsum(shocks, axis=1))  # shape (n_sims, horizon)

    future_dates = pd.bdate_range(y.index[-1] + pd.Timedelta(days=1), periods=horizon)
    return {
        "dates": future_dates,
        "p10": np.percentile(paths, 10, axis=0),
        "p50": np.percentile(paths, 50, axis=0),
        "p90": np.percentile(paths, 90, axis=0),
        "prob_up": float((paths[:, -1] > S0).mean()),
        "S0": S0,
    }


def detect_volume_anomalies(volume: pd.Series, k: float = 3.0) -> np.ndarray:
    """Flag volume spikes using a one-sided MAD threshold on log-volume.

    Uses the Iglewicz & Hoaglin modified Z-score on log1p(volume). Only flags
    abnormally HIGH volume days (genuine spikes) — low-volume days like holidays
    are intentionally ignored.
    k: modified Z-score threshold (default 3.0 ≈ ~1% false-positive rate).
    """
    v = volume.fillna(0)
    if len(v) < 10:
        return np.zeros(len(v), dtype=bool)
    log_v = np.log1p(v.to_numpy().astype(float))
    median = np.median(log_v)
    mad = np.median(np.abs(log_v - median))
    if mad < 1e-8:
        return np.zeros(len(v), dtype=bool)
    modified_z = 0.6745 * (log_v - median) / mad
    return modified_z > k  # one-sided: spikes only


# --------------------------------------------------------------------------- #
# Network / scraping
# --------------------------------------------------------------------------- #
def _fetch_page(page: int, page_size: int) -> str:
    """Download a single Capitol Trades results page (raises on HTTP error)."""
    resp = requests.get(
        BASE_URL,
        params={"txType": "buy", "page": page, "pageSize": page_size},
        headers=HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.text


def _parse_rows(html: str, revolut_norm: set, today: date, cutoff: date) -> list:
    """Parse one results page into a list of qualifying trade dicts."""
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for tr in soup.select("table tbody tr"):
        cells = tr.find_all("td")
        if len(cells) < 9:
            continue
        if cells[6].get_text(" ", strip=True).lower() != "buy":
            continue

        published = parse_published(cells[2].get_text(" ", strip=True), today)
        if published is None or published < cutoff:
            continue

        issuer = cells[1].get_text(" ", strip=True)
        match = TICKER_RE.search(issuer)
        if not match or norm(match.group(1)) not in revolut_norm:
            continue

        size_str = cells[7].get_text(" ", strip=True)
        rows.append({
            "politician": cells[0].get_text(" ", strip=True),
            "issuer": issuer,
            "ticker": match.group(1),
            "published": published,
            "traded": parse_date_token(cells[3].get_text(" ", strip=True)),
            "owner": cells[5].get_text(" ", strip=True),
            "size_str": size_str,
            "size_num": parse_size(size_str),
            "price": cells[8].get_text(" ", strip=True),
        })
    return rows


@st.cache_data(ttl=900, show_spinner=False)
def scrape_trades(tickers: tuple, days_back: int = 7,
                  max_pages: int = 25, page_size: int = 96) -> pd.DataFrame:
    """Scrape recent buy trades, filter to `tickers`, return a sorted DataFrame.

    Sorted by estimated volume (descending), then traded date (ascending).
    Cached for 15 minutes; `tickers` is a hashable tuple so caching works.
    """
    revolut_norm = {norm(t) for t in tickers}
    today = date.today()
    cutoff = today - timedelta(days=days_back)

    rows, empty_streak = [], 0
    for page in range(1, max_pages + 1):
        page_rows = _parse_rows(_fetch_page(page, page_size), revolut_norm, today, cutoff)
        rows.extend(page_rows)
        if not page_rows:
            empty_streak += 1
            if empty_streak >= 2:  # two empty pages in a row -> assume we're past the window
                break
        else:
            empty_streak = 0

    df = pd.DataFrame(rows)
    if not df.empty:
        df["traded"] = pd.to_datetime(df["traded"])
        df["published"] = pd.to_datetime(df["published"])
        df = df.sort_values(
            by=["size_num", "traded"],
            ascending=[False, True],
            na_position="last",
        ).reset_index(drop=True)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_history(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """Download historical OHLCV data via yfinance. Cached for 1 hour.

    Raises ValueError on empty results so transient failures are NOT cached
    (Streamlit does not cache exceptions) and will be retried on the next run.
    """
    df = yf.Ticker(to_yahoo(ticker)).history(period=period, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError("no data returned")
    # Strip timezone so the index plays nicely with tz-naive forecast dates.
    df = df.copy()
    df.index = df.index.tz_localize(None)
    return df


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def build_price_chart(ticker, hist, ind, forecast, anomaly_mask, sma_window, rsi_period):
    """Three-row figure: price + SMA + forecast, volume (anomalies flagged), RSI."""
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.20, 0.25], vertical_spacing=0.045,
        subplot_titles=(
            f"{ticker} — close & {sma_window}-day SMA",
            "Volume",
            f"RSI ({rsi_period})",
        ),
    )
    close = hist["Close"]

    # Row 1 — price and moving average
    fig.add_trace(
        go.Scatter(x=close.index, y=close, name="Close",
                   line=dict(color=COLOR_PRICE, width=1.8)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=ind["sma_series"].index, y=ind["sma_series"], name=f"SMA{sma_window}",
                   line=dict(color=COLOR_SMA, width=1.5, dash="dot")),
        row=1, col=1,
    )

    # Row 1 — MC fan chart (p10–p90 band + p50 median), connected to last real close
    if forecast is not None:
        fdates = forecast["dates"]
        p10, p50, p90 = forecast["p10"], forecast["p50"], forecast["p90"]
        S0 = forecast["S0"]
        all_dates = [close.index[-1], *fdates]

        # Shaded band between p10 and p90
        fig.add_trace(
            go.Scatter(
                x=[*all_dates, *reversed(all_dates)],
                y=[S0, *p90, *reversed([S0, *p10])],
                fill="toself",
                fillcolor="rgba(5,150,105,0.12)",
                line=dict(width=0),
                name="MC 10–90%",
                showlegend=True,
            ),
            row=1, col=1,
        )
        # p50 median line
        fig.add_trace(
            go.Scatter(
                x=all_dates, y=[S0, *p50],
                name="MC median",
                mode="lines+markers",
                line=dict(color=COLOR_FORECAST, width=2, dash="dash"),
                marker=dict(size=5),
            ),
            row=1, col=1,
        )

    # Row 1 — anomaly markers on the price line
    if anomaly_mask is not None and anomaly_mask.any():
        fig.add_trace(
            go.Scatter(
                x=close.index[anomaly_mask], y=close[anomaly_mask],
                name="Volume anomaly", mode="markers",
                marker=dict(color=COLOR_ANOMALY, size=9, symbol="x"),
            ),
            row=1, col=1,
        )

    # Row 2 — volume (anomalous days highlighted)
    if anomaly_mask is not None:
        vol_colors = np.where(anomaly_mask, COLOR_ANOMALY, COLOR_VOLUME)
    else:
        vol_colors = COLOR_VOLUME
    fig.add_trace(
        go.Bar(x=hist.index, y=hist["Volume"], marker_color=vol_colors,
               name="Volume", showlegend=False),
        row=2, col=1,
    )

    # Row 3 — RSI with 30/70 reference bands
    fig.add_trace(
        go.Scatter(x=ind["rsi_series"].index, y=ind["rsi_series"], name="RSI",
                   line=dict(color=COLOR_RSI, width=1.5), showlegend=False),
        row=3, col=1,
    )
    fig.add_hline(y=70, line=dict(color=COLOR_ANOMALY, width=1, dash="dot"), row=3, col=1)
    fig.add_hline(y=30, line=dict(color=COLOR_FORECAST, width=1, dash="dot"), row=3, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)

    fig.update_layout(
        height=640, template="plotly_white", hovermode="x unified",
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
        bargap=0.1,
    )
    return fig


# --------------------------------------------------------------------------- #
# Sidebar — settings + ticker management
# --------------------------------------------------------------------------- #
def _cb_add_ticker():
    raw = st.session_state.add_ticker_input.strip().upper()
    if raw:
        st.session_state.revolut_tickers.add(raw)
    st.session_state.add_ticker_input = ""


def _cb_remove_tickers():
    for t in st.session_state.remove_select:
        st.session_state.revolut_tickers.discard(t)


def _cb_reset_tickers():
    st.session_state.revolut_tickers = set(DEFAULT_TICKERS)


def render_sidebar() -> dict:
    st.sidebar.title("Controls")

    st.sidebar.subheader("Scraping")
    days_back = st.sidebar.slider("Look-back window (days)", 1, 30, 7)
    top_n = st.sidebar.slider("Number of top trades", 5, 30, 10)

    st.sidebar.subheader("Analysis")
    period = st.sidebar.select_slider(
        "Price history", options=["1mo", "3mo", "6mo", "1y", "2y"], value="6mo"
    )
    sma_window = st.sidebar.slider("SMA window (days)", 10, 100, 50, step=5)
    rsi_period = st.sidebar.slider("RSI period (days)", 5, 30, 14)
    horizon = st.sidebar.slider("Forecast horizon (trading days)", 3, 15, 7)
    anomaly_k = st.sidebar.slider(
        "Anomaly sensitivity (σ)", 1.5, 5.0, 3.0, step=0.5,
        help="Modified Z-score threshold for volume spike detection. Higher = stricter, fewer flags.",
    )

    _render_ticker_manager()

    return {
        "days_back": days_back, "top_n": top_n, "period": period,
        "sma_window": sma_window, "rsi_period": rsi_period,
        "horizon": horizon, "anomaly_k": anomaly_k,
    }


def _render_ticker_manager():
    tickers = st.session_state.revolut_tickers
    st.sidebar.subheader(f"Tickers ({len(tickers)})")

    with st.sidebar.expander("Manage tickers"):
        col_in, col_btn = st.columns([2, 1])
        with col_in:
            st.text_input("Add a ticker", key="add_ticker_input",
                          placeholder="e.g. NVDA", label_visibility="collapsed")
        with col_btn:
            st.button("Add", on_click=_cb_add_ticker, use_container_width=True)

        st.multiselect("Remove tickers", options=sorted(tickers), key="remove_select",
                       placeholder="Select tickers to remove")
        st.button("Remove selected", on_click=_cb_remove_tickers, use_container_width=True)

        col_reset, col_dl = st.columns(2)
        with col_reset:
            st.button("Reset", on_click=_cb_reset_tickers, use_container_width=True)
        with col_dl:
            st.download_button("Export", "\n".join(sorted(tickers)),
                               file_name="revolut_tickers.txt", use_container_width=True)

        st.caption("Current universe:")
        st.caption(", ".join(sorted(tickers)) or "—")


# --------------------------------------------------------------------------- #
# Main view sections
# --------------------------------------------------------------------------- #
def render_summary_metrics(top: pd.DataFrame, full: pd.DataFrame):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Qualifying trades", len(full))
    c2.metric("Unique tickers (top)", top["ticker"].nunique())
    c3.metric("Est. volume (top)", f"${top['size_num'].sum():,.0f}")
    c4.metric("Largest single trade", f"${top['size_num'].max():,.0f}")


def render_trades_table(top: pd.DataFrame):
    disp = pd.DataFrame({
        "Ticker": top["ticker"],
        "Est. Size ($)": top["size_num"],
        "Size Range": top["size_str"],
        "Traded": top["traded"].dt.date,
        "Published": top["published"].dt.date,
        "Politician": top["politician"],
        "Issuer": top["issuer"],
        "Owner": top["owner"],
        "Price": top["price"],
    })
    st.dataframe(
        disp, hide_index=True, use_container_width=True,
        column_config={
            "Est. Size ($)": st.column_config.NumberColumn(format="$%.0f"),
            "Traded": st.column_config.DateColumn(format="YYYY-MM-DD"),
            "Published": st.column_config.DateColumn(format="YYYY-MM-DD"),
        },
    )


def render_indicator_summary(results: dict, settings: dict):
    rows = []
    for ticker, (status, payload) in results.items():
        if status != "ok":
            continue
        ind, forecast, anomalies = payload["ind"], payload["forecast"], payload["anomalies"]
        fpct = float("nan")
        if forecast is not None:
            fpct = (forecast["p50"][-1] / ind["latest_close"] - 1) * 100
        rsi = ind["latest_rsi"]
        signal = "Overbought" if rsi >= 70 else "Oversold" if rsi <= 30 else "Neutral"
        rows.append({
            "Ticker": ticker,
            "Last ($)": ind["latest_close"],
            f"SMA{settings['sma_window']} ($)": ind["latest_sma"],
            "RSI": rsi,
            "Signal": signal,
            "Daily vol (%)": ind["daily_vol"] * 100,
            "Ann. vol (%)": ind["ann_vol"] * 100,
            f"{settings['horizon']}d trend (%)": fpct,
            "Vol anomalies": int(anomalies.sum()),
        })
    if not rows:
        return
    st.markdown("**Indicator summary**")
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, use_container_width=True,
        column_config={
            "Last ($)": st.column_config.NumberColumn(format="$%.2f"),
            f"SMA{settings['sma_window']} ($)": st.column_config.NumberColumn(format="$%.2f"),
            "RSI": st.column_config.NumberColumn(format="%.1f"),
            "Daily vol (%)": st.column_config.NumberColumn(format="%.2f"),
            "Ann. vol (%)": st.column_config.NumberColumn(format="%.1f"),
            f"{settings['horizon']}d trend (%)": st.column_config.NumberColumn(format="%.2f"),
        },
    )


def render_ticker_detail(ticker: str, payload: dict, settings: dict):
    hist, ind = payload["hist"], payload["ind"]
    forecast, anomalies = payload["forecast"], payload["anomalies"]

    chart_col, insight_col = st.columns([3, 1])

    with chart_col:
        fig = build_price_chart(
            ticker, hist, ind, forecast, anomalies,
            settings["sma_window"], settings["rsi_period"],
        )
        st.plotly_chart(fig, use_container_width=True)

    with insight_col:
        st.metric("Last close", f"${ind['latest_close']:.2f}")

        if not np.isnan(ind["latest_sma"]):
            st.metric(
                f"SMA{settings['sma_window']}", f"${ind['latest_sma']:.2f}",
                delta=round(ind["latest_close"] - ind["latest_sma"], 2),
                help="Delta shows price relative to the moving average.",
            )
        else:
            st.metric(f"SMA{settings['sma_window']}", "n/a",
                      help="Not enough history for this SMA window.")

        if not np.isnan(ind["latest_rsi"]):
            st.metric("RSI", f"{ind['latest_rsi']:.1f}",
                      help="Above 70 = overbought, below 30 = oversold.")
        st.metric("Daily volatility", f"{ind['daily_vol'] * 100:.2f}%"
                  if not np.isnan(ind["daily_vol"]) else "n/a")

        st.markdown("**ML insights**")
        if forecast is not None:
            pct_med = (forecast["p50"][-1] / ind["latest_close"] - 1) * 100
            pct_p10 = (forecast["p10"][-1] / ind["latest_close"] - 1) * 100
            pct_p90 = (forecast["p90"][-1] / ind["latest_close"] - 1) * 100
            arrow = "📈" if pct_med >= 0 else "📉"
            st.write(f"{arrow} {settings['horizon']}-day median: **{pct_med:+.2f}%**")
            st.write(f"↑ Prob. up: **{forecast['prob_up']:.0%}**")
            st.caption(f"80% range: {pct_p10:+.1f}% → {pct_p90:+.1f}%")
        else:
            st.write("Trend: not enough data")
        st.write(f"🚩 Volume anomalies: **{int(anomalies.sum())}** day(s)")
        st.caption("Trend = naive linear extrapolation. Illustrative only.")


def render_analysis(tickers: list, settings: dict):
    # Fetch everything up front with a progress bar, then render.
    results = {}
    progress = st.progress(0.0, text="Fetching price history…")
    for i, ticker in enumerate(tickers):
        try:
            hist = fetch_history(ticker, settings["period"])
        except Exception as exc:  # noqa: BLE001 — surface any yfinance failure per ticker
            results[ticker] = ("error", str(exc))
        else:
            results[ticker] = ("ok", {
                "hist": hist,
                "ind": compute_indicators(hist, settings["sma_window"], settings["rsi_period"]),
                "forecast": monte_carlo_forecast(hist["Close"], settings["horizon"]),
                "anomalies": detect_volume_anomalies(hist["Volume"], settings["anomaly_k"]),
            })
        progress.progress((i + 1) / len(tickers), text=f"Fetched {ticker}")
    progress.empty()

    render_indicator_summary(results, settings)

    failed = [t for t, (s, _) in results.items() if s == "error"]
    if failed:
        st.warning(
            "Couldn't fetch price data for: " + ", ".join(failed)
            + ". Yahoo Finance may be rate-limiting or the symbol may be unavailable. "
            "Try again in a moment."
        )

    tabs = st.tabs([ticker for ticker in tickers])
    for tab, ticker in zip(tabs, tickers):
        with tab:
            status, payload = results[ticker]
            if status == "error":
                st.info(f"No chart for {ticker} — data fetch failed.")
                continue
            render_ticker_detail(ticker, payload, settings)


# --------------------------------------------------------------------------- #
# App entry point
# --------------------------------------------------------------------------- #
def render_app():
    st.set_page_config(page_title="Capitol Trades Dashboard", page_icon="🏛️", layout="wide")

    # Session state
    if "revolut_tickers" not in st.session_state:
        st.session_state.revolut_tickers = set(DEFAULT_TICKERS)
    if "trades_df" not in st.session_state:
        st.session_state.trades_df = None

    # Light cosmetic polish only — Streamlit owns the component styling.
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2.2rem;}
        [data-testid="stMetricValue"] {font-size: 1.25rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🏛️ Capitol Trades — Politician Buy Tracker")
    st.markdown(
        "Recent **buy** disclosures by US politicians from "
        "[capitoltrades.com](https://www.capitoltrades.com/trades), filtered to "
        "supported tickers and ranked by estimated trade size. Each top "
        "name is enriched with technical indicators, an interactive chart, and two "
        "scikit-learn signals — a short-term trend forecast and a volume-anomaly flag."
    )

    settings = render_sidebar()

    col_btn, col_caption = st.columns([1, 3])
    with col_btn:
        fetch = st.button("Fetch latest trades", type="primary", use_container_width=True)
    with col_caption:
        st.caption(
            f"Tracking {len(st.session_state.revolut_tickers)} tickers · "
            f"last {settings['days_back']} days · results cached for 15 min"
        )

    if fetch:
        with st.spinner("Scraping capitoltrades.com…"):
            try:
                st.session_state.trades_df = scrape_trades(
                    tuple(sorted(st.session_state.revolut_tickers)),
                    settings["days_back"],
                )
            except requests.RequestException as exc:
                st.session_state.trades_df = None
                st.error(f"Couldn't reach capitoltrades.com: {exc}")
            except Exception as exc:  # noqa: BLE001
                st.session_state.trades_df = None
                st.error(f"Scraping failed: {exc}")

    df = st.session_state.trades_df
    if df is None:
        st.info("Press **Fetch latest trades** to load the most recent buys.")
        return
    if df.empty:
        st.warning(
            "No qualifying buy trades in this window. Either no listed buys "
            "were filed, or the site's table structure changed and the parser needs "
            "an update."
        )
        return

    top = df.head(settings["top_n"]).copy()

    render_summary_metrics(top, df)

    st.subheader("📋 Top trades")
    render_trades_table(top)

    st.subheader("🔬 Technical & ML analysis")
    unique_tickers = list(dict.fromkeys(top["ticker"].tolist()))  # de-dupe, keep order
    render_analysis(unique_tickers, settings)

    st.divider()
    st.caption("Data: capitoltrades.com · Prices: Yahoo Finance · ML: illustrative only, not financial advice.")


if __name__ == "__main__":
    render_app()