#!/usr/bin/env python3
"""
capitol_buys.py — TOP 10 'buy' trgovina s capitoltrades.com (zadnjih 7 dana),
samo dionice dostupne na Revolutu, sortirano po volumenu DESC pa po datumu
izvršenja ASC.
"""
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.capitoltrades.com/trades"
HEADERS  = {"User-Agent": "Mozilla/5.0 (CapitolBuysReporter/1.0)"}
TOP_N    = 10

# Default popis Revolut tickera (aproks. ~400). Override stavljanjem
# revolut_tickers.txt (jedan ticker po retku) u isti folder.
DEFAULT_REVOLUT_TICKERS = {
    "A","AAL","AAP","AAPL","ABBV","ABNB","ABT","ACN","ADBE","ADI","ADM","ADP",
    "ADSK","AEE","AEP","AES","AFL","AIG","AIZ","AJG","AKAM","ALB","ALGN","ALL",
    "ALLE","ALLY","AMAT","AMC","AMCR","AMD","AME","AMGN","AMP","AMT","AMZN",
    "ANET","ANSS","AON","AOS","APA","APD","APH","APTV","ARE","ARM","ATO","AVB",
    "AVGO","AVY","AWK","AXON","AXP","AZO","BA","BABA","BAC","BALL","BAX","BBWI",
    "BBY","BDX","BEN","BG","BIIB","BIO","BK","BKNG","BKR","BLK","BMY","BR",
    "BRK.B","BRO","BSX","BWA","BX","BXP","C","CAG","CAH","CARR","CAT","CB",
    "CBOE","CBRE","CCI","CCL","CDNS","CDW","CE","CEG","CF","CFG","CHD","CHRW",
    "CHTR","CI","CINF","CL","CLX","CMA","CMCSA","CME","CMG","CMI","CMS","CNC",
    "CNP","COF","COIN","COO","COP","COR","COST","CPB","CPRT","CRL","CRM","CRWD",
    "CSCO","CSGP","CSX","CTAS","CTRA","CTSH","CTVA","CVS","CVX","CZR","D","DAL",
    "DD","DE","DELL","DFS","DG","DGX","DHI","DHR","DIS","DLR","DLTR","DOC",
    "DOV","DOW","DPZ","DRI","DTE","DUK","DVA","DVN","DXCM","EA","EBAY","ECL",
    "ED","EFX","EG","EIX","EL","ELV","EMN","EMR","ENPH","EOG","EPAM","EQIX",
    "EQR","EQT","ES","ESS","ETN","ETR","ETSY","EVRG","EW","EXC","EXPD","EXPE",
    "EXR","F","FANG","FAST","FCX","FDS","FDX","FE","FFIV","FI","FICO","FIS",
    "FITB","FMC","FOX","FOXA","FRT","FSLR","FTNT","FTV","GD","GE","GEHC","GEN",
    "GEV","GILD","GIS","GL","GLW","GM","GNRC","GOOG","GOOGL","GPC","GPN","GRMN",
    "GS","GWW","HAL","HAS","HBAN","HCA","HD","HES","HIG","HII","HLT","HOLX",
    "HON","HPE","HPQ","HRL","HSIC","HST","HSY","HUBB","HUM","HWM","IBM","ICE",
    "IDXX","IEX","IFF","ILMN","INCY","INTC","INTU","INVH","IP","IPG","IQV","IR",
    "IRM","ISRG","IT","ITW","IVZ","J","JBHT","JBL","JCI","JKHY","JNJ","JNPR",
    "JPM","K","KDP","KEY","KEYS","KHC","KIM","KKR","KLAC","KMB","KMI","KMX",
    "KO","KR","KVUE","L","LDOS","LEN","LH","LHX","LIN","LKQ","LLY","LMT","LNT",
    "LOW","LRCX","LULU","LUV","LVS","LW","LYB","LYV","MA","MAA","MAR","MAS",
    "MCD","MCHP","MCK","MCO","MDLZ","MDT","MET","META","MGM","MHK","MKC","MKTX",
    "MLM","MMC","MMM","MNST","MO","MOH","MOS","MPC","MPWR","MRK","MRNA","MRO",
    "MS","MSCI","MSFT","MSI","MTB","MTCH","MTD","MU","NCLH","NDAQ","NDSN","NEE",
    "NEM","NFLX","NI","NKE","NOC","NOW","NRG","NSC","NTAP","NTRS","NUE","NVDA",
    "NVR","NWS","NWSA","NXPI","O","ODFL","OKE","OMC","ON","ORCL","ORLY","OTIS",
    "OXY","PANW","PARA","PAYC","PAYX","PCAR","PCG","PEG","PEP","PFE","PFG","PG",
    "PGR","PH","PHM","PKG","PLD","PLTR","PM","PNC","PNR","PNW","POOL","PPG",
    "PPL","PRU","PSA","PSX","PTC","PWR","PYPL","QCOM","QRVO","RCL","REG","REGN",
    "RF","RJF","RL","RMD","ROK","ROL","ROP","ROST","RSG","RTX","RVTY","SBAC",
    "SBUX","SCHW","SHW","SJM","SLB","SMCI","SNA","SNPS","SO","SOLV","SPG","SPGI",
    "SQ","SRE","STE","STLD","STT","STX","STZ","SWK","SWKS","SYF","SYK","SYY",
    "T","TAP","TDG","TDY","TECH","TEL","TER","TFC","TFX","TGT","TJX","TMO",
    "TMUS","TPR","TRGP","TRMB","TROW","TRV","TSCO","TSLA","TSN","TT","TTWO",
    "TXN","TXT","TYL","UAL","UBER","UDR","UHS","ULTA","UNH","UNP","UPS","URI",
    "USB","V","VEEV","VFC","VICI","VLO","VLTO","VMC","VRSK","VRSN","VRTX","VST",
    "VTR","VTRS","VZ","WAB","WAT","WBA","WBD","WDC","WEC","WELL","WFC","WM",
    "WMB","WMT","WRB","WST","WTW","WY","WYNN","XEL","XOM","XRAY","XYL","YUM",
    "ZBH","ZBRA","ZTS",
}

def load_revolut_tickers() -> set:
    p = Path("revolut_tickers.txt")
    if p.exists():
        return {l.strip().upper() for l in p.read_text().splitlines()
                if l.strip() and not l.startswith("#")}
    return set(DEFAULT_REVOLUT_TICKERS)

REVOLUT_NORM = {re.sub(r"[^A-Z0-9]", "", t.upper()) for t in load_revolut_tickers()}

def norm(t: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", t.upper())

def parse_size(s: str) -> float:
    """'15K–50K' → 32500; '1M–5M' → 3_000_000; '50M+' → 50_000_000."""
    s = s.replace("–", "-").replace("—", "-").replace(",", "").strip()
    def tok(x: str) -> float:
        x = x.strip().upper().replace("+", "")
        if not x: return 0.0
        m = 1.0
        if x.endswith("K"): m, x = 1_000, x[:-1]
        elif x.endswith("M"): m, x = 1_000_000, x[:-1]
        elif x.endswith("B"): m, x = 1_000_000_000, x[:-1]
        try: return float(x) * m
        except ValueError: return 0.0
    if "-" in s:
        a, b = s.split("-", 1); a, b = tok(a), tok(b)
        return (a + b) / 2 if b else a
    return tok(s)

def parse_published(text: str, today: date):
    t = text.strip().lower()
    if "today" in t: return today
    if "yesterday" in t: return today - timedelta(days=1)
    m = re.search(r"(\d+)\s*days?\s*ago", t)
    if m: return today - timedelta(days=int(m.group(1)))
    m = re.search(r"(\d{1,2}\s+[A-Za-z]+\s+\d{4})", text)
    if m:
        for fmt in ("%d %b %Y", "%d %B %Y"):
            try: return datetime.strptime(m.group(1), fmt).date()
            except ValueError: pass
    return None

def parse_date_token(text: str):
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", text)
    if not m: return None
    for fmt in ("%d %b %Y", "%d %B %Y"):
        try: return datetime.strptime(" ".join(m.groups()), fmt).date()
        except ValueError: pass
    return None

TICKER_RE = re.compile(r"\b([A-Z][A-Z0-9./\-]{0,9}):US\b")

def fetch(page: int) -> str:
    r = requests.get(BASE_URL,
                     params={"txType": "buy", "page": page, "pageSize": 96},
                     headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.text

def parse_rows(html: str, today: date, cutoff: date):
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for row in soup.select("table tbody tr"):
        c = row.find_all("td")
        if len(c) < 9: continue
        if c[6].get_text(" ", strip=True).lower() != "buy": continue
        pub = parse_published(c[2].get_text(" ", strip=True), today)
        if pub is None or pub < cutoff: continue
        issuer = c[1].get_text(" ", strip=True)
        tm = TICKER_RE.search(issuer)
        if not tm or norm(tm.group(1)) not in REVOLUT_NORM: continue
        size_str = c[7].get_text(" ", strip=True)
        out.append({
            "politician": c[0].get_text(" ", strip=True),
            "issuer":     issuer,
            "ticker":     tm.group(1),
            "published":  pub,
            "traded":     parse_date_token(c[3].get_text(" ", strip=True)),
            "owner":      c[5].get_text(" ", strip=True),
            "size_str":   size_str,
            "size_num":   parse_size(size_str),
            "price":      c[8].get_text(" ", strip=True),
        })
    return out

def main():
    today  = date.today()
    cutoff = today - timedelta(days=7)
    rows, empty = [], 0
    for page in range(1, 25):
        page_rows = parse_rows(fetch(page), today, cutoff)
        rows.extend(page_rows)
        if not page_rows:
            empty += 1
            if empty >= 2: break
        else:
            empty = 0

    # 1) volumen DESC, 2) traded date ASC (najraniji prvi)
    rows.sort(key=lambda r: (-r["size_num"], r["traded"] or date.max))
    top = rows[:TOP_N]

    print(f"=== Capitol Trades — TOP {TOP_N} BUY (Revolut) — {today.isoformat()} ===")
    print(f"Razdoblje objave: {cutoff.isoformat()} → {today.isoformat()}")
    print(f"Kvalificiranih trgovina: {len(rows)}\n")
    for i, t in enumerate(top, 1):
        traded = t["traded"].isoformat() if t["traded"] else "?"
        print(f"{i:2}. {t['ticker']:<6} | size {t['size_str']:<12} "
              f"(~${t['size_num']:>11,.0f}) | traded {traded} | pub {t['published'].isoformat()}")
        print(f"     {t['politician']}")
        print(f"     {t['issuer']} | owner: {t['owner']} | price: {t['price']}\n")

if __name__ == "__main__":
    sys.exit(main())
