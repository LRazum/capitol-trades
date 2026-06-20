# Capitol Trades — Execution Guide (keyless / CSV edition)

Spin the whole pipeline up from a blank folder. **No API keys are required** — the
congressional-trade source is a local CSV scraped from capitoltrades.com, and every other
data source is a free public download. Phases run **in order**; each later phase assumes the
earlier ones produced their artifacts.

```
Phase 0  Environment + folders
Phase 1  Static data (Stewart-Woon committees, SEC Form 4, congress-legislators) -> tidy parquets
Phase 2  Historical backfill (capitoltrades CSV -> Bronze -> compact -> Silver)
Phase 3  Research / reality check (Gold -> event study + DSR/PBO -> save model)
Phase 4  Live execution (DailyScreener -> Streamlit dashboard)
```

Commands assume macOS/Linux `bash`. On Windows use `py -m venv .venv` and
`.venv\Scripts\activate`.

> **What changed vs the old guide:** Phases 2 and 4 no longer use Quiver. The congressional
> feed now comes from your scraped `capitol_trades_cache.csv` via the keyless
> `CapitolTradesCsvAdapter`. There is no `QUIVER_API_KEY` anywhere, and `OPENFIGI_API_KEY` is
> now the *only* optional key (it merely raises a rate limit). The keyless path adds no new
> Python dependencies — it relies on `pandas`, which is already required.

---

## Phase 0 — Environment & directory structure

```bash
mkdir capitol-trades && cd capitol-trades

# put the package + scripts + top-level files in place:
#   capitol_ingest/   (all the modules, INCLUDING the new capitol_trades_csv.py)
#   scripts/          (00..05)
#   config.py  app_signals.py  requirements.txt
#   capitol_trades_cache.csv   (your scraped buy feed; see Phase 2)

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Target layout** once everything is downloaded/built (note the two new/changed files marked
`<— keyless`):

```
capitol-trades/
├── capitol_ingest/                # the library (bronze, silver, gold, committee, insider, ...)
│   ├── __init__.py                #   (exports CapitolTradesCsvAdapter)            <— edited
│   ├── capitol_trades_csv.py      #   keyless congressional-trade adapter          <— NEW
│   ├── quiver.py                  #   (kept; no longer used by the scripts)
│   └── ...                        #   models, amounts, normalize, silver, gold, ...
├── scripts/
│   ├── 00_download_raw_data.py
│   ├── 01_build_committee.py
│   ├── 02_build_insider.py
│   ├── 03_backfill.py             #   reads the CSV, not Quiver                     <— patched
│   ├── 04_research.py
│   └── 05_run_screener.py         #   reads the CSV, not Quiver                     <— patched
├── config.py                      #   adds CAPITOL_TRADES_CSV + get_congress_adapter()  <— edited
├── capitol_trades_cache.csv       #   your scraped capitoltrades buy feed          <— NEW (default location)
├── data/
│   ├── raw/                       # downloads land here
│   │   ├── stewart_woon/          #   House + Senate committee assignment files
│   │   ├── sec_form4/             #   one subdir per quarter of unzipped TSVs
│   │   ├── legislators-current.yaml
│   │   └── legislators-historical.yaml
│   ├── bronze/                    # partitioned parquet (filing_year=/filing_month=)
│   └── processed/                 # committee_membership.parquet, name_to_icpsr.json,
│                                  #   insider_tidy.parquet, sectors.csv, secmaster cache
├── models/                        # screening_model.joblib, feature_columns.json
└── signals/                       # actionable_latest.parquet (the dashboard reads this)
```

**Exactly where the new/edited files go**

| File | Destination |
|---|---|
| `capitol_trades_csv.py` (new module) | `capitol-trades/capitol_ingest/` — beside `quiver.py`, `models.py`, etc. |
| `03_backfill.py` (patched) | `capitol-trades/scripts/` — overwrite the existing one |
| `05_run_screener.py` (patched) | `capitol-trades/scripts/` — overwrite the existing one |
| `config.py` edit | `capitol-trades/config.py` (project root) — add the path + factory |
| `__init__.py` edit | `capitol-trades/capitol_ingest/__init__.py` — register the adapter |
| `capitol_trades_cache.csv` | `capitol-trades/` (project root, the default) — or anywhere; point `CAPITOL_TRADES_CSV` at it |

`config.py` creates `data/`, `models/`, and `signals/` automatically on first import.

**The two small edits** (in case you haven't applied them yet):

```python
# config.py — add near your other paths, then a factory in the provider-factories section
CAPITOL_TRADES_CSV = os.getenv("CAPITOL_TRADES_CSV", str(ROOT / "capitol_trades_cache.csv"))

def get_congress_adapter():
    """Keyless congressional-trade source: the scraped capitoltrades CSV."""
    from capitol_ingest import CapitolTradesCsvAdapter
    return CapitolTradesCsvAdapter(CAPITOL_TRADES_CSV)
```

```python
# capitol_ingest/__init__.py — register the adapter next to the Quiver one
from .capitol_trades_csv import CapitolTradesCsvAdapter
# ...and add "CapitolTradesCsvAdapter" to __all__, beside "QuiverCongressAdapter".
```

**Smoke-test the install with zero real data** (uses synthetic fixtures):

```bash
python live_demo.py        # committee + insider + screening logic
python research_demo.py    # event study, DSR/PBO, costs, hold-out
```

**Credentials:** none required. The only optional one:

```bash
export OPENFIGI_API_KEY="your_openfigi_key"   # OPTIONAL; only raises the security-master rate limit
```

---

## Phase 1 — Static data acquisition & preprocessing

All of this is free. `scripts/00_download_raw_data.py` automates the parts that can be
scripted (congress-legislators + SEC Form 4) and *checks* for the part that can't
(Stewart-Woon on the Harvard Dataverse).

### 1a. Automated download (congress-legislators + SEC Form 4)

The SEC blocks requests without a descriptive User-Agent, so identify yourself:

```bash
python scripts/00_download_raw_data.py --years-back 5 --user-agent "Your Name your@email.com"
# congress-legislators YAML -> data/raw/   (tries main then master branch)
# SEC Form 4 quarterly zips  -> data/raw/sec_form4/YYYYqX/   (unzipped, zips removed)
```

You can also set `SEC_USER_AGENT` instead of `--user-agent`, and re-run with `--force` to
re-download. The script ends by reminding you whether Stewart-Woon is still missing.

### 1b. Stewart-Woon committee assignments (manual)

The Harvard Dataverse can't be scripted (Terms of Use + a JavaScript download UI), so grab the
files by hand from Charles Stewart's *congdata* collection
(`https://dataverse.harvard.edu/dataverse/congdata`). The assignments ship as **two files** —
House and Senate — so download both and drop them in `data/raw/stewart_woon/`:

```
data/raw/stewart_woon/house_assignments_1031151.xls
data/raw/stewart_woon/senate_assignments_1031151.xls
```

Then build the membership table + name→ICPSR crosswalk. `--assignments` takes **both files**
(it concatenates them); add `--codes` only if your files store committee codes rather than
names:

```bash
python scripts/01_build_committee.py \
  --assignments data/raw/stewart_woon/house_assignments_1031151.xls \
                data/raw/stewart_woon/senate_assignments_1031151.xls
# optional: --codes data/raw/stewart_woon/committee_codes.csv
```

This writes `data/processed/committee_membership.parquet` and `name_to_icpsr.json`, and prints
the distinct committee names found. **Check that list** against the jurisdiction map in
`capitol_ingest/committee.py` (it should contain `armed services`, `financial services`,
`energy and commerce`, etc.); the `committee_relevant` flag matches on those names.

### 1c. Build the insider table from SEC Form 4

`00_download_raw_data.py` already fetched and unzipped the quarters into
`data/raw/sec_form4/YYYYqX/` (each containing `SUBMISSION.tsv`, `REPORTINGOWNER.tsv`,
`NONDERIV_TRANS.tsv`). Build the tidy table:

```bash
python scripts/02_build_insider.py --quarters-dir data/raw/sec_form4
# -> data/processed/insider_tidy.parquet
```

### 1d. (optional but recommended) sector map for committee relevance

`committee_relevant` needs a ticker → GICS (sector, industry) map. Create
`data/processed/sectors.csv` with columns `ticker,sector,industry`. A quick keyless way is
yfinance for the tickers you care about:

```bash
python - <<'PY'
import yfinance as yf, pandas as pd, config
tickers = ["LMT","RTX","JPM","BAC","PFE","XOM","NVDA"]   # your universe
rows = []
for t in tickers:
    info = yf.Ticker(t).info
    rows.append({"ticker": t, "sector": info.get("sector"), "industry": info.get("industry")})
pd.DataFrame(rows).to_csv(config.SECTORS_CSV, index=False)
print("wrote", config.SECTORS_CSV)
PY
```

Without `sectors.csv`, the pipeline still runs but `committee_relevant` is always `False`.

---

## Phase 2 — Historical backfill (from your CSV)

This is the keyless heart of the change. Instead of pulling Quiver, the backfill reads your
scraped capitoltrades CSV, commits to Bronze year-by-year (resumable), compacts
cross-partition amendments, and reports Silver counts. The first run resolves each unique
ticker via OpenFIGI (keyless = rate-limited, then cached in
`data/processed/secmaster_cache.sqlite`), so it is slower than reruns.

### 2a. Place the CSV and point the pipeline at it

Drop your scraped file at the default location, or set the env var to wherever it lives:

```bash
# default: project root
cp /path/to/capitol_trades_cache.csv ./capitol_trades_cache.csv
# OR point at it explicitly:
export CAPITOL_TRADES_CSV=/path/to/capitol_trades_cache.csv
```

**Expected CSV columns** (case-insensitive headers; extra columns are preserved verbatim in
`raw_payload` and otherwise ignored):

| column | example | maps to |
|---|---|---|
| `politician` | `Ro Khanna Democrat House CA` | politician name + chamber (parsed) |
| `issuer` | `Visa Inc V:US` | issuer |
| `ticker` | `V` / `BRK/B` | ticker (verbatim) |
| `published` | `2026-06-11` | **filing date** (the only tradeable timestamp; drives the window) |
| `traded` | `2026-05-01` | transaction date (self-reported) |
| `owner` | `Self` / `Spouse` / `Child` / `Joint` | owner |
| `size_str` | `15K–50K` / `50M+` | STOCK Act bracket (expanded to a dollar range) |
| `size_num` | `32500` | bracket midpoint (used only as a fallback) |
| `price` | `$208.03` / `N/A` | carried through; not used by the pipeline |

### 2b. Run the backfill

```bash
python scripts/03_backfill.py --start 2016-01-01 --end "$(date +%F)"
```

Artifacts: `data/bronze/filing_year=YYYY/filing_month=MM/data.parquet`. Re-running is
idempotent (deterministic `source_record_id`); interrupted runs resume from
`data/bronze/.backfill_checkpoint.json`. In practice `--start` is moot below the CSV's reach —
the buy feed only goes back ~8 months, which is the real ceiling on sample depth (see Gotchas).

> **Direction caveat (important).** capitoltrades' buy feed has no buy/sell column, so the
> adapter stamps every row `Purchase`. That is correct for a pure buy feed. If your CSV ever
> mixes buys and sells, add a direction column and pass `txn_type_col="..."` to the adapter —
> otherwise sales are silently mislabelled as buys and the forward-return labels invert for
> those rows.

---

## Phase 3 — The research phase (reality check)

Builds Gold from Bronze with the real committee + insider + price providers, seals the
out-of-time hold-out, and on the **dev** block only runs the event study, conditional tests,
purged-CV, and Deflated Sharpe / PBO — then trains and saves the screening model.

```bash
python scripts/04_research.py
# -> models/screening_model.joblib, models/feature_columns.json
```

Read the output critically. If purged-CV AUC sits at ~0.5, the conditional differences are
non-significant, DSR is low, and PBO is high, **there is no edge** — do not proceed to live
trading on it. Only when something survives all of that should you run the final, single
hold-out evaluation:

```bash
python scripts/04_research.py --evaluate-holdout    # consumes the sealed block ONCE
```

(`HoldoutProtocol` refuses a second evaluation in the same process — by design.)

---

## Phase 4 — Live execution (keyless)

### 4a. Keep the CSV fresh, then run the daily screener

The screener pulls "recent" disclosures the same way the backfill does — by reading the CSV
and filtering to a recent window of `published` dates. With a local CSV, "recent" means
"recent **in the file**", so **re-run your scraper to refresh `capitol_trades_cache.csv`
before each screen**, otherwise the screener keeps seeing the same rows. Re-ingesting is
idempotent, so running it repeatedly against the same CSV won't duplicate Bronze rows.

```bash
# 1) refresh the feed (your scraper)         -> updates capitol_trades_cache.csv
# 2) screen
python scripts/05_run_screener.py --pull-lookback-days 14 --signal-window-days 7
# -> signals/actionable_latest.parquet (+ a dated copy)
```

Schedule it on weekday evenings (after disclosures post). Edit your crontab with
`crontab -e` and add (adjust the paths and your scraper command). Note there is **no
`QUIVER_API_KEY`** — the only env vars are the optional `OPENFIGI_API_KEY` and, if your CSV
isn't at the default location, `CAPITOL_TRADES_CSV`:

```cron
30 18 * * 1-5  cd /ABSOLUTE/PATH/capitol-trades && \
  .venv/bin/python scrape_capitoltrades.py && \
  CAPITOL_TRADES_CSV=/ABSOLUTE/PATH/capitol-trades/capitol_trades_cache.csv \
  .venv/bin/python scripts/05_run_screener.py --compact >> logs/screener.log 2>&1
```

(Replace `scrape_capitoltrades.py` with whatever refreshes your CSV. If you'd rather refresh
on a separate schedule, drop that line and just ensure the CSV is updated before the screen
runs.)

### 4b. Launch the dashboard

```bash
streamlit run app_signals.py
```

Open the printed URL (default `http://localhost:8501`). The app does no computation — it
renders `signals/actionable_latest.parquet` written by the screener, with filters, summary
metrics, and a CSV export.

---

## One-shot quick reference

```bash
# setup (no keys required)
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
# optional: export OPENFIGI_API_KEY=...

# static data
python scripts/00_download_raw_data.py --years-back 5 --user-agent "Your Name your@email.com"
# (manually place the two Stewart-Woon files in data/raw/stewart_woon/)
python scripts/01_build_committee.py \
  --assignments data/raw/stewart_woon/house_assignments_1031151.xls \
                data/raw/stewart_woon/senate_assignments_1031151.xls
python scripts/02_build_insider.py --quarters-dir data/raw/sec_form4

# congressional feed
cp /path/to/capitol_trades_cache.csv ./capitol_trades_cache.csv   # or set CAPITOL_TRADES_CSV

# backfill -> research -> live
python scripts/03_backfill.py
python scripts/04_research.py
python scripts/05_run_screener.py
streamlit run app_signals.py
```

---

## Notes & gotchas

- **The whole pipeline is keyless.** No `QUIVER_API_KEY`. The only optional key is
  `OPENFIGI_API_KEY`, which just raises the security-master rate limit; without it the first
  backfill resolves tickers slowly, then everything caches in
  `data/processed/secmaster_cache.sqlite`. Only OpenFIGI (resolution) and yfinance (prices)
  touch the network; both are keyless.
- **Direction is assumed, not read.** The buy feed has no buy/sell column, so every row is a
  `Purchase`. This is the one assumption that would corrupt results if the CSV ever contains
  sells — add a direction column and pass `txn_type_col="..."` if so. (`gold.py` treats
  `is_purchase` as a *feature*, not a filter, so nothing is dropped — but a mislabelled sale
  would carry the wrong sign.)
- **No asset-type column** → every trade is `AssetType.UNKNOWN`. Safe: `is_option` is a 0/1
  feature, not a filter, so no rows are dropped. Pass `asset_type_col="..."` if you later
  scrape one.
- **Amount fidelity.** The adapter expands the `15K–50K` shorthand into a canonical dollar
  range (`$15,000 - $50,000`) that `amounts.parse_amount` snaps to the standard OGE brackets —
  identical midpoints/labels to the old Quiver path, so dev/hold-out comparisons stay
  apples-to-apples. The literal shorthand is preserved in `raw_payload`. (If you'd rather
  Bronze store the verbatim `15K–50K`, add K/M handling to `parse_amount` instead.)
- **Data depth is the binding constraint.** The capitoltrades buy feed reaches back only
  ~8 months, so that — not `--start` — caps your sample. Going keyless removes the
  subscription, not the depth ceiling, and does not change the established finding that the
  strategy trails SPY.
- **Dual-class tickers** (`BRK/B`) pass through verbatim because adapters don't map tickers.
  Downstream, `YFinancePriceProvider` expects `BRK-B`, so a handful of slash/dot symbols may
  fail to price-resolve and drop out of the labeled set — visibly (in reject/NaN counts), not
  silently.
- **No delisting data by default:** `config.get_security_master()` uses `NullListingHistory`.
  For true survivorship handling, swap in a CRSP/Sharadar/EODHD-backed `ListingHistoryProvider`.
- **Committee membership is Congress-granular** (2-year windows) and covers ~1993–2017 in the
  public Stewart-Woon files; trades in Congresses not covered won't flag relevant. Extend the
  membership table if you need recent Congresses — which, given the CSV's ~8-month reach, you
  almost certainly do for `committee_relevant` to fire on current trades.
- **This produces candidate signals, not validated trades.** A green dashboard is not an edge.
  Run Phase 3 on real history and let the single hold-out evaluation — not the signal table —
  decide whether any of this is real. Not investment advice.
```
