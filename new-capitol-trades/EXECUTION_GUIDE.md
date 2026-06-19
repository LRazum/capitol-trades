# Capitol Trades — Execution Guide

Spin the whole pipeline up from a blank folder. Phases run **in order**; each later phase
assumes the earlier ones produced their artifacts.

```
Phase 0  Environment + folders
Phase 1  Static data (Stewart-Woon committees, SEC Form 4) -> tidy parquets
Phase 2  Historical backfill (Quiver -> Bronze -> compact -> Silver)
Phase 3  Research / reality check (Gold -> event study + DSR/PBO -> save model)
Phase 4  Live execution (DailyScreener -> Streamlit dashboard)
```

Commands assume macOS/Linux `bash`. On Windows use `py -m venv .venv` and
`.venv\Scripts\activate`.

---

## Phase 0 — Environment & directory structure

```bash
mkdir capitol-trades && cd capitol-trades

# put the package + scripts + top-level files in place:
#   capitol_ingest/   (all the modules)
#   scripts/          (01..05)
#   config.py  app_signals.py  requirements.txt

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Target layout once everything is downloaded/built:

```
capitol-trades/
├── capitol_ingest/                # the library (bronze, silver, gold, committee, insider, ...)
├── scripts/
│   ├── 01_build_committee.py
│   ├── 02_build_insider.py
│   ├── 03_backfill.py
│   ├── 04_research.py
│   └── 05_run_screener.py
├── data/
│   ├── raw/                       # downloads land here
│   │   ├── stewart_woon/          #   committee assignment file (+ codes crosswalk)
│   │   ├── sec_form4/             #   one subdir per quarter of unzipped TSVs
│   │   ├── legislators-current.yaml
│   │   └── legislators-historical.yaml
│   ├── bronze/                    # partitioned parquet (filing_year=/filing_month=)
│   └── processed/                 # committee_membership.parquet, name_to_icpsr.json,
│                                  #   insider_tidy.parquet, sectors.csv, secmaster cache
├── models/                        # screening_model.joblib, feature_columns.json
└── signals/                       # actionable_latest.parquet (the dashboard reads this)
```

`config.py` creates `data/`, `models/`, and `signals/` automatically on first import.

**Smoke-test the install with zero real data** (uses synthetic fixtures):

```bash
python live_demo.py        # committee + insider + screening logic
python research_demo.py    # event study, DSR/PBO, costs, hold-out
```

Set credentials (needed from Phase 2 on):

```bash
export QUIVER_API_KEY="your_quiver_key"
export OPENFIGI_API_KEY="your_openfigi_key"   # optional; raises the security-master rate limit
```

---

## Phase 1 — Static data acquisition & preprocessing

### 1a. congress-legislators (name → ICPSR crosswalk)

```bash
cd data/raw
curl -L -o legislators-current.yaml \
  https://raw.githubusercontent.com/unitedstates/congress-legislators/main/legislators-current.yaml
curl -L -o legislators-historical.yaml \
  https://raw.githubusercontent.com/unitedstates/congress-legislators/main/legislators-historical.yaml
cd ../..
```

### 1b. Stewart-Woon committee assignments

Download from the Harvard Dataverse (Charles Stewart's *congdata*):
`https://dataverse.harvard.edu/dataverse/congdata`. Grab the combined assignment file
(`allCongressDataPublishV2.tab`) via the file's **Download → Original File Format**, plus the
**committee codes** crosswalk if your file stores codes rather than names. Place them in
`data/raw/stewart_woon/`.

```bash
python scripts/01_build_committee.py \
  --assignments data/raw/stewart_woon/allCongressDataPublishV2.tab \
  --codes       data/raw/stewart_woon/committee_codes.csv     # omit if names are present
```

This writes `data/processed/committee_membership.parquet` and `name_to_icpsr.json`, and
prints the distinct committee names found. **Check that list** against the jurisdiction map
in `capitol_ingest/committee.py` (e.g. it should contain `armed services`, `financial
services`, `energy and commerce`); the `committee_relevant` flag matches on those names.

### 1c. SEC Form 4 bulk datasets

From the SEC page
`https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets`,
download the quarterly ZIPs you want (each named like `2024q1_form345.zip`) and unzip each
into its own subdirectory. **The SEC blocks requests without a User-Agent**, so identify
yourself:

```bash
mkdir -p data/raw/sec_form4/2024q1
curl -L -A "yourname your@email.com" \
  -o /tmp/2024q1.zip "<paste the 2024q1 zip URL from the SEC page>"
unzip /tmp/2024q1.zip -d data/raw/sec_form4/2024q1
# repeat per quarter -> data/raw/sec_form4/2024q2, 2024q3, ...
```

Each quarter dir should now contain `SUBMISSION.tsv`, `REPORTINGOWNER.tsv`,
`NONDERIV_TRANS.tsv` (others are ignored). Build the tidy table:

```bash
python scripts/02_build_insider.py --quarters-dir data/raw/sec_form4
# -> data/processed/insider_tidy.parquet
```

### 1d. (optional but recommended) sector map for committee relevance

`committee_relevant` needs a ticker → GICS (sector, industry) map. Create
`data/processed/sectors.csv` with columns `ticker,sector,industry` from FMP, or generate a
quick one from yfinance for the tickers you care about:

```bash
python - <<'PY'
import yfinance as yf, pandas as pd, config
tickers = ["LMT","RTX","JPM","BAC","PFE","XOM","NVDA"]   # your universe
rows=[]
for t in tickers:
    info = yf.Ticker(t).info
    rows.append({"ticker":t,"sector":info.get("sector"),"industry":info.get("industry")})
pd.DataFrame(rows).to_csv(config.SECTORS_CSV, index=False)
print("wrote", config.SECTORS_CSV)
PY
```

Without `sectors.csv`, the pipeline still runs but `committee_relevant` is always `False`.

---

## Phase 2 — Historical backfill

Pulls the Quiver bulk feed once, commits to Bronze year-by-year (resumable + retrying),
compacts cross-partition amendments, and reports Silver counts. The first run resolves every
unique ticker via OpenFIGI (cached afterward), so it is slower than subsequent runs.

```bash
python scripts/03_backfill.py --start 2016-01-01 --end "$(date +%F)"
```

Artifacts: `data/bronze/filing_year=YYYY/filing_month=MM/data.parquet`. Re-running is
idempotent; interrupted runs resume from `data/bronze/.backfill_checkpoint.json`.

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

## Phase 4 — Live execution

### 4a. Run the daily screener (cron entry point)

```bash
python scripts/05_run_screener.py --pull-lookback-days 14 --signal-window-days 7
# -> signals/actionable_latest.parquet (+ a dated copy)
```

Schedule it on weekday evenings (after disclosures post). Edit your crontab with
`crontab -e` and add (adjust the paths):

```cron
30 18 * * 1-5  cd /ABSOLUTE/PATH/capitol-trades && \
  QUIVER_API_KEY=xxx OPENFIGI_API_KEY=yyy \
  .venv/bin/python scripts/05_run_screener.py --compact >> logs/screener.log 2>&1
```

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
# setup
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
export QUIVER_API_KEY=... ; export OPENFIGI_API_KEY=...

# static data
python scripts/01_build_committee.py --assignments data/raw/stewart_woon/allCongressDataPublishV2.tab --codes data/raw/stewart_woon/committee_codes.csv
python scripts/02_build_insider.py   --quarters-dir data/raw/sec_form4

# backfill -> research -> live
python scripts/03_backfill.py
python scripts/04_research.py
python scripts/05_run_screener.py
streamlit run app_signals.py
```

---

## Notes & gotchas

- **Phases 2–4 need network + a Quiver subscription.** Phases 0–1 are local except the
  downloads.
- **OpenFIGI rate limit:** without a key the security master resolves tickers slowly on the
  first backfill; set `OPENFIGI_API_KEY`. All resolutions are cached in
  `data/processed/secmaster_cache.sqlite`.
- **No delisting data by default:** `config.get_security_master()` uses `NullListingHistory`.
  For true survivorship handling, swap in a CRSP/Sharadar/EODHD-backed `ListingHistoryProvider`.
- **Committee membership is Congress-granular** (2-year windows) and covers ~1993–2017 in the
  public Stewart-Woon files; trades in Congresses not covered won't flag relevant. Extend the
  membership table if you need recent Congresses.
- **This produces candidate signals, not validated trades.** A green dashboard is not an
  edge. Run Phase 3 on real history and let the single hold-out evaluation — not the signal
  table — decide whether any of this is real. Not investment advice.
```
