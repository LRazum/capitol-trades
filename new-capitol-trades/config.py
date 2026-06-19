"""
config.py — single source of truth for paths, credentials, and pipeline parameters.

All scripts import this. Secrets come from environment variables (never hard-code keys):
    export QUIVER_API_KEY=...        # required for backfill + live screening
    export OPENFIGI_API_KEY=...      # optional, raises the security-master rate limit
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from capitol_ingest.gold import GoldConfig

# --- directories ----------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"               # downloaded source files
BRONZE = DATA / "bronze"         # partitioned parquet (filing_year/filing_month)
PROCESSED = DATA / "processed"   # tidy parquets + caches
MODELS = ROOT / "models"
SIGNALS = ROOT / "signals"
for _d in (RAW, BRONZE, PROCESSED, MODELS, SIGNALS):
    _d.mkdir(parents=True, exist_ok=True)

# --- processed artifacts --------------------------------------------------- #
MEMBERSHIP_PARQUET = PROCESSED / "committee_membership.parquet"
NAME_TO_ICPSR_JSON = PROCESSED / "name_to_icpsr.json"
INSIDER_TIDY_PARQUET = PROCESSED / "insider_tidy.parquet"
SECTORS_CSV = PROCESSED / "sectors.csv"               # columns: ticker,sector,industry
SECMASTER_CACHE = PROCESSED / "secmaster_cache.sqlite"
MODEL_PATH = MODELS / "screening_model.joblib"
FEATURES_JSON = MODELS / "feature_columns.json"

# --- credentials ----------------------------------------------------------- #
QUIVER_API_KEY = os.getenv("QUIVER_API_KEY", "")
OPENFIGI_API_KEY = os.getenv("OPENFIGI_API_KEY", "")

# --- parameters ------------------------------------------------------------ #
BACKFILL_START = date(2016, 1, 1)
BACKFILL_END = date.today()
HOLDOUT_FRAC = 0.2
PROBA_THRESHOLD = 0.55
GOLD = GoldConfig(
    use_original_filing_date=False,
    benchmark_ticker="SPY",
    pt_mult=2.0, sl_mult=2.0, max_holding_days=21,
    vol_lookback=60, momentum_windows=(21, 63, 126, 252), ma_window=200,
    insider_lookback_days=90,
)


# --- provider factories (lazy; build the real components on demand) -------- #
def get_security_master():
    from capitol_ingest import NullListingHistory, OpenFigiResolver, SecurityMaster
    # Swap NullListingHistory for a CRSP/Sharadar/EODHD-backed provider to add delisting data.
    return SecurityMaster(
        OpenFigiResolver(OPENFIGI_API_KEY or None),
        NullListingHistory(),
        cache_path=str(SECMASTER_CACHE),
    )


def get_sector_provider():
    import pandas as pd
    from capitol_ingest import StaticSectorProvider
    if SECTORS_CSV.exists():
        df = pd.read_csv(SECTORS_CSV)
        mapping = {str(r.ticker).upper(): (r.sector, r.industry) for r in df.itertuples()}
        return StaticSectorProvider(mapping)
    return StaticSectorProvider({})  # empty -> committee_relevant stays False until populated


def get_committee_provider():
    import json
    import pandas as pd
    from capitol_ingest import StewartWoonCommitteeProvider
    membership = pd.read_parquet(MEMBERSHIP_PARQUET)
    name_to_icpsr = json.loads(NAME_TO_ICPSR_JSON.read_text())
    return StewartWoonCommitteeProvider(membership, name_to_icpsr, get_sector_provider())


def get_insider_provider():
    from capitol_ingest import Form4InsiderProvider
    return Form4InsiderProvider(str(INSIDER_TIDY_PARQUET), officers_only=True)


def get_price_provider():
    from capitol_ingest import YFinancePriceProvider
    return YFinancePriceProvider()
