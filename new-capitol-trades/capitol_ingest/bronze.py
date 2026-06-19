"""
bronze.py — wire a VendorAdapter into the idempotent BronzeStore.

``disclosures_to_frame`` flattens ``RawDisclosure`` objects into a Parquet-ready frame
(parsing only the date used for partitioning + the recency signal; everything else stays
verbatim, with the full original kept as a JSON blob for audit/re-derivation).
``ingest_bronze`` runs adapter -> frame -> store.upsert.
"""
from __future__ import annotations

import json
from datetime import date
from typing import Iterable

import pandas as pd

from .models import RawDisclosure
from .normalize import VendorAdapter
from .storage import BronzeStore


def disclosures_to_frame(raws: Iterable[RawDisclosure]) -> pd.DataFrame:
    records = []
    for r in raws:
        d = r.model_dump(mode="python")  # includes extra fields (extra="allow")
        raw_payload = d.get("raw_payload") or {}
        records.append(
            {
                # provenance + keys
                "source": d["source"],
                "source_record_id": d["source_record_id"],
                "pulled_at": pd.Timestamp(d["pulled_at"]),
                # parsed for partitioning / recency only
                "filing_date": pd.to_datetime(d.get("filing_date_raw"), errors="coerce"),
                "transaction_date": pd.to_datetime(d.get("transaction_date_raw"), errors="coerce"),
                "_last_modified": pd.to_datetime(d.get("last_modified"), errors="coerce"),
                # verbatim fields
                "politician_raw": d.get("politician_raw"),
                "chamber_raw": d.get("chamber_raw"),
                "issuer_raw": d.get("issuer_raw"),
                "ticker_raw": d.get("ticker_raw"),
                "cusip_raw": d.get("cusip_raw"),
                "transaction_date_raw": d.get("transaction_date_raw"),
                "filing_date_raw": d.get("filing_date_raw"),
                "transaction_type_raw": d.get("transaction_type_raw"),
                "owner_raw": d.get("owner_raw"),
                "asset_type_raw": d.get("asset_type_raw"),
                "amount_raw": d.get("amount_raw"),
                # full untouched original, for re-derivation when a parser is fixed later
                "raw_payload": json.dumps(raw_payload, default=str),
            }
        )
    return pd.DataFrame.from_records(records)


def ingest_bronze(
    adapter: VendorAdapter,
    store: BronzeStore,
    *,
    start: date,
    end: date,
) -> dict:
    """Fetch from the adapter and idempotently upsert into the bronze store."""
    frame = disclosures_to_frame(adapter.fetch_raw(start=start, end=end))
    if frame.empty:
        return {"fetched": 0, "written": {}}
    written = store.upsert(frame)
    return {"fetched": int(len(frame)), "written": written}


# --------------------------------------------------------------------------- #
# Runnable demo (no network): mock Quiver payloads exercise the real adapter +
# store, proving idempotency, amendment supersession, and cross-partition compaction.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import tempfile

    from .quiver import QuiverCongressAdapter

    class _MockQuiver(QuiverCongressAdapter):
        """Inject Quiver-shaped rows instead of hitting the network."""
        def __init__(self, rows):
            super().__init__(api_key="DEMO")
            self._rows = rows
        def _fetch_json(self, url):
            return self._rows

    JANE_BASE = dict(Representative="Jane Doe", House="Representatives", Ticker="AAPL",
                      TransactionDate="2024-02-01", Transaction="Purchase", Owner="Self")
    JOHN = dict(Representative="John Roe", House="Senate", Ticker="MSFT",
                TransactionDate="2024-02-10", Transaction="Sale (Full)", Owner="Spouse",
                ReportDate="2024-02-20", Range="$50,001 - $100,000", last_modified="2024-02-20")

    def jane(report_date, rng, last_mod):
        return {**JANE_BASE, "ReportDate": report_date, "Range": rng, "last_modified": last_mod}

    store = BronzeStore(tempfile.mkdtemp())
    win = dict(start=date(2024, 1, 1), end=date(2024, 12, 31))

    def show(label):
        df = store.read()
        jrows = df[df["politician_raw"] == "Jane Doe"]
        amt = jrows["amount_raw"].tolist()
        print(f"{label:42} total={len(df):2d}  jane_rows={len(jrows)}  jane_amount={amt}")

    # 1) initial load
    ingest_bronze(_MockQuiver([jane("2024-03-05", "$15,001 - $50,000", "2024-03-05"), JOHN]), store, **win)
    show("1. initial load")

    # 2) exact re-run of the same window -> idempotent (no duplicates)
    ingest_bronze(_MockQuiver([jane("2024-03-05", "$15,001 - $50,000", "2024-03-05"), JOHN]), store, **win)
    show("2. re-run same window (idempotent)")

    # 3) amendment in the SAME partition: corrected Range, newer last_modified
    ingest_bronze(_MockQuiver([jane("2024-03-05", "$50,001 - $100,000", "2024-03-20")]), store, **win)
    show("3. amendment, same partition")

    # 4) stale re-fetch of the ORIGINAL (older last_modified) must NOT clobber the amendment
    ingest_bronze(_MockQuiver([jane("2024-03-05", "$15,001 - $50,000", "2024-03-05")]), store, **win)
    show("4. stale original re-fetched (ignored)")

    # 5) amendment in a DIFFERENT partition (later ReportDate) -> needs compaction
    ingest_bronze(_MockQuiver([jane("2024-05-10", "$100,001 - $250,000", "2024-05-10")]), store, **win)
    show("5. cross-partition amendment (pre-compact)")
    store.compact()
    show("5. after compact()")

    print("\nPartitions on disk:")
    for d in sorted(store._all_partition_dirs()):
        print("  ", d.relative_to(store.root))
