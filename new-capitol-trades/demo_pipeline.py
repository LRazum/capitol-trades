"""
demo_pipeline.py — end-to-end: mock backfill -> compact -> silver.

Run from the directory containing the `capitol_ingest` package:  python demo_pipeline.py

Uses offline stubs (a mock Quiver feed + a stub security master) so it runs without a
network or API keys. Swap in the real QuiverCongressAdapter + OpenFigiResolver/listing
provider for production.
"""
from __future__ import annotations

import tempfile
from datetime import date

from capitol_ingest.backfill import HistoricalBackfill
from capitol_ingest.quiver import QuiverCongressAdapter
from capitol_ingest.security_master import (
    IdentifierResolver,
    ListingHistoryProvider,
    ListingStatus,
    SecurityMaster,
)
from capitol_ingest.silver import SilverDriver
from capitol_ingest.storage import BronzeStore


# --------------------------------------------------------------------------- #
# Offline stubs (no network)
# --------------------------------------------------------------------------- #
class MockQuiver(QuiverCongressAdapter):
    """Inject Quiver-shaped rows instead of hitting the API."""
    def __init__(self, rows):
        super().__init__(api_key="DEMO", mode="bulk")
        self._rows = rows

    def _fetch_json(self, url):
        return self._rows


class StubResolver(IdentifierResolver):
    def by_cusip(self, cusip):
        return self.by_ticker("UNKNOWN")

    def by_ticker(self, ticker):
        return [{
            "figi": f"BBG-{ticker}", "composite_figi": f"BBG-{ticker}",
            "ticker": ticker.upper(), "name": f"{ticker.upper()} INC",
            "exch": "US", "security_type": "Common Stock",
        }]


class StubHistory(ListingHistoryProvider):
    def status(self, *, figi, cik, ticker, as_of):
        if ticker == "ZNGA":  # pretend this name was acquired and left the tape
            return ListingStatus(
                ticker_as_of="ZNGA", is_delisted=True, delisting_date=date(2022, 5, 23),
                delisting_action="acquired", still_trading_on_as_of=False,
            )
        return ListingStatus(ticker_as_of=ticker, is_delisted=False, still_trading_on_as_of=True)


def row(rep, house, ticker, txd, ttype, owner, report, rng, last_mod):
    return dict(
        Representative=rep, House=house, Ticker=ticker, TransactionDate=txd,
        Transaction=ttype, Owner=owner, ReportDate=report, Range=rng, last_modified=last_mod,
    )


MOCK_FEED = [
    row("Jane Doe", "Representatives", "AAPL", "2021-02-01", "Purchase", "Self",
        "2021-03-05", "$15,001 - $50,000", "2021-03-05"),
    row("John Roe", "Senate", "MSFT", "2022-06-10", "Sale (Full)", "Spouse",
        "2022-07-01", "$50,001 - $100,000", "2022-07-01"),
    row("Sam Poe", "Representatives", "ZNGA", "2022-01-15", "Purchase", "Self",
        "2022-02-10", "$1,001 - $15,000", "2022-02-10"),
    # Amendment of Jane's AAPL trade, filed a year later (cross-partition), amount corrected,
    # newer last_modified -> must supersede the original in both compaction and silver.
    row("Jane Doe", "Representatives", "AAPL", "2021-02-01", "Purchase", "Self",
        "2022-01-20", "$50,001 - $100,000", "2022-01-20"),
    # Malformed: unparseable transaction date -> rejected in silver.
    row("Bad Date", "Senate", "TSLA", "not-a-date", "Purchase", "Self",
        "2023-04-01", "$1,001 - $15,000", "2023-04-01"),
    # Malformed: filing precedes transaction -> rejected in silver.
    row("Time Traveler", "Senate", "NVDA", "2023-05-01", "Purchase", "Self",
        "2023-04-01", "$1,001 - $15,000", "2023-04-01"),
]


def main():
    store = BronzeStore(tempfile.mkdtemp())
    adapter = MockQuiver(MOCK_FEED)

    print("=== 1) Historical backfill into bronze (bulk-once, by year) ===")
    bf = HistoricalBackfill(adapter, store, granularity="year")
    print("   ", bf.run_bulk_once(start=date(2021, 1, 1), end=date(2024, 12, 31)))
    print("    bronze partitions:", [str(p.relative_to(store.root)) for p in sorted(store._all_partition_dirs())])

    print("\n=== 2) Compact bronze (resolve cross-partition amendments on disk) ===")
    print("   ", store.compact())
    print("    bronze partitions:", [str(p.relative_to(store.root)) for p in sorted(store._all_partition_dirs())])

    print("\n=== 3) Silver driver -> clean, resolved NormalizedTrade ===")
    sm = SecurityMaster(StubResolver(), StubHistory(), cache_path=":memory:")
    result = SilverDriver(store, sm).build()
    print(f"    clean trades: {result.n_trades}    rejected: {result.n_rejected}")

    tf = SilverDriver.trades_to_frame(result.trades).sort_values("filing_date")
    cols = ["politician", "ticker_as_of", "transaction_type", "transaction_date",
            "filing_date", "disclosure_lag_days", "amount_point", "is_delisted",
            "resolution_method"]
    print("\nClean silver dataset (feedstock for the research event table):")
    print(tf[cols].to_string(index=False))

    print("\nRejections log (visible, with reasons — nothing silently dropped):")
    print(SilverDriver.rejections_to_frame(result.rejections).to_string(index=False))


if __name__ == "__main__":
    main()
