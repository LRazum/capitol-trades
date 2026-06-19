"""
normalize.py — bronze -> silver orchestration.

``VendorAdapter`` is the ingestion boundary (FMP / Quiver / Apify implement it).
``normalize_one`` turns one verbatim :class:`RawDisclosure` into a validated
:class:`NormalizedTrade`, wiring together strict date handling, amount parsing, and
point-in-time security resolution. ``run_silver`` maps a whole batch and — importantly —
returns the rejects *with reasons* rather than dropping them, so data loss is visible
and can't quietly reintroduce survivorship bias.
"""
from __future__ import annotations

import abc
import hashlib
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, Iterator, Optional

from .amounts import parse_amount
from .models import (
    AssetType,
    Chamber,
    NormalizedTrade,
    Owner,
    RawDisclosure,
    TransactionType,
)
from .security_master import SecurityMaster


# --------------------------------------------------------------------------- #
# Ingestion boundary (bronze)
# --------------------------------------------------------------------------- #
class VendorAdapter(abc.ABC):
    """Fetch raw disclosures verbatim from a vendor. Implementations should NOT clean,
    map tickers, or compute dates — that is silver's job. They only adapt the vendor's
    wire format into :class:`RawDisclosure` and stamp provenance."""

    source: str

    @abc.abstractmethod
    def fetch_raw(self, *, start: date, end: date) -> Iterable[RawDisclosure]:
        ...


# --------------------------------------------------------------------------- #
# Field mappers
# --------------------------------------------------------------------------- #
_DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%d %b %Y", "%b %d, %Y", "%Y/%m/%d")


def parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    s = s.strip()
    try:  # fast path: ISO
        return date.fromisoformat(s[:10])
    except ValueError:
        pass
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _map_enum(value: Optional[str], mapping: dict[str, object], default):
    if not value:
        return default
    v = value.strip().lower()
    for needle, out in mapping.items():
        if needle in v:
            return out
    return default


_CHAMBER = {"house": Chamber.HOUSE, "rep": Chamber.HOUSE, "senate": Chamber.SENATE, "sen": Chamber.SENATE}
_TTYPE = {
    "purchase": TransactionType.PURCHASE,
    "buy": TransactionType.PURCHASE,
    "full sale": TransactionType.SALE_FULL,
    "partial sale": TransactionType.SALE_PARTIAL,
    "sale": TransactionType.SALE,
    "sell": TransactionType.SALE,
    "exchange": TransactionType.EXCHANGE,
    "receive": TransactionType.RECEIVE,
}
_OWNER = {"spouse": Owner.SPOUSE, "joint": Owner.JOINT, "child": Owner.DEPENDENT, "dependent": Owner.DEPENDENT, "self": Owner.SELF}
_ASSET = {"option": AssetType.OPTION, "etf": AssetType.ETF, "fund": AssetType.FUND, "bond": AssetType.BOND, "crypto": AssetType.CRYPTO, "stock": AssetType.STOCK}


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
@dataclass
class Reject:
    source_record_id: str
    reason: str
    raw: RawDisclosure


def _trade_id(raw: RawDisclosure, tx: date) -> str:
    h = hashlib.sha1(
        f"{raw.source}|{raw.source_record_id}|{raw.ticker_raw}|{tx.isoformat()}".encode()
    ).hexdigest()
    return h[:16]


def normalize_one(
    raw: RawDisclosure,
    sm: SecurityMaster,
    *,
    amount_default: str = "midpoint",
) -> NormalizedTrade:
    """Normalize a single raw row. Raises ``ValueError`` (caught by ``run_silver``) on
    any condition that makes the row unusable — we never fabricate missing dates."""
    tx = parse_date(raw.transaction_date_raw)
    fd = parse_date(raw.filing_date_raw)
    if tx is None:
        raise ValueError(f"unparseable transaction_date: {raw.transaction_date_raw!r}")
    if fd is None:
        raise ValueError(f"unparseable filing_date: {raw.filing_date_raw!r}")

    security = sm.resolve(
        as_of=fd,  # resolve identity AS OF the filing date, never 'today'
        issuer_name=raw.issuer_raw,
        cusip=raw.cusip_raw,
        ticker=raw.ticker_raw,
    )
    amount = parse_amount(raw.amount_raw, default=amount_default)

    # NormalizedTrade's own validator enforces filing_date >= transaction_date.
    return NormalizedTrade(
        trade_id=_trade_id(raw, tx),
        source=raw.source,
        source_record_id=raw.source_record_id,
        politician=(raw.politician_raw or "UNKNOWN").strip(),
        chamber=_map_enum(raw.chamber_raw, _CHAMBER, Chamber.UNKNOWN),
        owner=_map_enum(raw.owner_raw, _OWNER, Owner.UNKNOWN),
        transaction_type=_map_enum(raw.transaction_type_raw, _TTYPE, TransactionType.UNKNOWN),
        asset_type=_map_enum(raw.asset_type_raw, _ASSET, AssetType.UNKNOWN),
        security=security,
        amount=amount,
        transaction_date=tx,
        filing_date=fd,
    )


def run_silver(
    adapter: VendorAdapter,
    sm: SecurityMaster,
    *,
    start: date,
    end: date,
    amount_default: str = "midpoint",
) -> tuple[list[NormalizedTrade], list[Reject]]:
    """Run the full bronze->silver pass. Returns (trades, rejects). Inspect rejects —
    a growing reject pile is a data-quality signal, not something to ignore."""
    trades: list[NormalizedTrade] = []
    rejects: list[Reject] = []
    for raw in adapter.fetch_raw(start=start, end=end):
        try:
            trades.append(normalize_one(raw, sm, amount_default=amount_default))
        except (ValueError, Exception) as exc:  # noqa: BLE001 - capture reason, never drop silently
            rejects.append(Reject(raw.source_record_id, str(exc), raw))
    return trades, rejects


# --------------------------------------------------------------------------- #
# Runnable demo (no network): a fake adapter + a stub resolver
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from .security_master import IdentifierResolver, ListingStatus, ListingHistoryProvider

    class _StubResolver(IdentifierResolver):
        """Pretend OpenFIGI so the demo runs offline."""
        def by_cusip(self, cusip):
            return [{"figi": "BBG000B9XRY4", "composite_figi": "BBG000B9XRY4",
                     "ticker": "AAPL", "name": "APPLE INC", "exch": "US",
                     "security_type": "Common Stock"}]
        def by_ticker(self, ticker):
            return [{"figi": f"BBG-{ticker}", "composite_figi": f"BBG-{ticker}",
                     "ticker": ticker.upper(), "name": f"{ticker.upper()} CORP",
                     "exch": "US", "security_type": "Common Stock"}]

    class _StubHistory(ListingHistoryProvider):
        def status(self, *, figi, cik, ticker, as_of):
            if ticker == "XYZ":  # pretend XYZ was acquired
                return ListingStatus(ticker_as_of="XYZ", is_delisted=True,
                                     delisting_date=date(2021, 6, 1),
                                     delisting_action="acquired",
                                     still_trading_on_as_of=False)
            return ListingStatus(ticker_as_of=ticker, is_delisted=False,
                                 still_trading_on_as_of=True)

    class _FakeAdapter(VendorAdapter):
        source = "demo"
        def fetch_raw(self, *, start, end):
            now = datetime.now()
            yield RawDisclosure(source="demo", source_record_id="1", pulled_at=now,
                                politician_raw="Jane Doe", chamber_raw="House",
                                issuer_raw="Apple Inc", ticker_raw="AAPL", cusip_raw="037833100",
                                transaction_date_raw="2023-02-01", filing_date_raw="2023-03-05",
                                transaction_type_raw="Purchase", owner_raw="Spouse",
                                asset_type_raw="Stock", amount_raw="$15,001 - $50,000")
            yield RawDisclosure(source="demo", source_record_id="2", pulled_at=now,
                                politician_raw="John Roe", chamber_raw="Senate",
                                issuer_raw="XYZ Holdings", ticker_raw="XYZ",
                                transaction_date_raw="03/10/2021", filing_date_raw="04/01/2021",
                                transaction_type_raw="Sale (Full)", owner_raw="Self",
                                asset_type_raw="Stock", amount_raw="over $50,000,000")
            # Bad row: filing before transaction -> must be rejected, not used.
            yield RawDisclosure(source="demo", source_record_id="3", pulled_at=now,
                                politician_raw="Bad Row", issuer_raw="Foo",
                                ticker_raw="FOO", transaction_date_raw="2022-05-01",
                                filing_date_raw="2022-04-01", amount_raw="$1,001 - $15,000")

    sm = SecurityMaster(_StubResolver(), _StubHistory(), cache_path=":memory:")
    trades, rejects = run_silver(_FakeAdapter(), sm, start=date(2020, 1, 1), end=date(2024, 1, 1))

    print(f"\nNORMALIZED {len(trades)} trades, {len(rejects)} rejected\n" + "-" * 60)
    for t in trades:
        warn = sm.survivorship_check(t.security)
        print(f"{t.politician:10} {t.transaction_type.value:12} "
              f"{t.security.ticker_as_of:5} tx={t.transaction_date} file={t.filing_date} "
              f"lag={t.disclosure_lag_days}d  ${t.amount.point_estimate:,.0f}"
              + (f"  [!] {warn}" if warn else ""))
    for r in rejects:
        print(f"REJECT id={r.source_record_id}: {r.reason}")
