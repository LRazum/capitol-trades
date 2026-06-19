"""
quiver.py — concrete VendorAdapter for the Quiver Quantitative congressional API.

Endpoints (from Quiver's own client):
  live : https://api.quiverquant.com/beta/live/congresstrading      (recent)
  bulk : https://api.quiverquant.com/beta/bulk/congresstrading      (full history)
Auth  : Authorization header. Current docs use "Bearer <key>"; the legacy client uses
        "Token <key>" — both are supported via ``auth_scheme``.

The adapter only adapts Quiver's wire format into ``RawDisclosure`` and stamps provenance.
It does NOT clean, map tickers, or compute anything — that is the silver layer's job.
Field access uses fallbacks because the beta endpoints are inconsistent across versions
(e.g. ``ReportDate`` vs ``Filed``, ``Range`` vs ``Amount``).
"""
from __future__ import annotations

import hashlib
from datetime import date, datetime, timezone
from typing import Any, Iterable, Optional

import requests

from .models import RawDisclosure
from .normalize import VendorAdapter, parse_date


class QuiverCongressAdapter(VendorAdapter):
    source = "quiver"

    LIVE = "https://api.quiverquant.com/beta/live/congresstrading"
    BULK = "https://api.quiverquant.com/beta/bulk/congresstrading"

    def __init__(
        self,
        api_key: str,
        *,
        mode: str = "bulk",                 # "bulk" (history) or "live" (recent)
        auth_scheme: str = "Bearer",        # "Bearer" (current docs) or "Token" (legacy)
        timeout: int = 60,
        session: Optional[requests.Session] = None,
    ):
        self.api_key = api_key
        self.url = self.BULK if mode == "bulk" else self.LIVE
        self.auth_scheme = auth_scheme
        self.timeout = timeout
        self.session = session or requests.Session()

    # ----------------------------------------------------------- networking --
    def _headers(self) -> dict[str, str]:
        return {
            "accept": "application/json",
            "Authorization": f"{self.auth_scheme} {self.api_key}",
        }

    def _fetch_json(self, url: str) -> list[dict[str, Any]]:
        """Testable seam: override in tests to return a mock payload (no network)."""
        resp = self.session.get(url, headers=self._headers(), timeout=self.timeout)
        resp.raise_for_status()
        payload = resp.json()
        # Newer endpoints wrap rows in {"data": [...]}; beta congresstrading returns a list.
        if isinstance(payload, dict):
            return payload.get("data", [])
        return payload

    # --------------------------------------------------------------- mapping --
    @staticmethod
    def _first(row: dict, *keys: str) -> Optional[Any]:
        for k in keys:
            v = row.get(k)
            if v not in (None, "", "NaN", "nan"):
                return v
        return None

    @classmethod
    def _record_key(cls, row: dict) -> str:
        """Deterministic natural key, stable across re-pulls and across amendments that
        only correct *mutable* fields (amount, report date). It deliberately excludes
        those mutable fields so a corrected row de-duplicates against its original.

        Limitation: if an amendment corrects an *immutable* field (ticker or transaction
        date), the key changes and the rows won't merge — that requires the filing-id
        linkage, which Quiver's feed does not expose. Documented, not hidden.
        """
        rep = (cls._first(row, "Representative", "Name") or "").strip().lower()
        tic = (cls._first(row, "Ticker") or "").strip().upper()
        txd = str(cls._first(row, "TransactionDate", "Traded") or "")[:10]
        typ = (cls._first(row, "Transaction", "Type") or "").strip().lower()
        own = (cls._first(row, "Owner") or "").strip().lower()
        basis = f"quiver|{rep}|{tic}|{txd}|{typ}|{own}"
        return hashlib.sha1(basis.encode()).hexdigest()[:20]

    def fetch_raw(self, *, start: date, end: date) -> Iterable[RawDisclosure]:
        # One provenance timestamp per pull, applied to every row from this fetch.
        pulled_at = datetime.now(timezone.utc)
        rows = self._fetch_json(self.url)

        for row in rows:
            filed_raw = self._first(row, "ReportDate", "Filed")
            filing_date = parse_date(str(filed_raw)) if filed_raw else None
            # Bulk returns all history; filter to the requested window by filing date.
            if filing_date is not None and not (start <= filing_date <= end):
                continue

            last_mod = self._first(row, "last_modified", "LastModified")

            yield RawDisclosure(
                # provenance / audit trail
                source=self.source,
                source_record_id=self._record_key(row),  # stable dedup key
                pulled_at=pulled_at,
                # verbatim payload (strings)
                politician_raw=self._first(row, "Representative", "Name"),
                chamber_raw=self._first(row, "House", "Chamber"),
                issuer_raw=self._first(row, "Company", "Issuer"),
                ticker_raw=self._first(row, "Ticker"),
                cusip_raw=self._first(row, "CUSIP"),  # Quiver typically omits CUSIP
                transaction_date_raw=self._first(row, "TransactionDate", "Traded"),
                filing_date_raw=str(filed_raw) if filed_raw else None,
                transaction_type_raw=self._first(row, "Transaction", "Type"),
                owner_raw=self._first(row, "Owner"),
                asset_type_raw=self._first(row, "TickerType", "AssetType"),
                amount_raw=self._first(row, "Range", "Amount"),
                # extra fields (allowed by the model) carried for the bronze recency tiebreak
                last_modified=str(last_mod) if last_mod else None,
                raw_payload=row,  # the untouched original, for re-derivation later
            )
