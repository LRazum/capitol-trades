"""
capitol_trades_csv.py — keyless VendorAdapter backed by a scraped capitoltrades.com CSV.

Drop-in replacement for :class:`QuiverCongressAdapter` that requires **no API key**.
It reads a local CSV (the buy feed scraped from capitoltrades.com) and adapts each row
into a :class:`RawDisclosure`, stamping provenance. Like every adapter it does NOT clean,
map tickers, or compute returns — that is the silver/gold layers' job. The only shaping it
does is what an adapter is *for*: translating this vendor's wire format into the canonical
raw fields (e.g. expanding the "1K–15K" size shorthand into a dollar range that
``amounts.parse_amount`` already understands, and splitting the packed politician string
into a name + chamber).

Expected CSV columns (case-insensitive; extra columns are preserved verbatim in
``raw_payload`` and ignored otherwise)::

    politician   e.g. "Ro Khanna Democrat House CA"   (name + party + chamber + state)
    issuer       e.g. "Visa Inc V:US"
    ticker       e.g. "V"  /  "BRK/B"
    published    e.g. "2026-06-11"   -> filing date (the only tradeable timestamp)
    traded       e.g. "2026-05-01"   -> transaction date (self-reported)
    owner        e.g. "Self" / "Spouse" / "Child" / "Joint"
    size_str     e.g. "15K–50K" / "100K–250K" / "50M+"   (STOCK Act bracket, shorthand)
    size_num     e.g. 32500         (pre-computed bracket midpoint; used as a fallback)
    price        e.g. "$208.03" / "N/A"   (carried through; not used by the pipeline)

IMPORTANT — transaction direction
---------------------------------
capitoltrades' buy feed carries **no buy/sell column**, so this adapter stamps every row
with ``default_txn_type`` (``"Purchase"`` by default, matching the buy feed). If your CSV
mixes buys and sells, add a direction column and pass ``txn_type_col=...`` so the sign is
read per-row instead of assumed — otherwise sales would be silently mislabelled as buys.
"""
from __future__ import annotations

import hashlib
import re
from datetime import date, datetime, timezone
from typing import Iterable, Optional

import pandas as pd

from .models import RawDisclosure
from .normalize import VendorAdapter, parse_date

# Dash variants capitoltrades uses in size ranges (en/em/figure dash, minus) -> "-".
_DASH = re.compile(r"[\u2010-\u2015\u2212]")
# Magnitude suffixes in the size shorthand ("15K", "1.5M", ...).
_MULT = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
_MONEY = re.compile(r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*([KMBT]?)", re.I)
_OPEN_ENDED = re.compile(r"\+|over|more than|greater than|at least", re.I)

# Party tokens dropped when splitting the packed politician string into a clean name.
_PARTIES = {
    "democrat", "democratic", "republican", "independent",
    "libertarian", "green", "nonpartisan",
}
_CHAMBERS = {"house": "House", "senate": "Senate"}


def _money(token: str) -> Optional[float]:
    """'15K' -> 15000.0 ; '1.5M' -> 1500000.0 ; '250' -> 250.0 ; junk -> None."""
    if token is None:
        return None
    m = _MONEY.search(str(token).replace("$", "").replace(",", "").strip())
    if not m:
        return None
    try:
        value = float(m.group(1))
    except ValueError:
        return None
    return value * _MULT.get(m.group(2).upper(), 1.0)


def amount_string(size_str: object, size_num: object = None) -> Optional[str]:
    """Turn the capitoltrades size shorthand into a string ``parse_amount`` understands.

    capitoltrades rounds the STOCK Act brackets to round numbers ("1K–15K" for the
    $1,001–$15,000 bracket); expanding the suffixes literally yields a low bound exactly
    $1 under the canonical bracket, which ``parse_amount`` snaps back within its $1
    tolerance — so amount semantics stay identical to the Quiver path (same midpoint,
    same bracket label). Falls back to the pre-computed ``size_num`` midpoint when the
    shorthand is missing or unparseable.
    """
    s = "" if size_str is None else str(size_str).strip()
    if s and s.lower() not in ("nan", "n/a", "none", "--"):
        s = _DASH.sub("-", s)
        open_ended = bool(_OPEN_ENDED.search(s))
        parts = [p for p in s.split("-") if p.strip()]
        if open_ended and parts:
            lo = _money(parts[0])
            if lo is not None:
                return f"over ${lo:,.0f}"
        if len(parts) >= 2:
            lo, hi = _money(parts[0]), _money(parts[1])
            if lo is not None and hi is not None:
                return f"${lo:,.0f} - ${hi:,.0f}"
        if len(parts) == 1:
            v = _money(parts[0])
            if v is not None:
                return f"${v:,.0f}"
    # Fallback: the pre-computed numeric midpoint.
    if size_num is not None:
        n = str(size_num).strip()
        if n and n.lower() not in ("nan", "n/a", "none"):
            try:
                return str(float(n))
            except ValueError:
                pass
    return None


def split_politician(raw: object) -> tuple[Optional[str], Optional[str]]:
    """'Ro Khanna Democrat House CA' -> ('Ro Khanna', 'House').

    Parses from the chamber token: everything before it (minus a trailing party token)
    is the name; the chamber word itself is the chamber. Degrades gracefully — if no
    chamber token is found the whole string is returned as the name and chamber is None.
    """
    if raw is None:
        return None, None
    tokens = str(raw).split()
    if not tokens:
        return None, None
    chamber: Optional[str] = None
    cut = len(tokens)
    for i, tok in enumerate(tokens):
        key = tok.strip().lower()
        if key in _CHAMBERS:
            chamber = _CHAMBERS[key]
            cut = i
            break
    name_tokens = tokens[:cut]
    if name_tokens and name_tokens[-1].strip().lower() in _PARTIES:
        name_tokens = name_tokens[:-1]
    name = " ".join(name_tokens).strip() or str(raw).strip()
    return name, chamber


class CapitolTradesCsvAdapter(VendorAdapter):
    """Adapt a scraped capitoltrades.com CSV into :class:`RawDisclosure` rows. No API key."""

    source = "capitoltrades"

    def __init__(
        self,
        csv_path: str,
        *,
        default_txn_type: str = "Purchase",   # buy feed has no direction column
        txn_type_col: Optional[str] = None,    # set if your CSV *does* carry buy/sell
        asset_type_col: Optional[str] = None,  # set if your CSV carries an asset type
        encoding: str = "utf-8",
    ):
        self.csv_path = csv_path
        self.default_txn_type = default_txn_type
        self.txn_type_col = txn_type_col
        self.asset_type_col = asset_type_col
        self.encoding = encoding

    # ----------------------------------------------------------------- io  --
    def _read(self) -> pd.DataFrame:
        """Testable seam: override in tests to inject a frame without touching disk."""
        df = pd.read_csv(self.csv_path, dtype=str, keep_default_na=False, encoding=self.encoding)
        # Normalize headers (strip + lower) so minor header drift doesn't break mapping.
        df.columns = [str(c).strip().lower() for c in df.columns]
        return df

    @staticmethod
    def _val(row: dict, key: str) -> Optional[str]:
        v = row.get(key)
        if v is None:
            return None
        v = str(v).strip()
        return v if v and v.lower() not in ("nan", "none") else None

    def _record_key(self, row: dict, *, politician: Optional[str], txn_type: str) -> str:
        """Deterministic natural key, stable across re-scrapes. Excludes the *mutable*
        fields (size, price) so a corrected re-pull de-duplicates against its original.
        Mirrors the Quiver adapter's keying strategy."""
        rep = (politician or self._val(row, "politician") or "").strip().lower()
        tic = (self._val(row, "ticker") or "").strip().upper()
        txd = (self._val(row, "traded") or "")[:10]
        pub = (self._val(row, "published") or "")[:10]
        own = (self._val(row, "owner") or "").strip().lower()
        basis = f"capitoltrades|{rep}|{tic}|{txd}|{pub}|{txn_type.lower()}|{own}"
        return hashlib.sha1(basis.encode()).hexdigest()[:20]

    # -------------------------------------------------------------- fetch  --
    def fetch_raw(self, *, start: date, end: date) -> Iterable[RawDisclosure]:
        pulled_at = datetime.now(timezone.utc)  # one provenance stamp per pull
        df = self._read()

        for record in df.to_dict(orient="records"):
            row = {str(k).strip().lower(): v for k, v in record.items()}

            # Filing (published) date drives the backfill window, exactly like Quiver bulk.
            published = self._val(row, "published")
            filing_date = parse_date(published) if published else None
            if filing_date is not None and not (start <= filing_date <= end):
                continue  # unparseable dates fall through to silver, which rejects them

            politician, chamber = split_politician(self._val(row, "politician"))

            txn_type = self.default_txn_type
            if self.txn_type_col:
                txn_type = self._val(row, self.txn_type_col.lower()) or self.default_txn_type

            asset_type = (
                self._val(row, self.asset_type_col.lower()) if self.asset_type_col else None
            )

            yield RawDisclosure(
                # provenance / audit trail
                source=self.source,
                source_record_id=self._record_key(row, politician=politician, txn_type=txn_type),
                pulled_at=pulled_at,
                # verbatim-ish payload (strings)
                politician_raw=politician,
                chamber_raw=chamber,
                issuer_raw=self._val(row, "issuer"),
                ticker_raw=self._val(row, "ticker"),
                cusip_raw=None,                         # capitoltrades does not expose CUSIP
                transaction_date_raw=self._val(row, "traded"),
                filing_date_raw=published,
                transaction_type_raw=txn_type,
                owner_raw=self._val(row, "owner"),
                asset_type_raw=asset_type,              # None -> AssetType.UNKNOWN (safe)
                amount_raw=amount_string(row.get("size_str"), row.get("size_num")),
                # the untouched original row, for re-derivation if a parser is fixed later
                raw_payload=record,
            )
