"""
security_master.py — point-in-time security identification + survivorship handling.

Separates the two operations people conflate:

  1. RESOLVE a (possibly stale) identifier -> a *permanent* anchor (FIGI/CIK).
     Implemented live against the free OpenFIGI API. NOTE: OpenFIGI returns the
     *current* mapping, not an as-of-date one, and (for licensing reasons) does not
     return CUSIP/ISIN/SEDOL in its output.

  2. Look up the AS-OF-DATE listing: which ticker/name was valid on a date, and whether
     the security was still trading (delisting date + reason). This is the survivorship
     layer and is provider-pluggable — back it with CRSP / Sharadar / EODHD / FMP.

``SecurityMaster`` is the facade: ``resolve(...)`` returns a :class:`~models.SecurityRef`
combining both, with a SQLite cache, manual overrides, and explicit ambiguity handling.
"""
from __future__ import annotations

import abc
import json
import os
import re
import sqlite3
import time
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Optional

from .models import ResolutionMethod, SecurityRef

# --------------------------------------------------------------------------- #
# Issuer-name normalization (for name-based fallback resolution)
# --------------------------------------------------------------------------- #
_NAME_NOISE = re.compile(
    r"\b(inc|incorporated|corp|corporation|co|company|ltd|limited|llc|l\.?p|plc|sa|nv|ag|"
    r"holdings?|group|the|class\s+[a-c]|cl\s+[a-c]|common\s+stock|ordinary\s+shares?|"
    r"american\s+depositary\s+shares?|adr)\b\.?",
    re.I,
)


def normalize_name(name: str) -> str:
    n = re.sub(r"[^a-z0-9 ]", " ", name.lower())
    n = _NAME_NOISE.sub(" ", n)
    return re.sub(r"\s+", " ", n).strip()


# --------------------------------------------------------------------------- #
# Provider interfaces (swap implementations without touching the pipeline)
# --------------------------------------------------------------------------- #
class IdentifierResolver(abc.ABC):
    """Maps an identifier to permanent anchor(s) + a (current) ticker/name."""

    @abc.abstractmethod
    def by_cusip(self, cusip: str) -> list[dict[str, Any]]:
        ...

    @abc.abstractmethod
    def by_ticker(self, ticker: str) -> list[dict[str, Any]]:
        ...


@dataclass
class ListingStatus:
    ticker_as_of: Optional[str] = None
    name_as_of: Optional[str] = None
    is_delisted: Optional[bool] = None
    delisting_date: Optional[date] = None
    delisting_action: Optional[str] = None
    still_trading_on_as_of: Optional[bool] = None


class ListingHistoryProvider(abc.ABC):
    """As-of-date listing + delisting facts. Back with CRSP / Sharadar / EODHD / FMP.
    Required to avoid survivorship bias: it tells you the ticker valid on the filing
    date and whether the name had already left the tape."""

    @abc.abstractmethod
    def status(
        self,
        *,
        figi: Optional[str],
        cik: Optional[str],
        ticker: Optional[str],
        as_of: date,
    ) -> ListingStatus:
        ...


class NullListingHistory(ListingHistoryProvider):
    """No-op so the pipeline runs before you wire a paid temporal source.
    Returns 'unknown' for everything (None), which downstream must treat as a flag."""

    def status(self, *, figi, cik, ticker, as_of) -> ListingStatus:  # noqa: D102
        return ListingStatus(ticker_as_of=ticker)


# --------------------------------------------------------------------------- #
# Live resolver: OpenFIGI (free)
# --------------------------------------------------------------------------- #
class OpenFigiResolver(IdentifierResolver):
    """Resolve CUSIP/ticker -> FIGI via the free OpenFIGI v3 mapping API.

    Set ``OPENFIGI_API_KEY`` for the higher rate limit. Results are filtered to US
    equities to cut down 1->N ambiguity. A key practical note baked in here: v3 returns
    a ``warning`` (not ``error``) for no-match, which we treat as an empty result.
    """

    BASE = "https://api.openfigi.com/v3/mapping"

    def __init__(self, api_key: Optional[str] = None, sleep: float = 0.25):
        self.api_key = api_key or os.getenv("OPENFIGI_API_KEY")
        self.sleep = sleep

    def _post(self, jobs: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        import requests  # lazy import so the module loads without the dep present

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-OPENFIGI-APIKEY"] = self.api_key
        resp = requests.post(self.BASE, headers=headers, data=json.dumps(jobs), timeout=30)
        resp.raise_for_status()
        out: list[list[dict[str, Any]]] = []
        for item in resp.json():
            out.append(item.get("data", []) if isinstance(item, dict) else [])
        if self.sleep:
            time.sleep(self.sleep)
        return out

    @staticmethod
    def _clean(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Prefer common stock / equity; dedupe to share-class level.
        eq = [r for r in rows if r.get("marketSector") == "Equity"]
        rows = eq or rows
        seen, cleaned = set(), []
        for r in rows:
            key = r.get("shareClassFIGI") or r.get("figi")
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(
                {
                    "figi": r.get("shareClassFIGI") or r.get("figi"),
                    "composite_figi": r.get("compositeFIGI"),
                    "ticker": r.get("ticker"),
                    "name": r.get("name"),
                    "exch": r.get("exchCode"),
                    "security_type": r.get("securityType"),
                }
            )
        return cleaned

    def by_cusip(self, cusip: str) -> list[dict[str, Any]]:
        jobs = [{"idType": "ID_CUSIP", "idValue": cusip, "marketSecDes": "Equity"}]
        return self._clean(self._post(jobs)[0])

    def by_ticker(self, ticker: str) -> list[dict[str, Any]]:
        jobs = [
            {
                "idType": "TICKER",
                "idValue": ticker,
                "exchCode": "US",
                "marketSecDes": "Equity",
            }
        ]
        return self._clean(self._post(jobs)[0])


# --------------------------------------------------------------------------- #
# Tiny SQLite cache (dependency-free, dependency on stdlib only)
# --------------------------------------------------------------------------- #
class _SqliteCache:
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT, ts REAL)"
        )
        self.conn.commit()

    def get(self, key: str) -> Optional[dict]:
        row = self.conn.execute("SELECT v FROM kv WHERE k = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key: str, value: dict) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO kv (k, v, ts) VALUES (?, ?, ?)",
            (key, json.dumps(value), time.time()),
        )
        self.conn.commit()


# --------------------------------------------------------------------------- #
# Facade
# --------------------------------------------------------------------------- #
class SecurityMaster:
    """Resolve issuer/CUSIP/ticker -> permanent, survivorship-aware ``SecurityRef``.

    Resolution precedence: manual override -> cache -> CUSIP -> ticker -> (name).
    Name resolution is intentionally weak (low confidence) because it is the most
    error-prone path; prefer feeding CUSIPs from the vendor when available.
    """

    def __init__(
        self,
        resolver: IdentifierResolver,
        history: Optional[ListingHistoryProvider] = None,
        cache_path: str = "security_master_cache.sqlite",
        overrides: Optional[dict[str, dict[str, Any]]] = None,
    ):
        self.resolver = resolver
        self.history = history or NullListingHistory()
        self.cache = _SqliteCache(cache_path)
        # overrides keyed by "cusip:<X>", "ticker:<X>", or "name:<normalized>"
        self.overrides = overrides or {}

    # -- internal helpers ---------------------------------------------------- #
    def _cache_key(self, cusip, ticker, name) -> str:
        if cusip:
            return f"cusip:{cusip}"
        if ticker:
            return f"ticker:{ticker.upper()}"
        return f"name:{normalize_name(name or '')}"

    def _attach_history(self, base: dict, as_of: date) -> dict:
        st = self.history.status(
            figi=base.get("figi"),
            cik=base.get("cik"),
            ticker=base.get("ticker"),
            as_of=as_of,
        )
        base.update(
            {
                "ticker_as_of": st.ticker_as_of or base.get("ticker"),
                "name_as_of": st.name_as_of or base.get("name"),
                "is_delisted": st.is_delisted,
                "delisting_date": st.delisting_date.isoformat() if st.delisting_date else None,
                "delisting_action": st.delisting_action,
                "still_trading_on_as_of": st.still_trading_on_as_of,
            }
        )
        return base

    # -- public API ---------------------------------------------------------- #
    def resolve(
        self,
        *,
        as_of: date,
        issuer_name: Optional[str] = None,
        cusip: Optional[str] = None,
        ticker: Optional[str] = None,
    ) -> SecurityRef:
        # 1) manual override
        for key in (
            f"cusip:{cusip}" if cusip else None,
            f"ticker:{ticker.upper()}" if ticker else None,
            f"name:{normalize_name(issuer_name)}" if issuer_name else None,
        ):
            if key and key in self.overrides:
                ov = dict(self.overrides[key])
                ov = self._attach_history(ov, as_of)
                return _to_ref(ov, ResolutionMethod.MANUAL, 1.0, 1, as_of)

        # 2) cache
        ckey = self._cache_key(cusip, ticker, issuer_name)
        cached = self.cache.get(ckey)
        if cached is not None:
            cached = self._attach_history(cached, as_of)  # history is as-of, recompute
            return _to_ref(
                cached,
                ResolutionMethod(cached.get("_method", "unresolved")),
                cached.get("_confidence", 0.0),
                cached.get("_candidates", 0),
                as_of,
            )

        # 3) resolve permanent anchor by best available identifier
        method = ResolutionMethod.UNRESOLVED
        candidates: list[dict] = []
        if cusip:
            candidates = self.resolver.by_cusip(cusip)
            method = ResolutionMethod.CUSIP
        if not candidates and ticker:
            candidates = self.resolver.by_ticker(ticker)
            method = ResolutionMethod.TICKER
        # NOTE: name->FIGI is left as an override/manual path on purpose; automated
        #       name matching belongs behind its own provider with fuzzy scoring.

        if not candidates:
            return SecurityRef(
                ticker_as_of=ticker,
                name_as_of=issuer_name,
                as_of=as_of,
                method=ResolutionMethod.UNRESOLVED,
                confidence=0.0,
                candidates=0,
            )

        # 4) ambiguity handling: choose the US-composite common-stock candidate;
        #    record the candidate count and damp confidence when >1.
        chosen = _pick_best(candidates)
        n = len(candidates)
        confidence = (1.0 if n == 1 else 0.6) * (1.0 if method is ResolutionMethod.CUSIP else 0.8)

        base = {
            "figi": chosen.get("figi"),
            "composite_figi": chosen.get("composite_figi"),
            "cik": None,
            "ticker": chosen.get("ticker"),
            "name": chosen.get("name"),
            "_method": method.value,
            "_confidence": confidence,
            "_candidates": n,
        }
        self.cache.set(ckey, base)
        base = self._attach_history(base, as_of)
        return _to_ref(base, method, confidence, n, as_of)

    def survivorship_check(self, ref: SecurityRef) -> Optional[str]:
        """Return a human-readable warning if this security needs special handling so
        the backtest doesn't silently drop it (delisted) or mis-time it (unresolved)."""
        if not ref.is_resolved:
            return "UNRESOLVED security — exclude or override; never default to a live ticker."
        if ref.is_delisted:
            return (
                f"DELISTED ({ref.delisting_action or 'unknown'} on {ref.delisting_date}). "
                "Carry terminal value (acquisition price / ~0 for bankruptcy); do not drop."
            )
        if ref.still_trading_on_as_of is False:
            return "Security was NOT trading on the filing date — check entry timing."
        return None


# --------------------------------------------------------------------------- #
# module-level helpers
# --------------------------------------------------------------------------- #
def _pick_best(candidates: list[dict]) -> dict:
    def score(r: dict) -> tuple:
        return (
            r.get("exch") == "US",
            (r.get("security_type") or "").lower() in {"common stock", "share"},
            bool(r.get("ticker")),
        )

    return sorted(candidates, key=score, reverse=True)[0]


def _to_ref(
    base: dict, method: ResolutionMethod, confidence: float, candidates: int, as_of: date
) -> SecurityRef:
    dl = base.get("delisting_date")
    return SecurityRef(
        figi=base.get("figi"),
        composite_figi=base.get("composite_figi"),
        cik=base.get("cik"),
        ticker_as_of=base.get("ticker_as_of") or base.get("ticker"),
        name_as_of=base.get("name_as_of") or base.get("name"),
        as_of=as_of,
        is_delisted=base.get("is_delisted"),
        delisting_date=date.fromisoformat(dl) if isinstance(dl, str) else dl,
        delisting_action=base.get("delisting_action"),
        still_trading_on_as_of=base.get("still_trading_on_as_of"),
        method=method,
        confidence=confidence,
        candidates=candidates,
    )
