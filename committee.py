"""
committee.py — real committee-relevance provider from the Stewart-Woon dataset.

Stewart-Woon Congressional Committee Assignments (Harvard Dataverse, doi:10.7910/DVN/MAANO7
and the combined congdata file) records membership per Congress, keyed by ICPSR id + `cong`.
Membership is therefore point-in-time at *Congress* granularity (a 2-year window), so we map
a politician's `tradeable_date` to its active Congress, then check whether that member sat on
a committee whose jurisdiction covers the stock's GICS sector/industry.

Join chain: politician name/bioguide -> ICPSR (via congress-legislators crosswalk, which
carries both ids) -> (icpsr, cong) -> committee names -> jurisdiction match.
"""
from __future__ import annotations

import abc
import re
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

# --------------------------------------------------------------------------- #
# GICS -> committee jurisdiction (editable; substrings matched against committee names)
# --------------------------------------------------------------------------- #
SECTOR_TO_COMMITTEES: dict[str, set[str]] = {
    "Energy": {"energy and commerce", "energy and natural resources", "natural resources"},
    "Materials": {"natural resources", "energy and commerce"},
    "Industrials": {"transportation and infrastructure", "commerce, science", "armed services"},
    "Consumer Discretionary": {"energy and commerce", "ways and means"},
    "Consumer Staples": {"agriculture", "energy and commerce"},
    "Health Care": {"energy and commerce", "health, education", "finance", "ways and means"},
    "Financials": {"financial services", "banking, housing", "ways and means"},
    "Information Technology": {"energy and commerce", "judiciary", "commerce, science", "science, space"},
    "Communication Services": {"energy and commerce", "commerce, science", "judiciary"},
    "Utilities": {"energy and commerce", "energy and natural resources"},
    "Real Estate": {"financial services", "banking, housing"},
}

# Finer industry overrides (take precedence + union with sector). Matches the canonical
# "Armed Services -> Aerospace & Defense" example.
INDUSTRY_TO_COMMITTEES: dict[str, set[str]] = {
    "Aerospace & Defense": {"armed services", "appropriations"},
    "Banks": {"financial services", "banking, housing"},
    "Capital Markets": {"financial services", "banking, housing"},
    "Insurance": {"financial services", "banking, housing"},
    "Pharmaceuticals": {"energy and commerce", "health, education", "finance"},
    "Biotechnology": {"energy and commerce", "health, education", "finance"},
    "Oil, Gas & Consumable Fuels": {"energy and commerce", "energy and natural resources", "natural resources"},
    "Semiconductors & Semiconductor Equipment": {"energy and commerce", "science, space", "commerce, science"},
}


def congress_for_date(d: date) -> int:
    """Congress number active on date d (new Congress convenes ~Jan 3 of odd years)."""
    start_year = d.year if d.year % 2 == 1 else d.year - 1
    if d < date(start_year, 1, 3):
        start_year -= 2
    return (start_year - 1789) // 2 + 1


_TITLE = re.compile(r"\b(rep|sen|representative|senator|dr|mr|mrs|ms|hon)\b\.?", re.I)


def normalize_politician_name(name: str) -> str:
    n = _TITLE.sub(" ", (name or "").lower())
    n = re.sub(r"[^a-z\s]", " ", n)
    return re.sub(r"\s+", " ", n).strip()


# --------------------------------------------------------------------------- #
# Sector classification (inject a real source; static for offline use)
# --------------------------------------------------------------------------- #
class SectorProvider(abc.ABC):
    @abc.abstractmethod
    def get(self, ticker: str) -> Optional[tuple[Optional[str], Optional[str]]]:
        """Return (gics_sector, gics_industry) or None if unknown."""
        ...


class StaticSectorProvider(SectorProvider):
    def __init__(self, mapping: dict[str, tuple[Optional[str], Optional[str]]]):
        self.mapping = {k.upper(): v for k, v in mapping.items()}

    def get(self, ticker: str):
        return self.mapping.get((ticker or "").upper())


# --------------------------------------------------------------------------- #
# Stewart-Woon ingestion
# --------------------------------------------------------------------------- #
class StewartWoonLoader:
    """Read a Stewart-Woon assignment file into a tidy membership table:
    columns [icpsr, cong, committee_code, committee_name]. Robust to the exact column
    names across the House/Senate/combined files via candidate matching."""

    ICPSR = ("icpsr", "id", "idno")
    CONG = ("cong", "congress")
    CODE = ("comcode", "committee", "cmte_code", "code", "committee_code")
    CNAME = ("committee_name", "comname", "name")

    @staticmethod
    def _read(path: str | Path) -> pd.DataFrame:
        p = Path(path)
        if p.suffix in {".tab", ".tsv"}:
            return pd.read_csv(p, sep="\t")
        if p.suffix == ".csv":
            return pd.read_csv(p)
        if p.suffix == ".dta":
            return pd.read_stata(p)
        if p.suffix in {".xlsx", ".xls"}:
            return pd.read_excel(p)
        return pd.read_csv(p, sep=None, engine="python")

    @classmethod
    def _pick(cls, cols, candidates) -> Optional[str]:
        low = {c.lower(): c for c in cols}
        for cand in candidates:
            for lc, orig in low.items():
                if lc == cand or cand in lc:
                    return orig
        return None

    @classmethod
    def tidy(cls, raw: pd.DataFrame, codes: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        cols = raw.columns
        icpsr = cls._pick(cols, cls.ICPSR)
        cong = cls._pick(cols, cls.CONG)
        code = cls._pick(cols, cls.CODE)
        cname = cls._pick(cols, cls.CNAME)
        if not (icpsr and cong and code):
            raise ValueError(f"could not locate icpsr/cong/committee columns in {list(cols)}")

        out = pd.DataFrame({
            "icpsr": pd.to_numeric(raw[icpsr], errors="coerce").astype("Int64"),
            "cong": pd.to_numeric(raw[cong], errors="coerce").astype("Int64"),
            "committee_code": raw[code].astype(str).str.strip(),
        })
        if cname:
            out["committee_name"] = raw[cname].astype(str)
        elif codes is not None:
            cc = codes.columns
            kc = cls._pick(cc, cls.CODE)
            kn = cls._pick(cc, cls.CNAME)
            xwalk = dict(zip(codes[kc].astype(str).str.strip(), codes[kn].astype(str)))
            out["committee_name"] = out["committee_code"].map(xwalk).fillna(out["committee_code"])
        else:
            out["committee_name"] = out["committee_code"]
        return out.dropna(subset=["icpsr", "cong"])

    @classmethod
    def load(cls, assignment_path, codes_path=None) -> pd.DataFrame:
        codes = cls._read(codes_path) if codes_path else None
        return cls.tidy(cls._read(assignment_path), codes)


# --------------------------------------------------------------------------- #
# Provider
# --------------------------------------------------------------------------- #
class StewartWoonCommitteeProvider:
    """Implements the gold CommitteeProvider interface against real membership data."""

    def __init__(
        self,
        membership: pd.DataFrame,
        name_to_icpsr: dict[str, int],
        sector_provider: SectorProvider,
        *,
        sector_map: dict[str, set[str]] = SECTOR_TO_COMMITTEES,
        industry_map: dict[str, set[str]] = INDUSTRY_TO_COMMITTEES,
    ):
        # index (icpsr, cong) -> set of lowercased committee names
        self._index: dict[tuple[int, int], set[str]] = {}
        for r in membership.itertuples():
            key = (int(r.icpsr), int(r.cong))
            self._index.setdefault(key, set()).add(str(r.committee_name).lower())
        # normalize the name->icpsr keys once
        self.name_to_icpsr = {normalize_politician_name(k): int(v) for k, v in name_to_icpsr.items()}
        self.sector_provider = sector_provider
        self.sector_map = sector_map
        self.industry_map = industry_map

    def is_relevant(self, *, politician: str, ticker: str, as_of: date) -> bool:
        icpsr = self.name_to_icpsr.get(normalize_politician_name(politician))
        if icpsr is None:
            return False
        cls_ = self.sector_provider.get(ticker)
        if not cls_:
            return False
        sector, industry = cls_
        targets = set()
        if industry:
            targets |= self.industry_map.get(industry, set())
        if sector:
            targets |= self.sector_map.get(sector, set())
        if not targets:
            return False
        committees = self._index.get((icpsr, congress_for_date(as_of)), set())
        return any(sub in cname for cname in committees for sub in targets)
