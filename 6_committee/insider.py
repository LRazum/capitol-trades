"""
insider.py — real Form 4 insider overlay from the SEC bulk Insider Transactions Data Sets.

The SEC publishes quarterly flat files (SUBMISSION / REPORTINGOWNER / NONDERIV_TRANS, keyed
by ACCESSION_NUMBER). We join them into a tidy per-transaction table, then compute trailing
net *officer* buying (purchases minus sales) as of a politician's tradeable_date.

Point-in-time correctness has two filters, both essential:
  * FILING_DATE <= as_of   — only Form 4s that were already public could have informed a
    decision on as_of (Form 4 files within 2 business days, so most window trades are known,
    but the last ~2 days may not be — this filter handles that exactly).
  * TRANS_DATE in (as_of - lookback, as_of]  — the trade itself falls in the trailing window.

Open-market trades are TRANS_CODE P (purchase) and S (sale); grants/exercises/gifts (A/M/G…)
are excluded.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _pick(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    low = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    for cand in candidates:  # substring fallback
        for lc, orig in low.items():
            if cand.lower() in lc:
                return orig
    return None


def _officer_flags(reportingowner: pd.DataFrame) -> pd.Series:
    """Robustly derive an is_officer boolean per row across the schema's variants."""
    col = _pick(reportingowner, "isofficer", "rptowner_isofficer", "officer_flag")
    if col is not None:
        s = reportingowner[col].astype(str).str.lower()
        return s.isin({"1", "true", "y", "yes", "t"})
    # else parse a relationship/title text column for "officer"
    txt_col = _pick(reportingowner, "rptowner_relationship", "rptowner_txt", "rptowner_title", "officertitle")
    if txt_col is not None:
        return reportingowner[txt_col].astype(str).str.contains("officer", case=False, na=False)
    return pd.Series(False, index=reportingowner.index)


class Form4BulkLoader:
    """Build a tidy transaction table from the SEC quarterly datasets."""

    @staticmethod
    def _read_tsv(path: str | Path) -> pd.DataFrame:
        return pd.read_csv(path, sep="\t", dtype=str, low_memory=False)

    @staticmethod
    def tidy_from_tables(
        submission: pd.DataFrame,
        reportingowner: pd.DataFrame,
        nonderiv: pd.DataFrame,
        *,
        include_amendments: bool = False,
    ) -> pd.DataFrame:
        acc_s = _pick(submission, "accession_number")
        sub = pd.DataFrame({
            "accession_number": submission[acc_s].astype(str),
            "filing_date": pd.to_datetime(submission[_pick(submission, "filing_date", "filed_date")], errors="coerce"),
            "ticker": submission[_pick(submission, "issuertradingsymbol", "issuerticker", "issuer_symbol")].astype(str).str.upper(),
            "issuer_cik": submission[_pick(submission, "issuercik", "issuer_cik")].astype(str),
            "doc_type": submission[_pick(submission, "document_type", "doc_type")].astype(str),
        })
        if include_amendments:
            sub = sub[sub["doc_type"].str.startswith("4")]
        else:
            sub = sub[sub["doc_type"] == "4"]

        # reporting owner -> officer flag (one filing may have several owners; take "any officer")
        ro = pd.DataFrame({
            "accession_number": reportingowner[_pick(reportingowner, "accession_number")].astype(str),
            "rptowner_cik": reportingowner[_pick(reportingowner, "rptownercik", "rptowner_cik")].astype(str),
            "is_officer": _officer_flags(reportingowner).to_numpy(),
        })

        acc_n = _pick(nonderiv, "accession_number")
        nd = pd.DataFrame({
            "accession_number": nonderiv[acc_n].astype(str),
            "trans_date": pd.to_datetime(nonderiv[_pick(nonderiv, "trans_date", "transaction_date")], errors="coerce"),
            "trans_code": nonderiv[_pick(nonderiv, "trans_code", "transaction_code")].astype(str).str.upper(),
            "shares": pd.to_numeric(nonderiv[_pick(nonderiv, "trans_shares", "transaction_shares")], errors="coerce"),
            "price": pd.to_numeric(nonderiv[_pick(nonderiv, "trans_pricepershare", "transaction_price")], errors="coerce"),
        })
        nd = nd[nd["trans_code"].isin({"P", "S"})]

        df = nd.merge(sub, on="accession_number", how="inner").merge(ro, on="accession_number", how="left")
        df["is_officer"] = df["is_officer"].fillna(False)
        df["value"] = (df["shares"].fillna(0) * df["price"].fillna(0)).abs()
        sign = np.where(df["trans_code"] == "P", 1.0, -1.0)
        df["signed_value"] = sign * df["value"]
        df["signed_shares"] = sign * df["shares"].fillna(0).abs()
        return df.dropna(subset=["trans_date", "filing_date", "ticker"]).reset_index(drop=True)

    @classmethod
    def load_quarter(cls, directory: str | Path, **kw) -> pd.DataFrame:
        d = Path(directory)
        return cls.tidy_from_tables(
            cls._read_tsv(d / "SUBMISSION.tsv"),
            cls._read_tsv(d / "REPORTINGOWNER.tsv"),
            cls._read_tsv(d / "NONDERIV_TRANS.tsv"),
            **kw,
        )

    @classmethod
    def build_tidy(cls, quarter_dirs: list, out_path: str | Path) -> pd.DataFrame:
        frames = [cls.load_quarter(d) for d in quarter_dirs]
        tidy = pd.concat(frames, ignore_index=True).sort_values(["ticker", "trans_date"])
        tidy.to_parquet(out_path, engine="pyarrow", index=False)
        return tidy


class Form4InsiderProvider:
    """Implements the gold InsiderProvider interface. Pre-groups the tidy table by ticker
    for fast windowed queries. (For very large universes, sort by (ticker, trans_date) and
    use searchsorted instead of boolean masks.)"""

    def __init__(self, tidy, *, officers_only: bool = True, windows=(30, 60)):
        if not isinstance(tidy, pd.DataFrame):
            tidy = pd.read_parquet(tidy)
        self.officers_only = officers_only
        self.windows = tuple(windows)
        self._by_ticker = {tkr: g.reset_index(drop=True) for tkr, g in tidy.groupby("ticker")}

    def overlay(self, *, ticker: str, as_of, lookback_days: int) -> dict:
        out = {f"insider_net_{w}": 0.0 for w in self.windows}
        out.update({"insider_net_buys": 0.0, "insider_n_buyers": 0.0,
                    "insider_buy_value": 0.0, "insider_available": True})
        g = self._by_ticker.get((ticker or "").upper())
        if g is None or g.empty:
            return out

        as_of = pd.Timestamp(as_of)
        # point-in-time: only filings public by as_of, transactions on/before as_of
        known = g[(g["filing_date"] <= as_of) & (g["trans_date"] <= as_of)]
        if self.officers_only:
            known = known[known["is_officer"]]
        if known.empty:
            return out

        def net(days):
            w = known[known["trans_date"] > as_of - pd.Timedelta(days=days)]
            return float(w["signed_value"].sum())

        for wdays in self.windows:
            out[f"insider_net_{wdays}"] = net(wdays)
        out["insider_net_buys"] = net(lookback_days)

        primary = known[known["trans_date"] > as_of - pd.Timedelta(days=lookback_days)]
        buys = primary[primary["trans_code"] == "P"]
        out["insider_n_buyers"] = float(buys["rptowner_cik"].nunique())
        out["insider_buy_value"] = float(buys["value"].sum())
        return out
