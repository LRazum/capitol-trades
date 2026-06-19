"""
storage.py — idempotent, crash-safe bronze storage on partitioned Parquet.

A domain-agnostic ``BronzeStore``:

* Partitions by ``filing_year`` / ``filing_month`` derived from a date column. (Day-level
  partitioning — one dir per filing_date — produces thousands of tiny files; month-level
  is the efficient realization of "partition by filing_date" and still prunes well. Set
  finer/coarser via the constructor if you really want day or year grain.)
* Idempotent upserts: re-running overlapping date ranges never duplicates, because each
  affected partition is read → merged → de-duplicated on a stable key → rewritten.
* Amendment-safe: on a key collision the row with the highest *recency* wins, so a newer
  amendment is never clobbered by an older record and a stale re-fetch can't overwrite a
  newer one. (Recency = ``recency_cols``, e.g. the vendor's ``_last_modified``.)
* Crash-safe: each partition is written to a temp file then ``os.replace``-d into place
  (atomic on one filesystem), so a mid-run failure leaves already-written partitions
  intact and the next run simply re-converges.
* ``compact()`` resolves the rarer case where an amendment landed in a *different*
  filing_date partition than its original (same key, two partitions) by global dedup.

Single-writer assumption (one ingest process at a time), which is the norm for batch ETL.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import pyarrow.dataset as ds

_DATA_FILE = "data.parquet"


class BronzeStore:
    def __init__(
        self,
        root: str | os.PathLike,
        *,
        key_col: str = "source_record_id",
        date_col: str = "filing_date",
        recency_cols: Sequence[str] = ("_last_modified", "pulled_at"),
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.key_col = key_col
        self.date_col = date_col
        self.recency_cols = tuple(recency_cols)

    # ----------------------------------------------------------------- paths --
    def _partition_dir(self, year: int, month: int) -> Path:
        return self.root / f"filing_year={year}" / f"filing_month={month:02d}"

    def _all_partition_dirs(self) -> list[Path]:
        return [p.parent for p in self.root.glob(f"**/{_DATA_FILE}")]

    def _has_data(self) -> bool:
        return any(self.root.glob(f"**/{_DATA_FILE}"))

    # ------------------------------------------------------------- internals --
    @staticmethod
    def _partition_keys(dates: pd.Series) -> tuple[pd.Series, pd.Series]:
        dt = pd.to_datetime(dates, errors="coerce")
        # NaT -> sentinel (0, 0) so unparseable filing dates are quarantined, not lost.
        return dt.dt.year.fillna(0).astype(int), dt.dt.month.fillna(0).astype(int)

    def _read_partition(self, pdir: Path) -> Optional[pd.DataFrame]:
        f = pdir / _DATA_FILE
        return pd.read_parquet(f) if f.exists() else None

    def _dedupe(self, df: pd.DataFrame) -> pd.DataFrame:
        rc = [c for c in self.recency_cols if c in df.columns]
        if rc:
            # Latest recency first, so drop_duplicates(keep="first") keeps the winner.
            df = df.sort_values(rc, ascending=False, na_position="last")
        df = df.drop_duplicates(subset=[self.key_col], keep="first")
        return df.reset_index(drop=True)

    def _atomic_write(self, pdir: Path, df: pd.DataFrame) -> None:
        pdir.mkdir(parents=True, exist_ok=True)
        tmp = pdir / f".tmp-{uuid.uuid4().hex}.parquet"
        df.to_parquet(tmp, engine="pyarrow", index=False)
        os.replace(tmp, pdir / _DATA_FILE)  # atomic commit
        # Consolidate: drop any stale shard files from older multi-file writes.
        for f in pdir.glob("*.parquet"):
            if f.name != _DATA_FILE:
                f.unlink(missing_ok=True)

    # ---------------------------------------------------------------- public --
    def upsert(self, df: pd.DataFrame) -> dict[tuple[int, int], int]:
        """Merge ``df`` into the store. Returns {(year, month): rows_in_partition}."""
        if df.empty:
            return {}
        missing = {self.key_col, self.date_col} - set(df.columns)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")

        df = df.copy()
        df["__yr"], df["__mo"] = self._partition_keys(df[self.date_col])

        written: dict[tuple[int, int], int] = {}
        for (yr, mo), part in df.groupby(["__yr", "__mo"], dropna=False):
            part = part.drop(columns=["__yr", "__mo"])
            pdir = self._partition_dir(yr, mo)
            existing = self._read_partition(pdir)
            combined = part if existing is None else pd.concat([existing, part], ignore_index=True)
            combined = self._dedupe(combined)
            self._atomic_write(pdir, combined)
            written[(yr, mo)] = len(combined)
        return written

    def read(
        self,
        *,
        columns: Optional[list[str]] = None,
        start: Optional[object] = None,
        end: Optional[object] = None,
    ) -> pd.DataFrame:
        """Read the dataset (optionally a filing_date window). Partition pruning applies."""
        if not self._has_data():
            return pd.DataFrame()
        dataset = ds.dataset(self.root, format="parquet", partitioning="hive")
        flt = None
        if start is not None:
            flt = ds.field(self.date_col) >= pd.Timestamp(start)
        if end is not None:
            f2 = ds.field(self.date_col) <= pd.Timestamp(end)
            flt = f2 if flt is None else (flt & f2)
        return dataset.to_table(columns=columns, filter=flt).to_pandas()

    def compact(self) -> dict[tuple[int, int], int]:
        """Global dedup across partitions — resolves amendments that crossed into a
        different filing_date partition than their original. Safe to run periodically."""
        if not self._has_data():
            return {}
        df = self.read()
        df = df.drop(columns=[c for c in ("filing_year", "filing_month") if c in df.columns])
        deduped = self._dedupe(df)
        deduped = deduped.copy()
        deduped["__yr"], deduped["__mo"] = self._partition_keys(deduped[self.date_col])

        existing_dirs = set(self._all_partition_dirs())
        new_dirs: set[Path] = set()
        written: dict[tuple[int, int], int] = {}
        for (yr, mo), part in deduped.groupby(["__yr", "__mo"], dropna=False):
            pdir = self._partition_dir(yr, mo)
            new_dirs.add(pdir)
            self._atomic_write(pdir, part.drop(columns=["__yr", "__mo"]))
            written[(yr, mo)] = len(part)
        # Drop partitions that lost all their rows to a newer version elsewhere.
        for d in existing_dirs - new_dirs:
            (d / _DATA_FILE).unlink(missing_ok=True)
        return written
