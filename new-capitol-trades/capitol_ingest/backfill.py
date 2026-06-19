"""
backfill.py — multi-year historical backfill into the bronze layer.

Two strategies, both checkpointed (resumable) and wrapped in retry-with-backoff:

* ``run_windowed`` — fetch the adapter once per time window. Fully resumable per window
  and ideal for vendors with date-filtered or paginated endpoints (e.g. FMP).
  WARNING: Quiver's *bulk* endpoint returns the entire history on every call, so this
  re-downloads everything per window — use ``run_bulk_once`` for Quiver bulk.

* ``run_bulk_once`` — one download of the full payload, then commit to bronze
  window-by-window (bounded upserts, progress, resumable commits). The right pattern for
  a return-everything bulk endpoint. For truly enormous payloads, switch to a
  date-filtered/paginated endpoint and ``run_windowed`` instead.

Retry handling is duck-typed (no hard dependency on ``requests``): HTTP 429 / 5xx and
timeout/connection errors are treated as transient with exponential backoff + jitter,
honoring a ``Retry-After`` header when present.
"""
from __future__ import annotations

import json
import random
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Iterator, Optional

from .bronze import disclosures_to_frame, ingest_bronze
from .normalize import VendorAdapter, parse_date
from .storage import BronzeStore

_TRANSIENT_EXC = {
    "Timeout", "ReadTimeout", "ConnectTimeout", "ConnectionError",
    "ChunkedEncodingError", "RequestException", "HTTPError",
}


def _add_month(d: date) -> date:
    return date(d.year + (1 if d.month == 12 else 0), 1 if d.month == 12 else d.month + 1, 1)


def month_windows(start: date, end: date) -> Iterator[tuple[date, date]]:
    cur = date(start.year, start.month, 1)
    while cur <= end:
        nxt = _add_month(cur)
        yield max(cur, start), min(nxt - timedelta(days=1), end)
        cur = nxt


def year_windows(start: date, end: date) -> Iterator[tuple[date, date]]:
    for y in range(start.year, end.year + 1):
        yield max(date(y, 1, 1), start), min(date(y, 12, 31), end)


class HistoricalBackfill:
    def __init__(
        self,
        adapter: VendorAdapter,
        store: BronzeStore,
        *,
        granularity: str = "year",          # "year" or "month"
        max_retries: int = 5,
        base_delay: float = 2.0,
        max_delay: float = 60.0,
        checkpoint_path: Optional[str] = None,
        log: Callable[[str], None] = print,
    ):
        self.adapter = adapter
        self.store = store
        self.granularity = granularity
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path else store.root / ".backfill_checkpoint.json"
        )
        self.log = log
        self._done: set[str] = self._load_ckpt()

    # ----------------------------------------------------------- checkpoint --
    def _load_ckpt(self) -> set[str]:
        if self.checkpoint_path.exists():
            try:
                return set(json.loads(self.checkpoint_path.read_text()))
            except (json.JSONDecodeError, OSError):
                return set()
        return set()

    def _save_ckpt(self) -> None:
        self.checkpoint_path.write_text(json.dumps(sorted(self._done)))

    # ---------------------------------------------------------------- retry --
    def _retry(self, fn: Callable, what: str):
        delay = self.base_delay
        for attempt in range(1, self.max_retries + 1):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                resp = getattr(exc, "response", None)
                status = getattr(resp, "status_code", None)
                name = type(exc).__name__
                transient, retry_after = False, None
                if status is not None:
                    transient = status == 429 or 500 <= status < 600
                    headers = getattr(resp, "headers", None) or {}
                    ra = headers.get("Retry-After")
                    if ra and str(ra).isdigit():
                        retry_after = float(ra)
                elif name in _TRANSIENT_EXC:
                    transient = True

                if not transient or attempt == self.max_retries:
                    self.log(f"  [{what}] giving up after attempt {attempt}: {name}: {exc}")
                    raise
                sleep = retry_after if retry_after is not None else delay + random.uniform(0, 0.25 * delay)
                self.log(f"  [{what}] {name} (attempt {attempt}/{self.max_retries}); retry in {sleep:.1f}s")
                time.sleep(sleep)
                delay = min(2 * delay, self.max_delay)

    # ------------------------------------------------------------- windows  --
    def _windows(self, start: date, end: date) -> Iterator[tuple[date, date]]:
        return month_windows(start, end) if self.granularity == "month" else year_windows(start, end)

    def _window_key(self, start: date, end: date) -> str:
        return f"{self.granularity}:{start.isoformat()}:{end.isoformat()}"

    def _row_window_key(self, fd: Optional[date]) -> str:
        if fd is None:
            return f"{self.granularity}:unparsed"
        return f"month:{fd.year}-{fd.month:02d}" if self.granularity == "month" else f"year:{fd.year}"

    # -------------------------------------------------------------- run: A  --
    def run_windowed(self, start: date, end: date) -> dict:
        summary = {"windows": 0, "skipped": 0, "fetched": 0}
        for ws, we in self._windows(start, end):
            key = self._window_key(ws, we)
            if key in self._done:
                summary["skipped"] += 1
                continue
            self.log(f"window {ws} .. {we}")
            res = self._retry(
                lambda ws=ws, we=we: ingest_bronze(self.adapter, self.store, start=ws, end=we),
                what=key,
            )
            summary["windows"] += 1
            summary["fetched"] += res["fetched"]
            self._done.add(key)
            self._save_ckpt()
        return summary

    # -------------------------------------------------------------- run: B  --
    def run_bulk_once(self, start: date, end: date) -> dict:
        self.log(f"bulk fetch {start} .. {end} (single download)")
        raws = self._retry(
            lambda: list(self.adapter.fetch_raw(start=start, end=end)), what="bulk-fetch"
        )
        self.log(f"  fetched {len(raws)} rows; committing by {self.granularity}")

        buckets: dict[str, list] = {}
        for r in raws:
            fd = parse_date(r.filing_date_raw) if r.filing_date_raw else None
            buckets.setdefault(self._row_window_key(fd), []).append(r)

        summary = {"fetched": len(raws), "windows": 0, "skipped": 0, "committed": 0}
        for wk in sorted(buckets):
            if wk in self._done:
                summary["skipped"] += 1
                continue
            frame = disclosures_to_frame(buckets[wk])
            self.store.upsert(frame)
            summary["windows"] += 1
            summary["committed"] += len(frame)
            self._done.add(wk)
            self._save_ckpt()
            self.log(f"  committed {wk}: {len(frame)} rows")
        return summary
