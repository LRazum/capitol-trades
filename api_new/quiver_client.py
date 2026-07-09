"""
quiver_client.py — slim, self-contained Quiver client for the ML screener.

Only the datasets the model actually consumes are wired here (congress / insiders /
gov contracts / lobbying / wsb / off-exchange). Each fetch returns a DataFrame; the
feature layer normalizes column-name drift defensively. `validate_key` is used by the
startup gate to confirm the token before the app proceeds.

Auth: Quiver's shipping client uses `Authorization: Token <key>`; docs also accept
`Bearer`. Default is Token; pass auth_scheme="Bearer" if you get 401s.
"""
from __future__ import annotations

import time
from typing import Any, Optional

import pandas as pd
import requests

BASE = "https://api.quiverquant.com/beta"

# dataset key -> (live_slug, historical_slug_or_None, ticker_in_path)
ENDPOINTS: dict[str, tuple[str, Optional[str], bool]] = {
    "congress":     ("live/congresstrading", "historical/congresstrading", True),
    "insiders":     ("live/insiders",        None,                          False),
    "gov_contracts":("live/govcontractsall", "historical/govcontractsall",  True),
    "lobbying":     ("live/lobbying",         "historical/lobbying",         True),
    "wsb":          ("live/wallstreetbets",   "historical/wallstreetbets",   True),
    "offexchange":  ("live/offexchange",      "historical/offexchange",      True),
}


class QuiverError(RuntimeError):
    pass


class QuiverClient:
    def __init__(self, api_key: str, *, auth_scheme: str = "Token",
                 timeout: int = 60, max_retries: int = 4, backoff: float = 1.5):
        if not api_key:
            raise QuiverError("Empty API key.")
        self.api_key = api_key
        self.auth_scheme = auth_scheme
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.session = requests.Session()

    def _headers(self) -> dict[str, str]:
        return {"accept": "application/json",
                "Authorization": f"{self.auth_scheme} {self.api_key}"}

    def _request(self, path: str, params: Optional[dict] = None) -> tuple[int, Any]:
        url = f"{BASE}/{path.lstrip('/')}"
        for attempt in range(self.max_retries + 1):
            try:
                r = self.session.get(url, headers=self._headers(),
                                     params=params or None, timeout=self.timeout)
            except requests.RequestException as exc:
                if attempt < self.max_retries:
                    time.sleep(self.backoff ** attempt); continue
                raise QuiverError(f"network error: {exc}") from exc
            if r.status_code == 429 or r.status_code >= 500:
                if attempt < self.max_retries:
                    time.sleep(float(r.headers.get("Retry-After", self.backoff ** attempt)))
                    continue
            try:
                return r.status_code, r.json()
            except ValueError:
                return r.status_code, r.text
        raise QuiverError("retries exhausted")

    @staticmethod
    def _rows(body: Any) -> list[dict]:
        if isinstance(body, list):
            return body
        if isinstance(body, dict):
            for k in ("data", "results", "rows"):
                if isinstance(body.get(k), list):
                    return body[k]
        return []

    def get(self, path: str, params: Optional[dict] = None) -> pd.DataFrame:
        status, body = self._request(path, params)
        if status >= 400:
            raise QuiverError(f"{status} for /{path} :: {str(body)[:160]}")
        return pd.DataFrame(self._rows(body))

    def fetch(self, key: str, ticker: Optional[str] = None) -> pd.DataFrame:
        if key not in ENDPOINTS:
            raise QuiverError(f"unknown dataset '{key}'")
        live, hist, in_path = ENDPOINTS[key]
        if ticker and hist:
            path = f"{hist}/{ticker.upper()}" if in_path else hist
            params = None if in_path else {"ticker": ticker.upper()}
            return self.get(path, params)
        if ticker and not in_path:           # e.g. insiders live ?ticker=
            return self.get(live, {"ticker": ticker.upper()})
        return self.get(live)

    def validate_key(self) -> tuple[bool, str]:
        """Cheap probe used by the startup gate. Returns (ok, human message)."""
        status, body = self._request("live/congresstrading")
        if status == 200:
            n = len(self._rows(body))
            return True, f"Key OK — congress feed returned {n} recent rows."
        if status in (401, 403):
            return False, "Key rejected (401/403). Check the token, or try Bearer auth."
        if status == 429:
            return False, "Rate-limited (429). Key may be valid but throttled — retry shortly."
        return False, f"Unexpected status {status}: {str(body)[:120]}"
