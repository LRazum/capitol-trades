"""
cache.py — joblib-backed cache for analyzed stock data and trained models.

Two uses:
  • memoize raw Quiver pulls and yfinance downloads (keyed by args + a date bucket)
  • store the full ML payload (feature matrix, fitted model, OOS metrics, predictions)
    keyed by a fingerprint of universe + date range + horizon + feature/model config,
    so identical runs reload instantly instead of re-fetching and re-training.

Everything is persisted with joblib.dump/.load under `root` (default ./.cache).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional

import joblib


def fingerprint(*parts: Any) -> str:
    """Stable short hash over JSON-able parts (sets/lists are order-normalized)."""
    def norm(x):
        if isinstance(x, (set, frozenset)):
            return sorted(map(str, x))
        if isinstance(x, dict):
            return {str(k): norm(v) for k, v in sorted(x.items())}
        if isinstance(x, (list, tuple)):
            return [norm(v) for v in x]
        return x
    blob = json.dumps([norm(p) for p in parts], sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


class JoblibCache:
    def __init__(self, root: str | Path = ".cache"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.root / f"{name}.joblib"

    def exists(self, name: str) -> bool:
        return self._path(name).exists()

    def age_seconds(self, name: str) -> Optional[float]:
        p = self._path(name)
        return (time.time() - p.stat().st_mtime) if p.exists() else None

    def save(self, name: str, obj: Any) -> Path:
        p = self._path(name)
        joblib.dump({"_saved_at": time.time(), "obj": obj}, p, compress=3)
        return p

    def load(self, name: str, ttl: Optional[float] = None) -> Optional[Any]:
        p = self._path(name)
        if not p.exists():
            return None
        if ttl is not None and (time.time() - p.stat().st_mtime) > ttl:
            return None
        try:
            return joblib.load(p)["obj"]
        except Exception:
            return None  # corrupt cache entry -> treat as miss

    def memoize(self, name: str, producer, *, ttl: Optional[float] = None,
                force: bool = False) -> Any:
        """Return cached value or compute via producer() and cache it."""
        if not force:
            hit = self.load(name, ttl=ttl)
            if hit is not None:
                return hit
        value = producer()
        self.save(name, value)
        return value

    def clear(self) -> int:
        n = 0
        for p in self.root.glob("*.joblib"):
            p.unlink(); n += 1
        return n

    def entries(self) -> list[dict]:
        out = []
        for p in sorted(self.root.glob("*.joblib")):
            out.append({"name": p.stem,
                        "size_kb": round(p.stat().st_size / 1024, 1),
                        "age_min": round((time.time() - p.stat().st_mtime) / 60, 1)})
        return out
