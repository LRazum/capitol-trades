"""
check_setup.py — 30-second environment doctor for Quiver Signals.

Run:  python check_setup.py
Exits 0 if everything is ready; prints exactly what to fix otherwise.
"""
from __future__ import annotations

import importlib
import sys

FAILED = []


def check(name: str, fn):
    try:
        detail = fn()
        print(f"  ✓ {name}" + (f" ({detail})" if detail else ""))
    except Exception as exc:
        print(f"  ✗ {name}: {exc}")
        FAILED.append(name)


def dep(mod: str, minimal: str = ""):
    m = importlib.import_module(mod)
    v = getattr(m, "__version__", "?")
    return f"{v}{' — need >= ' + minimal if minimal else ''}"


def main() -> int:
    print(f"Python: {sys.version.split()[0]}  @  {sys.executable}")
    if sys.version_info < (3, 10):
        print("  ✗ Python 3.10+ required.")
        FAILED.append("python")

    print("\nDependencies:")
    for mod in ("streamlit", "plotly", "sklearn", "joblib", "pandas", "numpy",
                "requests", "yfinance"):
        check(mod, lambda mod=mod: dep(mod))
    # The one that has actually bitten this project: old starlette breaking streamlit import.
    check("starlette gzip symbol (streamlit compatibility)", lambda: (
        importlib.import_module("starlette.middleware.gzip").DEFAULT_EXCLUDED_CONTENT_TYPES
        and dep("starlette")))

    print("\nApp modules:")
    for mod in ("quiver_client", "cache", "features", "ml", "viz"):
        check(mod, lambda mod=mod: importlib.import_module(mod) and "")

    print("\nQuick engine smoke test (synthetic, ~10s):")
    def smoke():
        import warnings; warnings.filterwarnings("ignore")
        import features as F, ml as ML
        prov = F.SyntheticProvider(n_tickers=8, days=260)
        prices = prov.prices(); feeds = F.normalize_feeds(prov.feeds())
        ds = F.build_dataset(feeds, prices, horizon=21)
        r = ML.evaluate(ds, n_splits=4, n_null=0)
        assert r.model is not None and len(ds.X) > 0
        return f"{len(ds.X)} samples, model trains & scores"
    check("features → model → scores", smoke)

    print()
    if FAILED:
        print(f"NOT READY — fix the ✗ items above ({', '.join(FAILED)}).")
        print("Usually:  python -m pip install -U -r requirements.txt")
        return 1
    print("READY — launch with:  streamlit run app.py   (or ./run.sh)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
