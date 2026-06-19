"""
amounts.py — normalize STOCK Act PTR amount strings into numeric estimates.

Disclosures report dollar amounts as *brackets*, not exact figures
(e.g. "$15,001 - $50,000"), with the occasional exact value ("$360.00") and an
open-ended top bracket ("over $50,000,000"). This module turns any of those into an
:class:`~models.AmountEstimate` with bounds plus several point estimates, snapping to
the canonical PTR bracket set where possible and *flagging* (never silently guessing)
anything it can't parse.
"""
from __future__ import annotations

import math
import re
from typing import Optional

from .models import AmountEstimate

# Canonical PTR amount brackets: (lower, upper). upper=None => open-ended.
# The first ($1–$1,000) appears on some filings; standard brackets start at $1,001.
PTR_BRACKETS: list[tuple[float, Optional[float]]] = [
    (1.0, 1_000.0),
    (1_001.0, 15_000.0),
    (15_001.0, 50_000.0),
    (50_001.0, 100_000.0),
    (100_001.0, 250_000.0),
    (250_001.0, 500_000.0),
    (500_001.0, 1_000_000.0),
    (1_000_001.0, 5_000_000.0),
    (5_000_001.0, 25_000_000.0),
    (25_000_001.0, 50_000_000.0),
    (50_000_001.0, None),
]

# Tolerance (in dollars) for snapping a parsed bound to a canonical bracket bound.
_SNAP_TOL = 1.0

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_OPEN_ENDED = re.compile(r"(over|more than|greater than|at least|and over|or more|\+)", re.I)
_DASH = re.compile(r"[\u2010-\u2015\u2212]")  # hyphen/en/em dash + minus sign -> "-"


def _to_float(token: str) -> Optional[float]:
    try:
        return float(token.replace(",", "").replace("$", "").strip())
    except (ValueError, AttributeError):
        return None


def geometric_mean(low: float, high: float) -> Optional[float]:
    if low is None or high is None or low <= 0 or high <= 0:
        return None
    return math.sqrt(low * high)


def _label_for(low: Optional[float], high: Optional[float]) -> Optional[str]:
    """Return a canonical bracket label if (low, high) matches one within tolerance."""
    for blo, bhi in PTR_BRACKETS:
        lo_ok = low is not None and abs(low - blo) <= _SNAP_TOL
        if bhi is None:
            hi_ok = high is None
        else:
            hi_ok = high is not None and abs(high - bhi) <= _SNAP_TOL
        if lo_ok and hi_ok:
            top = "+" if bhi is None else f"{bhi:,.0f}"
            return f"${blo:,.0f}-{top}"
    return None


def _fill_high_from_bracket(low: float) -> tuple[Optional[float], bool]:
    """Given only a lower bound matching a canonical bracket, return its upper bound.
    Returns (high, is_open_ended)."""
    for blo, bhi in PTR_BRACKETS:
        if abs(low - blo) <= _SNAP_TOL:
            return bhi, (bhi is None)
    return None, False


def parse_amount(
    raw: Optional[str],
    *,
    default: str = "midpoint",
    open_ended_factor: float = 1.5,
) -> AmountEstimate:
    """Parse a reported amount string into an :class:`AmountEstimate`.

    Parameters
    ----------
    default : {"midpoint", "geometric", "low", "high"}
        Which estimate to expose as ``point_estimate``. For the open-ended top
        bracket (no upper bound) the point estimate falls back to
        ``low * open_ended_factor`` and the result is flagged ``is_open_ended``.
    open_ended_factor : float
        Multiplier applied to the lower bound for the open-ended bracket. This is an
        explicit, documented convention — not a hidden assumption — and the row is
        flagged so you can exclude or sensitivity-test it downstream.
    """
    if not raw or not str(raw).strip():
        return AmountEstimate(raw=raw, parse_ok=False, parse_error="empty")

    s = _DASH.sub("-", str(raw)).strip()
    s_low = s.lower()
    nums = [_to_float(m.group()) for m in _NUM.finditer(s)]
    nums = [n for n in nums if n is not None]
    open_ended = bool(_OPEN_ENDED.search(s_low))

    low: Optional[float] = None
    high: Optional[float] = None
    is_exact = False
    is_open_ended = False

    if len(nums) >= 2:
        low, high = nums[0], nums[1]
    elif len(nums) == 1:
        n = nums[0]
        has_dash_range = ("-" in s) or (" to " in s_low)
        if open_ended:
            low, high, is_open_ended = n, None, True
        elif has_dash_range:
            # e.g. "$15,001 -" with the upper omitted: recover from the bracket.
            low = n
            high, is_open_ended = _fill_high_from_bracket(n)
            if high is None and not is_open_ended:
                high = n  # could not recover; degenerate point
        else:
            low = high = n
            is_exact = True
    else:
        return AmountEstimate(raw=raw, parse_ok=False, parse_error="no_numbers")

    # Open-ended top bracket: no arithmetic midpoint is meaningful.
    if high is None:
        is_open_ended = True
        point = (low * open_ended_factor) if (low is not None and default != "low") else low
        return AmountEstimate(
            low=low,
            high=None,
            midpoint=None,
            geometric_mean=None,
            point_estimate=point,
            is_exact=False,
            is_open_ended=True,
            bracket_label=_label_for(low, None),
            raw=raw,
            parse_ok=low is not None,
            parse_error=None if low is not None else "no_lower_bound",
        )

    midpoint = (low + high) / 2.0 if (low is not None and high is not None) else None
    gmean = geometric_mean(low, high) if (low is not None and high is not None) else None

    point = {
        "midpoint": midpoint,
        "geometric": gmean,
        "low": low,
        "high": high,
    }.get(default, midpoint)

    return AmountEstimate(
        low=low,
        high=high,
        midpoint=midpoint,
        geometric_mean=gmean,
        point_estimate=point,
        is_exact=is_exact,
        is_open_ended=is_open_ended,
        bracket_label=_label_for(low, high),
        raw=raw,
        parse_ok=True,
        parse_error=None,
    )
