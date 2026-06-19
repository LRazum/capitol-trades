"""
models.py — Pydantic v2 data models for the Capitol Trades ingestion/normalization pipeline.

Design goals
------------
1.  STRICT date separation. ``transaction_date`` (self-reported, when the trade
    happened) and ``filing_date`` (a.k.a. reported/published date, when the
    disclosure became public) are *both required, separately named, and validated
    against each other*. Only the filing date is ever tradeable. The model exposes
    ``tradeable_date`` and deliberately provides NO generic ``.date`` attribute, so
    downstream code cannot grab the wrong one by accident.
2.  Bronze vs Silver. ``RawDisclosure`` preserves vendor output verbatim (immutable
    bronze). ``NormalizedTrade`` is the cleaned, type-safe silver record that carries
    full provenance back to the raw row.
3.  Survivorship-aware identity. Every normalized trade is anchored to a *permanent*
    security key (FIGI / CIK) plus the ticker that was valid *as of the filing date* —
    never the current ticker.
"""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    model_validator,
)

# Periodic Transaction Reports must be filed within 45 days of the transaction.
STOCK_ACT_MAX_LAG_DAYS = 45


# --------------------------------------------------------------------------- #
# Controlled vocabularies
# --------------------------------------------------------------------------- #
class Chamber(str, Enum):
    HOUSE = "house"
    SENATE = "senate"
    UNKNOWN = "unknown"


class TransactionType(str, Enum):
    PURCHASE = "purchase"
    SALE = "sale"
    SALE_PARTIAL = "sale_partial"
    SALE_FULL = "sale_full"
    EXCHANGE = "exchange"
    RECEIVE = "receive"
    UNKNOWN = "unknown"


class Owner(str, Enum):
    SELF = "self"
    SPOUSE = "spouse"
    JOINT = "joint"
    DEPENDENT = "dependent"
    UNKNOWN = "unknown"


class AssetType(str, Enum):
    STOCK = "stock"
    OPTION = "option"
    ETF = "etf"
    FUND = "fund"
    BOND = "bond"
    CRYPTO = "crypto"
    OTHER = "other"
    UNKNOWN = "unknown"


class ResolutionMethod(str, Enum):
    """How a SecurityRef's permanent anchor was obtained (most → least reliable)."""
    CUSIP = "cusip"
    TICKER = "ticker"
    NAME = "name"
    MANUAL = "manual_override"
    UNRESOLVED = "unresolved"


# --------------------------------------------------------------------------- #
# Security identity (survivorship-aware, point-in-time)
# --------------------------------------------------------------------------- #
class SecurityRef(BaseModel):
    """A resolved security.

    Splits *permanent* anchors (which survive ticker/name changes) from the
    *as-of-date* view (which ticker/name was valid on a given date, and whether the
    security was still trading then). Frozen so a resolved identity can't be mutated
    in place downstream.
    """
    model_config = ConfigDict(frozen=True)

    # Permanent anchors — never change across corporate actions.
    figi: Optional[str] = Field(None, description="shareClassFIGI preferred (most stable)")
    composite_figi: Optional[str] = None
    cik: Optional[str] = Field(None, description="SEC Central Index Key (permanent per filer)")

    # As-of-date view — what was true on `as_of`, NOT today.
    ticker_as_of: Optional[str] = None
    name_as_of: Optional[str] = None
    as_of: Optional[date] = None

    # Survivorship.
    is_delisted: Optional[bool] = None
    delisting_date: Optional[date] = None
    delisting_action: Optional[str] = Field(
        None, description="acquired / merged / bankrupt / unknown"
    )
    still_trading_on_as_of: Optional[bool] = None

    # Bookkeeping.
    method: ResolutionMethod = ResolutionMethod.UNRESOLVED
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    candidates: int = Field(0, description=">1 means the mapping was ambiguous")

    @property
    def is_resolved(self) -> bool:
        return self.method is not ResolutionMethod.UNRESOLVED and bool(self.figi or self.cik)


# --------------------------------------------------------------------------- #
# Amount estimate (from STOCK Act brackets — see amounts.py)
# --------------------------------------------------------------------------- #
class AmountEstimate(BaseModel):
    """Numeric interpretation of a reported amount string.

    Amounts are reported as ranges, not exact values, so we keep the bounds *and*
    several point estimates and let the caller choose which to use downstream.
    """
    model_config = ConfigDict(frozen=True)

    low: Optional[float] = None
    high: Optional[float] = Field(None, description="None => open-ended top bracket")
    midpoint: Optional[float] = None
    geometric_mean: Optional[float] = None
    point_estimate: Optional[float] = Field(
        None, description="The single value to use downstream (configurable)"
    )

    is_exact: bool = False
    is_open_ended: bool = False
    bracket_label: Optional[str] = Field(None, description="Canonical bracket matched, if any")

    raw: Optional[str] = None
    parse_ok: bool = True
    parse_error: Optional[str] = None


# --------------------------------------------------------------------------- #
# Bronze: raw, verbatim vendor row
# --------------------------------------------------------------------------- #
class RawDisclosure(BaseModel):
    """Immutable bronze record. Preserve vendor output as-is; coerce as little as
    possible. ``extra='allow'`` keeps any vendor-specific fields we didn't model."""
    model_config = ConfigDict(extra="allow")

    # Provenance (required for idempotent upserts + audit).
    source: str
    source_record_id: str = Field(..., description="Vendor/filing id — the upsert key")
    pulled_at: datetime

    # Verbatim payload (strings).
    politician_raw: Optional[str] = None
    chamber_raw: Optional[str] = None
    issuer_raw: Optional[str] = None
    ticker_raw: Optional[str] = None
    cusip_raw: Optional[str] = None
    transaction_date_raw: Optional[str] = None
    filing_date_raw: Optional[str] = None
    transaction_type_raw: Optional[str] = None
    owner_raw: Optional[str] = None
    asset_type_raw: Optional[str] = None
    amount_raw: Optional[str] = None

    # The untouched original, for re-derivation when a parser is fixed later.
    raw_payload: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Silver: normalized, type-safe trade
# --------------------------------------------------------------------------- #
class NormalizedTrade(BaseModel):
    """Cleaned silver record. The two dates are intentionally separate and both
    required; the validator forbids the physically impossible ``filing < transaction``.
    """
    model_config = ConfigDict(frozen=False)

    # Identity / provenance.
    trade_id: str
    source: str
    source_record_id: str

    # Who.
    politician: str
    chamber: Chamber = Chamber.UNKNOWN
    owner: Owner = Owner.UNKNOWN

    # What.
    transaction_type: TransactionType = TransactionType.UNKNOWN
    asset_type: AssetType = AssetType.UNKNOWN
    security: SecurityRef
    amount: AmountEstimate

    # WHEN — deliberately two fields, both required.
    transaction_date: date = Field(
        ..., description="Self-reported trade date. NOT tradeable (hindsight)."
    )
    filing_date: date = Field(
        ..., description="Public disclosure date. The ONLY tradeable timestamp."
    )

    @model_validator(mode="after")
    def _check_date_order(self) -> "NormalizedTrade":
        if self.filing_date < self.transaction_date:
            raise ValueError(
                f"filing_date {self.filing_date} precedes transaction_date "
                f"{self.transaction_date}: impossible (cannot file before trading). "
                "Treat as a data-quality reject, not a usable row."
            )
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def disclosure_lag_days(self) -> int:
        """Days between the trade and its public disclosure — itself a candidate feature."""
        return (self.filing_date - self.transaction_date).days

    @computed_field  # type: ignore[prop-decorator]
    @property
    def lag_exceeds_stock_act(self) -> bool:
        return self.disclosure_lag_days > STOCK_ACT_MAX_LAG_DAYS

    @property
    def tradeable_date(self) -> date:
        """The earliest date this information was public. Use THIS for entry timing,
        never ``transaction_date``. The actual fill is the next session >= this date."""
        return self.filing_date

    # NOTE: there is intentionally no generic ``.date`` attribute, to force callers
    # to choose explicitly between transaction_date and filing_date.
