"""Core Pydantic models for the trading bot."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from datetime import datetime

from pydantic import BaseModel, Field, model_validator


class Timeframe(StrEnum):
    """Supported timeframes."""

    M5 = "M5"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"


class Direction(StrEnum):
    """Trade direction."""

    LONG = "LONG"
    SHORT = "SHORT"


class ImpactLevel(StrEnum):
    """News event impact level."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Candle(BaseModel):
    """OHLCV candle data."""

    time: datetime
    instrument: str
    timeframe: Timeframe
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    spread: float | None = None

    @model_validator(mode="after")
    def validate_ohlc(self) -> Candle:
        """Ensure high >= low and both contain open/close."""
        if self.high < self.low:
            msg = f"high ({self.high}) must be >= low ({self.low})"
            raise ValueError(msg)
        if self.high < max(self.open, self.close):
            msg = f"high ({self.high}) must be >= max(open, close)"
            raise ValueError(msg)
        if self.low > min(self.open, self.close):
            msg = f"low ({self.low}) must be <= min(open, close)"
            raise ValueError(msg)
        return self


class Trade(BaseModel):
    """Represents a trade (live or backtest)."""

    id: UUID = Field(default_factory=uuid4)
    opened_at: datetime
    closed_at: datetime | None = None
    instrument: str
    direction: Direction
    entry_price: float | None = None
    exit_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    size: float | None = None
    pnl: float | None = None
    pnl_percent: float | None = None
    r_multiple: float | None = None
    confluence_score: float | None = None
    setup_type: dict[str, Any] | None = None
    context: dict[str, Any] | None = None
    news_context: dict[str, Any] | None = None
    is_backtest: bool = False
    backtest_run_id: UUID | None = None


class NewsEvent(BaseModel):
    """Represents a news event."""

    id: UUID = Field(default_factory=uuid4)
    time: datetime
    source: str
    event_type: str
    title: str | None = None
    content: str | None = None
    currency: str | None = None
    actual: str | None = None
    forecast: str | None = None
    previous: str | None = None
    impact_level: ImpactLevel | None = None
    llm_analysis: dict[str, Any] | None = None
    instruments: list[str] = Field(default_factory=list)


class BacktestRun(BaseModel):
    """Represents a backtest run."""

    id: UUID = Field(default_factory=uuid4)
    started_at: datetime
    completed_at: datetime | None = None
    config: dict[str, Any]
    walk_forward: dict[str, Any] | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    git_tag: str | None = None
    improvement_type: str | None = None


class ImprovementRecord(BaseModel):
    """Represents an improvement iteration record."""

    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime
    type: str
    proposal: dict[str, Any]
    baseline_metrics: dict[str, Any]
    new_metrics: dict[str, Any]
    accepted: bool
    reason: str | None = None
    git_tag_before: str | None = None
    git_tag_after: str | None = None
