"""News event action manager.

Translates news interpretations into trading actions:
- pause: stop all trading during uncertain/volatile news
- trigger_entry: enter in the direction of strong sentiment
- close_opposing: close positions that go against the news sentiment
- tighten_stops: reduce risk on open positions
- none: no action needed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from src.common.logging import get_logger
from src.news.interpreter import NewsAction

logger = get_logger(__name__)


@dataclass
class EventActionState:
    """Tracks active news-driven trading modifications."""

    paused_until: datetime | None = None
    tighten_stops_until: datetime | None = None
    pending_triggers: list[dict[str, Any]] = field(default_factory=list)
    close_opposing_until: datetime | None = None
    instrument_sentiments: dict[str, str] = field(default_factory=dict)


class EventManager:
    """Manages trading actions triggered by news events.

    Args:
        pre_event_pause_minutes: Minutes to pause before high-impact events.
        post_event_resume_minutes: Minutes after event before resuming.
    """

    def __init__(
        self,
        pre_event_pause_minutes: int = 30,
        post_event_resume_minutes: int = 15,
    ) -> None:
        self._pre_pause = pre_event_pause_minutes
        self._post_resume = post_event_resume_minutes
        self._state = EventActionState()

    def apply_action(
        self,
        action: NewsAction,
        event_time: datetime,
        analysis: dict[str, Any],
    ) -> None:
        """Apply a news-driven action.

        Args:
            action: The action to take.
            event_time: When the event occurs.
            analysis: Full LLM analysis dict.
        """
        if action == NewsAction.PAUSE:
            pause_end = event_time + timedelta(minutes=self._post_resume)
            self._state.paused_until = pause_end
            logger.info(
                "trading_paused",
                until=pause_end.isoformat(),
                reason=analysis.get("reasoning", ""),
            )

        elif action == NewsAction.TIGHTEN_STOPS:
            self._state.tighten_stops_until = event_time + timedelta(
                minutes=self._post_resume
            )

        elif action == NewsAction.DIRECTIONAL:
            # Store per-instrument sentiments for directional action
            inst_sentiments = analysis.get("instrument_sentiments", {})
            if inst_sentiments:
                self._state.instrument_sentiments = dict(inst_sentiments)
            else:
                # Fallback to global sentiment for all instruments
                sentiment = analysis.get("sentiment", "neutral")
                if sentiment in ("bullish", "bearish"):
                    self._state.instrument_sentiments = {
                        "__all__": sentiment
                    }
            self._state.close_opposing_until = event_time + timedelta(
                minutes=self._post_resume
            )
            # Also trigger entries
            self._state.pending_triggers.append(analysis)
            logger.info(
                "directional_action",
                sentiments=self._state.instrument_sentiments,
                reason=analysis.get("reasoning", ""),
            )

    def is_paused(self, current_time: datetime) -> bool:
        """Check if trading is currently paused."""
        if self._state.paused_until is None:
            return False
        return current_time < self._state.paused_until

    def should_tighten_stops(self, current_time: datetime) -> bool:
        """Check if stops should be tightened."""
        if self._state.tighten_stops_until is None:
            return False
        return current_time < self._state.tighten_stops_until

    def get_instrument_sentiments(
        self, current_time: datetime
    ) -> dict[str, str]:
        """Get per-instrument sentiments for directional actions.

        Returns:
            Dict of instrument → "bullish"/"bearish".
            If "__all__" key exists, applies to all instruments.
            Empty dict if no active directional action.
        """
        if self._state.close_opposing_until is None:
            return {}
        if current_time >= self._state.close_opposing_until:
            self._state.close_opposing_until = None
            self._state.instrument_sentiments = {}
            return {}
        return self._state.instrument_sentiments

    def pop_triggers(self) -> list[dict[str, Any]]:
        """Get and clear pending entry triggers."""
        triggers = self._state.pending_triggers
        self._state.pending_triggers = []
        return triggers
