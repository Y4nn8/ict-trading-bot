"""News event action manager.

Translates news interpretations into trading actions:
pause trading, tighten stops, trigger entries, or do nothing.
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

        elif action == NewsAction.TRIGGER_ENTRY:
            self._state.pending_triggers.append(analysis)

    def is_paused(self, current_time: datetime) -> bool:
        """Check if trading is currently paused.

        Args:
            current_time: Current timestamp.

        Returns:
            True if trading should be paused.
        """
        if self._state.paused_until is None:
            return False
        return current_time < self._state.paused_until

    def should_tighten_stops(self, current_time: datetime) -> bool:
        """Check if stops should be tightened.

        Args:
            current_time: Current timestamp.

        Returns:
            True if stops should be tighter than normal.
        """
        if self._state.tighten_stops_until is None:
            return False
        return current_time < self._state.tighten_stops_until

    def pop_triggers(self) -> list[dict[str, Any]]:
        """Get and clear pending entry triggers.

        Returns:
            List of trigger analysis dicts.
        """
        triggers = self._state.pending_triggers
        self._state.pending_triggers = []
        return triggers
