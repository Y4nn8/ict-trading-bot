"""LLM-based news interpretation using Claude Haiku.

Analyzes news events and determines their likely market impact
on specific instruments.
"""

from __future__ import annotations

import contextlib
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from src.common.logging import get_logger

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic

    from src.common.models import NewsEvent

logger = get_logger(__name__)


class NewsAction(StrEnum):
    """Action to take based on news interpretation."""

    NONE = "none"
    PAUSE = "pause"
    TIGHTEN_STOPS = "tighten_stops"
    TRIGGER_ENTRY = "trigger_entry"


class NewsInterpreter:
    """Interprets news events using Claude Haiku LLM.

    Args:
        client: Anthropic async client.
        model: Model to use for interpretation.
    """

    def __init__(
        self,
        client: AsyncAnthropic,
        model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        self._client = client
        self._model = model

    async def interpret(
        self,
        event: NewsEvent,
        instruments: list[str],
    ) -> dict[str, Any]:
        """Interpret a news event's impact on instruments.

        Args:
            event: The news event to analyze.
            instruments: List of instruments to assess impact for.

        Returns:
            Dict with keys: action, sentiment, impact_score, reasoning,
            affected_instruments.
        """
        prompt = self._build_prompt(event, instruments)

        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )

            block = response.content[0] if response.content else None
            text = block.text if block and hasattr(block, "text") else ""
            analysis = self._parse_response(text)

            await logger.ainfo(
                "news_interpreted",
                event_title=event.title,
                action=analysis.get("action"),
            )
            return analysis

        except Exception as e:
            await logger.awarning("news_interpretation_failed", error=str(e))
            return {
                "action": NewsAction.NONE,
                "sentiment": "neutral",
                "impact_score": 0.0,
                "reasoning": f"Interpretation failed: {e}",
                "affected_instruments": [],
            }

    def _build_prompt(self, event: NewsEvent, instruments: list[str]) -> str:
        """Build the LLM prompt for news interpretation."""
        return f"""Analyze this economic news event for forex/CFD trading impact.

Event: {event.title}
Type: {event.event_type}
Currency: {event.currency or 'N/A'}
Actual: {event.actual or 'N/A'}
Forecast: {event.forecast or 'N/A'}
Previous: {event.previous or 'N/A'}
Impact Level: {event.impact_level or 'N/A'}

Instruments being traded: {', '.join(instruments)}

Respond with EXACTLY this format:
ACTION: [none|pause|tighten_stops|trigger_entry]
SENTIMENT: [bullish|bearish|neutral]
IMPACT_SCORE: [0.0-1.0]
AFFECTED: [comma-separated instrument list]
REASONING: [one line explanation]"""

    def _parse_response(self, text: str) -> dict[str, Any]:
        """Parse the LLM response into structured data."""
        result: dict[str, Any] = {
            "action": NewsAction.NONE,
            "sentiment": "neutral",
            "impact_score": 0.0,
            "reasoning": "",
            "affected_instruments": [],
        }

        for line in text.strip().split("\n"):
            line = line.strip()
            if line.startswith("ACTION:"):
                action_str = line.split(":", 1)[1].strip().lower()
                try:
                    result["action"] = NewsAction(action_str)
                except ValueError:
                    result["action"] = NewsAction.NONE
            elif line.startswith("SENTIMENT:"):
                result["sentiment"] = line.split(":", 1)[1].strip().lower()
            elif line.startswith("IMPACT_SCORE:"):
                with contextlib.suppress(ValueError):
                    result["impact_score"] = float(line.split(":", 1)[1].strip())
            elif line.startswith("AFFECTED:"):
                instruments_str = line.split(":", 1)[1].strip()
                result["affected_instruments"] = [
                    i.strip() for i in instruments_str.split(",") if i.strip()
                ]
            elif line.startswith("REASONING:"):
                result["reasoning"] = line.split(":", 1)[1].strip()

        return result
