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
    DIRECTIONAL = "directional"  # Close opposing + trigger entry in sentiment direction


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
        instruments_str = ", ".join(instruments)
        return f"""Analyze this news event for forex/CFD trading impact.

Event: {event.title}
Type: {event.event_type}
Currency: {event.currency or 'N/A'}
Actual: {event.actual or 'N/A'}
Forecast: {event.forecast or 'N/A'}
Previous: {event.previous or 'N/A'}
Impact Level: {event.impact_level or 'N/A'}

Instruments traded: {instruments_str}

Choose ACTION:
- "directional": clear direction — close opposing + enter
- "pause": unclear/mixed — stop trading
- "tighten_stops": moderate — reduce risk
- "none": no action

Give sentiment PER INSTRUMENT (different instruments react
differently to the same news, e.g. BOJ rate hike is bearish
for NIKKEI225 but could be neutral for EUR/USD).

Respond EXACTLY:
ACTION: [none|pause|tighten_stops|directional]
IMPACT_SCORE: [0.0-1.0]
INSTRUMENTS:
  {instruments_str.split(', ')[0]}: [bullish|bearish|none]
  (repeat for each affected instrument, skip unaffected)
REASONING: [one line]"""

    def _parse_response(self, text: str) -> dict[str, Any]:
        """Parse the LLM response into structured data.

        New format includes per-instrument sentiment:
        INSTRUMENTS:
          EUR/USD: bullish
          NIKKEI225: bearish
        """
        result: dict[str, Any] = {
            "action": NewsAction.NONE,
            "sentiment": "neutral",  # Legacy: overall sentiment
            "instrument_sentiments": {},  # New: per-instrument
            "impact_score": 0.0,
            "reasoning": "",
        }

        in_instruments_block = False

        for line in text.strip().split("\n"):
            line = line.strip()
            if line.startswith("ACTION:"):
                action_str = line.split(":", 1)[1].strip().lower()
                try:
                    result["action"] = NewsAction(action_str)
                except ValueError:
                    result["action"] = NewsAction.NONE
                in_instruments_block = False
            elif line.startswith("IMPACT_SCORE:"):
                with contextlib.suppress(ValueError):
                    result["impact_score"] = float(line.split(":", 1)[1].strip())
                in_instruments_block = False
            elif line.startswith("REASONING:"):
                result["reasoning"] = line.split(":", 1)[1].strip()
                in_instruments_block = False
            elif line.startswith("INSTRUMENTS:"):
                in_instruments_block = True
            elif in_instruments_block and ":" in line:
                parts = line.split(":", 1)
                instrument = parts[0].strip()
                sentiment = parts[1].strip().lower()
                if sentiment in ("bullish", "bearish", "none"):
                    result["instrument_sentiments"][instrument] = sentiment
            # Legacy: SENTIMENT line (backwards compat)
            elif line.startswith("SENTIMENT:"):
                result["sentiment"] = line.split(":", 1)[1].strip().lower()
                in_instruments_block = False

        # Derive overall sentiment from instrument sentiments if available
        sentiments = result["instrument_sentiments"]
        if sentiments:
            bullish = sum(1 for s in sentiments.values() if s == "bullish")
            bearish = sum(1 for s in sentiments.values() if s == "bearish")
            if bullish > bearish:
                result["sentiment"] = "bullish"
            elif bearish > bullish:
                result["sentiment"] = "bearish"
            else:
                result["sentiment"] = "neutral"

        return result
