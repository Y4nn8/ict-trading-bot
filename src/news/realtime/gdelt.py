"""GDELT Project adapter for historical breaking news.

GDELT (Global Database of Events, Language, and Tone) provides free
access to worldwide news events with sentiment scoring.
No API key required.

Uses the GDELT DOC 2.0 API for full-text search with date filtering.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import TYPE_CHECKING, Any

import aiohttp

from src.common.logging import get_logger
from src.common.models import ImpactLevel, NewsEvent
from src.news.base import NewsSource

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from datetime import datetime

logger = get_logger(__name__)

_GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

# Keywords that indicate market-moving financial news
_FINANCE_KEYWORDS = [
    "central bank",
    "interest rate",
    "inflation rate",
    "federal reserve",
    "bank of japan",
    "trade tariff",
    "economic sanctions",
    "stock market crash",
    "currency crisis",
    "oil price surge",
]

# Map GDELT source countries to currencies we trade
_COUNTRY_CURRENCY_MAP = {
    "US": "USD",
    "UK": "GBP",
    "EU": "EUR",
    "JP": "JPY",
    "DE": "EUR",
    "FR": "EUR",
}

# GDELT rate limit: be conservative (no official limit but be polite)
_MIN_SECONDS_BETWEEN_REQUESTS = 10  # GDELT requires 5s, we use 10 for safety


class GDELTNewsSource(NewsSource):
    """Historical breaking news from GDELT Project.

    Free, no API key, covers global events with tone/sentiment.

    Args:
        max_articles_per_query: Maximum articles per query.
    """

    def __init__(self, max_articles_per_query: int = 50) -> None:
        self._max_articles = max_articles_per_query
        self._session: aiohttp.ClientSession | None = None
        self._request_times: deque[float] = deque()

    async def connect(self) -> None:
        """Create HTTP session."""
        self._session = aiohttp.ClientSession()
        await logger.ainfo("gdelt_connected")

    async def disconnect(self) -> None:
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def subscribe(self, instruments: list[str]) -> None:
        """No-op for query API."""

    async def _rate_limit(self) -> None:
        """GDELT requires at least 5 seconds between requests."""
        if self._request_times:
            elapsed = time.monotonic() - self._request_times[-1]
            if elapsed < _MIN_SECONDS_BETWEEN_REQUESTS:
                wait = _MIN_SECONDS_BETWEEN_REQUESTS - elapsed
                await asyncio.sleep(wait)
        self._request_times.append(time.monotonic())

    async def get_events(
        self, start: datetime, end: datetime
    ) -> list[NewsEvent]:
        """Fetch breaking news from GDELT for a date range.

        Queries GDELT DOC API with finance-related keywords,
        filtered by date range.

        Args:
            start: Start datetime.
            end: End datetime.

        Returns:
            List of NewsEvent objects.
        """
        if not self._session:
            await self.connect()
        assert self._session is not None

        all_events: list[NewsEvent] = []

        # GDELT requires OR'd terms wrapped in parentheses
        terms = " OR ".join(f'"{kw}"' for kw in _FINANCE_KEYWORDS)
        query = f"({terms}) sourcelang:english"

        start_str = start.strftime("%Y%m%d%H%M%S")
        end_str = end.strftime("%Y%m%d%H%M%S")

        await self._rate_limit()

        params = {
            "query": query,
            "mode": "ArtList",
            "maxrecords": str(self._max_articles),
            "format": "json",
            "startdatetime": start_str,
            "enddatetime": end_str,
            "sort": "DateDesc",
        }

        try:
            async with self._session.get(_GDELT_DOC_API, params=params) as resp:
                if resp.status != 200:
                    await logger.awarning(
                        "gdelt_request_failed", status=resp.status
                    )
                    return []

                content_type = resp.headers.get("content-type", "")
                if "json" not in content_type:
                    text = await resp.text()
                    await logger.awarning(
                        "gdelt_not_json",
                        content_type=content_type,
                        body=text[:200],
                    )
                    return []

                data = await resp.json()

            articles = data.get("articles", [])
            for article in articles:
                event = _parse_article(article)
                if event:
                    all_events.append(event)

            await logger.ainfo(
                "gdelt_fetched",
                articles=len(articles),
                parsed=len(all_events),
            )

        except Exception as e:
            await logger.awarning("gdelt_fetch_error", error=str(e))

        return all_events

    async def get_events_chunked(
        self, start: datetime, end: datetime, chunk_days: int = 7
    ) -> list[NewsEvent]:
        """Fetch events in chunks to cover a long period.

        Args:
            start: Start datetime.
            end: End datetime.
            chunk_days: Days per chunk.

        Returns:
            All events across all chunks.
        """
        from datetime import timedelta

        all_events: list[NewsEvent] = []
        chunk_start = start

        while chunk_start < end:
            chunk_end = min(chunk_start + timedelta(days=chunk_days), end)
            events = await self.get_events(chunk_start, chunk_end)
            all_events.extend(events)
            chunk_start = chunk_end

        return all_events

    async def stream(self) -> AsyncIterator[NewsEvent]:
        """Not supported for historical source."""
        return
        yield


def _parse_article(article: dict[str, Any]) -> NewsEvent | None:
    """Parse a GDELT article into a NewsEvent."""
    from datetime import UTC, datetime

    title = str(article.get("title", ""))
    if not title:
        return None

    # Parse date
    date_str = str(article.get("seendate", ""))
    try:
        event_time = datetime.strptime(date_str, "%Y%m%dT%H%M%SZ").replace(
            tzinfo=UTC
        )
    except (ValueError, TypeError):
        return None

    # Extract tone (GDELT provides a tone score: positive = bullish sentiment)
    raw_tone = article.get("tone", 0)
    tone = float(raw_tone) if raw_tone else 0.0

    # Determine source country/currency
    source_country = str(article.get("sourcecountry", ""))
    currency = _COUNTRY_CURRENCY_MAP.get(source_country)

    # Determine impact from tone magnitude
    abs_tone = abs(tone)
    if abs_tone > 5:
        impact = ImpactLevel.HIGH
    elif abs_tone > 2:
        impact = ImpactLevel.MEDIUM
    else:
        impact = ImpactLevel.LOW

    return NewsEvent(
        time=event_time,
        source="gdelt",
        event_type="breaking_news",
        title=title,
        content=str(article.get("url", "")),
        currency=currency,
        impact_level=impact,
        llm_analysis={"gdelt_tone": tone},
    )
