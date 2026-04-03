"""Finnhub economic calendar adapter with rate limiting.

Free tier: 60 requests/minute. This adapter tracks request timestamps
and sleeps when approaching the limit.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from datetime import timedelta
from typing import TYPE_CHECKING

import aiohttp

from src.common.logging import get_logger
from src.common.models import ImpactLevel, NewsEvent
from src.news.base import NewsSource

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from datetime import datetime

logger = get_logger(__name__)

_MAX_REQUESTS_PER_MINUTE = 55  # Stay below 60 to be safe

_IMPACT_MAP = {"1": ImpactLevel.LOW, "2": ImpactLevel.MEDIUM, "3": ImpactLevel.HIGH}

_COUNTRY_TO_CURRENCY = {
    "US": "USD", "EU": "EUR", "GB": "GBP",
    "JP": "JPY", "DE": "EUR", "FR": "EUR",
}

_RELEVANT_COUNTRIES = set(_COUNTRY_TO_CURRENCY.keys())


class FinnhubCalendarSource(NewsSource):
    """Fetches economic calendar data from Finnhub REST API.

    Includes rate limiting to stay within free tier (60 req/min).

    Args:
        api_key: Finnhub API key.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._base_url = "https://finnhub.io/api/v1"
        self._session: aiohttp.ClientSession | None = None
        self._request_times: deque[float] = deque()

    async def connect(self) -> None:
        """Create HTTP session."""
        self._session = aiohttp.ClientSession()
        await logger.ainfo("finnhub_connected")

    async def disconnect(self) -> None:
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def subscribe(self, instruments: list[str]) -> None:
        """No-op for REST API source."""

    async def _rate_limit(self) -> None:
        """Wait if we're approaching the rate limit."""
        now = time.monotonic()
        while self._request_times and now - self._request_times[0] > 60:
            self._request_times.popleft()

        if len(self._request_times) >= _MAX_REQUESTS_PER_MINUTE:
            wait_time = 60 - (now - self._request_times[0]) + 0.5
            if wait_time > 0:
                await logger.ainfo(
                    "finnhub_rate_limit_waiting", seconds=round(wait_time, 1)
                )
                await asyncio.sleep(wait_time)

        self._request_times.append(time.monotonic())

    async def get_events(
        self, start: datetime, end: datetime
    ) -> list[NewsEvent]:
        """Fetch economic calendar events from Finnhub.

        Fetches in weekly chunks with rate limiting.

        Args:
            start: Start date.
            end: End date.

        Returns:
            List of NewsEvent objects.
        """
        if not self._session:
            await self.connect()
        assert self._session is not None

        all_events: list[NewsEvent] = []
        chunk_start = start

        while chunk_start < end:
            chunk_end = min(chunk_start + timedelta(days=7), end)
            await self._rate_limit()

            params = {
                "from": chunk_start.strftime("%Y-%m-%d"),
                "to": chunk_end.strftime("%Y-%m-%d"),
                "token": self._api_key,
            }

            try:
                async with self._session.get(
                    f"{self._base_url}/calendar/economic", params=params
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        await logger.awarning(
                            "finnhub_request_failed",
                            status=resp.status,
                            body=text[:200],
                        )
                        chunk_start = chunk_end
                        continue
                    data = await resp.json()

                for item in data.get("economicCalendar", []):
                    event = _parse_event(item)
                    if event:
                        all_events.append(event)

                await logger.ainfo(
                    "finnhub_chunk_fetched",
                    start=chunk_start.strftime("%Y-%m-%d"),
                    events=len(data.get("economicCalendar", [])),
                )

            except Exception as e:
                await logger.awarning(
                    "finnhub_chunk_error",
                    start=chunk_start.strftime("%Y-%m-%d"),
                    error=str(e),
                )

            chunk_start = chunk_end

        await logger.ainfo("finnhub_fetch_complete", total=len(all_events))
        return all_events

    async def stream(self) -> AsyncIterator[NewsEvent]:
        """Not supported for REST API."""
        return
        yield


def _parse_event(item: dict[str, object]) -> NewsEvent | None:
    """Parse a Finnhub calendar item into a NewsEvent."""
    from datetime import UTC, datetime

    country = str(item.get("country", ""))
    if country not in _RELEVANT_COUNTRIES:
        return None

    currency = _COUNTRY_TO_CURRENCY.get(country, country)
    impact = _IMPACT_MAP.get(str(item.get("impact", "")))

    date_str = str(item.get("date", ""))
    try:
        event_time = datetime.fromisoformat(date_str).replace(tzinfo=UTC)
    except (ValueError, TypeError):
        try:
            event_time = datetime.strptime(date_str, "%Y-%m-%d").replace(
                tzinfo=UTC
            )
        except (ValueError, TypeError):
            return None

    return NewsEvent(
        time=event_time,
        source="finnhub",
        event_type="economic_calendar",
        title=str(item.get("event", "")),
        currency=currency,
        actual=str(item.get("actual", "")),
        forecast=str(item.get("estimate", "")),
        previous=str(item.get("prev", "")),
        impact_level=impact,
    )
