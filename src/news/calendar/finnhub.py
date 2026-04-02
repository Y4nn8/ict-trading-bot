"""Finnhub economic calendar adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

import aiohttp

from src.common.logging import get_logger
from src.common.models import ImpactLevel, NewsEvent
from src.news.base import NewsSource

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from datetime import datetime

logger = get_logger(__name__)


class FinnhubCalendarSource(NewsSource):
    """Fetches economic calendar data from Finnhub REST API.

    Args:
        api_key: Finnhub API key.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._base_url = "https://finnhub.io/api/v1"
        self._session: aiohttp.ClientSession | None = None

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

    async def get_events(
        self, start: datetime, end: datetime
    ) -> list[NewsEvent]:
        """Fetch economic calendar events from Finnhub.

        Args:
            start: Start date.
            end: End date.

        Returns:
            List of NewsEvent objects.
        """
        if not self._session:
            await self.connect()
        assert self._session is not None

        params = {
            "from": start.strftime("%Y-%m-%d"),
            "to": end.strftime("%Y-%m-%d"),
            "token": self._api_key,
        }

        async with self._session.get(
            f"{self._base_url}/calendar/economic", params=params
        ) as resp:
            data = await resp.json()

        events: list[NewsEvent] = []
        for item in data.get("economicCalendar", []):
            impact = _map_impact(item.get("impact"))
            events.append(NewsEvent(
                time=start,  # Finnhub doesn't always provide exact time
                source="finnhub",
                event_type="economic_calendar",
                title=item.get("event", ""),
                currency=item.get("country", ""),
                actual=str(item.get("actual", "")),
                forecast=str(item.get("estimate", "")),
                previous=str(item.get("prev", "")),
                impact_level=impact,
            ))

        await logger.ainfo("finnhub_events_fetched", count=len(events))
        return events

    async def stream(self) -> AsyncIterator[NewsEvent]:
        """Not supported for REST API."""
        return
        yield


def _map_impact(impact_str: str | None) -> ImpactLevel | None:
    """Map Finnhub impact string to ImpactLevel enum."""
    if not impact_str:
        return None
    mapping = {"1": ImpactLevel.LOW, "2": ImpactLevel.MEDIUM, "3": ImpactLevel.HIGH}
    return mapping.get(str(impact_str))
