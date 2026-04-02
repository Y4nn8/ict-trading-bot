"""ForexFactory News API historical calendar adapter."""

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

_FF_IMPACT_MAP = {
    "High": ImpactLevel.HIGH,
    "Medium": ImpactLevel.MEDIUM,
    "Low": ImpactLevel.LOW,
}


class FFApiHistoricalSource(NewsSource):
    """Fetches historical economic calendar from ForexFactory-style API.

    Args:
        base_url: API base URL.
    """

    def __init__(
        self,
        base_url: str = "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
    ) -> None:
        self._base_url = base_url
        self._session: aiohttp.ClientSession | None = None

    async def connect(self) -> None:
        """Create HTTP session."""
        self._session = aiohttp.ClientSession()

    async def disconnect(self) -> None:
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def subscribe(self, instruments: list[str]) -> None:
        """No-op for REST source."""

    async def get_events(
        self, start: datetime, end: datetime
    ) -> list[NewsEvent]:
        """Fetch calendar events.

        Args:
            start: Start date.
            end: End date.

        Returns:
            List of NewsEvent objects.
        """
        if not self._session:
            await self.connect()
        assert self._session is not None

        async with self._session.get(self._base_url) as resp:
            data = await resp.json()

        events: list[NewsEvent] = []
        for item in data if isinstance(data, list) else []:
            impact = _FF_IMPACT_MAP.get(item.get("impact", ""))
            events.append(NewsEvent(
                time=start,
                source="ff_api",
                event_type="economic_calendar",
                title=item.get("title", ""),
                currency=item.get("country", ""),
                actual=str(item.get("actual", "")),
                forecast=str(item.get("forecast", "")),
                previous=str(item.get("previous", "")),
                impact_level=impact,
            ))

        await logger.ainfo("ff_events_fetched", count=len(events))
        return events

    async def stream(self) -> AsyncIterator[NewsEvent]:
        """Not supported for historical source."""
        return
        yield
