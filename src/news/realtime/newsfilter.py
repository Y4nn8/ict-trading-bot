"""Newsfilter.io WebSocket + Query API adapter."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import aiohttp
import websockets

from src.common.logging import get_logger
from src.common.models import NewsEvent
from src.news.base import NewsSource

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from datetime import datetime

logger = get_logger(__name__)


class NewsfilterRealtimeSource(NewsSource):
    """Real-time news via newsfilter.io WebSocket.

    Args:
        ws_url: WebSocket endpoint URL.
    """

    def __init__(
        self,
        ws_url: str = "wss://newsfilter.io/live",
    ) -> None:
        self._ws_url = ws_url
        self._ws: websockets.ClientConnection | None = None
        self._subscribed_instruments: list[str] = []

    async def connect(self) -> None:
        """Establish WebSocket connection."""
        self._ws = await websockets.connect(self._ws_url)
        await logger.ainfo("newsfilter_ws_connected")

    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def subscribe(self, instruments: list[str]) -> None:
        """Subscribe to news for instruments."""
        self._subscribed_instruments = instruments
        if self._ws:
            await self._ws.send(json.dumps({
                "action": "subscribe",
                "instruments": instruments,
            }))

    async def get_events(
        self, start: datetime, end: datetime
    ) -> list[NewsEvent]:
        """Not supported for realtime source. Use NewsfilterHistoricalSource."""
        return []

    async def stream(self) -> AsyncIterator[NewsEvent]:
        """Stream real-time news events.

        Yields:
            NewsEvent objects as they arrive.
        """
        if not self._ws:
            await self.connect()
        assert self._ws is not None

        async for message in self._ws:
            try:
                data = json.loads(message)
                event = _parse_newsfilter_event(data)
                if event:
                    yield event
            except (json.JSONDecodeError, KeyError) as e:
                await logger.awarning("newsfilter_parse_error", error=str(e))


class NewsfilterHistoricalSource(NewsSource):
    """Historical news via newsfilter.io Query API.

    Args:
        api_url: Query API endpoint.
        api_key: API key.
    """

    def __init__(
        self,
        api_url: str = "https://newsfilter.io/api/v1/query",
        api_key: str = "",
    ) -> None:
        self._api_url = api_url
        self._api_key = api_key
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
        """No-op for query API."""

    async def get_events(
        self, start: datetime, end: datetime
    ) -> list[NewsEvent]:
        """Query historical news events.

        Args:
            start: Start datetime.
            end: End datetime.

        Returns:
            List of NewsEvent objects.
        """
        if not self._session:
            await self.connect()
        assert self._session is not None

        payload = {
            "queryString": "*",
            "from": 0,
            "size": 100,
            "dateRange": {
                "gte": start.isoformat(),
                "lte": end.isoformat(),
            },
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}

        async with self._session.post(
            self._api_url, json=payload, headers=headers
        ) as resp:
            data = await resp.json()

        events: list[NewsEvent] = []
        for hit in data.get("hits", []):
            event = _parse_newsfilter_event(hit.get("_source", {}))
            if event:
                events.append(event)

        await logger.ainfo("newsfilter_historical_fetched", count=len(events))
        return events

    async def stream(self) -> AsyncIterator[NewsEvent]:
        """Not supported for historical source."""
        return
        yield


def _parse_newsfilter_event(data: dict[str, object]) -> NewsEvent | None:
    """Parse a newsfilter.io event into a NewsEvent."""
    from datetime import UTC, datetime

    title = data.get("title")
    if not title:
        return None

    return NewsEvent(
        time=datetime.now(tz=UTC),
        source="newsfilter",
        event_type="breaking_news",
        title=str(title),
        content=str(data.get("description", "")),
    )
