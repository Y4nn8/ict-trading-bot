"""NewsSource abstract base class (adapter pattern).

All news sources implement this interface. Adding a new source
means implementing this ABC, nothing else.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from datetime import datetime

    from src.common.models import NewsEvent


class NewsSource(ABC):
    """Abstract base class for news data sources."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the news source."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the news source."""

    @abstractmethod
    async def subscribe(self, instruments: list[str]) -> None:
        """Subscribe to news for specific instruments.

        Args:
            instruments: List of instrument names or currencies.
        """

    @abstractmethod
    async def get_events(
        self, start: datetime, end: datetime
    ) -> list[NewsEvent]:
        """Get historical news events in a time range.

        Args:
            start: Start datetime (inclusive).
            end: End datetime (exclusive).

        Returns:
            List of NewsEvent objects.
        """

    @abstractmethod
    async def stream(self) -> AsyncIterator[NewsEvent]:
        """Stream real-time news events.

        Yields:
            NewsEvent objects as they arrive.
        """
        yield  # type: ignore[misc]
