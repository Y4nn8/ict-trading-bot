"""News event database storage with aligned timestamps."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.common.logging import get_logger

if TYPE_CHECKING:
    from datetime import datetime

    from src.common.db import Database
    from src.common.models import NewsEvent

logger = get_logger(__name__)


class NewsStore:
    """Stores and retrieves news events from the database.

    Args:
        db: Database connection manager.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    async def save_event(self, event: NewsEvent) -> None:
        """Save a news event to the database.

        Args:
            event: NewsEvent to save.
        """
        import json

        query = """
            INSERT INTO news_events
                (id, time, source, event_type, title, content, currency,
                 actual, forecast, previous, impact_level, llm_analysis, instruments)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            ON CONFLICT (id) DO NOTHING
        """
        await self._db.execute(
            query,
            event.id,
            event.time,
            event.source,
            event.event_type,
            event.title,
            event.content,
            event.currency,
            event.actual,
            event.forecast,
            event.previous,
            event.impact_level.value if event.impact_level else None,
            json.dumps(event.llm_analysis) if event.llm_analysis else None,
            event.instruments,
        )

    async def save_events(self, events: list[NewsEvent]) -> int:
        """Save multiple news events (batched).

        Args:
            events: List of NewsEvent objects.

        Returns:
            Number of events saved.
        """
        if not events:
            return 0

        import json

        query = """
            INSERT INTO news_events
                (id, time, source, event_type, title, content, currency,
                 actual, forecast, previous, impact_level, llm_analysis, instruments)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            ON CONFLICT (id) DO NOTHING
        """
        args = [
            (
                event.id,
                event.time,
                event.source,
                event.event_type,
                event.title,
                event.content,
                event.currency,
                event.actual,
                event.forecast,
                event.previous,
                event.impact_level.value if event.impact_level else None,
                json.dumps(event.llm_analysis) if event.llm_analysis else None,  # asyncpg needs str for jsonb
                event.instruments,
            )
            for event in events
        ]
        await self._db.executemany(query, args)
        await logger.ainfo("news_events_saved", count=len(events))
        return len(events)

    async def get_events(
        self,
        start: datetime,
        end: datetime,
        currency: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch news events from the database.

        Args:
            start: Start datetime (inclusive).
            end: End datetime (exclusive).
            currency: Optional currency filter.

        Returns:
            List of event dicts.
        """
        conditions = ["time >= $1", "time < $2"]
        params: list[object] = [start, end]

        if currency:
            conditions.append("currency = $3")
            params.append(currency)

        where = " AND ".join(conditions)
        query = f"""
            SELECT id, time, source, event_type, title, content, currency,
                   actual, forecast, previous, impact_level, llm_analysis, instruments
            FROM news_events
            WHERE {where}
            ORDER BY time ASC
        """

        records = await self._db.fetch(query, *params)
        return [dict(r) for r in records]

    async def get_events_at_time(
        self,
        target_time: datetime,
        tolerance_minutes: int = 5,
    ) -> list[dict[str, Any]]:
        """Get news events near a specific timestamp.

        Used for backtest replay to check if news is active at a candle time.

        Args:
            target_time: The target timestamp.
            tolerance_minutes: Minutes of tolerance around the target.

        Returns:
            List of event dicts near the target time.
        """
        from datetime import timedelta

        start = target_time - timedelta(minutes=tolerance_minutes)
        end = target_time + timedelta(minutes=tolerance_minutes)
        return await self.get_events(start, end)
