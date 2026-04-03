"""Market data ingestion: fetch from IG and store in TimescaleDB."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from src.common.exceptions import MarketDataError
from src.common.logging import get_logger

_TF_MINUTES: dict[str, int] = {"M5": 5, "H1": 60, "H4": 240, "D1": 1440}

if TYPE_CHECKING:
    from src.common.config import InstrumentConfig
    from src.market_data.ig_client import IGClient
    from src.market_data.storage import CandleStorage

logger = get_logger(__name__)


class MarketDataIngester:
    """Orchestrates fetching market data from IG and storing it.

    Args:
        ig_client: Authenticated IG Markets client.
        storage: Candle storage manager.
    """

    def __init__(self, ig_client: IGClient, storage: CandleStorage) -> None:
        self._ig_client = ig_client
        self._storage = storage

    async def ingest_historical(
        self,
        instrument: InstrumentConfig,
        days: int = 180,
        timeframe: str = "M5",
    ) -> int:
        """Fetch and store historical candle data for an instrument.

        Automatically resumes from the last stored candle if data already exists.
        Fetches in weekly chunks to respect IG API limits.

        Args:
            instrument: Instrument configuration.
            days: Number of days of history to fetch.
            timeframe: Base timeframe to fetch.

        Returns:
            Total number of candles stored.
        """
        end = datetime.now(tz=UTC)
        start = end - timedelta(days=days)

        # Resume from last stored candle (offset by base timeframe interval)
        latest = await self._storage.get_latest_candle_time(instrument.name, timeframe)
        if latest is not None:
            start = latest + timedelta(minutes=_TF_MINUTES.get(timeframe, 5))
            await logger.ainfo(
                "resuming_ingestion",
                instrument=instrument.name,
                from_time=start.isoformat(),
            )

        if start >= end:
            await logger.ainfo(
                "data_up_to_date", instrument=instrument.name, timeframe=timeframe
            )
            return 0

        total_stored = 0
        chunk_start = start
        chunk_size = timedelta(days=7)  # Weekly chunks for IG API limits

        while chunk_start < end:
            chunk_end = min(chunk_start + chunk_size, end)

            try:
                df = self._ig_client.fetch_historical_candles(
                    epic=instrument.epic,
                    resolution=timeframe,
                    start=chunk_start,
                    end=chunk_end,
                )
            except MarketDataError as e:
                if "exceeded" in str(e).lower() or "allowance" in str(e).lower():
                    await logger.awarning(
                        "rate_limit_reached",
                        instrument=instrument.name,
                        stored_so_far=total_stored,
                    )
                    break
                raise

            if not df.is_empty():
                stored = await self._storage.upsert_candles(
                    instrument=instrument.name,
                    timeframe=timeframe,
                    df=df,
                )
                total_stored += stored

            chunk_start = chunk_end

        await logger.ainfo(
            "ingestion_complete",
            instrument=instrument.name,
            timeframe=timeframe,
            total_candles=total_stored,
        )
        return total_stored

    async def refresh_aggregates(self) -> None:
        """Manually refresh TimescaleDB continuous aggregates.

        Triggers a refresh of H1, H4, D1 views. This is normally handled
        automatically by TimescaleDB policies, but can be called manually
        after bulk inserts.
        """
        for view in ("candles_h1", "candles_h4", "candles_d1"):
            try:
                await self._storage.execute_raw(
                    f"CALL refresh_continuous_aggregate('{view}', NULL, NULL)"
                )
                await logger.ainfo("aggregate_refreshed", view=view)
            except Exception as e:
                await logger.awarning(
                    "aggregate_refresh_failed", view=view, error=str(e)
                )
