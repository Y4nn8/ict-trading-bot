"""Tick replay engine for the Midas scalping system.

Streams ticks from TimescaleDB, builds 10s candles, runs all feature
extractors at every tick, and collects the feature matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from src.common.logging import get_logger
from src.midas.candle_builder import CandleBuilder
from src.midas.extractors.tick_features import TickFeatureExtractor
from src.midas.feature_extractor import FeatureRegistry
from src.midas.types import Tick

if TYPE_CHECKING:
    from pathlib import Path

    from src.common.db import Database

logger = get_logger(__name__)


@dataclass
class ReplayResult:
    """Result of a tick replay run."""

    total_ticks: int = 0
    total_candles: int = 0
    feature_rows: int = 0
    output_path: Path | None = None


@dataclass
class ReplayConfig:
    """Configuration for a replay run.

    Args:
        instrument: Instrument to replay.
        start: Replay start time (inclusive).
        end: Replay end time (exclusive).
        bucket_seconds: Candle duration in seconds.
        chunk_size: Number of ticks per DB fetch.
        output_path: Path to write Parquet output.
            None means features are collected in memory.
        sample_rate: Extract features every N ticks (1 = every tick).
            Higher values reduce output size for initial exploration.
    """

    instrument: str = "XAUUSD"
    start: datetime = field(default_factory=datetime.now)
    end: datetime = field(default_factory=datetime.now)
    bucket_seconds: int = 10
    chunk_size: int = 500_000
    output_path: Path | None = None
    sample_rate: int = 1


class ReplayEngine:
    """Stream ticks from DB, build candles, extract features.

    Args:
        db: Database connection.
        registry: Feature extractor registry.
        config: Replay configuration.
    """

    def __init__(
        self,
        db: Database,
        registry: FeatureRegistry,
        config: ReplayConfig,
    ) -> None:
        self._db = db
        self._registry = registry
        self._config = config
        self._builder = CandleBuilder(
            bucket_seconds=config.bucket_seconds,
        )

    async def run(self) -> ReplayResult:
        """Run the full replay and return results.

        Streams ticks in chunks via asyncpg cursor for memory efficiency.
        Writes features to Parquet in streaming chunks if output_path is set.
        """
        import polars as pl

        cfg = self._config
        result = ReplayResult()
        feature_chunks: list[list[dict[str, float]]] = []
        current_chunk: list[dict[str, float]] = []
        chunk_flush_size = 100_000
        tick_counter = 0

        await logger.ainfo(
            "replay_start",
            instrument=cfg.instrument,
            start=str(cfg.start),
            end=str(cfg.end),
            sample_rate=cfg.sample_rate,
        )

        async with self._db.pool.acquire() as conn, conn.transaction():
                cursor = conn.cursor(
                    "SELECT time, bid, ask FROM ticks "
                    "WHERE instrument = $1 AND time >= $2 AND time < $3 "
                    "ORDER BY time ASC",
                    cfg.instrument,
                    cfg.start,
                    cfg.end,
                )

                while True:
                    rows = await cursor.fetch(cfg.chunk_size)
                    if not rows:
                        break

                    for row in rows:
                        tick = Tick(
                            time=row["time"],
                            bid=float(row["bid"]),
                            ask=float(row["ask"]),
                        )
                        result.total_ticks += 1
                        tick_counter += 1

                        # Process tick through candle builder
                        closed = self._builder.process_tick(tick)
                        if closed is not None:
                            result.total_candles += 1
                            self._registry.on_candle_close(
                                closed, self._builder.candle_index - 1,
                            )
                            # Record spread for tick extractor
                            self._record_spread_on_close(tick.spread)

                        # Extract features (respecting sample rate)
                        if tick_counter % cfg.sample_rate == 0:
                            partial = self._builder.partial
                            if partial is not None:
                                features = self._registry.extract_all(
                                    tick,
                                    partial,
                                    self._builder.candle_index,
                                )
                                features["_time"] = tick.time.timestamp()
                                features["_bid"] = tick.bid
                                features["_ask"] = tick.ask
                                current_chunk.append(features)
                                result.feature_rows += 1

                        # Flush chunk to list
                        if len(current_chunk) >= chunk_flush_size:
                            feature_chunks.append(current_chunk)
                            current_chunk = []
                            await logger.ainfo(
                                "replay_progress",
                                ticks=result.total_ticks,
                                candles=result.total_candles,
                                features=result.feature_rows,
                            )

        # Flush last partial candle
        last_candle = self._builder.flush()
        if last_candle is not None:
            result.total_candles += 1
            self._registry.on_candle_close(
                last_candle, self._builder.candle_index - 1,
            )

        # Flush remaining features
        if current_chunk:
            feature_chunks.append(current_chunk)

        await logger.ainfo(
            "replay_complete",
            total_ticks=result.total_ticks,
            total_candles=result.total_candles,
            feature_rows=result.feature_rows,
        )

        # Write output
        if feature_chunks:
            all_rows = [
                row for chunk in feature_chunks for row in chunk
            ]
            df = pl.DataFrame(all_rows)
            if cfg.output_path is not None:
                df.write_parquet(cfg.output_path)
                result.output_path = cfg.output_path
                await logger.ainfo(
                    "parquet_written",
                    path=str(cfg.output_path),
                    rows=len(df),
                    columns=len(df.columns),
                )

        return result

    def _record_spread_on_close(self, spread: float) -> None:
        """Record spread in TickFeatureExtractor on candle close."""
        for ext in self._registry.extractors:
            if isinstance(ext, TickFeatureExtractor):
                ext._record_spread(spread)


def build_default_registry(instrument: str = "XAUUSD") -> FeatureRegistry:
    """Create a FeatureRegistry with all default extractors.

    Args:
        instrument: Instrument name for ICT/HTF extractors.

    Returns:
        Configured FeatureRegistry.
    """
    from src.midas.extractors.ict_features import ICTFeatureExtractor
    from src.midas.extractors.scalping_features import ScalpingFeatureExtractor
    from src.midas.extractors.tick_features import TickFeatureExtractor

    registry = FeatureRegistry()
    registry.register(TickFeatureExtractor())
    registry.register(ScalpingFeatureExtractor())
    registry.register(ICTFeatureExtractor(instrument=instrument))
    return registry
