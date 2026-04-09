"""Tick replay engine for the Midas scalping system.

Streams ticks from TimescaleDB, builds 10s candles, runs all feature
extractors at every tick, and collects the feature matrix.

Supports two modes:
  - **Collect mode** (default): accumulates features → Parquet file.
  - **Callback mode**: streams features to a user callback for live
    prediction + trade simulation (used during walk-forward testing).

Optionally integrates a TickLabeler for training data generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Protocol

from src.common.logging import get_logger
from src.midas.candle_builder import CandleBuilder
from src.midas.feature_extractor import FeatureRegistry
from src.midas.types import Tick

if TYPE_CHECKING:
    from pathlib import Path

    from src.common.db import Database
    from src.midas.labeler import LabelResult, TickLabeler

logger = get_logger(__name__)


class TickCallback(Protocol):
    """Protocol for sampled tick callbacks (features + signal)."""

    def __call__(self, tick: Tick, features: dict[str, float]) -> None: ...


class EveryTickHook(Protocol):
    """Protocol for per-tick hook (called on EVERY tick for exit checks)."""

    def __call__(self, tick: Tick) -> None: ...


@dataclass
class ReplayResult:
    """Result of a tick replay run."""

    total_ticks: int = 0
    total_candles: int = 0
    feature_rows: int = 0
    output_path: Path | None = None
    label_result: LabelResult | None = None


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
        sample_on_candle: Extract features on each candle close (recommended).
            When True, sample_rate is ignored.
        sample_rate: Legacy: extract features every N ticks.
            Only used when sample_on_candle is False.
    """

    instrument: str = "XAUUSD"
    start: datetime = field(default_factory=datetime.now)
    end: datetime = field(default_factory=datetime.now)
    bucket_seconds: int = 10
    chunk_size: int = 500_000
    output_path: Path | None = None
    sample_on_candle: bool = True
    sample_rate: int = 1


class ReplayEngine:
    """Stream ticks from DB, build candles, extract features.

    Args:
        db: Database connection.
        registry: Feature extractor registry.
        config: Replay configuration.
        labeler: Optional tick labeler for training data generation.
        tick_callback: Optional callback for streaming test evaluation.
            When set, features are passed to the callback instead of
            being accumulated in memory.
    """

    def __init__(
        self,
        db: Database,
        registry: FeatureRegistry,
        config: ReplayConfig,
        *,
        labeler: TickLabeler | None = None,
        tick_callback: TickCallback | None = None,
        every_tick_hook: EveryTickHook | None = None,
    ) -> None:
        self._db = db
        self._registry = registry
        self._config = config
        self._builder = CandleBuilder(
            bucket_seconds=config.bucket_seconds,
        )
        self._labeler = labeler
        self._tick_callback = tick_callback
        self._every_tick_hook = every_tick_hook

    async def run(self) -> ReplayResult:
        """Run the full replay and return results.

        Streams ticks in chunks via asyncpg cursor for memory efficiency.
        When labeler is set, extends query range by timeout_seconds for
        lookahead data.
        """
        import polars as pl

        cfg = self._config
        result = ReplayResult()
        feature_chunks: list[list[dict[str, float]]] = []
        current_chunk: list[dict[str, float]] = []
        chunk_flush_size = 100_000
        tick_counter = 0
        candle_just_closed = False
        is_callback_mode = self._tick_callback is not None

        # Extend query range for labeler lookahead
        query_end = cfg.end
        if self._labeler is not None:
            lookahead = timedelta(
                seconds=self._labeler.timeout_seconds,
            )
            query_end = cfg.end + lookahead

        await logger.ainfo(
            "replay_start",
            instrument=cfg.instrument,
            start=str(cfg.start),
            end=str(cfg.end),
            sample_rate=cfg.sample_rate,
            mode="callback" if is_callback_mode else "collect",
            labeling=self._labeler is not None,
        )

        async with self._db.pool.acquire() as conn, conn.transaction():
            cursor = conn.cursor(
                "SELECT time, bid, ask FROM ticks "
                "WHERE instrument = $1 AND time >= $2 AND time < $3 "
                "ORDER BY time ASC",
                cfg.instrument,
                cfg.start,
                query_end,
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

                    # Labeler sees every tick for lookahead resolution
                    if self._labeler is not None:
                        self._labeler.on_tick(tick)

                    # Per-tick hook (e.g. TradeSimulator exit checks)
                    if self._every_tick_hook is not None:
                        self._every_tick_hook(tick)

                    # Process tick through candle builder
                    closed = self._builder.process_tick(tick)
                    candle_just_closed = False
                    if closed is not None:
                        result.total_candles += 1
                        self._registry.on_candle_close(
                            closed, self._builder.candle_index - 1,
                        )
                        candle_just_closed = True

                    # Only extract features within the nominal window
                    in_window = tick.time < cfg.end

                    # Determine if we should extract features
                    should_extract = (
                        candle_just_closed
                        if cfg.sample_on_candle
                        else (tick_counter % cfg.sample_rate == 0)
                    )

                    if in_window and should_extract:
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

                            # Register entry for labeling
                            if self._labeler is not None:
                                self._labeler.add_entry(tick)

                            # Callback mode: stream directly
                            if is_callback_mode:
                                assert self._tick_callback is not None
                                self._tick_callback(tick, features)
                            else:
                                current_chunk.append(features)

                            result.feature_rows += 1

                    # Flush chunk to list (collect mode only)
                    if (
                        not is_callback_mode
                        and len(current_chunk) >= chunk_flush_size
                    ):
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
        if not is_callback_mode and current_chunk:
            feature_chunks.append(current_chunk)

        # Finalize labeler
        if self._labeler is not None:
            result.label_result = self._labeler.finalize()

        await logger.ainfo(
            "replay_complete",
            total_ticks=result.total_ticks,
            total_candles=result.total_candles,
            feature_rows=result.feature_rows,
        )

        # Write output (collect mode only)
        if not is_callback_mode and feature_chunks:
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


def build_default_registry(instrument: str = "XAUUSD") -> FeatureRegistry:
    """Create a FeatureRegistry with all default extractors.

    Args:
        instrument: Instrument name for ICT extractors.

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
