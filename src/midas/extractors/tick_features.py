"""TickFeatureExtractor: features from the current partial candle and raw tick.

No candle history required — purely instantaneous features.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

from src.midas.feature_extractor import ExtractorParam, FeatureExtractor

if TYPE_CHECKING:
    from src.midas.types import PartialCandle, Tick


class TickFeatureExtractor(FeatureExtractor):
    """Extract features from the current tick and partial candle.

    Features:
        tick__spread: current bid-ask spread.
        tick__spread_z: spread z-score vs recent candle close spreads.
        tick__partial_range: current partial candle range.
        tick__position_in_range: close position in candle range (0-1).
        tick__elapsed_pct: fraction of bucket duration elapsed (0-1).
        tick__tick_count: number of ticks in current partial candle.
        tick__mid: current mid price.

    Args:
        bucket_seconds: Duration of candle buckets (must match CandleBuilder).
    """

    def __init__(self, bucket_seconds: int = 10) -> None:
        self._spread_avg_period: int = 30
        self._candle_spreads: deque[float] = deque()
        self._bucket_seconds: float = float(bucket_seconds)

    @property
    def name(self) -> str:
        return "tick"

    def tunable_params(self) -> list[ExtractorParam]:
        return [
            ExtractorParam("spread_avg_period", 30, 10, 100, "int"),
        ]

    def configure(self, params: dict[str, float]) -> None:
        self._spread_avg_period = int(params.get(
            "spread_avg_period", 30,
        ))
        self._candle_spreads = deque(maxlen=self._spread_avg_period)

    def on_candle_close(
        self,
        closed_candle: dict[str, Any],
        candle_index: int,
    ) -> None:
        # Record the candle's close spread (bid/ask from last tick of candle)
        bid = closed_candle.get("bid")
        ask = closed_candle.get("ask")
        if bid is not None and ask is not None:
            self._candle_spreads.append(float(ask) - float(bid))

    def extract(
        self,
        tick: Tick,
        partial_candle: PartialCandle,
        candle_index: int,
    ) -> dict[str, float]:
        spread = tick.spread

        # Spread z-score vs recent candle close spreads
        if len(self._candle_spreads) >= 2:
            spreads = list(self._candle_spreads)
            avg = sum(spreads) / len(spreads)
            var = sum((s - avg) ** 2 for s in spreads) / len(spreads)
            std = var**0.5
            spread_z = (spread - avg) / std if std > 0 else 0.0
        else:
            spread_z = 0.0

        return {
            "tick__spread": spread,
            "tick__spread_z": spread_z,
            "tick__partial_range": partial_candle.range,
            "tick__position_in_range": partial_candle.position_in_range,
            "tick__elapsed_pct": min(
                partial_candle.elapsed_seconds / self._bucket_seconds, 1.0,
            ),
            "tick__tick_count": float(partial_candle.tick_count),
            "tick__mid": tick.mid,
        }

    def reset(self) -> None:
        self._candle_spreads.clear()
