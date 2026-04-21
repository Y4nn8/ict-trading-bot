"""ScalpingFeatureExtractor: momentum, mean-reversion, and range features.

Computed on 10s and M1 timeframes using circular buffers.
10s candles are aggregated internally into M1 via CandleAggregator.
M5/H1 momentum is covered by the ICT extractor (trend + FVG/OB distances).
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

from src.midas.candle_builder import CandleAggregator
from src.midas.feature_extractor import ExtractorParam, FeatureExtractor

if TYPE_CHECKING:
    from src.midas.types import PartialCandle, Tick

# Timeframe labels and their aggregation ratio from 10s candles.
# M5/H1 momentum is already covered by htf__ (trend + FVG/OB distances).
_TF_RATIOS: dict[str, int] = {
    "10s": 1,   # no aggregation (base TF)
    "m1": 6,    # 60s / 10s
}


class _TFBuffers:
    """Per-timeframe circular buffers for scalping features."""

    def __init__(self, max_len: int) -> None:
        self.closes: deque[float] = deque(maxlen=max_len)
        self.ranges: deque[float] = deque(maxlen=max_len)
        self.bodies: deque[float] = deque(maxlen=max_len)
        self.directions: deque[int] = deque(maxlen=max_len)

    def update(self, candle: dict[str, Any]) -> None:
        """Record a closed candle."""
        o = float(candle["open"])
        h = float(candle["high"])
        lo = float(candle["low"])
        c = float(candle["close"])
        self.closes.append(c)
        self.ranges.append(h - lo)
        self.bodies.append(abs(c - o))
        self.directions.append(1 if c >= o else -1)

    def clear(self) -> None:
        """Clear all buffers."""
        self.closes.clear()
        self.ranges.clear()
        self.bodies.clear()
        self.directions.clear()


class ScalpingFeatureExtractor(FeatureExtractor):
    """Extract scalping features on 10s and M1 timeframes.

    For each TF (10s, M1) produces:
        scalp__{tf}_roc_fast: rate of change over fast period.
        scalp__{tf}_roc_slow: rate of change over slow period.
        scalp__{tf}_mean_rev_z: z-score of price vs rolling mean.
        scalp__{tf}_atr: average true range (volatility proxy).
        scalp__{tf}_range_ratio: current candle range vs average range.
        scalp__{tf}_body_ratio: average body/range ratio.
        scalp__{tf}_direction_streak: consecutive candles in same direction.
        scalp__{tf}_vol_regime: short ATR / long ATR (>1 = vol expanding).
        scalp__{tf}_trend_regime: net directional bias over long window,
            in [-1, 1]. +1 = trending up, -1 = trending down, 0 = chop.
    """

    def __init__(self) -> None:
        self._roc_fast: int = 5
        self._roc_slow: int = 20
        self._mean_rev_period: int = 30
        self._atr_period: int = 14
        self._regime_period: int = 100
        self._buffers: dict[str, _TFBuffers] = {}
        self._aggregators: dict[str, CandleAggregator] = {}
        self._init_tf_state()

    def _init_tf_state(self) -> None:
        """Initialize per-TF buffers and aggregators."""
        max_len = max(
            self._roc_slow, self._mean_rev_period, self._atr_period,
            self._regime_period,
        ) + 1
        self._buffers = {tf: _TFBuffers(max_len) for tf in _TF_RATIOS}
        self._aggregators = {
            tf: CandleAggregator(candles_per_bucket=ratio)
            for tf, ratio in _TF_RATIOS.items()
            if ratio > 1
        }

    @property
    def name(self) -> str:
        return "scalp"

    def tunable_params(self) -> list[ExtractorParam]:
        return [
            ExtractorParam("roc_fast", 5, 2, 15, "int"),
            ExtractorParam("roc_slow", 20, 10, 60, "int"),
            ExtractorParam("mean_rev_period", 30, 10, 100, "int"),
            ExtractorParam("atr_period", 14, 5, 50, "int"),
            ExtractorParam("regime_period", 100, 40, 300, "int"),
        ]

    def configure(self, params: dict[str, float]) -> None:
        self._roc_fast = int(params.get("roc_fast", 5))
        self._roc_slow = int(params.get("roc_slow", 20))
        self._mean_rev_period = int(params.get("mean_rev_period", 30))
        self._atr_period = int(params.get("atr_period", 14))
        self._regime_period = int(params.get("regime_period", 100))
        self._init_tf_state()

    def on_candle_close(
        self,
        closed_candle: dict[str, Any],
        candle_index: int,
    ) -> None:
        # 10s: update directly
        self._buffers["10s"].update(closed_candle)

        # Higher TFs: aggregate then update buffer when a HTF candle closes
        for tf, agg in self._aggregators.items():
            htf_candle = agg.add(closed_candle)
            if htf_candle is not None:
                self._buffers[tf].update(htf_candle)

    def _compute_tf_features(
        self,
        tf: str,
        mid_price: float,
        partial_range: float,
    ) -> dict[str, float]:
        """Compute features for one timeframe."""
        buf = self._buffers[tf]
        n = len(buf.closes)

        # Rate of change
        roc_fast = 0.0
        if n > self._roc_fast and buf.closes[-self._roc_fast] != 0:
            roc_fast = (
                (buf.closes[-1] - buf.closes[-self._roc_fast])
                / buf.closes[-self._roc_fast]
            )

        roc_slow = 0.0
        if n > self._roc_slow and buf.closes[-self._roc_slow] != 0:
            roc_slow = (
                (buf.closes[-1] - buf.closes[-self._roc_slow])
                / buf.closes[-self._roc_slow]
            )

        # Mean reversion z-score
        mean_rev_z = 0.0
        if n >= self._mean_rev_period:
            recent = list(buf.closes)[-self._mean_rev_period:]
            avg = sum(recent) / len(recent)
            var = sum((x - avg) ** 2 for x in recent) / len(recent)
            std = var**0.5
            mean_rev_z = (mid_price - avg) / std if std > 0 else 0.0

        # ATR
        atr = 0.0
        if n >= self._atr_period:
            recent_ranges = list(buf.ranges)[-self._atr_period:]
            atr = sum(recent_ranges) / len(recent_ranges)

        # Range ratio (only meaningful for 10s — partial_range is always 10s)
        range_ratio = 0.0
        if tf == "10s" and atr > 0:
            range_ratio = partial_range / atr

        # Body ratio
        body_ratio = 0.0
        if n >= self._atr_period:
            recent_bodies = list(buf.bodies)[-self._atr_period:]
            recent_ranges = list(buf.ranges)[-self._atr_period:]
            total_range = sum(recent_ranges)
            if total_range > 0:
                body_ratio = sum(recent_bodies) / total_range

        # Direction streak
        streak = 0.0
        if buf.directions:
            dirs = list(buf.directions)
            last_dir = dirs[-1]
            count = 0
            for d in reversed(dirs):
                if d == last_dir:
                    count += 1
                else:
                    break
            streak = float(count * last_dir)

        # Volatility regime: short-term ATR / long-term ATR.
        # > 1 = volatility expanding, < 1 = compressing.
        vol_regime = 0.0
        if n >= self._regime_period:
            long_ranges = list(buf.ranges)[-self._regime_period:]
            long_atr = sum(long_ranges) / len(long_ranges)
            if long_atr > 0 and atr > 0:
                vol_regime = atr / long_atr

        # Trend regime: net directional bias in long window, in [-1, 1].
        # +1 = consistently up, -1 = consistently down, 0 = choppy.
        trend_regime = 0.0
        if n >= self._regime_period:
            long_dirs = list(buf.directions)[-self._regime_period:]
            trend_regime = sum(long_dirs) / len(long_dirs)

        return {
            f"scalp__{tf}_roc_fast": roc_fast,
            f"scalp__{tf}_roc_slow": roc_slow,
            f"scalp__{tf}_mean_rev_z": mean_rev_z,
            f"scalp__{tf}_atr": atr,
            f"scalp__{tf}_range_ratio": range_ratio,
            f"scalp__{tf}_body_ratio": body_ratio,
            f"scalp__{tf}_direction_streak": streak,
            f"scalp__{tf}_vol_regime": vol_regime,
            f"scalp__{tf}_trend_regime": trend_regime,
        }

    def extract(
        self,
        tick: Tick,
        partial_candle: PartialCandle,
        candle_index: int,
    ) -> dict[str, float]:
        features: dict[str, float] = {}
        for tf in _TF_RATIOS:
            features.update(
                self._compute_tf_features(
                    tf, tick.mid, partial_candle.range,
                ),
            )
        return features

    def reset(self) -> None:
        for buf in self._buffers.values():
            buf.clear()
        for agg in self._aggregators.values():
            agg.reset()
