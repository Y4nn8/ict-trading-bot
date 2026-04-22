"""MACDFeatureExtractor: classic MACD on M1 and M5 timeframes.

Standard MACD: fast=12, slow=26, signal=9 (candles of the TF).
The histogram captures momentum inflections; lagged histogram values
(hist_lag1..hist_lag5) expose divergence patterns and regime flips.

M1 and M5 are complementary: M1 catches short-term momentum shifts,
M5 gives a broader context that the scalping-level signal can confirm.

No look-ahead: EMAs update only on closed HTF candles (via
CandleAggregator). The values reported in extract() are always those of
the last fully-closed M1/M5 candle — the in-progress partial candle is
never read.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

from src.midas.candle_builder import CandleAggregator
from src.midas.feature_extractor import ExtractorParam, FeatureExtractor

if TYPE_CHECKING:
    from src.midas.types import PartialCandle, Tick

# Timeframe labels and aggregation ratio from 10s candles.
_MACD_TF_RATIOS: dict[str, int] = {
    "m1": 6,    # 60s / 10s
    "m5": 30,   # 300s / 10s
}

# Number of past histogram values to expose (hist_lag1..hist_lag5).
_HIST_LAGS: int = 5

# Classic MACD periods. Kept non-tunable to focus the Optuna outer
# search on upstream extractor params that have a larger effect.
_MACD_FAST: int = 12
_MACD_SLOW: int = 26
_MACD_SIGNAL: int = 9


class _EMA:
    """Exponential moving average, seeded with the first observation."""

    def __init__(self, period: int) -> None:
        self._alpha = 2.0 / (period + 1)
        self.value: float | None = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = self._alpha * x + (1.0 - self._alpha) * self.value
        return self.value

    def reset(self) -> None:
        self.value = None


class _MACDState:
    """MACD state for one timeframe: EMAs + histogram history."""

    def __init__(self, fast: int, slow: int, signal: int) -> None:
        self._ema_fast = _EMA(fast)
        self._ema_slow = _EMA(slow)
        self._ema_signal = _EMA(signal)
        # Current + 5 lags.
        self._hist_history: deque[float] = deque(maxlen=_HIST_LAGS + 1)

    def update(self, close: float) -> None:
        """Advance state by one closed HTF candle."""
        fast = self._ema_fast.update(close)
        slow = self._ema_slow.update(close)
        macd = fast - slow
        sig = self._ema_signal.update(macd)
        self._hist_history.append(macd - sig)

    @property
    def macd(self) -> float:
        if self._ema_fast.value is None or self._ema_slow.value is None:
            return 0.0
        return self._ema_fast.value - self._ema_slow.value

    @property
    def signal(self) -> float:
        return self._ema_signal.value if self._ema_signal.value is not None else 0.0

    @property
    def hist(self) -> float:
        return self._hist_history[-1] if self._hist_history else 0.0

    def hist_lag(self, n: int) -> float:
        """Histogram value n candles ago (n=1 = previous, n=5 = 5 ago)."""
        if len(self._hist_history) > n:
            return self._hist_history[-(n + 1)]
        return 0.0

    def reset(self) -> None:
        self._ema_fast.reset()
        self._ema_slow.reset()
        self._ema_signal.reset()
        self._hist_history.clear()


class MACDFeatureExtractor(FeatureExtractor):
    """MACD features on M1 and M5 timeframes.

    For each TF (m1, m5) produces:
        macd__{tf}_macd: EMA(fast) - EMA(slow).
        macd__{tf}_signal: EMA(signal_period) of macd.
        macd__{tf}_hist: macd - signal (momentum inflection indicator).
        macd__{tf}_hist_lag1..hist_lag5: past histogram values
            (enables divergence and regime-flip detection).

    Total: 8 features per TF, 16 features total.
    """

    def __init__(self) -> None:
        self._states: dict[str, _MACDState] = {}
        self._aggregators: dict[str, CandleAggregator] = {}
        self._init_state()

    def _init_state(self) -> None:
        self._states = {
            tf: _MACDState(_MACD_FAST, _MACD_SLOW, _MACD_SIGNAL)
            for tf in _MACD_TF_RATIOS
        }
        self._aggregators = {
            tf: CandleAggregator(candles_per_bucket=ratio)
            for tf, ratio in _MACD_TF_RATIOS.items()
        }

    @property
    def name(self) -> str:
        return "macd"

    def tunable_params(self) -> list[ExtractorParam]:
        return []

    def configure(self, params: dict[str, float]) -> None:
        self._init_state()

    def on_candle_close(
        self,
        closed_candle: dict[str, Any],
        candle_index: int,
    ) -> None:
        for tf, agg in self._aggregators.items():
            htf_candle = agg.add(closed_candle)
            if htf_candle is not None:
                self._states[tf].update(float(htf_candle["close"]))

    def extract(
        self,
        tick: Tick,
        partial_candle: PartialCandle,
        candle_index: int,
    ) -> dict[str, float]:
        features: dict[str, float] = {}
        for tf, state in self._states.items():
            features[f"macd__{tf}_macd"] = state.macd
            features[f"macd__{tf}_signal"] = state.signal
            features[f"macd__{tf}_hist"] = state.hist
            for lag in range(1, _HIST_LAGS + 1):
                features[f"macd__{tf}_hist_lag{lag}"] = state.hist_lag(lag)
        return features

    def reset(self) -> None:
        for state in self._states.values():
            state.reset()
        for agg in self._aggregators.values():
            agg.reset()
