"""ICTFeatureExtractor: ICT/SMC features on M5 and H1 timeframes.

Internally aggregates closed 10s candles into M5 and H1, runs the full
ICT detector suite on those, and reports distances/directions to key levels.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

from src.midas.candle_builder import CandleAggregator
from src.midas.feature_extractor import ExtractorParam, FeatureExtractor
from src.structure.fvg import FVG, FVGType
from src.structure.liquidity import LiquidityPool, LiquidityState, detect_liquidity_incremental
from src.structure.market_structure import Trend
from src.structure.premium_discount import (
    PriceZone,
    classify_price_zone,
    compute_pd_levels,
)
from src.structure.sessions import Killzone, get_killzone
from src.structure.state import MarketStructureState
from src.structure.swings import SwingType

if TYPE_CHECKING:
    from src.midas.types import PartialCandle, Tick
    from src.structure.order_blocks import OrderBlock
    from src.structure.state import CandleEvents

# Timeframes and their 10s-candle aggregation ratios
_ICT_TFS: dict[str, int] = {
    "m5": 30,   # 300s / 10s
    "h1": 360,  # 3600s / 10s
}


class _ICTTFState:
    """Per-timeframe ICT detector state and tracked levels."""

    def __init__(self, level_max_age: int, lookback: int) -> None:
        self.level_max_age = level_max_age
        self.lookback = lookback
        self.active_fvgs: deque[FVG] = deque()
        self.active_obs: deque[OrderBlock] = deque()
        self.active_liq_pools: deque[LiquidityPool] = deque()
        self.recent_events: deque[tuple[int, CandleEvents]] = deque()
        self.atr_buffer: deque[float] = deque(maxlen=14)
        self.current_atr: float = 0.0
        self.liq_state = LiquidityState()
        self.latest_swing_high: float = 0.0
        self.latest_swing_low: float = 0.0
        self.has_pd_range: bool = False
        self.candle_count: int = 0

    def process_events(
        self,
        events: CandleEvents,
        h: float,
        lo: float,
    ) -> None:
        """Update state from a closed HTF candle's detected events."""
        idx = self.candle_count
        self.candle_count += 1

        # ATR
        self.atr_buffer.append(h - lo)
        if self.atr_buffer:
            self.current_atr = sum(self.atr_buffer) / len(self.atr_buffer)

        # Recent events for recency features
        self.recent_events.append((idx, events))
        cutoff = idx - self.lookback
        while self.recent_events and self.recent_events[0][0] < cutoff:
            self.recent_events.popleft()

        # FVGs
        for fvg in events.fvgs:
            self.active_fvgs.append(fvg)
        age_cutoff = idx - self.level_max_age
        while self.active_fvgs and self.active_fvgs[0].index < age_cutoff:
            self.active_fvgs.popleft()

        # OBs
        for ob in events.order_blocks:
            self.active_obs.append(ob)
        while self.active_obs and self.active_obs[0].index < age_cutoff:
            self.active_obs.popleft()

        # Mitigate FVGs
        self.active_fvgs = deque(
            fvg for fvg in self.active_fvgs
            if not _is_fvg_mitigated(fvg, lo, h)
        )

        # Liquidity detection from swings
        for swing in events.swings:
            swing_dict: dict[str, object] = {
                "time": swing.time,
                "price": swing.price,
                "swing_type": swing.swing_type.value,
                "index": swing.index,
            }
            pools = detect_liquidity_incremental(swing_dict, self.liq_state)
            for pool in pools:
                self.active_liq_pools.append(pool)

            if swing.swing_type == SwingType.HIGH:
                self.latest_swing_high = swing.price
            else:
                self.latest_swing_low = swing.price
            if self.latest_swing_high > 0 and self.latest_swing_low > 0:
                self.has_pd_range = True

        # Age out / sweep liquidity pools
        while (
            self.active_liq_pools
            and self.active_liq_pools[0].index < age_cutoff
        ):
            self.active_liq_pools.popleft()
        self.active_liq_pools = deque(
            p for p in self.active_liq_pools
            if not (lo <= p.price <= h)
        )

    def reset(self) -> None:
        """Clear all state."""
        self.active_fvgs.clear()
        self.active_obs.clear()
        self.active_liq_pools.clear()
        self.recent_events.clear()
        self.atr_buffer.clear()
        self.current_atr = 0.0
        self.liq_state = LiquidityState()
        self.latest_swing_high = 0.0
        self.latest_swing_low = 0.0
        self.has_pd_range = False
        self.candle_count = 0


def _is_fvg_mitigated(fvg: FVG, low: float, high: float) -> bool:
    """Check if a candle mitigated (filled) an FVG."""
    if fvg.fvg_type == FVGType.BULLISH:
        return low <= fvg.bottom
    return high >= fvg.top


class ICTFeatureExtractor(FeatureExtractor):
    """Extract ICT/SMC features on M5 and H1 timeframes.

    Internally aggregates 10s candles into M5 and H1, runs the full
    detector suite (swings, BOS/CHoCH, FVG, OB, displacement) on each,
    and computes distances from the current tick price.

    Per TF (m5, h1):
        ict__{tf}_fvg_distance: distance to nearest unmitigated FVG.
        ict__{tf}_fvg_direction: +1 bullish, -1 bearish, 0 none.
        ict__{tf}_ob_distance: distance to nearest OB zone.
        ict__{tf}_ob_direction: +1 bullish, -1 bearish, 0 none.
        ict__{tf}_bos_recent: 1 if BOS in last N candles.
        ict__{tf}_choch_recent: 1 if CHoCH in last N candles.
        ict__{tf}_trend: +1 bullish, -1 bearish, 0 undefined.
        ict__{tf}_displacement_recent: 1 if displacement in last N candles.
        ict__{tf}_liq_sweep_distance: distance to nearest liquidity pool.
        ict__{tf}_premium_discount: +1 premium, -1 discount, 0 unknown.

    Global:
        ict__killzone: 1 if in killzone, 0 otherwise.

    Args:
        instrument: Instrument name for MarketStructureState.
    """

    def __init__(self, instrument: str = "XAUUSD") -> None:
        self._instrument = instrument
        self._lookback: int = 20
        self._level_max_age: int = 100
        self._ms_state = MarketStructureState(
            instruments=[instrument],
            timeframes=list(_ICT_TFS.keys()),
        )
        self._aggregators: dict[str, CandleAggregator] = {
            tf: CandleAggregator(candles_per_bucket=ratio)
            for tf, ratio in _ICT_TFS.items()
        }
        self._tf_states: dict[str, _ICTTFState] = {
            tf: _ICTTFState(self._level_max_age, self._lookback)
            for tf in _ICT_TFS
        }

    @property
    def name(self) -> str:
        return "ict"

    def tunable_params(self) -> list[ExtractorParam]:
        return [
            ExtractorParam("lookback", 20, 5, 50, "int"),
            ExtractorParam("level_max_age", 100, 20, 300, "int"),
        ]

    def configure(self, params: dict[str, float]) -> None:
        self._lookback = int(params.get("lookback", 20))
        self._level_max_age = int(params.get("level_max_age", 100))
        for state in self._tf_states.values():
            state.lookback = self._lookback
            state.level_max_age = self._level_max_age

    def on_candle_close(
        self,
        closed_candle: dict[str, Any],
        candle_index: int,
    ) -> None:
        # Aggregate 10s → M5/H1 and run detectors on closed HTF candles
        for tf, agg in self._aggregators.items():
            htf_candle = agg.add(closed_candle)
            if htf_candle is not None:
                events = self._ms_state.process_candle(
                    self._instrument, tf, htf_candle,
                )
                h = float(htf_candle["high"])
                lo = float(htf_candle["low"])
                self._tf_states[tf].process_events(events, h, lo)

    def _extract_tf(self, tf: str, price: float) -> dict[str, float]:
        """Compute features for one timeframe."""
        state = self._tf_states[tf]
        atr = state.current_atr if state.current_atr > 0 else 1.0

        # Nearest FVG
        fvg_dist = 0.0
        fvg_dir = 0.0
        if state.active_fvgs:
            best = min(state.active_fvgs, key=lambda f: abs(price - f.midpoint))
            fvg_dist = abs(price - best.midpoint) / atr
            fvg_dir = 1.0 if best.fvg_type == FVGType.BULLISH else -1.0

        # Nearest OB
        ob_dist = 0.0
        ob_dir = 0.0
        if state.active_obs:
            best_ob = min(
                state.active_obs,
                key=lambda o: abs(price - (o.top + o.bottom) / 2),
            )
            ob_mid = (best_ob.top + best_ob.bottom) / 2
            ob_dist = abs(price - ob_mid) / atr
            ob_dir = (
                1.0 if best_ob.ob_type.value.startswith("bullish") else -1.0
            )

        # Recent BOS / CHoCH / displacement
        bos_recent = 0.0
        choch_recent = 0.0
        disp_recent = 0.0
        for _idx, ev in state.recent_events:
            for ms in ev.ms_breaks:
                if ms.break_type.value.startswith("bos"):
                    bos_recent = 1.0
                else:
                    choch_recent = 1.0
            if ev.displacements:
                disp_recent = 1.0

        # Trend
        trend = self._ms_state.get_trend(self._instrument, tf)
        trend_val = 0.0
        if trend == Trend.BULLISH:
            trend_val = 1.0
        elif trend == Trend.BEARISH:
            trend_val = -1.0

        # Nearest liquidity pool
        liq_dist = 0.0
        if state.active_liq_pools:
            best_liq = min(
                state.active_liq_pools,
                key=lambda p: abs(price - p.price),
            )
            liq_dist = abs(price - best_liq.price) / atr

        # Premium / discount
        pd_val = 0.0
        if state.has_pd_range:
            levels = compute_pd_levels(
                state.latest_swing_high, state.latest_swing_low,
            )
            zone = classify_price_zone(price, levels)
            if zone == PriceZone.PREMIUM:
                pd_val = 1.0
            elif zone == PriceZone.DISCOUNT:
                pd_val = -1.0

        return {
            f"ict__{tf}_fvg_distance": fvg_dist,
            f"ict__{tf}_fvg_direction": fvg_dir,
            f"ict__{tf}_ob_distance": ob_dist,
            f"ict__{tf}_ob_direction": ob_dir,
            f"ict__{tf}_bos_recent": bos_recent,
            f"ict__{tf}_choch_recent": choch_recent,
            f"ict__{tf}_trend": trend_val,
            f"ict__{tf}_displacement_recent": disp_recent,
            f"ict__{tf}_liq_sweep_distance": liq_dist,
            f"ict__{tf}_premium_discount": pd_val,
        }

    def extract(
        self,
        tick: Tick,
        partial_candle: PartialCandle,
        candle_index: int,
    ) -> dict[str, float]:
        price = tick.mid
        features: dict[str, float] = {}

        for tf in _ICT_TFS:
            features.update(self._extract_tf(tf, price))

        # Killzone (time-based, TF-independent)
        kz = get_killzone(tick.time.hour)
        features["ict__killzone"] = 0.0 if kz == Killzone.NONE else 1.0

        return features

    def reset(self) -> None:
        self._ms_state = MarketStructureState(
            instruments=[self._instrument],
            timeframes=list(_ICT_TFS.keys()),
        )
        for agg in self._aggregators.values():
            agg.reset()
        for state in self._tf_states.values():
            state.reset()
