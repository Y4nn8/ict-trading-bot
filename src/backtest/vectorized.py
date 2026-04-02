"""Vectorized pre-computation pipeline for backtest.

Runs all structure detectors on full historical data at once,
producing indexed results that the event loop can look up by time/price.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.structure.displacement import detect_displacement_vectorized
from src.structure.fvg import detect_fvg_vectorized
from src.structure.liquidity import detect_liquidity_vectorized
from src.structure.market_structure import Trend, detect_market_structure_vectorized
from src.structure.order_blocks import detect_order_blocks_vectorized
from src.structure.sessions import add_session_columns_vectorized
from src.structure.swings import detect_swings_vectorized

if TYPE_CHECKING:
    import polars as pl


@dataclass
class PrecomputedData:
    """All pre-computed structure data for a single instrument/timeframe."""

    instrument: str
    timeframe: str
    candles: pl.DataFrame
    swings: pl.DataFrame
    market_structure: pl.DataFrame
    final_trend: Trend
    fvgs: pl.DataFrame
    order_blocks: pl.DataFrame
    liquidity: pl.DataFrame
    displacements: pl.DataFrame


def precompute(
    candles: pl.DataFrame,
    instrument: str,
    timeframe: str,
    swing_left: int = 2,
    swing_right: int = 2,
    ob_displacement_factor: float = 2.0,
    ob_atr_period: int = 14,
    disp_atr_period: int = 14,
    disp_threshold: float = 1.5,
    liq_tolerance_pct: float = 0.02,
    liq_lookback: int = 50,
    liq_min_touches: int = 2,
) -> PrecomputedData:
    """Run all vectorized detectors on candle data.

    Args:
        candles: DataFrame with OHLCV columns.
        instrument: Instrument name.
        timeframe: Timeframe string.
        swing_left: Swing detection left bars.
        swing_right: Swing detection right bars.
        ob_displacement_factor: Order block displacement factor.
        ob_atr_period: Order block ATR period.
        disp_atr_period: Displacement ATR period.
        disp_threshold: Displacement threshold.
        liq_tolerance_pct: Liquidity tolerance percentage.
        liq_lookback: Liquidity lookback period.
        liq_min_touches: Liquidity minimum touches.

    Returns:
        PrecomputedData with all detector results.
    """
    # Add session/killzone columns
    candles_with_sessions = add_session_columns_vectorized(candles)

    # Run all detectors
    swings = detect_swings_vectorized(candles, swing_left, swing_right)

    ms_breaks, final_trend = detect_market_structure_vectorized(
        candles, swing_left, swing_right
    )

    fvgs = detect_fvg_vectorized(candles)

    order_blocks = detect_order_blocks_vectorized(
        candles, ob_displacement_factor, ob_atr_period
    )

    liquidity = detect_liquidity_vectorized(
        candles, liq_tolerance_pct, liq_lookback, liq_min_touches
    )

    displacements = detect_displacement_vectorized(
        candles, disp_atr_period, disp_threshold
    )

    return PrecomputedData(
        instrument=instrument,
        timeframe=timeframe,
        candles=candles_with_sessions,
        swings=swings,
        market_structure=ms_breaks,
        final_trend=final_trend,
        fvgs=fvgs,
        order_blocks=order_blocks,
        liquidity=liquidity,
        displacements=displacements,
    )
