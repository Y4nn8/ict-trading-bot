"""Vectorized pre-computation pipeline for backtest.

Runs all structure detectors on full historical data at once,
producing indexed results that the event loop can look up by time/price.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.structure.displacement import detect_displacement_vectorized
from src.structure.fvg import detect_fvg_vectorized
from src.structure.liquidity import detect_liquidity_vectorized
from src.structure.market_structure import Trend, detect_market_structure_vectorized
from src.structure.order_blocks import detect_order_blocks_vectorized
from src.structure.sessions import add_session_columns_vectorized
from src.structure.swings import detect_swings_vectorized

if TYPE_CHECKING:
    import polars as pl

    from src.strategy.params import StrategyParams


def _compute_atr_column(candles: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """Add a rolling ATR column to candles DataFrame.

    Args:
        candles: DataFrame with open, high, low, close columns.
        period: ATR lookback period.

    Returns:
        DataFrame with an additional 'atr' column.
    """
    import polars as pl_mod

    highs = candles["high"].to_numpy()
    lows = candles["low"].to_numpy()
    closes = candles["close"].to_numpy()

    n = len(candles)
    atr = np.full(n, np.nan)

    if n < 2:
        return candles.with_columns(pl_mod.Series("atr", atr))

    # True Range: max(H-L, |H-prev_close|, |L-prev_close|)
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1]),
        ),
    )

    # Rolling mean of TR
    for i in range(period, len(tr)):
        atr[i + 1] = float(np.mean(tr[i - period + 1 : i + 1]))

    return candles.with_columns(pl_mod.Series("atr", atr))


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
    params: StrategyParams | None = None,
) -> PrecomputedData:
    """Run all vectorized detectors on candle data.

    Args:
        candles: DataFrame with OHLCV columns.
        instrument: Instrument name.
        timeframe: Timeframe string.
        params: Strategy parameters. Uses defaults if None.

    Returns:
        PrecomputedData with all detector results.
    """
    if params is None:
        from src.strategy.params import StrategyParams

        params = StrategyParams()

    candles_with_atr = _compute_atr_column(candles, params.disp_atr_period)
    candles_with_sessions = add_session_columns_vectorized(candles_with_atr)

    swings = detect_swings_vectorized(
        candles, params.swing_left_bars, params.swing_right_bars
    )

    ms_breaks, final_trend = detect_market_structure_vectorized(
        candles, params.swing_left_bars, params.swing_right_bars
    )

    fvgs = detect_fvg_vectorized(candles)

    order_blocks = detect_order_blocks_vectorized(
        candles, params.ob_displacement_factor, params.ob_atr_period
    )

    liquidity = detect_liquidity_vectorized(
        candles,
        params.liq_tolerance_pct,
        params.liq_lookback,
        params.liq_min_touches,
    )

    displacements = detect_displacement_vectorized(
        candles, params.disp_atr_period, params.disp_threshold
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
