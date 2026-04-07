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

    # Rolling mean of TR (vectorized).
    # tr[0] corresponds to candle index 1. First ATR at candle index `period`
    # uses tr[0:period].
    if len(tr) >= period:
        kernel = np.ones(period, dtype=float) / period
        rolling_atr = np.convolve(tr, kernel, mode="valid")  # length = len(tr) - period + 1
        atr[period : period + len(rolling_atr)] = rolling_atr

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
    htf_trend: np.ndarray | None = None  # H1 trend per M5 candle


def build_htf_trend_array(
    m5_candles: pl.DataFrame,
    h1_candles: pl.DataFrame,
    swing_left_bars: int = 2,
    swing_right_bars: int = 2,
) -> np.ndarray:
    """Build an array mapping each M5 candle to the current H1 trend.

    Runs market structure detection on H1 candles and forward-fills
    the trend for each M5 timestamp. An H1 bar is only visible after
    its close (timestamp + 1 hour) to prevent look-ahead bias.

    Args:
        m5_candles: M5 candles DataFrame with a 'time' column.
        h1_candles: H1 candles DataFrame with OHLCV columns.
        swing_left_bars: Swing detection left bars for H1.
        swing_right_bars: Swing detection right bars for H1.

    Returns:
        1-D numpy array of Trend string values, one per M5 candle.
    """
    from datetime import timedelta

    import polars as pl_mod

    n = len(m5_candles)
    result = np.full(n, Trend.UNDEFINED, dtype=object)

    if h1_candles.is_empty() or len(h1_candles) < 10:
        return result

    h1_breaks, _ = detect_market_structure_vectorized(
        h1_candles, swing_left_bars, swing_right_bars,
    )

    if h1_breaks.is_empty():
        return result

    # H1 bar at time T closes at T+1h — only apply trend after close
    h1_offset = timedelta(hours=1)
    break_times = h1_breaks["time"].to_list()
    break_visible_times = np.array(
        [t.replace(tzinfo=None) + h1_offset for t in break_times],
        dtype="datetime64[us]",
    )
    break_directions = h1_breaks["direction"].to_list()

    # Build cumulative trend from breaks: each break sets the new trend
    # BOS bullish / CHoCH bullish → bullish, BOS bearish / CHoCH bearish → bearish
    trends_at_breaks = []
    current_trend = Trend.UNDEFINED
    for d in break_directions:
        if d == "bullish":
            current_trend = Trend.BULLISH
        elif d == "bearish":
            current_trend = Trend.BEARISH
        trends_at_breaks.append(current_trend)

    m5_times = m5_candles["time"].cast(pl_mod.Datetime("us", "UTC")).to_numpy()
    m5_times_ns = m5_times.astype("datetime64[us]")

    # For each M5 candle, find the latest H1 break that is visible
    indices = np.searchsorted(break_visible_times, m5_times_ns, side="right") - 1

    for i in range(n):
        idx = indices[i]
        if idx >= 0:
            result[i] = trends_at_breaks[idx]

    return result


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
        swings=swings,
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
