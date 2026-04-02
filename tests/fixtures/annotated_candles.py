"""Annotated test fixtures with known ICT structure events.

This module provides hand-crafted OHLCV data with known swing highs/lows,
BOS/CHoCH, FVGs, and order blocks for testing detectors.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

_BASE = datetime(2024, 1, 15, 10, 0, tzinfo=UTC)


def _times(n: int) -> list[datetime]:
    return [_BASE + timedelta(minutes=i * 5) for i in range(n)]


# === Swing fixture: clear 5-bar Williams fractals ===
# Candles designed with clear swing high at index 4 and swing low at index 9
SWING_FIXTURE_DATA = {
    "time": [*_times(15)],
    "open": [
        1.080, 1.081, 1.082, 1.084, 1.085,  # rising to swing high at 4
        1.086, 1.084, 1.082, 1.080, 1.078,  # falling to swing low at 9
        1.076, 1.078, 1.080, 1.082, 1.084,  # rising again
    ],
    "high": [
        1.081, 1.082, 1.083, 1.085, 1.090,  # SH at 4: 1.090
        1.087, 1.085, 1.083, 1.081, 1.079,  # declining
        1.077, 1.079, 1.081, 1.083, 1.085,  # rising
    ],
    "low": [
        1.079, 1.080, 1.081, 1.083, 1.084,  # rising lows
        1.083, 1.081, 1.079, 1.077, 1.074,  # SL at 9: 1.074
        1.075, 1.077, 1.079, 1.081, 1.083,  # rising lows
    ],
    "close": [
        1.081, 1.082, 1.083, 1.085, 1.086,  # bullish candles
        1.084, 1.082, 1.080, 1.078, 1.076,  # bearish candles
        1.078, 1.080, 1.082, 1.084, 1.085,  # bullish candles
    ],
    "volume": [100.0] * 15,
}

SWING_FIXTURE = pl.DataFrame(SWING_FIXTURE_DATA)

# Expected swings: SH at index 4 (price=1.090), SL at index 9 (price=1.074)
EXPECTED_SWING_HIGHS = [{"index": 4, "price": 1.090}]
EXPECTED_SWING_LOWS = [{"index": 9, "price": 1.074}]

# === FVG fixture: bullish and bearish FVGs ===
# Bullish FVG: candle1.high < candle3.low (gap up)
FVG_FIXTURE_DATA = {
    "time": [*_times(8)],
    "open": [
        1.080, 1.081, 1.082,  # normal
        1.085, 1.090, 1.095,  # bullish FVG: c1(idx=2).high=1.084 < c3(idx=4).low=1.089
        1.094, 1.090,         # setup for bearish FVG
    ],
    "high": [
        1.081, 1.082, 1.084,  # c1 high = 1.084
        1.086, 1.096, 1.097,  # c3 (idx=4) low=1.089 > c1(idx=2) high=1.084 → bullish FVG
        1.095, 1.091,
    ],
    "low": [
        1.079, 1.080, 1.081,
        1.084, 1.089, 1.094,  # c3 low = 1.089
        1.088, 1.084,         # c3(idx=7).high=1.091 < c1(idx=5).low=1.094 → bearish FVG
    ],
    "close": [
        1.081, 1.082, 1.083,
        1.086, 1.095, 1.095,
        1.089, 1.085,
    ],
    "volume": [100.0] * 8,
}

FVG_FIXTURE = pl.DataFrame(FVG_FIXTURE_DATA)

# Expected FVGs:
# Bullish at index 3 (middle candle): top=1.089, bottom=1.084
# Bearish at index 6 (middle candle): top=1.094, bottom=1.091
EXPECTED_BULLISH_FVG = {"index": 3, "top": 1.089, "bottom": 1.084}
EXPECTED_BEARISH_FVG = {"index": 6, "top": 1.094, "bottom": 1.091}

# === BOS/CHoCH fixture: trend continuation and reversal ===
# Designed to produce: uptrend → BOS (break swing high) → CHoCH (break swing low)
BOS_CHOCH_FIXTURE_DATA = {
    "time": [*_times(25)],
    "open": [
        # Phase 1: establish uptrend with swing lows at ~1.078 and highs at ~1.090
        1.080, 1.082, 1.084, 1.086, 1.088,  # 0-4: rise
        1.086, 1.084, 1.082, 1.080, 1.078,  # 5-9: pullback (SL at 9)
        1.080, 1.082, 1.084, 1.086, 1.091,  # 10-14: rise above SH (BOS at ~14)
        1.090, 1.088, 1.086, 1.084, 1.080,  # 15-19: pullback
        1.078, 1.076, 1.074, 1.072, 1.070,  # 20-24: break below SL (CHoCH)
    ],
    "high": [
        1.083, 1.085, 1.087, 1.089, 1.090,  # SH at 4: 1.090
        1.088, 1.086, 1.084, 1.082, 1.080,
        1.083, 1.085, 1.087, 1.089, 1.095,  # breaks above 1.090 → BOS
        1.092, 1.090, 1.088, 1.086, 1.082,
        1.079, 1.077, 1.075, 1.073, 1.071,
    ],
    "low": [
        1.079, 1.081, 1.083, 1.085, 1.087,
        1.083, 1.081, 1.079, 1.077, 1.076,  # SL at 9: 1.076
        1.079, 1.081, 1.083, 1.085, 1.089,
        1.088, 1.086, 1.084, 1.082, 1.078,
        1.075, 1.073, 1.071, 1.069, 1.068,  # breaks below 1.076 → CHoCH
    ],
    "close": [
        1.082, 1.084, 1.086, 1.088, 1.089,
        1.084, 1.082, 1.080, 1.078, 1.077,
        1.082, 1.084, 1.086, 1.088, 1.093,
        1.089, 1.087, 1.085, 1.083, 1.079,
        1.076, 1.074, 1.072, 1.070, 1.069,
    ],
    "volume": [100.0] * 25,
}

BOS_CHOCH_FIXTURE = pl.DataFrame(BOS_CHOCH_FIXTURE_DATA)

# === Order Block fixture: displacement after opposing candle ===
# Large ATR environment, then a big bullish candle after a bearish candle
OB_FIXTURE_DATA = {
    "time": [*_times(20)],
    "open": [
        # Establish ATR with normal-sized candles (body ~0.002)
        1.080, 1.082, 1.081, 1.083, 1.082,
        1.084, 1.083, 1.085, 1.084, 1.086,
        1.085, 1.087, 1.086, 1.088, 1.087,
        1.089, 1.088,
        1.086,  # idx 17: bearish candle (OB candidate)
        1.084,  # idx 18: big bullish displacement
        1.096,
    ],
    "high": [
        1.083, 1.084, 1.083, 1.085, 1.084,
        1.086, 1.085, 1.087, 1.086, 1.088,
        1.087, 1.089, 1.088, 1.090, 1.089,
        1.091, 1.090,
        1.087,  # OB candle high
        1.098,  # displacement high (body=0.014, ~7x ATR of ~0.002)
        1.098,
    ],
    "low": [
        1.079, 1.081, 1.080, 1.082, 1.081,
        1.083, 1.082, 1.084, 1.083, 1.085,
        1.084, 1.086, 1.085, 1.087, 1.086,
        1.088, 1.087,
        1.083,  # OB candle low
        1.083,  # displacement low
        1.095,
    ],
    "close": [
        1.082, 1.083, 1.082, 1.084, 1.083,
        1.085, 1.084, 1.086, 1.085, 1.087,
        1.086, 1.088, 1.087, 1.089, 1.088,
        1.090, 1.089,
        1.084,  # bearish close (OB: bearish candle before bullish displacement)
        1.098,  # bullish displacement close
        1.097,
    ],
    "volume": [100.0] * 20,
}

OB_FIXTURE = pl.DataFrame(OB_FIXTURE_DATA)

# Expected: bullish OB at index 17 (bearish candle before displacement at 18)
EXPECTED_BULLISH_OB = {"index": 17, "top": 1.087, "bottom": 1.083}
