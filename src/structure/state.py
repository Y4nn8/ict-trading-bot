"""MarketStructureState: multi-timeframe, multi-instrument state manager.

Maintains independent detector states per (instrument, timeframe) pair.
Processes new candles and returns all detected events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.structure.displacement import (
    Displacement,
    DisplacementState,
    detect_displacement_incremental,
)
from src.structure.fvg import FVG, FVGState, detect_fvg_incremental
from src.structure.market_structure import (
    MarketStructureIncrState,
    MSBreak,
    Trend,
    detect_market_structure_incremental,
)
from src.structure.order_blocks import (
    OBState,
    OrderBlock,
    detect_order_blocks_incremental,
)
from src.structure.swings import (
    SwingPoint,
    SwingState,
    detect_swings_incremental,
)


@dataclass
class DetectorStates:
    """All incremental detector states for one (instrument, timeframe) pair."""

    swing: SwingState = field(default_factory=SwingState)
    market_structure: MarketStructureIncrState = field(
        default_factory=MarketStructureIncrState
    )
    fvg: FVGState = field(default_factory=FVGState)
    order_block: OBState = field(default_factory=OBState)
    displacement: DisplacementState = field(default_factory=DisplacementState)


@dataclass
class CandleEvents:
    """All events detected from processing a single candle."""

    swings: list[SwingPoint] = field(default_factory=list)
    ms_breaks: list[MSBreak] = field(default_factory=list)
    fvgs: list[FVG] = field(default_factory=list)
    order_blocks: list[OrderBlock] = field(default_factory=list)
    displacements: list[Displacement] = field(default_factory=list)

    @property
    def has_events(self) -> bool:
        """Return True if any events were detected."""
        return bool(
            self.swings
            or self.ms_breaks
            or self.fvgs
            or self.order_blocks
            or self.displacements
        )


class MarketStructureState:
    """Manages detector states across multiple instruments and timeframes.

    Each (instrument, timeframe) pair gets its own independent set of
    detector states. This allows processing candles from different
    instruments and timeframes without interference.

    Args:
        instruments: List of instrument names to track.
        timeframes: List of timeframe strings to track.
    """

    def __init__(
        self,
        instruments: list[str],
        timeframes: list[str],
    ) -> None:
        self._states: dict[tuple[str, str], DetectorStates] = {}
        self._trends: dict[tuple[str, str], Trend] = {}

        for instrument in instruments:
            for tf in timeframes:
                key = (instrument, tf)
                self._states[key] = DetectorStates()
                self._trends[key] = Trend.UNDEFINED

    def process_candle(
        self,
        instrument: str,
        timeframe: str,
        candle: dict[str, Any],
    ) -> CandleEvents:
        """Process a new candle through all detectors.

        Args:
            instrument: Instrument identifier.
            timeframe: Timeframe string.
            candle: Dict with keys: time, open, high, low, close.

        Returns:
            CandleEvents with all detected events from this candle.
        """
        key = (instrument, timeframe)
        states = self._states.get(key)
        if states is None:
            # Auto-register unknown pairs
            states = DetectorStates()
            self._states[key] = states
            self._trends[key] = Trend.UNDEFINED

        events = CandleEvents()

        # 1. Swing detection
        events.swings = detect_swings_incremental(candle, states.swing)

        # 2. Market structure (BOS/CHoCH)
        events.ms_breaks = detect_market_structure_incremental(
            candle, events.swings, states.market_structure
        )

        # 3. FVG detection
        events.fvgs = detect_fvg_incremental(candle, states.fvg)

        # 4. Order Block detection
        events.order_blocks = detect_order_blocks_incremental(candle, states.order_block)

        # 5. Displacement detection
        events.displacements = detect_displacement_incremental(
            candle, states.displacement
        )

        # Update trend tracking
        self._trends[key] = states.market_structure.trend

        return events

    def get_trend(self, instrument: str, timeframe: str) -> Trend:
        """Get the current trend for an instrument/timeframe pair.

        Args:
            instrument: Instrument identifier.
            timeframe: Timeframe string.

        Returns:
            Current Trend state.
        """
        return self._trends.get((instrument, timeframe), Trend.UNDEFINED)

    def get_states(self, instrument: str, timeframe: str) -> DetectorStates | None:
        """Get the detector states for an instrument/timeframe pair.

        Args:
            instrument: Instrument identifier.
            timeframe: Timeframe string.

        Returns:
            DetectorStates or None if not tracked.
        """
        return self._states.get((instrument, timeframe))
