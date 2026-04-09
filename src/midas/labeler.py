"""TickLabeler: label ticks with BUY/SELL outcome via SL/TP lookahead.

For each sampled tick, evaluates two hypothetical trades (BUY and SELL)
by scanning future ticks to see if TP or SL is hit first.

Labels:
    1  = TP hit before SL (win)
    0  = SL hit before TP (loss)
    -1 = timeout / unresolved (skip in training)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    from src.midas.types import LabelConfig, Tick


@dataclass
class _PendingLabel:
    """A pending label awaiting resolution from future ticks."""

    feature_index: int
    entry_time: datetime
    # BUY: enter at ask, evaluate exit against future bid
    buy_sl: float
    buy_tp: float
    # SELL: enter at bid, evaluate exit against future ask
    sell_sl: float
    sell_tp: float
    buy_label: int = -1
    buy_resolved: bool = False
    sell_label: int = -1
    sell_resolved: bool = False

    @property
    def fully_resolved(self) -> bool:
        """Both BUY and SELL labels are determined."""
        return self.buy_resolved and self.sell_resolved


@dataclass
class LabelResult:
    """Output of the labeling process."""

    buy_labels: list[int] = field(default_factory=list)
    sell_labels: list[int] = field(default_factory=list)
    total_labeled: int = 0
    buy_wins: int = 0
    buy_losses: int = 0
    sell_wins: int = 0
    sell_losses: int = 0
    timeouts: int = 0


class TickLabeler:
    """Labels ticks with BUY/SELL outcome based on SL/TP lookahead.

    Usage during replay:
        1. Call add_entry() for each tick that has a feature row
        2. Call on_tick() for EVERY tick (including non-sampled ones)
        3. Call finalize() at end to get labels

    Args:
        config: Label configuration (SL/TP distances, timeout).
    """

    def __init__(self, config: LabelConfig) -> None:
        self._config = config
        self._pending: deque[_PendingLabel] = deque()
        self._resolved: list[_PendingLabel] = []
        self._next_index: int = 0

    def add_entry(self, tick: Tick) -> int:
        """Register a tick as a candidate entry point.

        Args:
            tick: The tick at the potential entry.

        Returns:
            The feature index assigned to this entry.
        """
        idx = self._next_index
        self._next_index += 1

        entry = _PendingLabel(
            feature_index=idx,
            entry_time=tick.time,
            # BUY enters at ask, exits evaluated against bid
            buy_sl=tick.ask - self._config.sl_points,
            buy_tp=tick.ask + self._config.tp_points,
            # SELL enters at bid, exits evaluated against ask
            sell_sl=tick.bid + self._config.sl_points,
            sell_tp=tick.bid - self._config.tp_points,
        )
        self._pending.append(entry)
        return idx

    def on_tick(self, tick: Tick) -> None:
        """Evaluate all pending entries against this tick.

        Call this for every tick in the stream, not just sampled ones.

        Args:
            tick: The current tick.
        """
        timeout = self._config.timeout_seconds
        resolved_count = 0

        for pending in self._pending:
            # BUY evaluation: check bid for SL/TP
            if not pending.buy_resolved:
                if tick.bid <= pending.buy_sl:
                    pending.buy_label = 0  # SL hit (loss)
                    pending.buy_resolved = True
                elif tick.bid >= pending.buy_tp:
                    pending.buy_label = 1  # TP hit (win)
                    pending.buy_resolved = True

            # SELL evaluation: check ask for SL/TP
            if not pending.sell_resolved:
                if tick.ask >= pending.sell_sl:
                    pending.sell_label = 0  # SL hit (loss)
                    pending.sell_resolved = True
                elif tick.ask <= pending.sell_tp:
                    pending.sell_label = 1  # TP hit (win)
                    pending.sell_resolved = True

            # Timeout check
            elapsed = (tick.time - pending.entry_time).total_seconds()
            if elapsed > timeout:
                if not pending.buy_resolved:
                    pending.buy_label = -1
                    pending.buy_resolved = True
                if not pending.sell_resolved:
                    pending.sell_label = -1
                    pending.sell_resolved = True

            if pending.fully_resolved:
                resolved_count += 1

        # Move resolved entries out of pending
        while self._pending and self._pending[0].fully_resolved:
            self._resolved.append(self._pending.popleft())

    def finalize(self) -> LabelResult:
        """Finalize all remaining pending entries and return labels.

        Unresolved entries get label=-1 (timeout).

        Returns:
            LabelResult with aligned buy/sell label arrays.
        """
        # Force-resolve remaining entries
        for pending in self._pending:
            if not pending.buy_resolved:
                pending.buy_label = -1
                pending.buy_resolved = True
            if not pending.sell_resolved:
                pending.sell_label = -1
                pending.sell_resolved = True
            self._resolved.append(pending)
        self._pending.clear()

        # Sort by feature index and build label arrays
        self._resolved.sort(key=lambda p: p.feature_index)

        result = LabelResult(total_labeled=len(self._resolved))
        for entry in self._resolved:
            result.buy_labels.append(entry.buy_label)
            result.sell_labels.append(entry.sell_label)
            if entry.buy_label == 1:
                result.buy_wins += 1
            elif entry.buy_label == 0:
                result.buy_losses += 1
            if entry.sell_label == 1:
                result.sell_wins += 1
            elif entry.sell_label == 0:
                result.sell_losses += 1
            if entry.buy_label == -1 or entry.sell_label == -1:
                result.timeouts += 1

        return result

    def reset(self) -> None:
        """Reset state for a new labeling run."""
        self._pending.clear()
        self._resolved.clear()
        self._next_index = 0
