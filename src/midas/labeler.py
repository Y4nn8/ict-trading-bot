"""TickLabeler: label ticks for entry model training.

For each sampled tick, evaluates hypothetical BUY/SELL trades with
SL/TP lookahead. Tracks actual PnL in points (not just win/loss).

Labels:
    buy_label/sell_label: 1=win, 0=loss, -1=timeout
    buy_pnl/sell_pnl: actual PnL in price points
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import nan
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    from src.midas.types import LabelConfig, Tick


@dataclass
class _PendingEntry:
    """A pending entry label awaiting SL/TP resolution."""

    feature_index: int
    entry_time: datetime
    entry_ask: float
    entry_bid: float
    # BUY: enter at ask, SL/TP evaluated against bid
    buy_sl: float
    buy_tp: float
    # SELL: enter at bid, SL/TP evaluated against ask
    sell_sl: float
    sell_tp: float
    # PnL in points (set on resolution)
    buy_pnl: float = nan
    sell_pnl: float = nan
    buy_resolved: bool = False
    sell_resolved: bool = False
    # Track best unrealized PnL for exit labeling
    buy_best_pnl: float = 0.0
    sell_best_pnl: float = 0.0

    @property
    def fully_resolved(self) -> bool:
        """Both BUY and SELL are resolved."""
        return self.buy_resolved and self.sell_resolved


@dataclass
class LabelResult:
    """Output of the labeling process."""

    # Entry labels (per sampled tick)
    buy_labels: list[int] = field(default_factory=list)
    sell_labels: list[int] = field(default_factory=list)
    buy_pnls: list[float] = field(default_factory=list)
    sell_pnls: list[float] = field(default_factory=list)
    total_labeled: int = 0
    buy_wins: int = 0
    buy_losses: int = 0
    sell_wins: int = 0
    sell_losses: int = 0
    timeouts: int = 0



class TickLabeler:
    """Labels ticks for entry and exit model training.

    Usage during replay:
        1. Call on_tick() for EVERY tick (resolves pending entries)
        2. Call add_entry() for sampled ticks (registers new candidates)
        3. Call finalize() at end to get labels

    Args:
        config: Label configuration (SL/TP distances, timeout).
    """

    def __init__(self, config: LabelConfig) -> None:
        self._config = config
        self._pending: deque[_PendingEntry] = deque()
        self._resolved: list[_PendingEntry] = []
        self._next_index: int = 0

    @property
    def timeout_seconds(self) -> float:
        """Label timeout in seconds (for lookahead query extension)."""
        return self._config.timeout_seconds

    def add_entry(self, tick: Tick) -> int:
        """Register a tick as a candidate entry point.

        Args:
            tick: The tick at the potential entry.

        Returns:
            The feature index assigned to this entry.
        """
        idx = self._next_index
        self._next_index += 1

        sl = self._config.sl_points
        tp = self._config.tp_points

        entry = _PendingEntry(
            feature_index=idx,
            entry_time=tick.time,
            entry_ask=tick.ask,
            entry_bid=tick.bid,
            buy_sl=tick.ask - sl,
            buy_tp=tick.ask + tp,
            sell_sl=tick.bid + sl,
            sell_tp=tick.bid - tp,
        )
        self._pending.append(entry)
        return idx

    def on_tick(self, tick: Tick) -> None:
        """Evaluate all pending entries against this tick.

        Args:
            tick: The current tick.
        """
        timeout = self._config.timeout_seconds

        for pending in self._pending:
            # BUY evaluation: exits against bid
            if not pending.buy_resolved:
                unrealized = tick.bid - pending.entry_ask
                pending.buy_best_pnl = max(
                    pending.buy_best_pnl, unrealized,
                )

                if tick.bid <= pending.buy_sl:
                    pending.buy_pnl = pending.buy_sl - pending.entry_ask
                    pending.buy_resolved = True
                elif tick.bid >= pending.buy_tp:
                    pending.buy_pnl = pending.buy_tp - pending.entry_ask
                    pending.buy_resolved = True

            # SELL evaluation: exits against ask
            if not pending.sell_resolved:
                unrealized = pending.entry_bid - tick.ask
                pending.sell_best_pnl = max(
                    pending.sell_best_pnl, unrealized,
                )

                if tick.ask >= pending.sell_sl:
                    pending.sell_pnl = pending.entry_bid - pending.sell_sl
                    pending.sell_resolved = True
                elif tick.ask <= pending.sell_tp:
                    pending.sell_pnl = pending.entry_bid - pending.sell_tp
                    pending.sell_resolved = True

            # Timeout
            elapsed = (tick.time - pending.entry_time).total_seconds()
            if elapsed > timeout:
                if not pending.buy_resolved:
                    # Close at current bid on timeout
                    pending.buy_pnl = tick.bid - pending.entry_ask
                    pending.buy_resolved = True
                if not pending.sell_resolved:
                    pending.sell_pnl = pending.entry_bid - tick.ask
                    pending.sell_resolved = True

        # Move resolved entries out of pending
        while self._pending and self._pending[0].fully_resolved:
            self._resolved.append(self._pending.popleft())

    def finalize(self) -> LabelResult:
        """Finalize all pending entries and return labels.

        Returns:
            LabelResult with entry labels + PnLs and exit labels.
        """
        # Force-resolve remaining entries
        for pending in self._pending:
            if not pending.buy_resolved:
                pending.buy_pnl = 0.0  # unknown
                pending.buy_resolved = True
            if not pending.sell_resolved:
                pending.sell_pnl = 0.0
                pending.sell_resolved = True
            self._resolved.append(pending)
        self._pending.clear()

        self._resolved.sort(key=lambda p: p.feature_index)

        result = LabelResult(total_labeled=len(self._resolved))
        for entry in self._resolved:
            # Entry labels: 1=win, 0=loss, -1=timeout (backward compat)
            buy_label = (
                1 if entry.buy_pnl > 0
                else (0 if entry.buy_pnl < 0 else -1)
            )
            sell_label = (
                1 if entry.sell_pnl > 0
                else (0 if entry.sell_pnl < 0 else -1)
            )
            result.buy_labels.append(buy_label)
            result.sell_labels.append(sell_label)
            result.buy_pnls.append(entry.buy_pnl)
            result.sell_pnls.append(entry.sell_pnl)

            if buy_label == 1:
                result.buy_wins += 1
            elif buy_label == 0:
                result.buy_losses += 1
            if sell_label == 1:
                result.sell_wins += 1
            elif sell_label == 0:
                result.sell_losses += 1
            if buy_label == -1 or sell_label == -1:
                result.timeouts += 1

        return result

    def reset(self) -> None:
        """Reset state for a new labeling run."""
        self._pending.clear()
        self._resolved.clear()
        self._next_index = 0
