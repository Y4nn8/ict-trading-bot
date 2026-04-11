"""TickLabeler: label ticks for entry model training.

Two modes:
  - Streaming: during replay, evaluates SL/TP lookahead tick-by-tick.
  - DataFrame: relabel a features DataFrame with new SL/TP params
    (fast, no DB access needed).

Labels:
    buy_label/sell_label: 1=win, 0=loss, -1=timeout
    buy_pnl/sell_pnl: actual PnL in price points
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import nan
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from datetime import datetime

    import polars as pl

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


def relabel_dataframe(
    df: pl.DataFrame,
    sl_points: float,
    tp_points: float,
    timeout_seconds: float,
) -> LabelResult:
    """Relabel a features DataFrame with new SL/TP params.

    Fast alternative to streaming labeling — works on already-extracted
    features without DB access. Scans forward in the DataFrame for each
    row to check SL/TP hit.

    Requires columns: _time, _bid, _ask.

    Args:
        df: Features DataFrame with _time, _bid, _ask columns.
        sl_points: Stop loss distance in price points.
        tp_points: Take profit distance in price points.
        timeout_seconds: Max lookahead in seconds.

    Returns:
        LabelResult with buy/sell labels and PnL arrays.
    """
    times = df["_time"].to_numpy()  # timestamps as float
    bids = df["_bid"].to_numpy()
    asks = df["_ask"].to_numpy()
    n = len(times)

    buy_labels = np.zeros(n, dtype=np.int32)
    sell_labels = np.zeros(n, dtype=np.int32)
    buy_pnls = np.zeros(n, dtype=np.float64)
    sell_pnls = np.zeros(n, dtype=np.float64)

    for i in range(n):
        entry_ask = asks[i]
        entry_bid = bids[i]
        entry_time = times[i]

        buy_sl = entry_ask - sl_points
        buy_tp = entry_ask + tp_points
        sell_sl = entry_bid + sl_points
        sell_tp = entry_bid - tp_points

        buy_resolved = False
        sell_resolved = False

        # Scan forward
        for j in range(i + 1, n):
            if times[j] - entry_time > timeout_seconds:
                # Timeout: close at market
                if not buy_resolved:
                    buy_pnls[i] = bids[j] - entry_ask
                    buy_labels[i] = 1 if buy_pnls[i] > 0 else (0 if buy_pnls[i] < 0 else -1)
                    buy_resolved = True
                if not sell_resolved:
                    sell_pnls[i] = entry_bid - asks[j]
                    sell_labels[i] = 1 if sell_pnls[i] > 0 else (0 if sell_pnls[i] < 0 else -1)
                    sell_resolved = True
                break

            if not buy_resolved:
                if bids[j] <= buy_sl:
                    buy_pnls[i] = buy_sl - entry_ask
                    buy_labels[i] = 0
                    buy_resolved = True
                elif bids[j] >= buy_tp:
                    buy_pnls[i] = buy_tp - entry_ask
                    buy_labels[i] = 1
                    buy_resolved = True

            if not sell_resolved:
                if asks[j] >= sell_sl:
                    sell_pnls[i] = entry_bid - sell_sl
                    sell_labels[i] = 0
                    sell_resolved = True
                elif asks[j] <= sell_tp:
                    sell_pnls[i] = entry_bid - sell_tp
                    sell_labels[i] = 1
                    sell_resolved = True

            if buy_resolved and sell_resolved:
                break

        # End of data: unresolved → timeout
        if not buy_resolved:
            buy_labels[i] = -1
        if not sell_resolved:
            sell_labels[i] = -1

    # Build result
    result = LabelResult(total_labeled=n)
    result.buy_labels = buy_labels.tolist()
    result.sell_labels = sell_labels.tolist()
    result.buy_pnls = buy_pnls.tolist()
    result.sell_pnls = sell_pnls.tolist()
    result.buy_wins = int((buy_labels == 1).sum())
    result.buy_losses = int((buy_labels == 0).sum())
    result.sell_wins = int((sell_labels == 1).sum())
    result.sell_losses = int((sell_labels == 0).sum())
    result.timeouts = int(((buy_labels == -1) | (sell_labels == -1)).sum())
    return result
