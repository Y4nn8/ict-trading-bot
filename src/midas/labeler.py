"""TickLabeler: label ticks for entry and exit model training.

Two modes:
  - Streaming: during replay, evaluates SL/TP lookahead tick-by-tick.
  - DataFrame: relabel a features DataFrame with new SL/TP params
    (fast, no DB access needed). JIT-compiled with numba.

Entry labels:
    buy_label/sell_label: 1=win, 0=loss, -1=timeout
    buy_pnl/sell_pnl: actual PnL in price points

Exit labels (optimal-close):
    For each candle while a position is open, labels CLOSE=1 if closing
    now yields better PnL than holding to the trade's natural exit.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import nan
from typing import TYPE_CHECKING

import numba as nb
import numpy as np

if TYPE_CHECKING:
    from datetime import datetime

    import polars as pl

    from src.midas.types import LabelConfig, Tick

from src.midas.types import ATR_COLUMN_DEFAULT


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


@dataclass(frozen=True, slots=True)
class ExitDatasetResult:
    """Output of the exit dataset builder.

    Contains parallel arrays of row indices, position context, and
    optimal-close labels for training the HOLD/CLOSE exit model.
    """

    row_indices: np.ndarray
    directions: np.ndarray  # 1.0=BUY, -1.0=SELL
    unrealized_pnls: np.ndarray
    durations: np.ndarray  # seconds since entry
    exit_labels: np.ndarray  # 0=HOLD, 1=CLOSE
    n_entries: int = 0
    n_rows: int = 0



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


@nb.njit(cache=True)  # type: ignore[untyped-decorator]
def _relabel_core(
    times: np.ndarray,
    bids: np.ndarray,
    asks: np.ndarray,
    sl_arr: np.ndarray,
    tp_arr: np.ndarray,
    timeout_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """JIT-compiled labeling core.

    Per-row SL/TP arrays allow both fixed and ATR-based modes.

    Args:
        times: Timestamps as float64 (epoch seconds).
        bids: Bid prices.
        asks: Ask prices.
        sl_arr: SL distance per row (in price points).
        tp_arr: TP distance per row (in price points).
        timeout_seconds: Max lookahead in seconds.

    Returns:
        Tuple of (buy_labels, sell_labels, buy_pnls, sell_pnls).
    """
    n = len(times)
    buy_labels = np.zeros(n, dtype=np.int32)
    sell_labels = np.zeros(n, dtype=np.int32)
    buy_pnls = np.zeros(n, dtype=np.float64)
    sell_pnls = np.zeros(n, dtype=np.float64)

    for i in range(n):
        entry_ask = asks[i]
        entry_bid = bids[i]
        entry_time = times[i]
        sl = sl_arr[i]
        tp = tp_arr[i]

        buy_sl = entry_ask - sl
        buy_tp = entry_ask + tp
        sell_sl = entry_bid + sl
        sell_tp = entry_bid - tp

        buy_resolved = False
        sell_resolved = False

        for j in range(i + 1, n):
            if times[j] - entry_time > timeout_seconds:
                if not buy_resolved:
                    pnl = bids[j] - entry_ask
                    buy_pnls[i] = pnl
                    if pnl > 0.0:
                        buy_labels[i] = 1
                    elif pnl < 0.0:
                        buy_labels[i] = 0
                    else:
                        buy_labels[i] = -1
                    buy_resolved = True
                if not sell_resolved:
                    pnl = entry_bid - asks[j]
                    sell_pnls[i] = pnl
                    if pnl > 0.0:
                        sell_labels[i] = 1
                    elif pnl < 0.0:
                        sell_labels[i] = 0
                    else:
                        sell_labels[i] = -1
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

        if not buy_resolved:
            buy_labels[i] = -1
        if not sell_resolved:
            sell_labels[i] = -1

    return buy_labels, sell_labels, buy_pnls, sell_pnls


def _build_price_arrays(
    df: pl.DataFrame,
    sl_points: float,
    tp_points: float,
    k_sl: float | None,
    k_tp: float | None,
    atr_column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract price arrays and build per-row SL/TP distances.

    Shared by ``relabel_dataframe`` and ``build_exit_dataset``.

    Returns:
        (times, bids, asks, sl_arr, tp_arr) as float64 numpy arrays.
    """
    times = df["_time"].to_numpy().astype(np.float64)
    bids = df["_bid"].to_numpy().astype(np.float64)
    asks = df["_ask"].to_numpy().astype(np.float64)
    n = len(times)

    if k_sl is not None and k_tp is not None:
        atrs = df[atr_column].to_numpy().astype(np.float64)
        sl_arr = np.where(atrs > 0.0, k_sl * atrs, sl_points)
        tp_arr = np.where(atrs > 0.0, k_tp * atrs, tp_points)
    else:
        sl_arr = np.full(n, sl_points, dtype=np.float64)
        tp_arr = np.full(n, tp_points, dtype=np.float64)

    return times, bids, asks, sl_arr, tp_arr


def relabel_dataframe(
    df: pl.DataFrame,
    sl_points: float,
    tp_points: float,
    timeout_seconds: float,
    *,
    k_sl: float | None = None,
    k_tp: float | None = None,
    atr_column: str = ATR_COLUMN_DEFAULT,
) -> LabelResult:
    """Relabel a features DataFrame with SL/TP params.

    JIT-compiled alternative to streaming labeling — works on
    already-extracted features without DB access.

    Supports two modes:
      - **Fixed**: uses ``sl_points``/``tp_points`` directly.
      - **ATR-based**: when ``k_sl``/``k_tp`` are provided, computes
        per-row SL = k_sl * ATR, TP = k_tp * ATR from ``atr_column``.
        Falls back to ``sl_points``/``tp_points`` when ATR is zero
        (cold-start rows).

    Requires columns: _time, _bid, _ask (and ``atr_column`` for ATR mode).

    Args:
        df: Features DataFrame with _time, _bid, _ask columns.
        sl_points: Stop loss distance in price points (fixed mode,
            or fallback when ATR is zero).
        tp_points: Take profit distance in price points (fixed mode,
            or fallback when ATR is zero).
        timeout_seconds: Max lookahead in seconds.
        k_sl: SL multiplier for ATR-based mode (optional).
        k_tp: TP multiplier for ATR-based mode (optional).
        atr_column: Column name containing ATR values.

    Returns:
        LabelResult with buy/sell labels and PnL arrays.
    """
    times, bids, asks, sl_arr, tp_arr = _build_price_arrays(
        df, sl_points, tp_points, k_sl, k_tp, atr_column,
    )

    buy_labels, sell_labels, buy_pnls, sell_pnls = _relabel_core(
        times, bids, asks, sl_arr, tp_arr, timeout_seconds,
    )

    result = LabelResult(total_labeled=len(times))
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


@nb.njit(cache=True)  # type: ignore[untyped-decorator]
def _exit_dataset_core(
    times: np.ndarray,
    bids: np.ndarray,
    asks: np.ndarray,
    sl_arr: np.ndarray,
    tp_arr: np.ndarray,
    timeout_seconds: float,
    entry_indices: np.ndarray,
    entry_directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """JIT-compiled exit dataset builder.

    For each entry, walks forward through subsequent candles. At each
    candle, records position context and an optimal-close label:
    CLOSE=1 if unrealized PnL now > final PnL (closing beats holding).

    Args:
        times: Timestamps as float64 (epoch seconds).
        bids: Bid prices.
        asks: Ask prices.
        sl_arr: SL distance per row (price points).
        tp_arr: TP distance per row (price points).
        timeout_seconds: Max trade duration.
        entry_indices: Row indices where entries occur.
        entry_directions: 1=BUY, 2=SELL per entry.

    Returns:
        Tuple of (row_indices, directions, unrealized_pnls,
        durations, exit_labels, n_resolved) — variable-length
        parallel arrays plus the count of resolved entries.
    """
    n = len(times)
    n_entries = len(entry_indices)

    # Pre-allocate with upper bound (each entry can generate up to n rows)
    max_rows = 0
    for e in range(n_entries):
        idx = entry_indices[e]
        max_rows += n - idx - 1
    # Cap to avoid excessive memory — practical trades are short
    max_rows = min(max_rows, n_entries * 500)

    out_row = np.empty(max_rows, dtype=np.int64)
    out_dir = np.empty(max_rows, dtype=np.float64)
    out_pnl = np.empty(max_rows, dtype=np.float64)
    out_dur = np.empty(max_rows, dtype=np.float64)
    out_lbl = np.empty(max_rows, dtype=np.int32)

    pos = 0  # write position
    n_resolved = 0

    for e in range(n_entries):
        idx = entry_indices[e]
        direction = entry_directions[e]
        entry_time = times[idx]
        sl = sl_arr[idx]
        tp = tp_arr[idx]

        if direction == 1:  # BUY
            entry_price = asks[idx]
            sl_price = entry_price - sl
            tp_price = entry_price + tp
        else:  # SELL
            entry_price = bids[idx]
            sl_price = entry_price + sl
            tp_price = entry_price - tp

        # First pass: find exit point and final PnL
        exit_j = -1
        final_pnl = 0.0
        for j in range(idx + 1, n):
            elapsed = times[j] - entry_time
            if elapsed > timeout_seconds:
                # Timeout: close at market
                if direction == 1:  # noqa: SIM108
                    final_pnl = bids[j] - entry_price
                else:
                    final_pnl = entry_price - asks[j]
                exit_j = j
                break

            if direction == 1:
                if bids[j] <= sl_price:
                    final_pnl = sl_price - entry_price
                    exit_j = j
                    break
                if bids[j] >= tp_price:
                    final_pnl = tp_price - entry_price
                    exit_j = j
                    break
            else:
                if asks[j] >= sl_price:
                    final_pnl = entry_price - sl_price
                    exit_j = j
                    break
                if asks[j] <= tp_price:
                    final_pnl = entry_price - tp_price
                    exit_j = j
                    break

        if exit_j == -1:
            continue

        n_resolved += 1
        # Second pass: generate exit rows for candles between entry+1 and exit
        dir_f = 1.0 if direction == 1 else -1.0
        for j in range(idx + 1, exit_j + 1):
            if pos >= max_rows:
                break

            elapsed = times[j] - entry_time
            if direction == 1:  # noqa: SIM108
                unrealized = bids[j] - entry_price
            else:
                unrealized = entry_price - asks[j]

            # Optimal-close label: CLOSE=1 if closing now beats holding
            label = 1 if unrealized > final_pnl else 0

            out_row[pos] = j
            out_dir[pos] = dir_f
            out_pnl[pos] = unrealized
            out_dur[pos] = elapsed
            out_lbl[pos] = label
            pos += 1

    return out_row[:pos], out_dir[:pos], out_pnl[:pos], out_dur[:pos], out_lbl[:pos], n_resolved


def build_exit_dataset(
    df: pl.DataFrame,
    entry_target: np.ndarray,
    sl_points: float,
    tp_points: float,
    timeout_seconds: float,
    *,
    k_sl: float | None = None,
    k_tp: float | None = None,
    atr_column: str = ATR_COLUMN_DEFAULT,
) -> ExitDatasetResult:
    """Build training dataset for the HOLD/CLOSE exit model.

    For each entry predicted by the entry model (target != 0), simulates
    the trade forward through subsequent candles. At each candle while
    the position is open, records the market features, position context,
    and an optimal-close label.

    The optimal-close label is CLOSE=1 when the unrealized PnL at the
    current candle exceeds the trade's final PnL — meaning closing now
    would have been better than holding to the natural SL/TP/timeout exit.

    Args:
        df: Features DataFrame with _time, _bid, _ask columns.
        entry_target: Entry target array (0=PASS, 1=BUY, 2=SELL).
        sl_points: SL distance (fixed mode / fallback).
        tp_points: TP distance (fixed mode / fallback).
        timeout_seconds: Max trade duration.
        k_sl: SL multiplier for ATR-based mode (optional).
        k_tp: TP multiplier for ATR-based mode (optional).
        atr_column: ATR column name for ATR-based mode.

    Returns:
        ExitDatasetResult with row indices, position context, and labels.

    Raises:
        ValueError: If entry_target length doesn't match df length.
    """
    if len(entry_target) != len(df):
        msg = (
            f"entry_target length ({len(entry_target)}) "
            f"!= df length ({len(df)})"
        )
        raise ValueError(msg)

    times, bids, asks, sl_arr, tp_arr = _build_price_arrays(
        df, sl_points, tp_points, k_sl, k_tp, atr_column,
    )

    entry_mask = entry_target != 0
    entry_indices = np.where(entry_mask)[0].astype(np.int64)
    entry_directions = entry_target[entry_mask].astype(np.int64)

    if len(entry_indices) == 0:
        return ExitDatasetResult(
            row_indices=np.array([], dtype=np.int64),
            directions=np.array([], dtype=np.float64),
            unrealized_pnls=np.array([], dtype=np.float64),
            durations=np.array([], dtype=np.float64),
            exit_labels=np.array([], dtype=np.int32),
        )

    row_indices, directions, unrealized_pnls, durations, exit_labels, n_resolved = (
        _exit_dataset_core(
            times, bids, asks, sl_arr, tp_arr,
            timeout_seconds, entry_indices, entry_directions,
        )
    )

    return ExitDatasetResult(
        row_indices=row_indices,
        directions=directions,
        unrealized_pnls=unrealized_pnls,
        durations=durations,
        exit_labels=exit_labels,
        n_entries=int(n_resolved),
        n_rows=len(row_indices),
    )
