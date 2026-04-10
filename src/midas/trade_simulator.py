"""Tick-level trade simulator for the Midas scalping engine.

Manages open positions tick-by-tick with fixed SL/TP.
Entry at ask (BUY) or bid (SELL), exit evaluated against bid/ask.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    from src.midas.types import Tick


@dataclass(frozen=True, slots=True)
class SimConfig:
    """Trade simulation configuration.

    Args:
        sl_points: Stop loss distance in price points.
        tp_points: Take profit distance in price points.
        initial_capital: Starting capital.
        size: Position size per trade.
        value_per_point: Currency value per price point per unit size.
        max_open_positions: Max simultaneous positions.
        max_spread: Skip entries when spread exceeds this.
    """

    sl_points: float = 3.0
    tp_points: float = 3.0
    initial_capital: float = 10_000.0
    size: float = 0.1
    value_per_point: float = 1.0  # 1€/pt for XAUUSD Contrat 1€
    max_open_positions: int = 1
    max_spread: float = 2.0


@dataclass
class MidasPosition:
    """An open position."""

    trade_id: str
    direction: str  # "BUY" or "SELL"
    entry_price: float
    entry_time: datetime
    sl_price: float
    tp_price: float
    size: float


@dataclass(frozen=True, slots=True)
class MidasTrade:
    """A completed trade."""

    trade_id: str
    direction: str
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    sl_price: float
    tp_price: float
    size: float
    pnl: float
    pnl_points: float
    is_win: bool


class TradeSimulator:
    """Tick-level position manager with fixed SL/TP.

    Args:
        config: Simulation configuration.
    """

    def __init__(self, config: SimConfig | None = None) -> None:
        self._config = config or SimConfig()
        self._positions: list[MidasPosition] = []
        self._trades: list[MidasTrade] = []
        self._capital: float = self._config.initial_capital

    @property
    def closed_trades(self) -> list[MidasTrade]:
        """All completed trades."""
        return list(self._trades)

    @property
    def capital(self) -> float:
        """Current capital."""
        return self._capital

    @property
    def open_count(self) -> int:
        """Number of open positions."""
        return len(self._positions)

    def on_signal(self, tick: Tick, signal: int) -> list[MidasTrade]:
        """Process a model signal at a tick.

        Args:
            tick: Current tick.
            signal: 0=PASS, 1=BUY, 2=SELL.

        Returns:
            List of trades closed on this tick (may be empty).
        """
        # First, check existing positions for SL/TP
        closed = self._check_exits(tick)

        # Then try to open new position
        if signal in (1, 2) and self._can_open(tick):
            self._open_position(tick, "BUY" if signal == 1 else "SELL")

        return closed

    def on_tick(self, tick: Tick) -> list[MidasTrade]:
        """Check SL/TP exits without a new signal.

        Args:
            tick: Current tick.

        Returns:
            List of trades closed on this tick.
        """
        return self._check_exits(tick)

    def early_close(self, tick: Tick, position_index: int = 0) -> MidasTrade | None:
        """Close a specific position early (exit model decision).

        Args:
            tick: Current tick for exit price.
            position_index: Index into open positions list.

        Returns:
            The closed trade, or None if index is invalid.
        """
        if position_index >= len(self._positions):
            return None

        pos = self._positions.pop(position_index)
        exit_price = tick.bid if pos.direction == "BUY" else tick.ask
        pnl_points = (
            (exit_price - pos.entry_price)
            if pos.direction == "BUY"
            else (pos.entry_price - exit_price)
        )
        pnl = pnl_points * pos.size * self._config.value_per_point

        trade = MidasTrade(
            trade_id=pos.trade_id,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            entry_time=pos.entry_time,
            exit_time=tick.time,
            sl_price=pos.sl_price,
            tp_price=pos.tp_price,
            size=pos.size,
            pnl=pnl,
            pnl_points=pnl_points,
            is_win=pnl > 0,
        )
        self._trades.append(trade)
        self._capital += pnl
        return trade

    def get_position_context(
        self, tick: Tick, position_index: int = 0,
    ) -> dict[str, float] | None:
        """Get position context features for the exit model.

        Args:
            tick: Current tick.
            position_index: Index into open positions list.

        Returns:
            Dict with pos_unrealized_pnl, pos_duration_sec, pos_direction.
            None if no position at that index.
        """
        if position_index >= len(self._positions):
            return None

        pos = self._positions[position_index]
        if pos.direction == "BUY":
            unrealized = tick.bid - pos.entry_price
            direction = 1.0
        else:
            unrealized = pos.entry_price - tick.ask
            direction = -1.0

        duration = (tick.time - pos.entry_time).total_seconds()

        return {
            "pos_unrealized_pnl": unrealized,
            "pos_duration_sec": duration,
            "pos_direction": direction,
        }

    def _can_open(self, tick: Tick) -> bool:
        """Check if we can open a new position."""
        if len(self._positions) >= self._config.max_open_positions:
            return False
        return tick.spread <= self._config.max_spread

    def _open_position(self, tick: Tick, direction: str) -> None:
        """Open a new position."""
        cfg = self._config

        if direction == "BUY":
            entry = tick.ask
            sl = entry - cfg.sl_points
            tp = entry + cfg.tp_points
        else:
            entry = tick.bid
            sl = entry + cfg.sl_points
            tp = entry - cfg.tp_points

        pos = MidasPosition(
            trade_id=str(uuid.uuid4())[:8],
            direction=direction,
            entry_price=entry,
            entry_time=tick.time,
            sl_price=sl,
            tp_price=tp,
            size=cfg.size,
        )
        self._positions.append(pos)

    def _check_exits(self, tick: Tick) -> list[MidasTrade]:
        """Check all open positions for SL/TP hit.

        SL is checked before TP (pessimistic assumption).
        Closes all eligible positions on the same tick.
        """
        closed: list[MidasTrade] = []
        remaining: list[MidasPosition] = []

        for pos in self._positions:
            exit_price: float | None = None
            is_win = False

            if pos.direction == "BUY":
                if tick.bid <= pos.sl_price:
                    exit_price = pos.sl_price
                elif tick.bid >= pos.tp_price:
                    exit_price = pos.tp_price
                    is_win = True
            else:
                if tick.ask >= pos.sl_price:
                    exit_price = pos.sl_price
                elif tick.ask <= pos.tp_price:
                    exit_price = pos.tp_price
                    is_win = True

            if exit_price is not None:
                pnl_points = (
                    (exit_price - pos.entry_price)
                    if pos.direction == "BUY"
                    else (pos.entry_price - exit_price)
                )
                pnl = pnl_points * pos.size * self._config.value_per_point

                trade = MidasTrade(
                    trade_id=pos.trade_id,
                    direction=pos.direction,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    entry_time=pos.entry_time,
                    exit_time=tick.time,
                    sl_price=pos.sl_price,
                    tp_price=pos.tp_price,
                    size=pos.size,
                    pnl=pnl,
                    pnl_points=pnl_points,
                    is_win=is_win,
                )
                self._trades.append(trade)
                self._capital += pnl
                closed.append(trade)
            else:
                remaining.append(pos)

        self._positions = remaining
        return closed

    def close_all(self, tick: Tick) -> list[MidasTrade]:
        """Force-close all open positions at current market price.

        Args:
            tick: Current tick for exit price.

        Returns:
            List of closed trades.
        """
        closed: list[MidasTrade] = []
        for pos in self._positions:
            exit_price = tick.bid if pos.direction == "BUY" else tick.ask

            pnl_points = (
                (exit_price - pos.entry_price)
                if pos.direction == "BUY"
                else (pos.entry_price - exit_price)
            )
            pnl = pnl_points * pos.size * self._config.value_per_point

            trade = MidasTrade(
                trade_id=pos.trade_id,
                direction=pos.direction,
                entry_price=pos.entry_price,
                exit_price=exit_price,
                entry_time=pos.entry_time,
                exit_time=tick.time,
                sl_price=pos.sl_price,
                tp_price=pos.tp_price,
                size=pos.size,
                pnl=pnl,
                pnl_points=pnl_points,
                is_win=pnl > 0,
            )
            self._trades.append(trade)
            self._capital += pnl
            closed.append(trade)

        self._positions.clear()
        return closed

    def reset(self) -> None:
        """Reset simulator for a new run."""
        self._positions.clear()
        self._trades.clear()
        self._capital = self._config.initial_capital
