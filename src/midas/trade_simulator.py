"""Tick-level trade simulator for the Midas scalping engine.

Manages open positions tick-by-tick with fixed or dynamic SL/TP.
Entry at ask (BUY) or bid (SELL), exit evaluated against bid/ask.

Dynamic sizing maps LightGBM probability → contract size via a
gamma-ramp curve, scaling up to the full available margin when
confidence is high.

Slippage simulation applies random adverse slippage to market orders
(entry, early_close, close_all).  SL/TP exits are guaranteed
stop/limit orders — no slippage.
"""

from __future__ import annotations

import math
import random
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    from src.midas.types import Tick


@dataclass(frozen=True, slots=True)
class SimConfig:
    """Trade simulation configuration.

    Supports two SL/TP modes:
      - **Fixed**: uses ``sl_points``/``tp_points`` directly.
      - **ATR-based**: when ``k_sl``/``k_tp`` are set, computes
        SL = k_sl * ATR, TP = k_tp * ATR from the ATR value passed
        to ``on_signal``. Falls back to ``sl_points``/``tp_points``
        when ATR is zero.

    Supports two sizing modes:
      - **Fixed**: uses ``size`` for every trade.
      - **Dynamic**: when ``gamma`` is set, computes position size from
        the model's probability via a gamma-ramp curve, scaling up to
        the full available margin at ``max_margin_proba``.

    Args:
        sl_points: Stop loss distance in price points (fixed mode / fallback).
        tp_points: Take profit distance in price points (fixed mode / fallback).
        k_sl: SL multiplier for ATR-based mode (None = fixed mode).
        k_tp: TP multiplier for ATR-based mode (None = fixed mode).
        initial_capital: Starting capital.
        size: Position size per trade (fixed sizing mode).
        value_per_point: Currency value per price point per unit size.
        max_open_positions: Max simultaneous positions.
        max_spread: Skip entries when spread exceeds this.
        gamma: Ramp curvature for dynamic sizing (None = fixed sizing).
        max_margin_proba: Probability threshold for 100% margin usage.
        margin_pct: Margin requirement as fraction of price (e.g. 0.05 = 5%).
        min_lot_size: Minimum tradeable lot size.
        sizing_threshold: Probability floor for the gamma ramp. Should
            match the model's entry_threshold so confidence=0 at the
            decision boundary.
        slippage_min_pts: Minimum slippage in price points per market
            order. Set >0 for conservative simulation.
        slippage_max_pts: Maximum random slippage in price points.
            Each market order draws uniform(min, max). Both at 0
            disables slippage.
        slippage_seed: RNG seed for reproducible slippage sequences.
    """

    sl_points: float = 3.0
    tp_points: float = 3.0
    k_sl: float | None = None
    k_tp: float | None = None
    initial_capital: float = 5_000.0
    size: float = 0.1
    value_per_point: float = 1.0  # 1€/pt for XAUUSD Contrat 1€
    max_open_positions: int = 1
    max_spread: float = 2.0
    gamma: float | None = None
    max_margin_proba: float = 0.85
    margin_pct: float = 0.05
    min_lot_size: float = 0.1
    sizing_threshold: float = 1 / 3
    slippage_min_pts: float = 0.0
    slippage_max_pts: float = 0.0
    slippage_seed: int | None = None


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
    margin: float = 0.0
    proba: float = 0.0


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
    proba: float = 0.0


class TradeSimulator:
    """Tick-level position manager with fixed or dynamic SL/TP and sizing.

    Args:
        config: Simulation configuration.
    """

    def __init__(self, config: SimConfig | None = None) -> None:
        self._config = config or SimConfig()
        self._positions: list[MidasPosition] = []
        self._trades: list[MidasTrade] = []
        self._capital: float = self._config.initial_capital
        self._rng = random.Random(self._config.slippage_seed)

    @property
    def closed_trades(self) -> list[MidasTrade]:
        """All completed trades."""
        return list(self._trades)

    @property
    def capital(self) -> float:
        """Current capital."""
        return self._capital

    @property
    def margin_used(self) -> float:
        """Total margin currently locked by open positions."""
        return sum(pos.margin for pos in self._positions)

    @property
    def open_count(self) -> int:
        """Number of open positions."""
        return len(self._positions)

    def _sample_slippage(self) -> float:
        """Draw a random adverse slippage amount in price points.

        Returns 0.0 when slippage simulation is disabled.
        """
        mn = self._config.slippage_min_pts
        mx = self._config.slippage_max_pts
        if mx <= 0.0 and mn <= 0.0:
            return 0.0
        return self._rng.uniform(mn, mx)

    def on_signal(
        self,
        tick: Tick,
        signal: int,
        *,
        atr: float = 0.0,
        proba: float = 0.0,
    ) -> list[MidasTrade]:
        """Process a model signal at a tick.

        Args:
            tick: Current tick.
            signal: 0=PASS, 1=BUY, 2=SELL.
            atr: Current ATR value (used in ATR-based SL/TP mode).
            proba: Model probability for the predicted class (used in
                dynamic sizing mode).

        Returns:
            List of trades closed on this tick (may be empty).
        """
        closed = self._check_exits(tick)

        if signal in (1, 2) and self._can_open(tick):
            self._open_position(
                tick, "BUY" if signal == 1 else "SELL",
                atr=atr, proba=proba,
            )

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
        slip = self._sample_slippage()
        exit_price = tick.bid - slip if pos.direction == "BUY" else tick.ask + slip
        return self._close_position(pos, exit_price, tick.time)

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
        if tick.spread > self._config.max_spread:
            return False
        if self._config.gamma is not None:
            # Use mid price for the pre-check; actual margin uses bid/ask in _open_position
            price = tick.mid
            min_margin = price * self._config.margin_pct * self._config.min_lot_size
            if min_margin > self._capital - self.margin_used:
                return False
        return True

    @staticmethod
    def compute_dynamic_size(
        proba: float,
        threshold: float,
        gamma: float,
        max_margin_proba: float,
        available_margin: float,
        margin_per_lot: float,
        min_lot_size: float,
    ) -> float | None:
        """Compute position size from model probability via gamma ramp.

        Args:
            proba: Model probability for the predicted class.
            threshold: Entry threshold (minimum proba to trade).
            gamma: Ramp curvature (>1 conservative, <1 aggressive).
            max_margin_proba: Proba at which 100% of available margin is used.
            available_margin: Capital minus margin already locked.
            margin_per_lot: Margin required per 1.0 lot.
            min_lot_size: Minimum tradeable lot.

        Returns:
            Position size in lots, or None if below min_lot_size.
        """
        if margin_per_lot <= 0 or available_margin <= 0 or min_lot_size <= 0:
            return None

        # Small epsilon avoids floor under-allocating due to IEEE 754 drift
        eps = 1e-9
        lot_steps = available_margin / margin_per_lot / min_lot_size
        size_max = math.floor(lot_steps + eps) * min_lot_size

        if size_max < min_lot_size:
            return None

        if proba >= max_margin_proba:
            size = size_max
        else:
            denom = max_margin_proba - threshold
            if denom <= 0:
                return None
            confidence = (proba - threshold) / denom
            confidence = max(confidence, 0.0)
            raw_steps = (confidence ** gamma) * size_max / min_lot_size
            size = math.floor(raw_steps + eps) * min_lot_size

        return size if size >= min_lot_size else None

    def _close_position(
        self,
        pos: MidasPosition,
        exit_price: float,
        exit_time: datetime,
    ) -> MidasTrade:
        """Build a MidasTrade, update capital, and record it.

        Args:
            pos: The position being closed.
            exit_price: Price at which the position exits.
            exit_time: Timestamp of exit.

        Returns:
            The completed trade.
        """
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
            exit_time=exit_time,
            sl_price=pos.sl_price,
            tp_price=pos.tp_price,
            size=pos.size,
            pnl=pnl,
            pnl_points=pnl_points,
            is_win=pnl > 0,
            proba=pos.proba,
        )
        self._trades.append(trade)
        self._capital += pnl
        return trade

    def _open_position(
        self,
        tick: Tick,
        direction: str,
        *,
        atr: float = 0.0,
        proba: float = 0.0,
    ) -> None:
        """Open a new position."""
        cfg = self._config

        # Compute SL/TP distances (ATR-based or fixed)
        if cfg.k_sl is not None and cfg.k_tp is not None and atr > 0.0:
            sl_dist = cfg.k_sl * atr
            tp_dist = cfg.k_tp * atr
        else:
            sl_dist = cfg.sl_points
            tp_dist = cfg.tp_points

        slip = self._sample_slippage()

        if direction == "BUY":
            entry = tick.ask + slip  # adverse: higher fill
            sl = entry - sl_dist
            tp = entry + tp_dist
        else:
            entry = tick.bid - slip  # adverse: lower fill
            sl = entry + sl_dist
            tp = entry - tp_dist

        # Compute position size
        if cfg.gamma is not None:
            price = tick.ask if direction == "BUY" else tick.bid
            margin_per_lot = price * cfg.margin_pct
            available = self._capital - self.margin_used
            size = self.compute_dynamic_size(
                proba=proba,
                threshold=cfg.sizing_threshold,
                gamma=cfg.gamma,
                max_margin_proba=cfg.max_margin_proba,
                available_margin=available,
                margin_per_lot=margin_per_lot,
                min_lot_size=cfg.min_lot_size,
            )
            if size is None:
                return
            margin = size * margin_per_lot
        else:
            size = cfg.size
            margin = 0.0

        pos = MidasPosition(
            trade_id=str(uuid.uuid4())[:8],
            direction=direction,
            entry_price=entry,
            entry_time=tick.time,
            sl_price=sl,
            tp_price=tp,
            size=size,
            margin=margin,
            proba=proba,
        )
        self._positions.append(pos)

    def _check_exits(self, tick: Tick) -> list[MidasTrade]:
        """Check all open positions for SL/TP hit.

        SL is checked before TP (pessimistic assumption).
        Closes all eligible positions on the same tick.
        """
        if not self._positions:
            return []

        closed: list[MidasTrade] = []
        i = 0
        while i < len(self._positions):
            pos = self._positions[i]
            exit_price: float | None = None

            if pos.direction == "BUY":
                if tick.bid <= pos.sl_price:
                    exit_price = pos.sl_price
                elif tick.bid >= pos.tp_price:
                    exit_price = pos.tp_price
            else:
                if tick.ask >= pos.sl_price:
                    exit_price = pos.sl_price
                elif tick.ask <= pos.tp_price:
                    exit_price = pos.tp_price

            if exit_price is not None:
                self._positions.pop(i)
                closed.append(self._close_position(pos, exit_price, tick.time))
            else:
                i += 1

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
            slip = self._sample_slippage()
            exit_price = tick.bid - slip if pos.direction == "BUY" else tick.ask + slip
            closed.append(self._close_position(pos, exit_price, tick.time))

        self._positions.clear()
        return closed

    def reset(self) -> None:
        """Reset simulator for a new run."""
        self._positions.clear()
        self._trades.clear()
        self._capital = self._config.initial_capital
        self._rng = random.Random(self._config.slippage_seed)
