"""Tests for Midas TradeSimulator."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.midas.trade_simulator import SimConfig, TradeSimulator
from src.midas.types import Tick


def _tick(seconds: float, bid: float, ask: float) -> Tick:
    base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    return Tick(time=base + timedelta(seconds=seconds), bid=bid, ask=ask)


class TestTradeSimulator:
    """Tests for TradeSimulator."""

    def test_buy_entry_at_ask(self) -> None:
        sim = TradeSimulator(SimConfig(sl_points=2.0, tp_points=2.0))
        tick = _tick(0, bid=100.0, ask=101.0)
        sim.on_signal(tick, signal=1)  # BUY

        assert sim.open_count == 1
        pos = sim._positions[0]
        assert pos.entry_price == 101.0  # entered at ask
        assert pos.sl_price == 99.0  # ask - SL
        assert pos.tp_price == 103.0  # ask + TP

    def test_sell_entry_at_bid(self) -> None:
        sim = TradeSimulator(SimConfig(sl_points=2.0, tp_points=2.0))
        tick = _tick(0, bid=100.0, ask=101.0)
        sim.on_signal(tick, signal=2)  # SELL

        pos = sim._positions[0]
        assert pos.entry_price == 100.0  # entered at bid
        assert pos.sl_price == 102.0  # bid + SL
        assert pos.tp_price == 98.0  # bid - TP

    def test_buy_tp_hit(self) -> None:
        sim = TradeSimulator(SimConfig(
            sl_points=2.0, tp_points=2.0,
            size=0.1, value_per_point=10.0,
        ))
        sim.on_signal(_tick(0, bid=100.0, ask=101.0), signal=1)

        # TP at 103, bid reaches 103
        trade = sim.on_tick(_tick(1, bid=103.0, ask=104.0))

        assert trade is not None
        assert trade.is_win is True
        assert trade.pnl_points == pytest.approx(2.0)  # 103 - 101
        assert trade.pnl == pytest.approx(2.0)  # 2pts * 0.1 * 10

    def test_buy_sl_hit(self) -> None:
        sim = TradeSimulator(SimConfig(
            sl_points=2.0, tp_points=2.0,
            size=0.1, value_per_point=10.0,
        ))
        sim.on_signal(_tick(0, bid=100.0, ask=101.0), signal=1)

        # SL at 99, bid drops to 99
        trade = sim.on_tick(_tick(1, bid=99.0, ask=100.0))

        assert trade is not None
        assert trade.is_win is False
        assert trade.pnl_points == pytest.approx(-2.0)  # 99 - 101
        assert trade.pnl == pytest.approx(-2.0)

    def test_sell_tp_hit(self) -> None:
        sim = TradeSimulator(SimConfig(
            sl_points=2.0, tp_points=2.0,
            size=0.1, value_per_point=10.0,
        ))
        sim.on_signal(_tick(0, bid=100.0, ask=101.0), signal=2)

        # SELL TP at 98, ask drops to 98
        trade = sim.on_tick(_tick(1, bid=97.0, ask=98.0))

        assert trade is not None
        assert trade.is_win is True
        assert trade.pnl_points == pytest.approx(2.0)  # 100 - 98

    def test_max_positions_enforced(self) -> None:
        sim = TradeSimulator(SimConfig(max_open_positions=1))
        sim.on_signal(_tick(0, bid=100.0, ask=101.0), signal=1)
        sim.on_signal(_tick(1, bid=100.0, ask=101.0), signal=2)

        assert sim.open_count == 1  # second entry rejected

    def test_pass_signal_no_entry(self) -> None:
        sim = TradeSimulator()
        sim.on_signal(_tick(0, bid=100.0, ask=101.0), signal=0)
        assert sim.open_count == 0

    def test_spread_filter(self) -> None:
        sim = TradeSimulator(SimConfig(max_spread=1.0))
        # Spread = 2.0, exceeds max
        sim.on_signal(_tick(0, bid=100.0, ask=102.0), signal=1)
        assert sim.open_count == 0

    def test_close_all(self) -> None:
        sim = TradeSimulator(SimConfig(
            max_open_positions=3,
            sl_points=10.0, tp_points=10.0,
        ))
        sim.on_signal(_tick(0, bid=100.0, ask=101.0), signal=1)

        closed = sim.close_all(_tick(5, bid=102.0, ask=103.0))
        assert len(closed) == 1
        assert closed[0].exit_price == 102.0  # BUY exits at bid
        assert sim.open_count == 0

    def test_capital_tracking(self) -> None:
        sim = TradeSimulator(SimConfig(
            initial_capital=10000.0,
            sl_points=2.0, tp_points=2.0,
            size=1.0, value_per_point=10.0,
        ))
        sim.on_signal(_tick(0, bid=100.0, ask=101.0), signal=1)
        sim.on_tick(_tick(1, bid=103.0, ask=104.0))  # TP hit, +20

        assert sim.capital == pytest.approx(10020.0)

    def test_reset(self) -> None:
        sim = TradeSimulator(SimConfig(initial_capital=5000.0))
        sim.on_signal(_tick(0, bid=100.0, ask=101.0), signal=1)
        sim.reset()
        assert sim.open_count == 0
        assert len(sim.closed_trades) == 0
        assert sim.capital == 5000.0
