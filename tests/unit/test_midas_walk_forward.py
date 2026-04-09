"""Tests for Midas walk-forward utility functions."""

from __future__ import annotations

from datetime import UTC, datetime

from src.midas.trade_simulator import MidasTrade
from src.midas.walk_forward import (
    _midas_to_common_trade,
    generate_windows,
)


class TestGenerateWindows:
    """Tests for window generation."""

    def test_basic_windows(self) -> None:
        start = datetime(2025, 1, 1, tzinfo=UTC)
        end = datetime(2025, 3, 1, tzinfo=UTC)  # 59 days

        windows = generate_windows(start, end, train_days=30, test_days=2, step_days=2)

        assert len(windows) > 0
        # First window: train 30d + test 2d = 32d from start
        t_start, t_end, test_s, test_e = windows[0]
        assert t_start == start
        assert (t_end - t_start).days == 30
        assert test_s == t_end
        assert (test_e - test_s).days == 2

    def test_step_advances(self) -> None:
        start = datetime(2025, 1, 1, tzinfo=UTC)
        end = datetime(2025, 3, 1, tzinfo=UTC)

        windows = generate_windows(start, end, train_days=30, test_days=2, step_days=5)

        if len(windows) >= 2:
            assert (windows[1][0] - windows[0][0]).days == 5

    def test_no_windows_if_range_too_short(self) -> None:
        start = datetime(2025, 1, 1, tzinfo=UTC)
        end = datetime(2025, 1, 10, tzinfo=UTC)  # 9 days

        windows = generate_windows(start, end, train_days=30, test_days=2, step_days=2)
        assert len(windows) == 0

    def test_window_count(self) -> None:
        start = datetime(2025, 1, 1, tzinfo=UTC)
        end = datetime(2025, 4, 1, tzinfo=UTC)  # 90 days

        windows = generate_windows(start, end, train_days=30, test_days=2, step_days=2)
        # ~(90 - 32) / 2 + 1 = ~30 windows
        assert len(windows) >= 25


class TestMidasToCommonTrade:
    """Tests for trade conversion."""

    def test_buy_trade_conversion(self) -> None:
        mt = MidasTrade(
            trade_id="abc",
            direction="BUY",
            entry_price=100.0,
            exit_price=103.0,
            entry_time=datetime(2025, 1, 1, 12, 0, tzinfo=UTC),
            exit_time=datetime(2025, 1, 1, 12, 5, tzinfo=UTC),
            sl_price=97.0,
            tp_price=103.0,
            size=0.1,
            pnl=3.0,
            pnl_points=3.0,
            is_win=True,
        )
        trade = _midas_to_common_trade(mt, "XAUUSD")

        assert trade.instrument == "XAUUSD"
        assert trade.direction.value == "LONG"
        assert trade.pnl == 3.0
        assert trade.r_multiple == 1.0  # 3pts / 3pts SL distance
        assert trade.is_backtest is True

    def test_sell_trade_conversion(self) -> None:
        mt = MidasTrade(
            trade_id="def",
            direction="SELL",
            entry_price=100.0,
            exit_price=98.0,
            entry_time=datetime(2025, 1, 1, 12, 0, tzinfo=UTC),
            exit_time=datetime(2025, 1, 1, 12, 5, tzinfo=UTC),
            sl_price=102.0,
            tp_price=98.0,
            size=0.1,
            pnl=2.0,
            pnl_points=2.0,
            is_win=True,
        )
        trade = _midas_to_common_trade(mt, "XAUUSD")
        assert trade.direction.value == "SHORT"
        assert trade.r_multiple == 1.0  # 2pts / 2pts
