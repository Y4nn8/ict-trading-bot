"""Tests for Midas walk-forward utility functions."""

from __future__ import annotations

from datetime import UTC, datetime

from src.midas.optimizer import OptimizationResult
from src.midas.trade_simulator import MidasTrade
from src.midas.walk_forward import (
    WalkForwardOptunaConfig,
    _midas_to_common_trade,
    _print_param_stability,
    _snap_to_monday,
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


class TestWalkForwardOptunaConfig:
    """Tests for WalkForwardOptunaConfig defaults."""

    def test_defaults(self) -> None:
        cfg = WalkForwardOptunaConfig()
        assert cfg.instrument == "XAUUSD"
        assert cfg.train_days == 30
        assert cfg.test_days == 7
        assert cfg.step_days == 7
        assert cfg.outer_trials == 10
        assert cfg.inner_trials == 20
        assert cfg.sample_on_candle is True

    def test_custom_values(self) -> None:
        cfg = WalkForwardOptunaConfig(
            train_days=14,
            test_days=3,
            outer_trials=5,
            inner_trials=10,
            slippage_min_pts=0.1,
            slippage_max_pts=0.5,
        )
        assert cfg.train_days == 14
        assert cfg.test_days == 3
        assert cfg.slippage_min_pts == 0.1


class TestParamStability:
    """Tests for param stability analysis."""

    def test_stable_params_detected(self, capsys: object) -> None:
        """Params with low CV should be marked stable."""
        results = [
            OptimizationResult(
                best_inner_params={"k_sl": 1.50, "k_tp": 2.00},
                best_outer_params={},
            ),
            OptimizationResult(
                best_inner_params={"k_sl": 1.52, "k_tp": 1.10},
                best_outer_params={},
            ),
            OptimizationResult(
                best_inner_params={"k_sl": 1.48, "k_tp": 2.80},
                best_outer_params={},
            ),
        ]

        _print_param_stability(results)
        captured = capsys.readouterr()  # type: ignore[union-attr]

        # k_sl has low CV (~1.3%), should be stable
        assert "YES" in captured.out
        assert "Converged:" in captured.out

    def test_skips_single_window(self, capsys: object) -> None:
        """Should not print anything with fewer than 2 windows."""
        results = [
            OptimizationResult(
                best_inner_params={"k_sl": 1.5},
                best_outer_params={},
            ),
        ]
        _print_param_stability(results)
        captured = capsys.readouterr()  # type: ignore[union-attr]
        assert captured.out == ""

    def test_includes_outer_params(self, capsys: object) -> None:
        """Outer params should also appear in stability analysis."""
        results = [
            OptimizationResult(
                best_inner_params={"k_sl": 1.5},
                best_outer_params={"atr_period": 14},
            ),
            OptimizationResult(
                best_inner_params={"k_sl": 1.6},
                best_outer_params={"atr_period": 14},
            ),
        ]
        _print_param_stability(results)
        captured = capsys.readouterr()  # type: ignore[union-attr]
        assert "outer__atr_period" in captured.out


class TestSnapToMonday:
    """Tests for Monday alignment."""

    def test_monday_stays(self) -> None:
        # 2026-04-06 is a Monday
        dt = datetime(2026, 4, 6, tzinfo=UTC)
        assert _snap_to_monday(dt).weekday() == 0
        assert _snap_to_monday(dt) == dt

    def test_wednesday_snaps_forward(self) -> None:
        # 2026-04-08 is Wednesday → next Monday is 2026-04-13
        dt = datetime(2026, 4, 8, tzinfo=UTC)
        snapped = _snap_to_monday(dt)
        assert snapped.weekday() == 0
        assert snapped == datetime(2026, 4, 13, tzinfo=UTC)

    def test_sunday_snaps_forward(self) -> None:
        # 2026-04-12 is Sunday → next Monday is 2026-04-13
        dt = datetime(2026, 4, 12, tzinfo=UTC)
        snapped = _snap_to_monday(dt)
        assert snapped == datetime(2026, 4, 13, tzinfo=UTC)


class TestAlignMonday:
    """Tests for generate_windows with align_monday."""

    def test_aligned_windows_start_monday(self) -> None:
        # 2025-10-01 is a Wednesday
        start = datetime(2025, 10, 1, tzinfo=UTC)
        end = datetime(2026, 4, 1, tzinfo=UTC)

        windows = generate_windows(
            start, end, train_days=28, test_days=7, step_days=7,
            align_monday=True,
        )
        assert len(windows) > 0
        # First window should start on Monday (2025-10-06)
        assert windows[0][0].weekday() == 0
        assert windows[0][0] == datetime(2025, 10, 6, tzinfo=UTC)

    def test_non_aligned_preserves_start(self) -> None:
        # Without align_monday, start stays as-is
        start = datetime(2025, 10, 1, tzinfo=UTC)  # Wednesday
        end = datetime(2026, 4, 1, tzinfo=UTC)

        windows = generate_windows(
            start, end, train_days=28, test_days=7, step_days=7,
            align_monday=False,
        )
        assert windows[0][0] == start


class TestWFOptunaConfigNewFields:
    """Tests for new WalkForwardOptunaConfig fields."""

    def test_new_defaults(self) -> None:
        cfg = WalkForwardOptunaConfig()
        assert cfg.split_oos is False
        assert cfg.fixed_inner_params is None
        assert cfg.align_monday is False
        assert cfg.min_daily_trades == 10
        assert cfg.trade_deficit_penalty == 10.0
