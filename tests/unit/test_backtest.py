"""Tests for backtest engine, metrics, simulator, walk-forward, and report."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.backtest.engine import OpenPosition
from src.backtest.metrics import compute_metrics
from src.backtest.report import format_report, generate_report
from src.backtest.simulator import SimulationConfig, compute_swap_cost, simulate_fill
from src.backtest.vectorized import PrecomputedData, precompute
from src.backtest.walk_forward import (
    aggregate_walk_forward,
    generate_windows,
    split_trades_by_time,
)
from src.common.models import Direction, Trade
from src.execution.position_sizer import PositionSizer
from src.execution.risk_manager import RiskManager
from src.strategy.confluence import ConfluenceScorer
from src.strategy.entry import EntryEvaluator
from src.strategy.exit import ExitEvaluator
from src.strategy.filters import TradeFilter
from tests.fixtures.annotated_candles import OB_FIXTURE, SWING_FIXTURE


class TestSimulator:
    """Tests for execution simulator."""

    def test_simulate_fill_success(self) -> None:
        config = SimulationConfig(slippage_max_pips=0, order_rejection_rate=0)
        result = simulate_fill(1.0800, is_buy=True, spread=0.0008, config=config)
        assert result.filled
        assert result.fill_price >= 1.0800  # Buy fills at or above target

    def test_simulate_fill_rejection(self) -> None:
        config = SimulationConfig(order_rejection_rate=1.0)  # Always reject
        result = simulate_fill(1.0800, is_buy=True, spread=0.0008, config=config)
        assert not result.filled

    def test_swap_cost(self) -> None:
        config = SimulationConfig(swap_long_per_day=-0.5)
        cost = compute_swap_cost(1.0, is_long=True, days_held=1.0, config=config)
        assert cost < 0  # Swap is a cost


class TestMetrics:
    """Tests for performance metrics."""

    @pytest.fixture
    def sample_trades(self) -> list[Trade]:
        base = datetime(2024, 1, 15, 10, 0, tzinfo=UTC)
        return [
            Trade(
                opened_at=base,
                closed_at=base + timedelta(hours=1),
                instrument="EUR/USD",
                direction=Direction.LONG,
                entry_price=1.0800,
                exit_price=1.0850,
                stop_loss=1.0770,
                take_profit=1.0860,
                size=1.0,
                pnl=50.0,
                pnl_percent=0.46,
                r_multiple=1.67,
                is_backtest=True,
            ),
            Trade(
                opened_at=base + timedelta(hours=2),
                closed_at=base + timedelta(hours=3),
                instrument="EUR/USD",
                direction=Direction.SHORT,
                entry_price=1.0850,
                exit_price=1.0870,
                stop_loss=1.0870,
                take_profit=1.0810,
                size=1.0,
                pnl=-20.0,
                pnl_percent=-0.18,
                r_multiple=-1.0,
                is_backtest=True,
            ),
            Trade(
                opened_at=base + timedelta(hours=4),
                closed_at=base + timedelta(hours=5),
                instrument="EUR/USD",
                direction=Direction.LONG,
                entry_price=1.0820,
                exit_price=1.0860,
                stop_loss=1.0800,
                take_profit=1.0860,
                size=1.0,
                pnl=40.0,
                pnl_percent=0.37,
                r_multiple=2.0,
                is_backtest=True,
            ),
        ]

    def test_compute_metrics(self, sample_trades: list[Trade]) -> None:
        metrics = compute_metrics(sample_trades)
        assert metrics.total_trades == 3
        assert metrics.winning_trades == 2
        assert metrics.losing_trades == 1
        assert metrics.win_rate == pytest.approx(2 / 3)
        assert metrics.total_pnl == pytest.approx(70.0)

    def test_profit_factor(self, sample_trades: list[Trade]) -> None:
        metrics = compute_metrics(sample_trades)
        # gross_profit=90, gross_loss=20 → PF=4.5
        assert metrics.profit_factor == pytest.approx(4.5)

    def test_empty_trades(self) -> None:
        metrics = compute_metrics([])
        assert metrics.total_trades == 0
        assert metrics.sharpe_ratio == 0.0

    def test_max_drawdown(self, sample_trades: list[Trade]) -> None:
        metrics = compute_metrics(sample_trades, initial_capital=10000)
        assert metrics.max_drawdown >= 0


class TestWalkForward:
    """Tests for walk-forward validation."""

    def test_generate_windows(self) -> None:
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 12, 31, tzinfo=UTC)
        windows = generate_windows(start, end, train_months=4, test_months=1)
        assert len(windows) > 0
        # Each window has 4 components
        assert len(windows[0]) == 4

    def test_split_trades_by_time(self) -> None:
        base = datetime(2024, 1, 15, tzinfo=UTC)
        trades = [
            Trade(opened_at=base, instrument="X", direction=Direction.LONG),
            Trade(
                opened_at=base + timedelta(days=60),
                instrument="X",
                direction=Direction.LONG,
            ),
        ]
        result = split_trades_by_time(trades, base, base + timedelta(days=30))
        assert len(result) == 1

    def test_aggregate_empty(self) -> None:
        result = aggregate_walk_forward([])
        assert result.total_test_trades == 0


class TestPrecompute:
    """Tests for vectorized pre-computation."""

    def test_precompute_returns_all_data(self) -> None:
        result = precompute(SWING_FIXTURE, "EUR/USD", "M5")
        assert isinstance(result, PrecomputedData)
        assert result.instrument == "EUR/USD"
        assert not result.candles.is_empty()
        assert "session" in result.candles.columns

    def test_precompute_with_ob_fixture(self) -> None:
        result = precompute(OB_FIXTURE, "EUR/USD", "M5")
        assert not result.displacements.is_empty()


class TestReport:
    """Tests for backtest report generation."""

    def test_generate_report(self) -> None:
        base = datetime(2024, 1, 15, 10, 0, tzinfo=UTC)
        trades = [
            Trade(
                opened_at=base,
                closed_at=base + timedelta(hours=1),
                instrument="EUR/USD",
                direction=Direction.LONG,
                pnl=100.0,
                r_multiple=2.0,
                is_backtest=True,
            ),
        ]
        report = generate_report(trades, initial_capital=10000)
        assert report["summary"]["total_trades"] == 1
        assert report["summary"]["total_pnl"] == 100.0
        assert report["summary"]["return_pct"] == 1.0

    def test_format_report(self) -> None:
        report = generate_report([], initial_capital=10000)
        formatted = format_report(report)
        assert "BACKTEST REPORT" in formatted


class TestStrategy:
    """Tests for strategy components."""

    def test_confluence_scorer(self) -> None:
        scorer = ConfluenceScorer()
        context: dict[str, object] = {"fvgs": [{"x": 1}], "in_killzone": True}
        score = scorer.score({}, context)
        assert score > 0

    def test_confluence_empty_context(self) -> None:
        scorer = ConfluenceScorer()
        score = scorer.score({}, {})
        assert score == 0.0

    def test_entry_evaluator_no_breaks(self) -> None:
        evaluator = EntryEvaluator()
        candle = {"close": 1.08, "high": 1.082, "low": 1.078}
        result = evaluator.evaluate(candle, {}, 0.5)
        assert result is None

    def test_entry_evaluator_with_break(self) -> None:
        evaluator = EntryEvaluator(min_confluence=0.3)
        candle = {"close": 1.08, "high": 1.082, "low": 1.078}
        context = {"ms_breaks": [{"direction": "bullish"}]}
        result = evaluator.evaluate(candle, context, 0.5)
        assert result is not None
        assert result.direction == Direction.LONG

    def test_exit_evaluator_max_hold(self) -> None:
        evaluator = ExitEvaluator(max_hold_candles=2)
        pos = OpenPosition(
            trade_id="test",
            instrument="EUR/USD",
            direction=Direction.LONG,
            entry_price=1.08,
            entry_time=datetime(2024, 1, 15, tzinfo=UTC),
            stop_loss=1.077,
            take_profit=1.086,
            size=1.0,
            confluence_score=0.5,
        )
        candle = {"close": 1.081, "high": 1.082, "low": 1.080}
        # First candle: no exit
        assert evaluator.evaluate(pos, candle) is None
        # Second candle: max hold
        assert evaluator.evaluate(pos, candle) is not None

    def test_trade_filter_max_positions(self) -> None:
        f = TradeFilter(max_positions=2)
        assert f.passes({}, {}, 1)
        assert not f.passes({}, {}, 2)

    def test_trade_filter_killzone(self) -> None:
        f = TradeFilter(require_killzone=True)
        assert not f.passes({}, {"in_killzone": False}, 0)
        assert f.passes({}, {"in_killzone": True}, 0)


class TestPositionSizer:
    """Tests for dynamic position sizing."""

    def test_high_confluence_bigger_size(self) -> None:
        sizer = PositionSizer()
        small = sizer.compute_size(10000, 0.3, 1.08, 1.077, value_per_point=1.0, min_size=0.5)
        large = sizer.compute_size(10000, 0.8, 1.08, 1.077, value_per_point=1.0, min_size=0.5)
        assert large > small

    def test_zero_sl_distance(self) -> None:
        sizer = PositionSizer()
        assert sizer.compute_size(10000, 0.5, 1.08, 1.08) == 0.0

    def test_respects_min_size(self) -> None:
        sizer = PositionSizer()
        # Even if raw calc gives 0.3, should return min_size=0.5
        size = sizer.compute_size(5000, 0.5, 23000, 22960, value_per_point=1.0, min_size=0.5)
        assert size >= 0.5

    def test_returns_zero_if_min_size_exceeds_budget(self) -> None:
        sizer = PositionSizer()
        # Capital=100, risk=0.5%=0.50€, SL=40pts, vpp=1€ → min_size risk=0.5*40=20€ > 0.50€
        size = sizer.compute_size(100, 0.3, 23000, 22960, value_per_point=1.0, min_size=0.5)
        assert size == 0.0

    def test_rounds_down_to_step(self) -> None:
        sizer = PositionSizer()
        # 5000€ capital, 1% risk=50€, SL=40pts, vpp=1€ → 50/40=1.25 → floor to 1.0
        size = sizer.compute_size(
            5000, 0.5, 23000, 22960, value_per_point=1.0, min_size=0.5, size_step=0.5
        )
        assert size % 0.5 == 0.0

    def test_dax_e1_realistic(self) -> None:
        """DAX €1/point, 5000€ capital, 1% risk, 40pt SL."""
        sizer = PositionSizer()
        size = sizer.compute_size(5000, 0.5, 23000, 22960, value_per_point=1.0, min_size=0.5)
        # Risk budget: 50€, risk per contract: 40*1=40€, raw=1.25, step=1.0
        assert size == 1.0


class TestRiskManager:
    """Tests for circuit breakers."""

    def test_no_circuit_break(self) -> None:
        rm = RiskManager()
        assert not rm.is_circuit_broken(0, 10000, 10000)

    def test_daily_dd_circuit_break(self) -> None:
        rm = RiskManager(max_daily_drawdown_pct=3.0)
        assert rm.is_circuit_broken(-350, 10000, 10000)

    def test_total_dd_circuit_break(self) -> None:
        rm = RiskManager(max_total_drawdown_pct=10.0)
        assert rm.is_circuit_broken(0, 8900, 10000)

    def test_can_open_position(self) -> None:
        rm = RiskManager(max_positions=3)
        assert rm.can_open_position(2)
        assert not rm.can_open_position(3)
