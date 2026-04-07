"""Tests for backtest engine, metrics, simulator, walk-forward, and report."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.backtest.engine import BacktestEngine, OpenPosition
from src.backtest.metrics import compute_metrics, compute_metrics_by_source
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

    def test_avg_risk_pct(self, sample_trades: list[Trade]) -> None:
        metrics = compute_metrics(sample_trades, initial_capital=10000)
        # Trade 1: |1.0800 - 1.0770| * 1.0 = 0.003 → 0.003/10000*100 = 0.03%
        # Trade 2: |1.0850 - 1.0870| * 1.0 = 0.002 → 0.002/10050*100 ≈ 0.0199%
        # Trade 3: |1.0820 - 1.0800| * 1.0 = 0.002 → 0.002/10030*100 ≈ 0.0199%
        assert metrics.avg_risk_pct > 0
        assert metrics.avg_risk_pct < 1.0  # Very small risk on these trades


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

    def test_daily_gain_circuit_break(self) -> None:
        rm = RiskManager(max_daily_gain_pct=3.0)
        # +3% of 10000 = 300 → should trigger
        assert rm.is_circuit_broken(300, 10000, 10000)
        # +2% → should not trigger
        assert not rm.is_circuit_broken(200, 10000, 10000)

    def test_daily_gain_disabled_by_default(self) -> None:
        rm = RiskManager()
        # Large gain should not trigger when disabled (default 0)
        assert not rm.is_circuit_broken(5000, 10000, 10000)

    def test_can_open_position(self) -> None:
        rm = RiskManager(max_positions=3)
        assert rm.can_open_position(2)
        assert not rm.can_open_position(3)


class TestBreakevenStop:
    """Tests for breakeven stop-loss mechanism."""

    def test_sl_moved_to_breakeven_long(self) -> None:
        """LONG: SL moves to entry when price reaches trigger % of TP distance."""
        precomputed = precompute(SWING_FIXTURE, "EUR/USD", "M5")
        engine = BacktestEngine(
            precomputed=precomputed,
            confluence_scorer=ConfluenceScorer(),
            entry_evaluator=EntryEvaluator(),
            exit_evaluator=ExitEvaluator(),
            trade_filter=TradeFilter(),
            position_sizer=PositionSizer(),
            risk_manager=RiskManager(),
            be_trigger_pct=0.2,  # trigger at 20% of TP distance
            be_offset_pct=0.0,   # exact breakeven
        )
        # Simulate a LONG position: entry=1.1, SL=1.09, TP=1.15
        pos = OpenPosition(
            trade_id="test-be",
            instrument="EUR/USD",
            direction=Direction.LONG,
            entry_price=1.1,
            entry_time=datetime(2024, 1, 1, tzinfo=UTC),
            stop_loss=1.09,
            take_profit=1.15,
            size=1.0,
            confluence_score=0.8,
        )
        engine._open_positions = [pos]
        # TP dist = 0.05, 20% = 0.01, so high >= 1.11 triggers
        candle = {
            "time": datetime(2024, 1, 1, 0, 5, tzinfo=UTC),
            "open": 1.105, "high": 1.112, "low": 1.104, "close": 1.11,
        }
        engine._check_exits(candle, 0)
        # SL should have moved to entry (1.1)
        assert pos.stop_loss == pytest.approx(1.1)
        assert pos.trade_id in engine._be_applied

    def test_sl_moved_with_offset_short(self) -> None:
        """SHORT: SL moves to entry - offset when trigger reached."""
        precomputed = precompute(SWING_FIXTURE, "EUR/USD", "M5")
        engine = BacktestEngine(
            precomputed=precomputed,
            confluence_scorer=ConfluenceScorer(),
            entry_evaluator=EntryEvaluator(),
            exit_evaluator=ExitEvaluator(),
            trade_filter=TradeFilter(),
            position_sizer=PositionSizer(),
            risk_manager=RiskManager(),
            be_trigger_pct=0.2,
            be_offset_pct=0.05,  # lock in 5% of TP distance
        )
        # SHORT: entry=1.15, SL=1.16, TP=1.10 → dist=0.05
        pos = OpenPosition(
            trade_id="test-be-short",
            instrument="EUR/USD",
            direction=Direction.SHORT,
            entry_price=1.15,
            entry_time=datetime(2024, 1, 1, tzinfo=UTC),
            stop_loss=1.16,
            take_profit=1.10,
            size=1.0,
            confluence_score=0.8,
        )
        engine._open_positions = [pos]
        # 20% of 0.05 = 0.01, low <= 1.14 triggers
        candle = {
            "time": datetime(2024, 1, 1, 0, 5, tzinfo=UTC),
            "open": 1.145, "high": 1.148, "low": 1.138, "close": 1.14,
        }
        engine._check_exits(candle, 0)
        # New SL = entry - 5% of dist = 1.15 - 0.0025 = 1.1475
        assert pos.stop_loss == pytest.approx(1.1475)

    def test_be_not_applied_twice(self) -> None:
        """BE should only be applied once per position."""
        precomputed = precompute(SWING_FIXTURE, "EUR/USD", "M5")
        engine = BacktestEngine(
            precomputed=precomputed,
            confluence_scorer=ConfluenceScorer(),
            entry_evaluator=EntryEvaluator(),
            exit_evaluator=ExitEvaluator(),
            trade_filter=TradeFilter(),
            position_sizer=PositionSizer(),
            risk_manager=RiskManager(),
            be_trigger_pct=0.2,
            be_offset_pct=0.0,
        )
        pos = OpenPosition(
            trade_id="test-be-once",
            instrument="EUR/USD",
            direction=Direction.LONG,
            entry_price=1.1,
            entry_time=datetime(2024, 1, 1, tzinfo=UTC),
            stop_loss=1.09,
            take_profit=1.15,
            size=1.0,
            confluence_score=0.8,
        )
        engine._open_positions = [pos]
        engine._be_applied.add("test-be-once")
        # Manually set SL to something else — should NOT be overwritten
        pos.stop_loss = 1.095
        candle = {
            "time": datetime(2024, 1, 1, 0, 5, tzinfo=UTC),
            "open": 1.105, "high": 1.112, "low": 1.104, "close": 1.11,
        }
        engine._check_exits(candle, 0)
        assert pos.stop_loss == pytest.approx(1.095)  # unchanged

    def test_be_disabled_when_zero(self) -> None:
        """No BE move when be_trigger_pct=0 (default)."""
        precomputed = precompute(SWING_FIXTURE, "EUR/USD", "M5")
        engine = BacktestEngine(
            precomputed=precomputed,
            confluence_scorer=ConfluenceScorer(),
            entry_evaluator=EntryEvaluator(),
            exit_evaluator=ExitEvaluator(),
            trade_filter=TradeFilter(),
            position_sizer=PositionSizer(),
            risk_manager=RiskManager(),
            be_trigger_pct=0.0,
        )
        pos = OpenPosition(
            trade_id="test-be-off",
            instrument="EUR/USD",
            direction=Direction.LONG,
            entry_price=1.1,
            entry_time=datetime(2024, 1, 1, tzinfo=UTC),
            stop_loss=1.09,
            take_profit=1.15,
            size=1.0,
            confluence_score=0.8,
        )
        engine._open_positions = [pos]
        candle = {
            "time": datetime(2024, 1, 1, 0, 5, tzinfo=UTC),
            "open": 1.105, "high": 1.14, "low": 1.104, "close": 1.13,
        }
        engine._check_exits(candle, 0)
        assert pos.stop_loss == pytest.approx(1.09)  # unchanged


class TestMarginTracking:
    """Tests for margin tracking in BacktestEngine."""

    def _make_engine(
        self,
        initial_capital: float = 10000.0,
        leverage: float = 30.0,
        pip_size: float = 1.0,
    ) -> BacktestEngine:
        """Create a minimal engine for margin testing.

        Uses pip_size=1.0 by default so value_per_point == value_per_price_unit,
        keeping test arithmetic simple. Pass pip_size=0.0001 to test forex conversion.
        """
        precomputed = precompute(SWING_FIXTURE, "EUR/USD", "M5")
        return BacktestEngine(
            precomputed=precomputed,
            confluence_scorer=ConfluenceScorer(),
            entry_evaluator=EntryEvaluator(),
            exit_evaluator=ExitEvaluator(),
            trade_filter=TradeFilter(),
            position_sizer=PositionSizer(),
            risk_manager=RiskManager(),
            sim_config=SimulationConfig(),
            initial_capital=initial_capital,
            leverage=leverage,
            pip_size=pip_size,
        )

    def test_used_margin_empty(self) -> None:
        engine = self._make_engine()
        assert engine._compute_used_margin() == 0.0

    def test_used_margin_with_positions(self) -> None:
        engine = self._make_engine(initial_capital=10000, leverage=30)
        engine._open_positions.append(OpenPosition(
            trade_id="t1",
            instrument="EUR/USD",
            direction=Direction.LONG,
            entry_price=1.08,
            entry_time=datetime(2024, 1, 15, tzinfo=UTC),
            stop_loss=1.077,
            take_profit=1.086,
            size=10000,  # ~1 mini lot
            confluence_score=0.5,
        ))
        # margin = 1.08 * 10000 / 30 = 360
        assert engine._compute_used_margin() == pytest.approx(360.0)

    def test_equity_with_unrealized_pnl(self) -> None:
        engine = self._make_engine(initial_capital=10000, leverage=30)
        engine._open_positions.append(OpenPosition(
            trade_id="t1",
            instrument="EUR/USD",
            direction=Direction.LONG,
            entry_price=1.0800,
            entry_time=datetime(2024, 1, 15, tzinfo=UTC),
            stop_loss=1.077,
            take_profit=1.086,
            size=10000,
            confluence_score=0.5,
        ))
        # Price moved up: unrealized PnL = (1.0850 - 1.0800) * 10000 = 50
        equity = engine._compute_equity(1.0850)
        assert equity == pytest.approx(10050.0)

    def test_cap_size_fits(self) -> None:
        engine = self._make_engine(initial_capital=10000, leverage=30)
        # No positions, full capital available
        # Max size = 10000 * 30 / 1.08 = 277,777. Requested 10000 fits easily.
        capped, used, equity = engine._cap_size_to_margin(1.08, 10000, 1.08)
        assert capped == 10000
        assert used == 0.0
        assert equity == pytest.approx(10000.0)

    def test_cap_size_reduces(self) -> None:
        engine = self._make_engine(initial_capital=1000, leverage=30)
        # Max size = 1000 * 30 / 1.08 ≈ 27,778. Requested 50,000 is too much.
        capped, _, _ = engine._cap_size_to_margin(1.08, 50000, 1.08)
        assert capped < 50000
        assert capped == pytest.approx(1000 * 30 / 1.08)

    def test_cap_size_considers_existing_positions(self) -> None:
        engine = self._make_engine(initial_capital=1000, leverage=30)
        # First position uses 1.08 * 20000 / 30 = 720 of margin
        engine._open_positions.append(OpenPosition(
            trade_id="t1",
            instrument="EUR/USD",
            direction=Direction.LONG,
            entry_price=1.08,
            entry_time=datetime(2024, 1, 15, tzinfo=UTC),
            stop_loss=1.077,
            take_profit=1.086,
            size=20000,
            confluence_score=0.5,
        ))
        # Available margin = 1000 - 720 = 280
        # Max size for new = 280 * 30 / 1.08 ≈ 7,778
        capped, _, _ = engine._cap_size_to_margin(1.08, 50000, 1.08)
        assert capped < 50000
        assert capped == pytest.approx(280 * 30 / 1.08)

    def test_cap_size_to_remaining_margin(self) -> None:
        engine = self._make_engine(initial_capital=100, leverage=30)
        # Position uses nearly all margin: 1.08 * 2700 / 30 = 97.2
        engine._open_positions.append(OpenPosition(
            trade_id="t1",
            instrument="EUR/USD",
            direction=Direction.LONG,
            entry_price=1.08,
            entry_time=datetime(2024, 1, 15, tzinfo=UTC),
            stop_loss=1.077,
            take_profit=1.086,
            size=2700,
            confluence_score=0.5,
        ))
        # Available margin ≈ 100 - 97.2 = 2.8
        # Max size = 2.8 * 30 / 1.08 ≈ 77.8
        capped, _, _ = engine._cap_size_to_margin(1.08, 50000, 1.08)
        assert capped == pytest.approx(2.8 * 30 / 1.08, rel=0.01)

    def test_margin_counters_tracked(self) -> None:
        """Engine tracks margin_rejected and margin_capped counters."""
        precomputed = precompute(SWING_FIXTURE, "EUR/USD", "M5")
        engine = BacktestEngine(
            precomputed=precomputed,
            confluence_scorer=ConfluenceScorer(),
            entry_evaluator=EntryEvaluator(min_confluence=0.01),
            exit_evaluator=ExitEvaluator(),
            trade_filter=TradeFilter(require_killzone=False, max_positions=100),
            position_sizer=PositionSizer(),
            risk_manager=RiskManager(max_positions=100),
            sim_config=SimulationConfig(order_rejection_rate=0),
            initial_capital=1.0,  # Extremely low capital
            leverage=1.0,  # 1:1 leverage = max restrictive
        )
        result = engine.run()
        assert result.margin_rejected >= 0
        assert result.margin_capped >= 0
        assert result.peak_margin_usage_pct >= 0


class TestComputeMetricsBySource:
    """Tests for compute_metrics_by_source."""

    def _make_trade(
        self, pnl: float, trigger_source: str = "ict"
    ) -> Trade:
        return Trade(
            opened_at=datetime(2024, 1, 15, 10, 0, tzinfo=UTC),
            closed_at=datetime(2024, 1, 15, 11, 0, tzinfo=UTC),
            instrument="EUR/USD",
            direction=Direction.LONG,
            entry_price=1.08,
            exit_price=1.08 + pnl / 100,
            stop_loss=1.07,
            take_profit=1.09,
            size=1.0,
            pnl=pnl,
            pnl_percent=pnl,
            r_multiple=pnl / 10 if pnl != 0 else 0,
            setup_type={"trigger_source": trigger_source},
            is_backtest=True,
        )

    def test_all_ict_trades(self) -> None:
        trades = [self._make_trade(100), self._make_trade(-50)]
        result = compute_metrics_by_source(trades)
        assert result.ict_trade_count == 2
        assert result.news_trade_count == 0
        assert result.ict.total_pnl == pytest.approx(50, abs=1)

    def test_all_news_trades(self) -> None:
        trades = [self._make_trade(200, "news")]
        result = compute_metrics_by_source(trades)
        assert result.ict_trade_count == 0
        assert result.news_trade_count == 1
        assert result.news.total_pnl == pytest.approx(200, abs=1)

    def test_mixed_trades(self) -> None:
        trades = [
            self._make_trade(100, "ict"),
            self._make_trade(200, "news"),
            self._make_trade(-30, "ict"),
        ]
        result = compute_metrics_by_source(trades)
        assert result.ict_trade_count == 2
        assert result.news_trade_count == 1

    def test_default_to_ict_when_no_setup_type(self) -> None:
        trade = self._make_trade(50)
        trade.setup_type = None
        result = compute_metrics_by_source([trade])
        assert result.ict_trade_count == 1
        assert result.news_trade_count == 0

    def test_empty_trades(self) -> None:
        result = compute_metrics_by_source([])
        assert result.ict_trade_count == 0
        assert result.news_trade_count == 0
