"""Tests for improvement loop: optimizer, analyzer, patch manager, validator."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

from src.improvement.llm_analyzer import LLMAnalyzer
from src.improvement.optuna_optimizer import OptunaOptimizer
from src.improvement.patch_manager import Patch, PatchManager
from src.improvement.trade_logger import TradeContext, TradeLogger
from src.improvement.validator import ImprovementValidator


class TestTradeLogger:
    """Tests for trade context logging."""

    def test_log_and_retrieve(self) -> None:
        tl = TradeLogger()
        ctx = TradeContext(
            trade_id="t1",
            instrument="EUR/USD",
            timeframe="M5",
            direction="LONG",
            entry_price=1.08,
            exit_price=1.085,
            stop_loss=1.077,
            take_profit=1.086,
            confluence_score=0.7,
            pnl=50.0,
            r_multiple=1.67,
            entry_time=datetime(2024, 1, 15, tzinfo=UTC),
            exit_time=datetime(2024, 1, 15, 1, tzinfo=UTC),
            setup_type="ict_confluence",
            active_fvgs=[],
            active_obs=[],
            ms_trend="bullish",
            session="london",
            killzone="london_open_kz",
        )
        tl.log_trade(ctx)
        assert len(tl.get_all_trades()) == 1
        assert tl.get_recent_trades(10)[0].trade_id == "t1"

    def test_clear(self) -> None:
        tl = TradeLogger()
        tl.log_trade(TradeContext(
            trade_id="t1", instrument="X", timeframe="M5",
            direction="LONG", entry_price=1, exit_price=2,
            stop_loss=0.5, take_profit=3, confluence_score=0.5,
            pnl=100, r_multiple=2, entry_time=datetime(2024, 1, 1, tzinfo=UTC),
            exit_time=None, setup_type="test", active_fvgs=[], active_obs=[],
            ms_trend="bullish", session="asian", killzone="none",
        ))
        tl.clear()
        assert len(tl.get_all_trades()) == 0


class TestOptunaOptimizer:
    """Tests for Optuna parameter optimization."""

    def test_optimize_accepts_improvement(self) -> None:
        def objective(params: dict[str, object]) -> float:
            return 1.5  # Always returns good Sharpe

        optimizer = OptunaOptimizer(
            objective_fn=objective,
            n_trials=5,
            min_improvement_pct=2.0,
            max_sharpe_jump_pct=200.0,
        )
        result = optimizer.optimize(
            param_space={"weight_fvg": (0.1, 0.3)},
            baseline_sharpe=1.0,
        )
        assert result.accepted
        assert result.best_sharpe == 1.5

    def test_optimize_rejects_small_improvement(self) -> None:
        def objective(params: dict[str, object]) -> float:
            return 1.01  # Barely better

        optimizer = OptunaOptimizer(
            objective_fn=objective,
            n_trials=5,
            min_improvement_pct=5.0,
        )
        result = optimizer.optimize(
            param_space={"x": (0.0, 1.0)},
            baseline_sharpe=1.0,
        )
        assert not result.accepted

    def test_optimize_rejects_overfitting(self) -> None:
        def objective(params: dict[str, object]) -> float:
            return 5.0  # Suspiciously good

        optimizer = OptunaOptimizer(
            objective_fn=objective,
            n_trials=5,
            max_sharpe_jump_pct=50.0,
        )
        result = optimizer.optimize(
            param_space={"x": (0.0, 1.0)},
            baseline_sharpe=1.0,
        )
        assert not result.accepted
        assert "overfitting" in result.reason.lower()


class TestPatchManager:
    """Tests for parameter patch management."""

    def test_apply_and_rollback(self) -> None:
        pm = PatchManager()
        pm.set_baseline({"a": 1, "b": 2})

        patch = Patch(patch_id="p1", params={"a": 10})
        new_params = pm.apply_patch(patch)
        assert new_params["a"] == 10
        assert new_params["b"] == 2

        rolled_back = pm.rollback_last()
        assert rolled_back["a"] == 1

    def test_rollback_empty(self) -> None:
        pm = PatchManager()
        pm.set_baseline({"x": 1})
        result = pm.rollback_last()
        assert result["x"] == 1

    def test_history(self) -> None:
        pm = PatchManager()
        pm.set_baseline({})
        pm.apply_patch(Patch(patch_id="p1", params={"x": 1}))
        pm.apply_patch(Patch(patch_id="p2", params={"y": 2}))
        assert len(pm.history) == 2


class TestValidator:
    """Tests for improvement validation."""

    def test_accepts_good_improvement(self) -> None:
        v = ImprovementValidator(min_improvement_pct=2.0)
        result = v.validate(
            {"sharpe_ratio": 1.0, "max_drawdown_pct": 5.0},
            {"sharpe_ratio": 1.1, "max_drawdown_pct": 5.5},
        )
        assert result.accepted

    def test_rejects_insufficient(self) -> None:
        v = ImprovementValidator(min_improvement_pct=5.0)
        result = v.validate(
            {"sharpe_ratio": 1.0, "max_drawdown_pct": 5.0},
            {"sharpe_ratio": 1.01, "max_drawdown_pct": 5.0},
        )
        assert not result.accepted

    def test_rejects_mdd_degradation(self) -> None:
        v = ImprovementValidator(max_mdd_degradation_pct=3.0)
        result = v.validate(
            {"sharpe_ratio": 1.0, "max_drawdown_pct": 5.0},
            {"sharpe_ratio": 1.5, "max_drawdown_pct": 12.0},
        )
        assert not result.accepted

    def test_rejects_overfitting(self) -> None:
        v = ImprovementValidator(max_sharpe_jump_pct=50.0)
        result = v.validate(
            {"sharpe_ratio": 1.0, "max_drawdown_pct": 5.0},
            {"sharpe_ratio": 3.0, "max_drawdown_pct": 4.0},
        )
        assert not result.accepted


class TestLLMAnalyzer:
    """Tests for LLM structural analysis."""

    def test_parse_proposals(self) -> None:
        analyzer = LLMAnalyzer(client=MagicMock())
        text = """PROPOSAL: Add session filter for Asian range trades
CATEGORY: filter
EXPECTED_IMPACT: Reduce losses during low-volatility Asian session
CONFIDENCE: 0.7
---
PROPOSAL: Increase OB displacement factor to 2.5
CATEGORY: parameter
EXPECTED_IMPACT: Fewer but higher quality order block entries
CONFIDENCE: 0.6"""
        proposals = analyzer._parse_proposals(text)
        assert len(proposals) == 2
        assert proposals[0].category == "filter"
        assert proposals[1].confidence == 0.6

    async def test_analyze_empty_trades(self) -> None:
        analyzer = LLMAnalyzer(client=AsyncMock())
        result = await analyzer.analyze_trades([], {})
        assert result == []
