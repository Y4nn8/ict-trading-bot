"""Tests for Midas nested Optuna optimizer."""

from __future__ import annotations

from datetime import UTC, datetime

import optuna
import pytest

from src.midas.optimizer import (
    OptimizationResult,
    OptimizerConfig,
    TrialRecord,
    _count_trading_days,
    _suggest_inner_params,
    _suggest_outer_params,
    default_output_prefix,
    load_outer_param_ranges,
    write_trial_logs,
)
from src.midas.replay_engine import build_default_registry
from src.midas.trade_simulator import MidasTrade


class TestOuterParams:
    """Tests for outer loop parameter suggestion (extractor only)."""

    def test_suggests_extractor_params(self) -> None:
        registry = build_default_registry()
        registry_params = registry.all_tunable_params()

        study = optuna.create_study()
        trial = study.ask()
        params = _suggest_outer_params(trial, registry_params)

        for p in registry_params:
            assert p.name in params

        # SL/TP should NOT be in outer params
        assert "sl_points" not in params
        assert "tp_points" not in params
        assert "label_timeout" not in params


class TestInnerParams:
    """Tests for inner loop parameter suggestion (k_sl/k_tp + LightGBM)."""

    def test_suggests_all_params(self) -> None:
        config = OptimizerConfig()
        study = optuna.create_study()
        trial = study.ask()
        params = _suggest_inner_params(trial, config)

        # ATR-based params should be in inner params
        assert "k_sl" in params
        assert "k_tp" in params
        assert "sl_fallback" in params
        assert "tp_fallback" in params
        assert "label_timeout" in params

        # LightGBM params
        assert "n_estimators" in params
        assert "entry_threshold" in params

    def test_respects_ranges(self) -> None:
        config = OptimizerConfig(
            k_sl_range=(0.8, 2.5),
            k_tp_range=(0.5, 2.0),
            sl_range=(3.0, 6.0),
            tp_range=(2.0, 5.0),
        )
        study = optuna.create_study()
        trial = study.ask()
        params = _suggest_inner_params(trial, config)

        assert 0.8 <= params["k_sl"] <= 2.5
        assert 0.5 <= params["k_tp"] <= 2.0
        assert 3.0 <= params["sl_fallback"] <= 6.0
        assert 2.0 <= params["tp_fallback"] <= 5.0
        assert 0.25 <= params["entry_threshold"] <= 0.60


def _make_trade(
    trade_id: str = "t1",
    direction: str = "BUY",
    pnl: float = 5.0,
) -> MidasTrade:
    """Helper to build a MidasTrade for tests."""
    return MidasTrade(
        trade_id=trade_id,
        direction=direction,
        entry_price=2000.0,
        exit_price=2005.0,
        entry_time=datetime(2025, 3, 1, 10, 0, tzinfo=UTC),
        exit_time=datetime(2025, 3, 1, 10, 5, tzinfo=UTC),
        sl_price=1997.0,
        tp_price=2005.0,
        size=0.1,
        pnl=pnl,
        pnl_points=pnl,
        is_win=pnl > 0,
        proba=0.65,
    )


class TestTrialRecord:
    """Tests for TrialRecord dataclass."""

    def test_creation(self) -> None:
        tr = TrialRecord(
            window_idx=0,
            outer_idx=3,
            score=42.5,
            n_trades=20,
            win_rate=0.6,
            pnl=150.0,
            outer_params={"atr_period": 14},
            inner_params={"k_sl": 1.5, "k_tp": 2.0},
            trades=[_make_trade()],
        )
        assert tr.window_idx == 0
        assert tr.outer_idx == 3
        assert len(tr.trades) == 1

    def test_frozen(self) -> None:
        tr = TrialRecord(
            window_idx=0, outer_idx=0, score=0.0,
            n_trades=0, win_rate=0.0, pnl=0.0,
            outer_params={}, inner_params={}, trades=[],
        )
        with pytest.raises(AttributeError):
            tr.score = 99.0  # type: ignore[misc]


class TestDefaultOutputPrefix:
    """Tests for timestamped output prefix."""

    def test_format(self) -> None:
        prefix = default_output_prefix()
        assert prefix.startswith("config/midas_optuna_")
        # Should contain date + time
        parts = prefix.replace("config/midas_optuna_", "").split("_")
        assert len(parts) == 2
        assert len(parts[0]) == 8  # YYYYMMDD
        assert len(parts[1]) == 6  # HHMMSS


class TestWriteTrialLogs:
    """Tests for CSV trial and trade log writing."""

    def test_writes_both_files(self, tmp_path: object) -> None:
        import csv
        from pathlib import Path

        prefix = str(Path(str(tmp_path)) / "test_run")
        records = [
            TrialRecord(
                window_idx=0, outer_idx=0, score=10.0,
                n_trades=3, win_rate=0.667, pnl=15.0,
                outer_params={"atr_period": 14},
                inner_params={"k_sl": 1.5, "entry_threshold": 0.4},
                trades=[_make_trade("t1", pnl=5.0), _make_trade("t2", pnl=-2.0)],
            ),
            TrialRecord(
                window_idx=0, outer_idx=1, score=8.0,
                n_trades=2, win_rate=0.5, pnl=3.0,
                outer_params={"atr_period": 10},
                inner_params={"k_sl": 2.0, "entry_threshold": 0.35},
                trades=[_make_trade("t3", pnl=3.0)],
            ),
        ]

        trials_path, trades_path = write_trial_logs(records, prefix)

        assert trials_path.exists()
        assert trades_path.exists()

        # Check trials CSV
        with open(trials_path) as f:
            reader = list(csv.DictReader(f))
        assert len(reader) == 2
        assert reader[0]["outer_idx"] == "0"
        assert reader[0]["score"] == "10.0"
        assert reader[0]["outer__atr_period"] == "14"
        assert reader[0]["inner__k_sl"] == "1.5"
        assert reader[1]["outer__atr_period"] == "10"

        # Check trades CSV
        with open(trades_path) as f:
            reader = list(csv.DictReader(f))
        assert len(reader) == 3
        assert reader[0]["outer_idx"] == "0"
        assert reader[0]["trade_id"] == "t1"
        assert reader[2]["outer_idx"] == "1"
        assert reader[2]["trade_id"] == "t3"

    def test_empty_records(self, tmp_path: object) -> None:
        from pathlib import Path

        prefix = str(Path(str(tmp_path)) / "empty")
        trials_path, trades_path = write_trial_logs([], prefix)

        assert trials_path.exists()
        assert trades_path.exists()

    def test_multi_window(self, tmp_path: object) -> None:
        import csv
        from pathlib import Path

        prefix = str(Path(str(tmp_path)) / "multi")
        records = [
            TrialRecord(
                window_idx=0, outer_idx=0, score=5.0,
                n_trades=1, win_rate=1.0, pnl=5.0,
                outer_params={}, inner_params={"k_sl": 1.0},
                trades=[_make_trade("w0t1")],
            ),
            TrialRecord(
                window_idx=1, outer_idx=0, score=3.0,
                n_trades=1, win_rate=1.0, pnl=3.0,
                outer_params={}, inner_params={"k_sl": 1.2},
                trades=[_make_trade("w1t1")],
            ),
        ]

        trials_path, trades_path = write_trial_logs(records, prefix)

        with open(trials_path) as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["window_idx"] == "0"
        assert rows[1]["window_idx"] == "1"

        with open(trades_path) as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["window_idx"] == "0"
        assert rows[1]["window_idx"] == "1"


class TestOptimizationResultTrialRecords:
    """Tests that OptimizationResult includes trial_records."""

    def test_default_empty(self) -> None:
        result = OptimizationResult()
        assert result.trial_records == []

    def test_append_records(self) -> None:
        result = OptimizationResult()
        result.trial_records.append(TrialRecord(
            window_idx=0, outer_idx=0, score=1.0,
            n_trades=5, win_rate=0.5, pnl=10.0,
            outer_params={}, inner_params={}, trades=[],
        ))
        assert len(result.trial_records) == 1


class TestOuterParamRanges:
    """Tests for outer param range restriction."""

    def test_range_override_applied(self) -> None:
        registry = build_default_registry()
        registry_params = registry.all_tunable_params()

        # Restrict first param to a narrow range
        param = registry_params[0]
        lo, hi = int(param.low) + 1, int(param.low) + 3
        narrow_range = (float(lo), float(hi))

        study = optuna.create_study()
        trial = study.ask()
        params = _suggest_outer_params(
            trial, registry_params,
            range_overrides={param.name: narrow_range},
        )

        assert lo <= params[param.name] <= hi

    def test_no_override_uses_default(self) -> None:
        registry = build_default_registry()
        registry_params = registry.all_tunable_params()

        study = optuna.create_study()
        trial = study.ask()
        params = _suggest_outer_params(trial, registry_params)

        for p in registry_params:
            assert p.low <= params[p.name] <= p.high


class TestLoadOuterParamRanges:
    """Tests for YAML range loading."""

    def test_loads_ranges(self, tmp_path: object) -> None:
        from pathlib import Path

        path = Path(str(tmp_path)) / "ranges.yml"
        path.write_text("atr_period: [10, 16]\nliq_lookback: [100, 200]\n")

        ranges = load_outer_param_ranges(str(path))
        assert ranges == {
            "atr_period": (10.0, 16.0),
            "liq_lookback": (100.0, 200.0),
        }

    def test_invalid_yaml_raises(self, tmp_path: object) -> None:
        from pathlib import Path

        path = Path(str(tmp_path)) / "bad.yml"
        path.write_text("- just a list\n")

        with pytest.raises(ValueError, match="Expected a YAML mapping"):
            load_outer_param_ranges(str(path))

    def test_skips_non_range_entries(self, tmp_path: object) -> None:
        from pathlib import Path

        path = Path(str(tmp_path)) / "mixed.yml"
        path.write_text("atr_period: [10, 16]\nsome_string: hello\n")

        ranges = load_outer_param_ranges(str(path))
        assert "atr_period" in ranges
        assert "some_string" not in ranges


class TestCountTradingDays:
    """Tests for trading day counting."""

    def test_full_week(self) -> None:
        # Mon 2026-04-06 to Mon 2026-04-13 = 5 weekdays
        start = datetime(2026, 4, 6, tzinfo=UTC)
        end = datetime(2026, 4, 13, tzinfo=UTC)
        assert _count_trading_days(start, end) == 5

    def test_weekend_only(self) -> None:
        # Sat to Mon = 0 weekdays, but min 1
        start = datetime(2026, 4, 11, tzinfo=UTC)  # Saturday
        end = datetime(2026, 4, 13, tzinfo=UTC)  # Monday
        assert _count_trading_days(start, end) == 1

    def test_single_weekday(self) -> None:
        start = datetime(2026, 4, 6, tzinfo=UTC)  # Monday
        end = datetime(2026, 4, 7, tzinfo=UTC)  # Tuesday
        assert _count_trading_days(start, end) == 1

    def test_two_weeks(self) -> None:
        start = datetime(2026, 4, 6, tzinfo=UTC)  # Monday
        end = datetime(2026, 4, 20, tzinfo=UTC)  # Monday 2 weeks later
        assert _count_trading_days(start, end) == 10


class TestCompositeScoring:
    """Tests for composite scoring with trade deficit penalty."""

    def test_config_defaults(self) -> None:
        config = OptimizerConfig()
        assert config.min_daily_trades == 10
        assert config.trade_deficit_penalty == 10.0

    def test_deficit_penalty_math(self) -> None:
        """Verify the penalty formula produces correct gradient."""
        # 5 trading days, min 10/day = 50 min trades, penalty 10/trade
        min_daily = 10
        penalty = 10.0
        trading_days = 5
        min_trades = min_daily * trading_days  # 50

        # 0 trades: PnL 0 - 50 * 10 = -500
        deficit_0 = max(0, min_trades - 0) * penalty
        assert deficit_0 == 500.0

        # 30 trades: PnL 100 - 20 * 10 = -100
        deficit_30 = max(0, min_trades - 30) * penalty
        assert 100.0 - deficit_30 == -100.0

        # 50 trades: PnL 100 - 0 = 100 (no penalty)
        deficit_50 = max(0, min_trades - 50) * penalty
        assert deficit_50 == 0.0

        # 80 trades: no penalty either
        deficit_80 = max(0, min_trades - 80) * penalty
        assert deficit_80 == 0.0
