"""Tests for Midas nested Optuna optimizer."""

from __future__ import annotations

import optuna

from src.midas.optimizer import (
    OptimizerConfig,
    _suggest_inner_params,
    _suggest_outer_params,
)
from src.midas.replay_engine import build_default_registry


class TestOuterParams:
    """Tests for outer loop parameter suggestion."""

    def test_suggests_extractor_params(self) -> None:
        registry = build_default_registry()
        registry_params = registry.all_tunable_params()
        config = OptimizerConfig()

        study = optuna.create_study()
        trial = study.ask()
        params = _suggest_outer_params(trial, registry_params, config)

        for p in registry_params:
            assert p.name in params

        assert "sl_points" in params
        assert "tp_points" in params
        assert "label_timeout" in params

    def test_sl_tp_ranges(self) -> None:
        registry = build_default_registry()
        registry_params = registry.all_tunable_params()
        config = OptimizerConfig(sl_range=(5.0, 8.0), tp_range=(3.0, 6.0))

        study = optuna.create_study()
        trial = study.ask()
        params = _suggest_outer_params(trial, registry_params, config)

        assert 5.0 <= params["sl_points"] <= 8.0
        assert 3.0 <= params["tp_points"] <= 6.0
        assert 60.0 <= params["label_timeout"] <= 600.0


class TestInnerParams:
    """Tests for inner loop parameter suggestion."""

    def test_suggests_lgbm_params(self) -> None:
        study = optuna.create_study()
        trial = study.ask()
        params = _suggest_inner_params(trial)

        expected_keys = {
            "n_estimators",
            "learning_rate",
            "max_depth",
            "num_leaves",
            "min_child_samples",
            "subsample",
            "colsample_bytree",
            "entry_threshold",
        }
        assert set(params.keys()) == expected_keys

    def test_param_ranges(self) -> None:
        study = optuna.create_study()
        trial = study.ask()
        params = _suggest_inner_params(trial)

        assert 50 <= params["n_estimators"] <= 1000
        assert 0.01 <= params["learning_rate"] <= 0.3
        assert 3 <= params["max_depth"] <= 10
        assert 0.25 <= params["entry_threshold"] <= 0.60
