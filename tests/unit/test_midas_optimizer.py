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
    """Tests for inner loop parameter suggestion (SL/TP + LightGBM)."""

    def test_suggests_all_params(self) -> None:
        config = OptimizerConfig()
        study = optuna.create_study()
        trial = study.ask()
        params = _suggest_inner_params(trial, config)

        # SL/TP should be in inner params
        assert "sl_points" in params
        assert "tp_points" in params
        assert "label_timeout" in params

        # LightGBM params
        assert "n_estimators" in params
        assert "entry_threshold" in params

    def test_respects_ranges(self) -> None:
        config = OptimizerConfig(sl_range=(5.0, 8.0), tp_range=(3.0, 6.0))
        study = optuna.create_study()
        trial = study.ask()
        params = _suggest_inner_params(trial, config)

        assert 5.0 <= params["sl_points"] <= 8.0
        assert 3.0 <= params["tp_points"] <= 6.0
        assert 0.25 <= params["entry_threshold"] <= 0.60
