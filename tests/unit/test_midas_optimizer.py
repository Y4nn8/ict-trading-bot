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
