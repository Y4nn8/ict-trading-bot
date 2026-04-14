"""Tests for Midas ensemble predictor."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.midas.ensemble import EnsembleMember, EnsemblePredictor
from src.midas.trainer import MidasTrainer, TrainerConfig


def _make_features(n: int = 500) -> pl.DataFrame:
    rng = np.random.default_rng(42)
    return pl.DataFrame({
        "feat_1": rng.normal(0, 1, n).tolist(),
        "feat_2": rng.normal(0, 1, n).tolist(),
        "feat_3": rng.normal(0, 1, n).tolist(),
        "_time": list(range(n)),
        "_bid": [100.0] * n,
        "_ask": [101.0] * n,
    })


def _make_target(n: int = 500) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.choice([0, 1, 2], size=n, p=[0.6, 0.2, 0.2]).astype(np.int32)


def _train_member(
    seed: int = 42,
    threshold: float = 0.4,
) -> tuple[MidasTrainer, dict[str, object]]:
    """Train a single member with a given seed."""
    rng = np.random.default_rng(seed)
    n = 500
    df = pl.DataFrame({
        "feat_1": rng.normal(0, 1, n).tolist(),
        "feat_2": rng.normal(0, 1, n).tolist(),
        "feat_3": rng.normal(0, 1, n).tolist(),
        "_time": list(range(n)),
        "_bid": [100.0] * n,
        "_ask": [101.0] * n,
    })
    target = rng.choice([0, 1, 2], size=n, p=[0.6, 0.2, 0.2]).astype(np.int32)
    config = TrainerConfig(
        n_estimators=20,
        entry_threshold=threshold,
        early_stopping_rounds=5,
    )
    trainer = MidasTrainer(config)
    trainer.train(df, target)
    params: dict[str, object] = {
        "k_sl": 1.5 + seed * 0.1,
        "k_tp": 2.0 + seed * 0.1,
        "sl_fallback": 3.0,
        "tp_fallback": 3.0,
        "gamma": 1.0,
        "max_margin_proba": 0.85,
        "entry_threshold": threshold,
        "min_risk_pct": 0.01,
    }
    return trainer, params


class TestEnsemblePredictor:
    """Tests for EnsemblePredictor."""

    def test_empty_ensemble_returns_pass(self) -> None:
        ep = EnsemblePredictor()
        signal, conf = ep.predict({"feat_1": 0.5, "feat_2": 0.3, "feat_3": 0.1})
        assert signal == 0
        assert conf == 0.0

    def test_single_member_delegates(self) -> None:
        trainer, params = _train_member(seed=42, threshold=0.3)
        ep = EnsemblePredictor(
            members=[EnsembleMember(trainer=trainer, inner_params=params, score=100.0)],
        )
        features = {"feat_1": 0.5, "feat_2": 0.3, "feat_3": 0.1}
        ens_sig, ens_conf = ep.predict(features)
        solo_sig, solo_conf = trainer.predict(features)
        assert ens_sig == solo_sig
        assert abs(ens_conf - solo_conf) < 1e-6

    def test_majority_vote_buy(self) -> None:
        """3 BUY vs 2 SELL → BUY wins."""
        ep = EnsemblePredictor()

        class FakeTrainer:
            has_exit_model = False
            def __init__(self, sig: int, conf: float) -> None:
                self._sig = sig
                self._conf = conf
            def predict(self, _f: dict[str, float]) -> tuple[int, float]:
                return self._sig, self._conf

        ep.members = [
            EnsembleMember(trainer=FakeTrainer(1, 0.7), inner_params={}, score=10),  # type: ignore[arg-type]
            EnsembleMember(trainer=FakeTrainer(1, 0.6), inner_params={}, score=9),  # type: ignore[arg-type]
            EnsembleMember(trainer=FakeTrainer(1, 0.8), inner_params={}, score=8),  # type: ignore[arg-type]
            EnsembleMember(trainer=FakeTrainer(2, 0.9), inner_params={}, score=7),  # type: ignore[arg-type]
            EnsembleMember(trainer=FakeTrainer(2, 0.85), inner_params={}, score=6),  # type: ignore[arg-type]
        ]
        signal, conf = ep.predict({})
        assert signal == 1
        assert abs(conf - 0.7) < 1e-6  # mean(0.7, 0.6, 0.8)

    def test_no_quorum_returns_pass(self) -> None:
        """2 BUY, 2 SELL, 1 PASS → no quorum → PASS."""
        ep = EnsemblePredictor()

        class FakeTrainer:
            has_exit_model = False
            def __init__(self, sig: int) -> None:
                self._sig = sig
            def predict(self, _f: dict[str, float]) -> tuple[int, float]:
                return self._sig, 0.5

        ep.members = [
            EnsembleMember(trainer=FakeTrainer(1), inner_params={}, score=10),  # type: ignore[arg-type]
            EnsembleMember(trainer=FakeTrainer(1), inner_params={}, score=9),  # type: ignore[arg-type]
            EnsembleMember(trainer=FakeTrainer(2), inner_params={}, score=8),  # type: ignore[arg-type]
            EnsembleMember(trainer=FakeTrainer(2), inner_params={}, score=7),  # type: ignore[arg-type]
            EnsembleMember(trainer=FakeTrainer(0), inner_params={}, score=6),  # type: ignore[arg-type]
        ]
        signal, _ = ep.predict({})
        assert signal == 0

    def test_size_property(self) -> None:
        ep = EnsemblePredictor()
        assert ep.size == 0
        trainer, params = _train_member()
        ep.members.append(
            EnsembleMember(trainer=trainer, inner_params=params, score=1.0),
        )
        assert ep.size == 1

    def test_has_exit_model_false_by_default(self) -> None:
        trainer, params = _train_member()
        ep = EnsemblePredictor(
            members=[EnsembleMember(trainer=trainer, inner_params=params, score=1.0)],
        )
        assert not ep.has_exit_model

    def test_median_param(self) -> None:
        members = [
            EnsembleMember(
                trainer=_train_member(seed=i)[0],
                inner_params={"k_sl": float(i)},
                score=float(i),
            )
            for i in range(1, 6)
        ]
        ep = EnsemblePredictor(members=members)
        assert ep.median_param("k_sl") == 3.0
        assert ep.median_param("missing", default=99.0) == 99.0

    def test_build_sim_config_overrides(self) -> None:
        members = [
            EnsembleMember(
                trainer=_train_member(seed=i)[0],
                inner_params={
                    "k_sl": 1.0 + i * 0.1,
                    "k_tp": 2.0 + i * 0.1,
                    "sl_fallback": 3.0,
                    "tp_fallback": 4.0,
                    "gamma": 1.0,
                    "max_margin_proba": 0.85,
                    "entry_threshold": 0.5,
                    "min_risk_pct": 0.01,
                },
                score=float(i),
            )
            for i in range(5)
        ]
        ep = EnsemblePredictor(members=members)
        ov = ep.build_sim_config_overrides()
        assert "k_sl" in ov
        assert "k_tp" in ov
        assert "gamma" in ov
        assert abs(ov["k_sl"] - 1.2) < 1e-6  # median of 1.0, 1.1, 1.2, 1.3, 1.4

    def test_predict_exit_majority_vote(self) -> None:
        """Test exit voting with fake trainers."""
        ep = EnsemblePredictor()

        class FakeTrainerExit:
            has_exit_model = True
            def __init__(self, should_close: bool) -> None:
                self._close = should_close
            def predict(self, _f: dict[str, float]) -> tuple[int, float]:
                return 0, 0.5
            def predict_exit(self, _f: dict[str, float], **_kw: float) -> tuple[bool, float]:
                return self._close, 0.8 if self._close else 0.2

        ep.members = [
            EnsembleMember(trainer=FakeTrainerExit(True), inner_params={}, score=10),  # type: ignore[arg-type]
            EnsembleMember(trainer=FakeTrainerExit(True), inner_params={}, score=9),  # type: ignore[arg-type]
            EnsembleMember(trainer=FakeTrainerExit(False), inner_params={}, score=8),  # type: ignore[arg-type]
        ]
        should_close, conf = ep.predict_exit(
            {}, pos_unrealized_pnl=1.0, pos_duration_sec=60.0, pos_direction=1.0,
        )
        assert should_close is True
        assert conf == pytest.approx((0.8 + 0.8 + 0.2) / 3, abs=1e-6)

    def test_predict_exit_no_exit_models(self) -> None:
        trainer, params = _train_member()
        ep = EnsemblePredictor(
            members=[EnsembleMember(trainer=trainer, inner_params=params, score=1.0)],
        )
        should_close, conf = ep.predict_exit(
            {}, pos_unrealized_pnl=0.0, pos_duration_sec=0.0, pos_direction=1.0,
        )
        assert should_close is False
        assert conf == 0.0

    def test_multi_member_real_models(self) -> None:
        """Ensemble of 3 real trained models produces valid output."""
        members = []
        for seed in [42, 123, 456]:
            trainer, params = _train_member(seed=seed, threshold=0.3)
            members.append(
                EnsembleMember(trainer=trainer, inner_params=params, score=float(seed)),
            )
        ep = EnsemblePredictor(members=members)
        features = {"feat_1": 0.5, "feat_2": -0.3, "feat_3": 1.2}
        signal, conf = ep.predict(features)
        assert signal in (0, 1, 2)
        assert 0.0 <= conf <= 1.0
