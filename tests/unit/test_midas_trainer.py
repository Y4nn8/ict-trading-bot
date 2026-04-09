"""Tests for Midas LightGBM trainer."""

from __future__ import annotations

import numpy as np
import polars as pl

from src.midas.trainer import MidasTrainer, TrainerConfig


def _make_features(n: int = 500) -> pl.DataFrame:
    """Create synthetic feature data."""
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
    """Create synthetic 3-class target."""
    rng = np.random.default_rng(42)
    return rng.choice([0, 1, 2], size=n, p=[0.6, 0.2, 0.2]).astype(
        np.int32,
    )


class TestMidasTrainer:
    """Tests for MidasTrainer."""

    def test_build_target(self) -> None:
        buy = [1, 0, -1, 1, 0]
        sell = [0, 1, -1, 1, 0]
        target = MidasTrainer.build_target(buy, sell)

        assert target[0] == 1  # buy wins, sell loses → BUY
        assert target[1] == 2  # sell wins, buy loses → SELL
        assert target[2] == 0  # both timeout → PASS
        assert target[3] == 0  # both win → PASS (ambiguous)
        assert target[4] == 0  # both lose → PASS

    def test_train_produces_result(self) -> None:
        df = _make_features(300)
        target = _make_target(300)

        trainer = MidasTrainer(TrainerConfig(
            n_estimators=10,
            early_stopping_rounds=5,
        ))
        result = trainer.train(df, target)

        assert trainer.is_trained
        assert len(result.feature_names) == 3  # feat_1, feat_2, feat_3
        assert len(result.feature_importance) == 3
        assert result.n_train > 0
        assert result.n_val > 0

    def test_predict_returns_signal(self) -> None:
        df = _make_features(300)
        target = _make_target(300)

        trainer = MidasTrainer(TrainerConfig(
            n_estimators=10,
            entry_threshold=0.3,
        ))
        trainer.train(df, target)

        features = {"feat_1": 0.5, "feat_2": -0.3, "feat_3": 1.0}
        signal, confidence = trainer.predict(features)

        assert signal in (0, 1, 2)
        assert 0.0 <= confidence <= 1.0

    def test_meta_columns_excluded(self) -> None:
        df = _make_features(200)
        target = _make_target(200)

        trainer = MidasTrainer(TrainerConfig(n_estimators=5))
        result = trainer.train(df, target)

        assert "_time" not in result.feature_names
        assert "_bid" not in result.feature_names
        assert "_ask" not in result.feature_names

    def test_predict_threshold(self) -> None:
        """High threshold should produce more PASS signals."""
        df = _make_features(500)
        target = _make_target(500)

        # Low threshold
        trainer_low = MidasTrainer(TrainerConfig(
            n_estimators=20, entry_threshold=0.2,
        ))
        trainer_low.train(df, target)

        # High threshold
        trainer_high = MidasTrainer(TrainerConfig(
            n_estimators=20, entry_threshold=0.9,
        ))
        trainer_high.train(df, target)

        # Test on 100 samples
        rng = np.random.default_rng(99)
        low_entries = 0
        high_entries = 0
        for _ in range(100):
            f = {"feat_1": rng.normal(), "feat_2": rng.normal(),
                 "feat_3": rng.normal()}
            s_low, _ = trainer_low.predict(f)
            s_high, _ = trainer_high.predict(f)
            if s_low != 0:
                low_entries += 1
            if s_high != 0:
                high_entries += 1

        # High threshold should produce fewer entries
        assert high_entries <= low_entries

    def test_sample_weights(self) -> None:
        """PnL-based weights should be larger for higher PnL trades."""
        buy_pnls = [3.0, -2.0, 0.0, 5.0, -1.0]
        sell_pnls = [-1.0, 2.0, 0.0, -3.0, 1.0]
        target = MidasTrainer.build_target(
            [1, 0, -1, 1, 0], [0, 1, -1, 0, 1],
        )
        weights = MidasTrainer.build_sample_weights(
            buy_pnls, sell_pnls, target,
        )
        # BUY target[0]: weight = |3.0| + 1 = 4.0
        assert weights[0] == 4.0
        # SELL target[1]: weight = |2.0| + 1 = 3.0
        assert weights[1] == 3.0
        # PASS target[2]: weight = 1.0
        assert weights[2] == 1.0

    def test_train_with_weights(self) -> None:
        df = _make_features(300)
        target = _make_target(300)
        weights = np.ones(300)

        trainer = MidasTrainer(TrainerConfig(n_estimators=10))
        result = trainer.train(df, target, sample_weights=weights)
        assert trainer.is_trained
        assert result.n_train > 0

    def test_exit_model_training(self) -> None:
        """Exit model should train on features + position context."""
        rng = np.random.default_rng(42)
        n = 300
        df = pl.DataFrame({
            "feat_1": rng.normal(0, 1, n).tolist(),
            "feat_2": rng.normal(0, 1, n).tolist(),
            "pos_unrealized_pnl": rng.normal(0, 2, n).tolist(),
            "pos_duration_sec": rng.uniform(0, 300, n).tolist(),
            "pos_direction": rng.choice([1.0, -1.0], n).tolist(),
            "_time": list(range(n)),
        })
        labels = rng.choice([0, 1], n).astype(np.int32)

        trainer = MidasTrainer(TrainerConfig(n_estimators=10))
        result = trainer.train_exit(df, labels)

        assert trainer.has_exit_model
        assert "pos_unrealized_pnl" in result.feature_names
        assert "pos_duration_sec" in result.feature_names
        assert result.n_train > 0

    def test_predict_exit(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        df = pl.DataFrame({
            "feat_1": rng.normal(0, 1, n).tolist(),
            "pos_unrealized_pnl": rng.normal(0, 2, n).tolist(),
            "pos_duration_sec": rng.uniform(0, 300, n).tolist(),
            "pos_direction": rng.choice([1.0, -1.0], n).tolist(),
        })
        labels = rng.choice([0, 1], n).astype(np.int32)

        trainer = MidasTrainer(TrainerConfig(
            n_estimators=10, exit_threshold=0.5,
        ))
        trainer.train_exit(df, labels)

        should_close, confidence = trainer.predict_exit(
            {"feat_1": 0.5},
            pos_unrealized_pnl=1.5,
            pos_duration_sec=60.0,
            pos_direction=1.0,
        )
        assert isinstance(should_close, bool)
        assert 0.0 <= confidence <= 1.0

    def test_predict_exit_without_model(self) -> None:
        trainer = MidasTrainer()
        should_close, confidence = trainer.predict_exit(
            {}, pos_unrealized_pnl=0, pos_duration_sec=0, pos_direction=1,
        )
        assert should_close is False
        assert confidence == 0.0
