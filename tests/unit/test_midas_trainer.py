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
