"""Tests for Midas LightGBM trainer."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

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


class TestImportanceThreshold:
    def _make_dataset(self, n: int = 400, n_features: int = 10) -> tuple[pl.DataFrame, np.ndarray]:
        rng = np.random.default_rng(42)
        data: dict[str, list[float]] = {}
        for i in range(n_features):
            data[f"f_{i}"] = rng.normal(0, 1, n).tolist()
        # Make f_0 a strong signal for the target
        df = pl.DataFrame(data)
        target = (df["f_0"].to_numpy() > 0).astype(np.int32)
        # Convert to 3-class (PASS=0, BUY=1)
        target_3c = np.where(target == 0, 1, 2).astype(np.int32)
        return df, target_3c

    def test_zero_threshold_keeps_all_features(self) -> None:
        df, target = self._make_dataset()
        trainer = MidasTrainer(TrainerConfig(
            n_estimators=20, importance_threshold=0.0,
        ))
        result = trainer.train(df, target)
        assert len(result.feature_names) == 10

    def test_threshold_filters_low_importance_features(self) -> None:
        df, target = self._make_dataset()
        trainer = MidasTrainer(TrainerConfig(
            n_estimators=20, importance_threshold=0.05,
        ))
        result = trainer.train(df, target)
        # At least one feature dropped since only f_0 carries the signal
        assert len(result.feature_names) < 10
        assert len(result.feature_names) >= 1

    def test_threshold_preserves_model_usability(self) -> None:
        df, target = self._make_dataset()
        trainer = MidasTrainer(TrainerConfig(
            n_estimators=20, importance_threshold=0.05,
        ))
        trainer.train(df, target)
        # Model still predicts with only the kept features
        features = {name: 0.5 for name in trainer._entry_features}
        signal, proba = trainer.predict(features)
        assert signal in (0, 1, 2)
        assert 0.0 <= proba <= 1.0

    def test_higher_threshold_drops_more(self) -> None:
        df, target = self._make_dataset()
        loose = MidasTrainer(TrainerConfig(
            n_estimators=20, importance_threshold=0.01,
        ))
        tight = MidasTrainer(TrainerConfig(
            n_estimators=20, importance_threshold=0.50,
        ))
        r_loose = loose.train(df, target)
        r_tight = tight.train(df, target)
        assert len(r_tight.feature_names) <= len(r_loose.feature_names)


class TestMetaLabeling:
    def _make_meta_dataset(
        self, n: int = 300, n_features: int = 5,
    ) -> tuple[pl.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(123)
        data: dict[str, list[float]] = {
            f"f_{i}": rng.normal(0, 1, n).tolist() for i in range(n_features)
        }
        df = pl.DataFrame(data)
        primary_proba = rng.uniform(0.5, 0.95, n).astype(np.float32)
        primary_direction = rng.choice([1, 2], n).astype(np.int32)
        # Meta label: profitable iff f_0 > 0 and primary_proba > 0.7
        meta_label = (
            (df["f_0"].to_numpy() > 0) & (primary_proba > 0.7)
        ).astype(np.int32)
        return df, primary_proba, primary_direction, meta_label

    def test_train_meta_produces_model(self) -> None:
        df, p_proba, p_dir, y = self._make_meta_dataset()
        trainer = MidasTrainer(TrainerConfig(n_estimators=30))
        result = trainer.train_meta(df, p_proba, p_dir, y)
        assert trainer.has_meta_model
        assert "primary_proba" in result.feature_names
        assert "primary_direction" in result.feature_names
        assert 0 <= result.val_log_loss <= 5

    def test_train_meta_raises_on_empty(self) -> None:
        trainer = MidasTrainer()
        empty = pl.DataFrame({"f_0": []})
        with pytest.raises(ValueError, match="empty"):
            trainer.train_meta(
                empty,
                np.array([], dtype=np.float32),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
            )

    def test_predict_without_meta_model_unchanged(self) -> None:
        # Train primary only, predict should behave as before
        rng = np.random.default_rng(42)
        n = 300
        df = pl.DataFrame({"f_0": rng.normal(0, 1, n).tolist()})
        target = (df["f_0"].to_numpy() > 0).astype(np.int32)
        target_3c = np.where(target == 0, 1, 2).astype(np.int32)
        trainer = MidasTrainer(TrainerConfig(
            n_estimators=30, entry_threshold=0.55,
        ))
        trainer.train(df, target_3c)
        signal, _ = trainer.predict({"f_0": 5.0})
        assert signal in (0, 1, 2)

    def test_meta_gate_can_suppress_primary_signal(self) -> None:
        # Train primary so it issues a signal
        rng = np.random.default_rng(0)
        n = 500
        df = pl.DataFrame({"f_0": rng.normal(0, 1, n).tolist()})
        target_3c = np.where(
            df["f_0"].to_numpy() > 0, 1, 2,
        ).astype(np.int32)
        trainer = MidasTrainer(TrainerConfig(
            n_estimators=30, entry_threshold=0.55,
            meta_threshold=0.99,  # very strict meta gate
        ))
        trainer.train(df, target_3c)
        signal_no_meta, _ = trainer.predict({"f_0": 3.0})

        # Fake a meta dataset where the label is always 0 (never take trade)
        meta_df, p_proba, p_dir, _ = self._make_meta_dataset()
        # Use same single feature name as the primary model for
        # realistic fit
        meta_df = pl.DataFrame({"f_0": rng.normal(0, 1, 500).tolist()})
        p_proba = rng.uniform(0.5, 0.95, 500).astype(np.float32)
        p_dir = rng.choice([1, 2], 500).astype(np.int32)
        y = np.zeros(500, dtype=np.int32)
        trainer.train_meta(meta_df, p_proba, p_dir, y)

        # Meta model learned "always 0" → predict should suppress signal
        signal_with_meta, _ = trainer.predict({"f_0": 3.0})
        assert signal_no_meta != 0  # primary would fire
        assert signal_with_meta == 0  # meta suppressed it

    def test_meta_properties(self) -> None:
        trainer = MidasTrainer()
        assert not trainer.has_meta_model
        df, p_proba, p_dir, y = self._make_meta_dataset()
        trainer.train_meta(df, p_proba, p_dir, y)
        assert trainer.has_meta_model
