"""LightGBM trainer for the Midas scalping engine.

Trains a 3-class model (BUY=1, SELL=2, PASS=0) from feature+label data.
Predicts probabilities and applies a threshold for entry signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import lightgbm as lgb
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    import polars as pl

# Columns excluded from training features
_META_COLUMNS = {"_time", "_bid", "_ask", "label_buy", "label_sell", "target"}


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    """LightGBM training configuration.

    Args:
        n_estimators: Number of boosting rounds.
        learning_rate: Step size shrinkage.
        max_depth: Maximum tree depth (-1 = no limit).
        num_leaves: Max leaves per tree.
        min_child_samples: Minimum samples in a leaf.
        subsample: Row subsampling ratio.
        colsample_bytree: Column subsampling ratio.
        entry_threshold: Min P(BUY) or P(SELL) to generate a signal.
        val_fraction: Fraction of training data for early-stop validation.
        early_stopping_rounds: Stop if val metric doesn't improve for N rounds.
    """

    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 6
    num_leaves: int = 31
    min_child_samples: int = 100
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    entry_threshold: float = 0.55
    val_fraction: float = 0.1
    early_stopping_rounds: int = 50


@dataclass
class TrainResult:
    """Output of a training run."""

    feature_names: list[str] = field(default_factory=list)
    feature_importance: dict[str, float] = field(default_factory=dict)
    val_log_loss: float = 0.0
    class_distribution: dict[int, int] = field(default_factory=dict)
    n_train: int = 0
    n_val: int = 0


class MidasTrainer:
    """Train and predict with LightGBM for tick-level signals.

    Args:
        config: Training configuration.
    """

    def __init__(self, config: TrainerConfig | None = None) -> None:
        self._config = config or TrainerConfig()
        self._model: lgb.Booster | None = None
        self._feature_names: list[str] = []

    @property
    def is_trained(self) -> bool:
        """Whether a model has been trained or loaded."""
        return self._model is not None

    @staticmethod
    def build_target(
        buy_labels: list[int],
        sell_labels: list[int],
    ) -> np.ndarray:
        """Build 3-class target from buy/sell labels.

        Args:
            buy_labels: Per-row buy outcome (1=win, 0=loss, -1=timeout).
            sell_labels: Per-row sell outcome.

        Returns:
            Array with 0=PASS, 1=BUY, 2=SELL.
        """
        buy = np.array(buy_labels)
        sell = np.array(sell_labels)
        target = np.zeros(len(buy), dtype=np.int32)

        # BUY wins and SELL doesn't → BUY
        target[(buy == 1) & (sell != 1)] = 1
        # SELL wins and BUY doesn't → SELL
        target[(sell == 1) & (buy != 1)] = 2
        # Both win → PASS (ambiguous), both lose → PASS, timeout → PASS
        return target

    def train(
        self,
        df: pl.DataFrame,
        target: np.ndarray,
    ) -> TrainResult:
        """Train LightGBM on features + target.

        Args:
            df: Feature DataFrame (may contain meta columns).
            target: Target array (0=PASS, 1=BUY, 2=SELL).

        Returns:
            TrainResult with metrics and feature importance.
        """
        # Filter to trainable rows (exclude timeout labels from target)
        # We keep PASS (0) rows since they're valid "don't trade" examples
        feature_cols = [
            c for c in df.columns if c not in _META_COLUMNS
        ]
        self._feature_names = feature_cols

        x_all = df.select(feature_cols).to_numpy()
        y = target

        # Temporal train/val split
        cfg = self._config
        n = len(x_all)
        n_val = max(1, int(n * cfg.val_fraction))
        n_train = n - n_val

        x_train, x_val = x_all[:n_train], x_all[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]

        train_data = lgb.Dataset(
            x_train, label=y_train,
            feature_name=feature_cols,
        )
        val_data = lgb.Dataset(
            x_val, label=y_val,
            reference=train_data,
        )

        params: dict[str, Any] = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "learning_rate": cfg.learning_rate,
            "max_depth": cfg.max_depth,
            "num_leaves": cfg.num_leaves,
            "min_child_samples": cfg.min_child_samples,
            "subsample": cfg.subsample,
            "colsample_bytree": cfg.colsample_bytree,
            "is_unbalance": True,
            "verbosity": -1,
            "seed": 42,
        }

        callbacks: list[Any] = [
            lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ]

        self._model = lgb.train(
            params,
            train_data,
            num_boost_round=cfg.n_estimators,
            valid_sets=[val_data],
            valid_names=["val"],
            callbacks=callbacks,
        )

        # Feature importance
        importance = dict(
            zip(
                feature_cols,
                self._model.feature_importance(importance_type="gain"),
                strict=True,
            ),
        )

        # Val loss
        val_loss = 0.0
        if self._model.best_score and "val" in self._model.best_score:
            val_loss = self._model.best_score["val"]["multi_logloss"]

        # Class distribution
        unique, counts = np.unique(y, return_counts=True)
        dist = dict(zip(unique.tolist(), counts.tolist(), strict=True))

        return TrainResult(
            feature_names=feature_cols,
            feature_importance=importance,
            val_log_loss=val_loss,
            class_distribution=dist,
            n_train=n_train,
            n_val=n_val,
        )

    def predict(self, features: dict[str, float]) -> tuple[int, float]:
        """Predict signal for a single feature row.

        Args:
            features: Feature dict (as produced by FeatureRegistry).

        Returns:
            (signal, confidence) where signal is 0=PASS, 1=BUY, 2=SELL
            and confidence is the winning class probability.
        """
        assert self._model is not None, "Model not trained"
        x = np.array(
            [[features.get(f, 0.0) for f in self._feature_names]],
        )
        proba = self._model.predict(x)[0]  # [p_pass, p_buy, p_sell]
        threshold = self._config.entry_threshold

        p_buy = float(proba[1])
        p_sell = float(proba[2])

        if p_buy >= threshold and p_buy > p_sell:
            return 1, p_buy
        if p_sell >= threshold and p_sell > p_buy:
            return 2, p_sell
        return 0, float(proba[0])

    def save(self, path: Path) -> None:
        """Save trained model to file."""
        assert self._model is not None
        self._model.save_model(str(path))

    def load(self, path: Path) -> None:
        """Load model from file."""
        self._model = lgb.Booster(model_file=str(path))
