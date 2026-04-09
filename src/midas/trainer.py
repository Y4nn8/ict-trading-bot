"""LightGBM trainer for the Midas scalping engine.

Entry model: 3-class (BUY=1, SELL=2, PASS=0), weighted by PnL magnitude.
Exit model: binary (HOLD=0, CLOSE=1), trained on ticks during open positions.
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
_META_COLUMNS = {
    "_time", "_bid", "_ask", "label_buy", "label_sell", "target",
    "pos_unrealized_pnl", "pos_duration_sec", "pos_direction",
    "exit_label",
}


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
        entry_threshold: Min P(BUY) or P(SELL) to generate entry signal.
        exit_threshold: Min P(CLOSE) to close a position early.
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
    exit_threshold: float = 0.55
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
    """Train and predict with LightGBM for tick-level entry + exit.

    Two models:
      - Entry model: 3-class (PASS/BUY/SELL), weighted by PnL magnitude
      - Exit model: binary (HOLD/CLOSE)

    Args:
        config: Training configuration.
    """

    def __init__(self, config: TrainerConfig | None = None) -> None:
        self._config = config or TrainerConfig()
        self._entry_model: lgb.Booster | None = None
        self._exit_model: lgb.Booster | None = None
        self._entry_features: list[str] = []
        self._exit_features: list[str] = []

    @property
    def is_trained(self) -> bool:
        """Whether the entry model has been trained."""
        return self._entry_model is not None

    @property
    def has_exit_model(self) -> bool:
        """Whether the exit model has been trained."""
        return self._exit_model is not None

    @staticmethod
    def build_target(
        buy_labels: list[int],
        sell_labels: list[int],
    ) -> np.ndarray:
        """Build 3-class target from buy/sell labels.

        Returns:
            Array with 0=PASS, 1=BUY, 2=SELL.
        """
        buy = np.array(buy_labels)
        sell = np.array(sell_labels)
        target = np.zeros(len(buy), dtype=np.int32)
        target[(buy == 1) & (sell != 1)] = 1
        target[(sell == 1) & (buy != 1)] = 2
        return target

    @staticmethod
    def build_sample_weights(
        buy_pnls: list[float],
        sell_pnls: list[float],
        target: np.ndarray,
    ) -> np.ndarray:
        """Build sample weights from PnL magnitude.

        Trades with larger PnL (positive or negative) get higher weight.
        This teaches the model to prioritize high-conviction setups.

        Returns:
            Weight array (same length as target).
        """
        buy_arr = np.array(buy_pnls)
        sell_arr = np.array(sell_pnls)
        weights = np.ones(len(target), dtype=np.float64)

        # BUY targets: weight by |buy_pnl|
        buy_mask = target == 1
        weights[buy_mask] = np.abs(buy_arr[buy_mask]) + 1.0

        # SELL targets: weight by |sell_pnl|
        sell_mask = target == 2
        weights[sell_mask] = np.abs(sell_arr[sell_mask]) + 1.0

        # PASS targets: weight=1 (baseline)
        return weights

    def train(
        self,
        df: pl.DataFrame,
        target: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> TrainResult:
        """Train the entry model (3-class: PASS/BUY/SELL).

        Args:
            df: Feature DataFrame (may contain meta columns).
            target: Target array (0=PASS, 1=BUY, 2=SELL).
            sample_weights: Optional PnL-based weights per sample.

        Returns:
            TrainResult with metrics and feature importance.
        """
        feature_cols = [
            c for c in df.columns if c not in _META_COLUMNS
        ]
        self._entry_features = feature_cols

        x_all = df.select(feature_cols).to_numpy()
        y = target

        cfg = self._config
        n = len(x_all)
        n_val = max(1, int(n * cfg.val_fraction))
        n_train = n - n_val

        x_train, x_val = x_all[:n_train], x_all[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]
        w_train = (
            sample_weights[:n_train] if sample_weights is not None else None
        )

        train_data = lgb.Dataset(
            x_train, label=y_train, weight=w_train,
            feature_name=feature_cols,
        )
        val_data = lgb.Dataset(
            x_val, label=y_val, reference=train_data,
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

        self._entry_model = lgb.train(
            params, train_data,
            num_boost_round=cfg.n_estimators,
            valid_sets=[val_data], valid_names=["val"],
            callbacks=callbacks,
        )

        importance = dict(zip(
            feature_cols,
            self._entry_model.feature_importance(importance_type="gain"),
            strict=True,
        ))

        val_loss = 0.0
        if self._entry_model.best_score and "val" in self._entry_model.best_score:
            val_loss = self._entry_model.best_score["val"]["multi_logloss"]

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

    def train_exit(
        self,
        df: pl.DataFrame,
        exit_labels: np.ndarray,
    ) -> TrainResult:
        """Train the exit model (binary: HOLD=0, CLOSE=1).

        The DataFrame should include position context columns:
        pos_unrealized_pnl, pos_duration_sec, pos_direction.

        Args:
            df: Feature DataFrame with position context.
            exit_labels: Binary array (0=HOLD, 1=CLOSE).

        Returns:
            TrainResult for the exit model.
        """
        # Exit model uses all features + position context columns
        exit_meta = {"_time", "_bid", "_ask", "exit_label"}
        feature_cols = [c for c in df.columns if c not in exit_meta]
        self._exit_features = feature_cols

        x_all = df.select(feature_cols).to_numpy()
        y = exit_labels

        cfg = self._config
        n = len(x_all)
        n_val = max(1, int(n * cfg.val_fraction))
        n_train = n - n_val

        x_train, x_val = x_all[:n_train], x_all[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]

        train_data = lgb.Dataset(
            x_train, label=y_train, feature_name=feature_cols,
        )
        val_data = lgb.Dataset(
            x_val, label=y_val, reference=train_data,
        )

        params: dict[str, Any] = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": cfg.learning_rate,
            "max_depth": cfg.max_depth,
            "num_leaves": cfg.num_leaves,
            "min_child_samples": min(cfg.min_child_samples, 20),
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

        self._exit_model = lgb.train(
            params, train_data,
            num_boost_round=cfg.n_estimators,
            valid_sets=[val_data], valid_names=["val"],
            callbacks=callbacks,
        )

        importance = dict(zip(
            feature_cols,
            self._exit_model.feature_importance(importance_type="gain"),
            strict=True,
        ))

        val_loss = 0.0
        if self._exit_model.best_score and "val" in self._exit_model.best_score:
            val_loss = self._exit_model.best_score["val"]["binary_logloss"]

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
        """Predict entry signal for a single feature row.

        Returns:
            (signal, confidence) where signal is 0=PASS, 1=BUY, 2=SELL.
        """
        assert self._entry_model is not None, "Entry model not trained"
        x = np.array(
            [[features.get(f, 0.0) for f in self._entry_features]],
        )
        proba = self._entry_model.predict(x)[0]
        threshold = self._config.entry_threshold

        p_buy = float(proba[1])
        p_sell = float(proba[2])

        if p_buy >= threshold and p_buy > p_sell:
            return 1, p_buy
        if p_sell >= threshold and p_sell > p_buy:
            return 2, p_sell
        return 0, float(proba[0])

    def predict_exit(
        self,
        features: dict[str, float],
        pos_unrealized_pnl: float,
        pos_duration_sec: float,
        pos_direction: float,
    ) -> tuple[bool, float]:
        """Predict whether to close an open position.

        Args:
            features: Market features (same 42 features).
            pos_unrealized_pnl: Current unrealized PnL in points.
            pos_duration_sec: Seconds since entry.
            pos_direction: 1.0 for BUY, -1.0 for SELL.

        Returns:
            (should_close, confidence) where confidence is P(CLOSE).
        """
        if self._exit_model is None:
            return False, 0.0

        exit_features = dict(features)
        exit_features["pos_unrealized_pnl"] = pos_unrealized_pnl
        exit_features["pos_duration_sec"] = pos_duration_sec
        exit_features["pos_direction"] = pos_direction

        x = np.array(
            [[exit_features.get(f, 0.0) for f in self._exit_features]],
        )
        p_close = float(self._exit_model.predict(x)[0])
        return p_close >= self._config.exit_threshold, p_close

    def save(self, path: Path) -> None:
        """Save entry model to file."""
        assert self._entry_model is not None
        self._entry_model.save_model(str(path))

    def load(self, path: Path) -> None:
        """Load entry model from file."""
        self._entry_model = lgb.Booster(model_file=str(path))
