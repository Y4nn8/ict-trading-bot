"""Ensemble predictor — majority vote across top-K LightGBM models.

Instead of deploying the single best Optuna trial, the ensemble
aggregates predictions from the K highest-scoring trials.  This
reduces selection overfitting: a signal must be confirmed by
multiple independently-tuned models before it is executed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from src.midas.trainer import MidasTrainer


@dataclass(frozen=True, slots=True)
class EnsembleMember:
    """A single model in the ensemble with its associated params."""

    trainer: MidasTrainer
    inner_params: dict[str, Any]
    score: float


@dataclass(slots=True)
class EnsemblePredictor:
    """Majority-vote predictor over multiple MidasTrainer models.

    Each member independently predicts BUY/SELL/PASS, then the
    ensemble returns the majority signal.  Ties with PASS are
    resolved in favour of PASS (conservative).

    For exit predictions, majority vote on CLOSE vs HOLD.

    Attributes:
        members: Ordered list of ensemble members (best score first).
        min_agreement: Minimum number of members that must agree for
            a non-PASS signal.  Defaults to ``ceil(len(members) / 2)``.
    """

    members: list[EnsembleMember] = field(default_factory=list)
    min_agreement: int | None = None

    @property
    def size(self) -> int:
        """Number of models in the ensemble."""
        return len(self.members)

    @property
    def has_exit_model(self) -> bool:
        """True if any member has an exit model."""
        return any(m.trainer.has_exit_model for m in self.members)

    def _quorum(self) -> int:
        if self.min_agreement is not None:
            return self.min_agreement
        return (self.size + 1) // 2  # ceil(n/2), i.e. strict majority

    def predict(self, features: dict[str, float]) -> tuple[int, float]:
        """Majority-vote entry prediction.

        Returns:
            ``(signal, confidence)`` where signal is 0=PASS, 1=BUY,
            2=SELL.  Confidence is the mean probability across the
            members that voted for the winning signal.
        """
        if not self.members:
            return 0, 0.0

        votes: list[tuple[int, float]] = [
            m.trainer.predict(features) for m in self.members
        ]

        signals = [v[0] for v in votes]
        counts = Counter(signals)

        quorum = self._quorum()

        for candidate in (1, 2):
            if counts.get(candidate, 0) >= quorum:
                confs = [v[1] for v in votes if v[0] == candidate]
                return candidate, float(np.mean(confs))

        pass_confs = [v[1] for v in votes if v[0] == 0]
        if pass_confs:
            return 0, float(np.mean(pass_confs))
        all_confs = [v[1] for v in votes]
        return 0, float(np.mean(all_confs))

    def predict_exit(
        self,
        features: dict[str, float],
        pos_unrealized_pnl: float,
        pos_duration_sec: float,
        pos_direction: float,
    ) -> tuple[bool, float]:
        """Majority-vote exit prediction.

        Returns:
            ``(should_close, confidence)`` — close if a majority of
            exit-capable members vote CLOSE.
        """
        close_votes = 0
        total_voters = 0
        p_close_sum = 0.0

        for m in self.members:
            if not m.trainer.has_exit_model:
                continue
            total_voters += 1
            should_close, p_close = m.trainer.predict_exit(
                features,
                pos_unrealized_pnl=pos_unrealized_pnl,
                pos_duration_sec=pos_duration_sec,
                pos_direction=pos_direction,
            )
            p_close_sum += p_close
            if should_close:
                close_votes += 1

        if total_voters == 0:
            return False, 0.0

        avg_p = p_close_sum / total_voters
        quorum = (total_voters + 1) // 2
        return close_votes >= quorum, avg_p

    def median_param(self, key: str, default: float = 0.0) -> float:
        """Return the median value of a param across ensemble members."""
        values = [
            float(m.inner_params[key])
            for m in self.members
            if key in m.inner_params
        ]
        if not values:
            return default
        return float(np.median(values))

    def build_sim_config_overrides(self) -> dict[str, float]:
        """Compute median SL/TP and sizing params for the simulator."""
        return {
            "k_sl": self.median_param("k_sl", 1.5),
            "k_tp": self.median_param("k_tp", 1.5),
            "sl_fallback": self.median_param("sl_fallback", 3.0),
            "tp_fallback": self.median_param("tp_fallback", 3.0),
            "gamma": self.median_param("gamma", 1.0),
            "max_margin_proba": self.median_param("max_margin_proba", 0.85),
            "sizing_threshold": self.median_param("entry_threshold", 0.5),
            "min_risk_pct": self.median_param("min_risk_pct", 0.005),
        }
