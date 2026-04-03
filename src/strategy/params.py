"""Centralized strategy parameters for Optuna optimization.

All tunable numeric parameters are defined here. This dataclass is
the single source of truth that gets passed through the backtest
pipeline and optimized by Optuna.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import optuna


@dataclass
class StrategyParams:
    """All tunable strategy parameters.

    Grouped by subsystem. Optuna optimizes these values; the backtest
    pipeline reads them.
    """

    # --- Detector params ---
    swing_left_bars: int = 2
    swing_right_bars: int = 2
    ob_displacement_factor: float = 2.0
    ob_atr_period: int = 14
    disp_atr_period: int = 14
    disp_threshold: float = 1.5
    liq_tolerance_pct: float = 0.02
    liq_lookback: int = 50
    liq_min_touches: int = 2

    # --- Confluence weights (normalized internally) ---
    weight_fvg: float = 0.15
    weight_ob: float = 0.20
    weight_ms: float = 0.25
    weight_displacement: float = 0.10
    weight_killzone: float = 0.15
    weight_pd: float = 0.15

    # --- Entry ---
    min_confluence: float = 0.4
    sl_atr_multiple: float = 1.5
    rr_ratio: float = 2.0

    # --- Exit ---
    max_hold_candles: int = 72  # 0 = disabled

    # --- Filter ---
    max_spread_pips: float = 3.0
    require_killzone: bool = True
    max_positions: int = 5

    # --- Position sizing ---
    risk_low_threshold: float = 0.4
    risk_high_threshold: float = 0.7
    risk_low_pct: float = 0.5
    risk_medium_pct: float = 1.0
    risk_high_pct: float = 2.0
    risk_max_pct: float = 2.0

    # --- Risk manager ---
    max_daily_drawdown_pct: float = 3.0
    max_total_drawdown_pct: float = 10.0

    @staticmethod
    def from_optuna_trial(trial: optuna.Trial) -> StrategyParams:
        """Create params from an Optuna trial with suggested ranges.

        Args:
            trial: Optuna trial object.

        Returns:
            StrategyParams with trial-suggested values.
        """
        # Confluence weights — suggest raw then normalize
        raw_fvg = trial.suggest_float("weight_fvg", 0.0, 1.0)
        raw_ob = trial.suggest_float("weight_ob", 0.0, 1.0)
        raw_ms = trial.suggest_float("weight_ms", 0.0, 1.0)
        raw_disp = trial.suggest_float("weight_displacement", 0.0, 1.0)
        raw_kz = trial.suggest_float("weight_killzone", 0.0, 1.0)
        raw_pd = trial.suggest_float("weight_pd", 0.0, 1.0)
        total = raw_fvg + raw_ob + raw_ms + raw_disp + raw_kz + raw_pd
        if total == 0:
            total = 1.0

        return StrategyParams(
            # Detectors
            swing_left_bars=trial.suggest_int("swing_left_bars", 1, 5),
            swing_right_bars=trial.suggest_int("swing_right_bars", 1, 5),
            ob_displacement_factor=trial.suggest_float("ob_displacement_factor", 1.0, 4.0),
            ob_atr_period=trial.suggest_int("ob_atr_period", 10, 30),
            disp_atr_period=trial.suggest_int("disp_atr_period", 10, 30),
            disp_threshold=trial.suggest_float("disp_threshold", 1.0, 3.0),
            liq_tolerance_pct=trial.suggest_float("liq_tolerance_pct", 0.01, 0.1),
            liq_lookback=trial.suggest_int("liq_lookback", 20, 200),
            liq_min_touches=trial.suggest_int("liq_min_touches", 2, 5),
            # Confluence (normalized)
            weight_fvg=raw_fvg / total,
            weight_ob=raw_ob / total,
            weight_ms=raw_ms / total,
            weight_displacement=raw_disp / total,
            weight_killzone=raw_kz / total,
            weight_pd=raw_pd / total,
            # Entry
            min_confluence=trial.suggest_float("min_confluence", 0.2, 0.8),
            sl_atr_multiple=trial.suggest_float("sl_atr_multiple", 0.5, 3.0),
            rr_ratio=trial.suggest_float("rr_ratio", 1.0, 5.0),
            # Exit
            max_hold_candles=trial.suggest_int("max_hold_candles", 12, 288),
            # Filter
            max_spread_pips=trial.suggest_float("max_spread_pips", 0.5, 10.0),
            require_killzone=trial.suggest_categorical(
                "require_killzone", [True, False]
            ),
            max_positions=trial.suggest_int("max_positions", 1, 10),
            # Position sizing
            risk_low_threshold=trial.suggest_float("risk_low_threshold", 0.2, 0.6),
            risk_high_threshold=trial.suggest_float("risk_high_threshold", 0.5, 0.9),
            risk_low_pct=trial.suggest_float("risk_low_pct", 0.1, 2.0),
            risk_medium_pct=trial.suggest_float("risk_medium_pct", 0.2, 3.0),
            risk_high_pct=trial.suggest_float("risk_high_pct", 0.5, 5.0),
            risk_max_pct=trial.suggest_float("risk_max_pct", 1.0, 5.0),
            # Risk
            max_daily_drawdown_pct=trial.suggest_float(
                "max_daily_drawdown_pct", 1.0, 5.0
            ),
            max_total_drawdown_pct=trial.suggest_float(
                "max_total_drawdown_pct", 5.0, 20.0
            ),
        )

    @staticmethod
    def from_dict(params: dict[str, Any]) -> StrategyParams:
        """Create params from a dict (e.g. from Optuna best_trial.params).

        Normalizes confluence weights automatically.

        Args:
            params: Dict of parameter name → value.

        Returns:
            StrategyParams instance.
        """
        # Normalize confluence weights if present as raw values
        weight_keys = [
            "weight_fvg", "weight_ob", "weight_ms",
            "weight_displacement", "weight_killzone", "weight_pd",
        ]
        weights = {k: float(params.get(k, 0.1)) for k in weight_keys}
        total = sum(weights.values()) or 1.0
        normalized = {k: v / total for k, v in weights.items()}

        return StrategyParams(
            swing_left_bars=int(params.get("swing_left_bars", 2)),
            swing_right_bars=int(params.get("swing_right_bars", 2)),
            ob_displacement_factor=float(params.get("ob_displacement_factor", 2.0)),
            ob_atr_period=int(params.get("ob_atr_period", 14)),
            disp_atr_period=int(params.get("disp_atr_period", 14)),
            disp_threshold=float(params.get("disp_threshold", 1.5)),
            liq_tolerance_pct=float(params.get("liq_tolerance_pct", 0.02)),
            liq_lookback=int(params.get("liq_lookback", 50)),
            liq_min_touches=int(params.get("liq_min_touches", 2)),
            weight_fvg=normalized["weight_fvg"],
            weight_ob=normalized["weight_ob"],
            weight_ms=normalized["weight_ms"],
            weight_displacement=normalized["weight_displacement"],
            weight_killzone=normalized["weight_killzone"],
            weight_pd=normalized["weight_pd"],
            min_confluence=float(params.get("min_confluence", 0.4)),
            sl_atr_multiple=float(params.get("sl_atr_multiple", 1.5)),
            rr_ratio=float(params.get("rr_ratio", 2.0)),
            max_hold_candles=int(params.get("max_hold_candles", 72)),
            max_spread_pips=float(params.get("max_spread_pips", 3.0)),
            require_killzone=bool(params.get("require_killzone", True)),
            max_positions=int(params.get("max_positions", 5)),
            risk_low_threshold=float(params.get("risk_low_threshold", 0.4)),
            risk_high_threshold=float(params.get("risk_high_threshold", 0.7)),
            risk_low_pct=float(params.get("risk_low_pct", 0.5)),
            risk_medium_pct=float(params.get("risk_medium_pct", 1.0)),
            risk_high_pct=float(params.get("risk_high_pct", 2.0)),
            risk_max_pct=float(params.get("risk_max_pct", 2.0)),
            max_daily_drawdown_pct=float(params.get("max_daily_drawdown_pct", 3.0)),
            max_total_drawdown_pct=float(params.get("max_total_drawdown_pct", 10.0)),
        )

    def to_dict(self) -> dict[str, object]:
        """Convert to dict for serialization."""
        return dict(self.__dict__)
