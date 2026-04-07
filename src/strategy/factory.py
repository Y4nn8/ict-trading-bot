"""Factory: build all strategy components from StrategyParams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.backtest.simulator import SimulationConfig
from src.execution.position_sizer import PositionSizer, RiskTiers
from src.execution.risk_manager import RiskManager
from src.strategy.confluence import ConfluenceScorer, ConfluenceWeights
from src.strategy.entry import EntryEvaluator
from src.strategy.exit import ExitEvaluator
from src.strategy.filters import TradeFilter

if TYPE_CHECKING:
    from src.strategy.params import StrategyParams


@dataclass
class StrategyComponents:
    """All components needed by the backtest engine."""

    confluence_scorer: ConfluenceScorer
    entry_evaluator: EntryEvaluator
    exit_evaluator: ExitEvaluator
    trade_filter: TradeFilter
    position_sizer: PositionSizer
    risk_manager: RiskManager
    sim_config: SimulationConfig


def build_strategy(params: StrategyParams) -> StrategyComponents:
    """Build all strategy components from a StrategyParams instance.

    Args:
        params: The strategy parameters.

    Returns:
        StrategyComponents with all components configured.
    """
    return StrategyComponents(
        confluence_scorer=ConfluenceScorer(
            weights=ConfluenceWeights(
                fvg=params.weight_fvg,
                order_block=params.weight_ob,
                market_structure=params.weight_ms,
                displacement=params.weight_displacement,
                killzone=params.weight_killzone,
                premium_discount=params.weight_pd,
            )
        ),
        entry_evaluator=EntryEvaluator(
            min_confluence=params.min_confluence,
            default_sl_atr_multiple=params.sl_atr_multiple,
            default_rr_ratio=params.rr_ratio,
        ),
        exit_evaluator=ExitEvaluator(
            max_hold_candles=params.max_hold_candles,
        ),
        trade_filter=TradeFilter(
            max_spread_pips=params.max_spread_pips,
            require_killzone=params.require_killzone,
            max_positions=params.max_positions,
        ),
        position_sizer=PositionSizer(
            tiers=RiskTiers(
                low_threshold=params.risk_low_threshold,
                high_threshold=params.risk_high_threshold,
                low_risk_pct=params.risk_low_pct,
                medium_risk_pct=params.risk_medium_pct,
                high_risk_pct=params.risk_high_pct,
            ),
            max_risk_pct=params.risk_max_pct,
        ),
        risk_manager=RiskManager(
            max_daily_drawdown_pct=params.max_daily_drawdown_pct,
            max_total_drawdown_pct=params.max_total_drawdown_pct,
            max_positions=params.max_positions,
            max_daily_gain_pct=params.max_daily_gain_pct,
        ),
        sim_config=SimulationConfig(),
    )
