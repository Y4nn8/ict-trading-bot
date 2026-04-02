"""Realistic execution simulation for backtesting.

Simulates spread, slippage, swap costs, and order rejection.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class SimulationConfig:
    """Configuration for execution simulation."""

    slippage_max_pips: float = 2.0
    order_rejection_rate: float = 0.01
    swap_long_per_day: float = -0.5  # pips per day
    swap_short_per_day: float = -0.3


@dataclass
class FillResult:
    """Result of simulating an order fill."""

    filled: bool
    fill_price: float
    spread_cost: float
    slippage: float


def simulate_fill(
    target_price: float,
    is_buy: bool,
    spread: float,
    config: SimulationConfig,
    volatility_factor: float = 1.0,
) -> FillResult:
    """Simulate filling an order with realistic execution.

    Args:
        target_price: The desired entry/exit price.
        is_buy: True for buy orders, False for sell.
        spread: Current spread in price units.
        config: Simulation configuration.
        volatility_factor: Multiplier for slippage (higher = more slippage).

    Returns:
        FillResult with actual fill price and costs.
    """
    # Order rejection
    if random.random() < config.order_rejection_rate:
        return FillResult(
            filled=False,
            fill_price=target_price,
            spread_cost=0.0,
            slippage=0.0,
        )

    # Spread cost: buy at ask (higher), sell at bid (lower)
    spread_adjustment = spread / 2 if is_buy else -spread / 2

    # Random slippage: 0 to max, scaled by volatility
    max_slip = config.slippage_max_pips * 0.0001 * volatility_factor
    slippage = random.uniform(0, max_slip)
    # Slippage is adverse: buy fills higher, sell fills lower
    slippage_direction = 1.0 if is_buy else -1.0

    fill_price = target_price + spread_adjustment + slippage * slippage_direction

    return FillResult(
        filled=True,
        fill_price=fill_price,
        spread_cost=abs(spread_adjustment),
        slippage=slippage,
    )


def compute_swap_cost(
    position_size: float,
    is_long: bool,
    days_held: float,
    config: SimulationConfig,
) -> float:
    """Compute overnight swap/financing cost.

    Args:
        position_size: Position size in lots.
        is_long: True for long positions.
        days_held: Number of days position is held (fractional OK).
        config: Simulation configuration.

    Returns:
        Swap cost in price units (negative = cost).
    """
    swap_rate = config.swap_long_per_day if is_long else config.swap_short_per_day
    return swap_rate * 0.0001 * position_size * days_held
