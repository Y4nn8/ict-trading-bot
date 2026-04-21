"""Vectorized trading environment over imagined trajectories.

Steps through pre-generated world model trajectories with continuous
position-based rewards.  The actor chooses a position (long/short/flat)
at each step; reward = position * return.  Spread is charged only when
the position changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from src.morpheus.actor_critic import BUY, HOLD, SELL, portfolio_features
from src.morpheus.dataset import BASE_OBS_COLUMNS

if TYPE_CHECKING:
    from src.morpheus.actor_critic import PolicyConfig
    from src.morpheus.dataset import NormStats

RET_CLOSE_IDX = BASE_OBS_COLUMNS.index("ret_close")


class ImaginationEnv:
    """Vectorized trading environment on imagined trajectories.

    Position-based reward: at each step, reward = position * return.
    BUY means "be long", SELL means "be short", HOLD means "keep
    current position".  Spread is charged on position changes only.

    Args:
        config: PolicyConfig with trading parameters.
        norm_stats: NormStats for denormalizing ret_close.
        device: Torch device.
    """

    def __init__(
        self,
        config: PolicyConfig,
        norm_stats: NormStats,
        device: torch.device,
    ) -> None:
        self._config = config
        self._device = device
        self._margin_rate = config.margin_rate
        self._initial_capital = config.initial_capital

        self._ret_mean = float(norm_stats.mean[RET_CLOSE_IDX])
        self._ret_std = float(norm_stats.std[RET_CLOSE_IDX])

    def reset(
        self, batch_size: int, start_prices: Tensor,
    ) -> dict[str, Tensor]:
        """Initialize portfolio state for a new batch.

        Args:
            batch_size: Number of parallel environments.
            start_prices: (batch,) close price at end of context window.
        """
        z = torch.zeros(batch_size, device=self._device)
        return {
            "position": z.long(),
            "price": start_prices.to(self._device).float(),
            "capital": torch.full(
                (batch_size,), self._initial_capital, device=self._device,
            ),
            "step_in_pos": z.clone(),
            "cumulative_pnl": z.clone(),
        }

    def step(
        self,
        actions: Tensor,
        obs_t: Tensor,
        portfolio: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, int]]:
        """Execute one environment step.

        Actions: BUY=be long, SELL=be short, HOLD=keep current position.
        Reward = position_sign * raw_return * notional / initial_capital.
        Spread is charged when position changes.

        Args:
            actions: (batch,) int64 — 0=HOLD, 1=BUY, 2=SELL.
            obs_t: (batch, obs_dim) — current imagined observation.
            portfolio: Current portfolio state dict.

        Returns:
            rewards, new_portfolio, stats.
        """
        pos = portfolio["position"].clone()
        price = portfolio["price"].clone()
        capital = portfolio["capital"].clone()
        step_in = portfolio["step_in_pos"].clone()
        cum_pnl = portfolio["cumulative_pnl"].clone()

        # Denormalize return and update price
        raw_ret = obs_t[:, RET_CLOSE_IDX] * self._ret_std + self._ret_mean
        new_price = price * (1.0 + raw_ret)

        # Position size in units
        size = capital / (self._margin_rate * price)

        # Position sign: long=+1, short=-1, flat=0
        pos_sign = torch.zeros_like(capital)
        pos_sign = torch.where(pos == BUY, torch.ones_like(pos_sign), pos_sign)
        pos_sign = torch.where(pos == SELL, -torch.ones_like(pos_sign), pos_sign)

        # Reward = position * price_change * size / initial_capital
        price_change = new_price - price
        rewards = pos_sign * price_change * size / self._initial_capital

        # -- Process position changes --
        # HOLD keeps current position
        new_pos = pos.clone()
        new_pos = torch.where(actions == BUY, torch.ones_like(pos), new_pos)
        new_pos = torch.where(actions == SELL, torch.full_like(pos, SELL), new_pos)

        # Detect position changes and charge spread
        pos_changed = new_pos != pos
        was_flat = pos == HOLD
        going_flat = new_pos == HOLD
        # Flipping (long→short or short→long) costs 2 spreads
        flipping = pos_changed & (~was_flat) & (~going_flat)
        opening = pos_changed & was_flat & (~going_flat)
        closing = pos_changed & (~was_flat) & going_flat

        spread_cost = torch.zeros_like(capital)
        spread_pts = self._config.spread_points
        spread_cost = torch.where(
            opening, spread_pts * size, spread_cost,
        )
        spread_cost = torch.where(
            closing, spread_pts * size, spread_cost,
        )
        spread_cost = torch.where(
            flipping, 2.0 * spread_pts * size, spread_cost,
        )

        rewards = rewards - spread_cost / self._initial_capital

        # Update step counter
        step_in = torch.where(new_pos != HOLD, step_in + 1, torch.zeros_like(step_in))

        # Idle penalty when flat
        idle = (new_pos == HOLD) & (actions == HOLD) & (pos == HOLD)
        rewards = torch.where(
            idle, rewards + self._config.idle_penalty, rewards,
        )

        cum_pnl = cum_pnl + rewards * self._initial_capital

        stats = {
            "position_changes": int(pos_changed.sum().item()),
            "long_steps": int((new_pos == BUY).sum().item()),
            "short_steps": int((new_pos == SELL).sum().item()),
            "flat_steps": int((new_pos == HOLD).sum().item()),
        }

        return rewards, {
            "position": new_pos,
            "price": new_price,
            "capital": capital,
            "step_in_pos": step_in,
            "cumulative_pnl": cum_pnl,
        }, stats

    def get_features(self, portfolio: dict[str, Tensor]) -> Tensor:
        """Build (batch, PORTFOLIO_DIM) feature vector."""
        return portfolio_features(
            position=portfolio["position"],
            unrealized_pnl=portfolio["cumulative_pnl"],
            step_in_pos=portfolio["step_in_pos"],
            capital=portfolio["capital"],
            margin_used=torch.zeros_like(portfolio["capital"]),
            initial_capital=self._initial_capital,
            horizon=self._config.horizon,
        )
