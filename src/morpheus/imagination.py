"""Vectorized trading environment over imagined trajectories.

Steps through pre-generated world model trajectories, simulates
trades with SL/TP in price points, tracks portfolio state, and
computes rewards.  Reconstructs synthetic prices from the world
model's relative returns using a real starting price.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from src.morpheus.actor_critic import BUY, HOLD, SELL, portfolio_features

if TYPE_CHECKING:
    from src.morpheus.actor_critic import PolicyConfig
    from src.morpheus.dataset import NormStats

RET_CLOSE_IDX = 3


class ImaginationEnv:
    """Vectorized trading environment on imagined trajectories.

    Reconstructs synthetic prices from relative returns, then checks
    SL/TP in price points (instrument-native units).  Position size
    uses 100% of available capital as margin.

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
        self._sl = config.sl_points
        self._tp = config.tp_points
        self._spread = config.spread_points
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

        Returns:
            Portfolio state dict.
        """
        z = torch.zeros(batch_size, device=self._device)
        return {
            "position": z.long(),
            "entry_price": z.clone(),
            "price": start_prices.to(self._device).float(),
            "capital": torch.full(
                (batch_size,), self._initial_capital, device=self._device,
            ),
            "margin_used": z.clone(),
            "step_in_pos": z.clone(),
            "unrealized_pnl": z.clone(),
            "size": z.clone(),
        }

    def step(
        self,
        actions: Tensor,
        obs_t: Tensor,
        portfolio: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, int | float]]:
        """Execute one environment step.

        Args:
            actions: (batch,) int64 — 0=HOLD, 1=BUY, 2=SELL.
            obs_t: (batch, obs_dim) — current imagined observation.
            portfolio: Current portfolio state dict.

        Returns:
            rewards: (batch,) float32.
            new_portfolio: Updated portfolio state dict.
        """
        pos = portfolio["position"].clone()
        entry_price = portfolio["entry_price"].clone()
        price = portfolio["price"].clone()
        capital = portfolio["capital"].clone()
        margin_used = portfolio["margin_used"].clone()
        step_in = portfolio["step_in_pos"].clone()
        size = portfolio["size"].clone()

        # Reconstruct price from denormalized return
        raw_ret = obs_t[:, RET_CLOSE_IDX] * self._ret_std + self._ret_mean
        price = price * (1.0 + raw_ret)

        rewards = torch.zeros_like(capital)

        in_long = pos == BUY
        in_short = pos == SELL

        # -- Compute unrealized PnL --
        long_pnl = (price - entry_price - self._spread) * size
        short_pnl = (entry_price - price - self._spread) * size
        unrealized = torch.where(in_long, long_pnl, torch.zeros_like(capital))
        unrealized = torch.where(in_short, short_pnl, unrealized)

        # -- Check SL/TP in price points --
        long_diff = price - entry_price - self._spread
        short_diff = entry_price - price - self._spread

        long_sl = in_long & (long_diff <= -self._sl)
        long_tp = in_long & (long_diff >= self._tp)
        short_sl = in_short & (short_diff <= -self._sl)
        short_tp = in_short & (short_diff >= self._tp)

        sl_mask = long_sl | short_sl
        tp_mask = long_tp | short_tp
        close_mask = sl_mask | tp_mask

        # Realized PnL (clamped to SL/TP)
        realized_pnl = torch.zeros_like(capital)
        realized_pnl = torch.where(long_sl, -self._sl * size, realized_pnl)
        realized_pnl = torch.where(long_tp, self._tp * size, realized_pnl)
        realized_pnl = torch.where(short_sl, -self._sl * size, realized_pnl)
        realized_pnl = torch.where(short_tp, self._tp * size, realized_pnl)

        # Close positions
        capital = torch.where(close_mask, capital + margin_used + realized_pnl, capital)
        margin_used = torch.where(close_mask, torch.zeros_like(margin_used), margin_used)
        pos = torch.where(close_mask, torch.zeros_like(pos), pos)
        entry_price = torch.where(close_mask, torch.zeros_like(entry_price), entry_price)
        step_in = torch.where(close_mask, torch.zeros_like(step_in), step_in)
        size = torch.where(close_mask, torch.zeros_like(size), size)
        unrealized = torch.where(close_mask, torch.zeros_like(unrealized), unrealized)

        rewards = torch.where(
            close_mask, realized_pnl / self._initial_capital, rewards,
        )

        # -- Process new actions (only when flat) --
        is_flat = pos == HOLD
        open_buy = is_flat & (actions == BUY)
        open_sell = is_flat & (actions == SELL)
        open_any = open_buy | open_sell

        new_margin = capital.clone()
        new_size = new_margin / (self._margin_rate * price)
        capital = torch.where(open_any, capital - new_margin, capital)
        margin_used = torch.where(open_any, new_margin, margin_used)
        size = torch.where(open_any, new_size, size)
        pos = torch.where(open_buy, torch.ones_like(pos), pos)
        pos = torch.where(open_sell, torch.full_like(pos, SELL), pos)
        entry_price = torch.where(open_any, price, entry_price)
        step_in = torch.where(open_any, torch.zeros_like(step_in), step_in)
        unrealized = torch.where(open_any, torch.zeros_like(unrealized), unrealized)

        # -- Invalid action penalty --
        invalid = (~is_flat) & (actions != HOLD)
        rewards = torch.where(
            invalid, rewards + self._config.invalid_penalty, rewards,
        )

        # -- Step penalty while in position --
        still_in = pos != HOLD
        rewards = torch.where(
            still_in, rewards + self._config.step_penalty, rewards,
        )

        # -- Idle penalty when flat and choosing HOLD --
        idle = is_flat & (actions == HOLD)
        rewards = torch.where(
            idle, rewards + self._config.idle_penalty, rewards,
        )

        step_in = torch.where(still_in, step_in + 1, step_in)

        stats = {
            "opened": open_any.sum().item(),
            "sl_hits": sl_mask.sum().item(),
            "tp_hits": tp_mask.sum().item(),
            "invalid": invalid.sum().item(),
        }

        return rewards, {
            "position": pos,
            "entry_price": entry_price,
            "price": price,
            "capital": capital,
            "margin_used": margin_used,
            "step_in_pos": step_in,
            "unrealized_pnl": unrealized,
            "size": size,
        }, stats

    def force_close(self, portfolio: dict[str, Tensor]) -> Tensor:
        """Force-close all open positions at current unrealized PnL.

        Returns:
            rewards: (batch,) realized PnL from force close.
        """
        in_position = portfolio["position"] != HOLD
        return torch.where(
            in_position,
            portfolio["unrealized_pnl"] / self._initial_capital,
            torch.zeros_like(portfolio["capital"]),
        )

    def get_features(self, portfolio: dict[str, Tensor]) -> Tensor:
        """Build (batch, PORTFOLIO_DIM) feature vector."""
        return portfolio_features(
            position=portfolio["position"],
            unrealized_pnl=portfolio["unrealized_pnl"],
            step_in_pos=portfolio["step_in_pos"],
            capital=portfolio["capital"],
            margin_used=portfolio["margin_used"],
            initial_capital=self._initial_capital,
            horizon=self._config.horizon,
        )
