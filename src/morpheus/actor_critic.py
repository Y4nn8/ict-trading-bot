"""Actor-critic networks and policy config for Phase B.

Simple MLP actor (state → action logits) and critic (state → value)
for discrete trading actions in imagination.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class PolicyConfig:
    """Configuration for actor/critic networks and trading environment."""

    # Network
    hidden_dim: int = 128
    n_hidden: int = 2

    # Trading
    spread_points: float = 0.5
    margin_rate: float = 0.05
    initial_capital: float = 5000.0

    # Reward shaping
    step_penalty: float = -0.0001
    invalid_penalty: float = -0.001
    idle_penalty: float = -0.0005

    # Training
    horizon: int = 64
    context_len: int = 256
    gamma: float = 0.99
    lambda_: float = 0.95
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    entropy_coef: float = 0.05
    grad_clip: float = 1.0
    epochs: int = 100
    batch_size: int = 32
    rollouts_per_epoch: int = 16
    seed: int = 0

    # AMP / perf
    amp: bool = False


HOLD, BUY, SELL = 0, 1, 2
N_ACTIONS = 3
PORTFOLIO_DIM = 7


def _build_mlp(
    in_dim: int, out_dim: int, hidden_dim: int, n_hidden: int,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for _ in range(n_hidden):
        layers.extend([nn.Linear(prev, hidden_dim), nn.ELU()])
        prev = hidden_dim
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Policy network: state → action logits.

    Args:
        state_dim: d_model + PORTFOLIO_DIM.
        n_actions: Number of discrete actions.
        hidden_dim: MLP hidden width.
        n_hidden: Number of hidden layers.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int = N_ACTIONS,
        hidden_dim: int = 128,
        n_hidden: int = 2,
    ) -> None:
        super().__init__()
        self.net = _build_mlp(state_dim, n_actions, hidden_dim, n_hidden)

    def forward(self, state: Tensor) -> Tensor:
        """Return action logits (*, n_actions)."""
        return self.net(state)  # type: ignore[no-any-return]


class Critic(nn.Module):
    """Value network: state → scalar value estimate.

    Args:
        state_dim: d_model + PORTFOLIO_DIM.
        hidden_dim: MLP hidden width.
        n_hidden: Number of hidden layers.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        n_hidden: int = 2,
    ) -> None:
        super().__init__()
        self.net = _build_mlp(state_dim, 1, hidden_dim, n_hidden)

    def forward(self, state: Tensor) -> Tensor:
        """Return value estimate (*, )."""
        return self.net(state).squeeze(-1)  # type: ignore[no-any-return]


def compute_state_dim(d_model: int) -> int:
    """Total state dimension for actor/critic input."""
    return d_model + PORTFOLIO_DIM


def build_actor_critic(
    d_model: int, config: PolicyConfig,
) -> tuple[Actor, Critic]:
    """Construct actor and critic from config."""
    state_dim = compute_state_dim(d_model)
    actor = Actor(state_dim, N_ACTIONS, config.hidden_dim, config.n_hidden)
    critic = Critic(state_dim, config.hidden_dim, config.n_hidden)
    return actor, critic


def portfolio_features(
    position: Tensor,
    unrealized_pnl: Tensor,
    step_in_pos: Tensor,
    capital: Tensor,
    margin_used: Tensor,
    initial_capital: float,
    horizon: int,
) -> Tensor:
    """Build (batch, PORTFOLIO_DIM) feature vector from portfolio state.

    Args:
        position: (batch,) int — 0=flat, 1=long, 2=short.
        unrealized_pnl: (batch,) float — current unrealized PnL.
        step_in_pos: (batch,) float — timesteps since entry.
        capital: (batch,) float — available capital.
        margin_used: (batch,) float — margin locked.
        initial_capital: Starting capital for normalization.
        horizon: Horizon length for time normalization.
    """
    device = position.device

    is_flat = (position == HOLD).float()
    is_long = (position == BUY).float()
    is_short = (position == SELL).float()
    pnl_norm = unrealized_pnl / initial_capital
    time_norm = step_in_pos / max(horizon, 1)
    cap_norm = capital / initial_capital
    margin_norm = margin_used / initial_capital

    return torch.stack(
        [is_flat, is_long, is_short, pnl_norm, time_norm, cap_norm, margin_norm],
        dim=-1,
    ).to(device)
