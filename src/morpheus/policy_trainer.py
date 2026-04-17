"""Policy training loop — actor/critic in imagination.

Samples context windows from the dataset, generates imagined
trajectories with the frozen world model, and trains actor/critic
using policy gradient with lambda-returns.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as f
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from src.common.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from src.morpheus.actor_critic import Actor, Critic, PolicyConfig
    from src.morpheus.dataset import MorpheusDataset
    from src.morpheus.imagination import ImaginationEnv
    from src.morpheus.transformer_wm import TransformerWorldModel

logger = get_logger(__name__)


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric log transform: sign(x) * ln(|x| + 1)."""
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog: sign(x) * (exp(|x|) - 1)."""
    return torch.sign(x) * (torch.exp(x.abs()) - 1)


@dataclass
class PolicyEpochMetrics:
    """Per-epoch metrics for policy training."""

    epoch: int
    actor_loss: float
    critic_loss: float
    mean_reward: float
    mean_return: float
    entropy: float
    mean_capital: float = field(default=float("nan"))
    position_changes: int = 0
    long_steps: int = 0
    short_steps: int = 0
    flat_steps: int = 0


def compute_mc_returns(
    rewards: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Compute Monte Carlo discounted returns (no bootstrapping).

    Args:
        rewards: (batch, horizon) per-step rewards.
        gamma: Discount factor.

    Returns:
        Discounted returns (batch, horizon).
    """
    _, horizon = rewards.shape
    returns = torch.zeros_like(rewards)
    running = torch.zeros(rewards.shape[0], device=rewards.device)

    for t in reversed(range(horizon)):
        running = rewards[:, t] + gamma * running
        returns[:, t] = running

    return returns


class PolicyTrainer:
    """Trains actor/critic in imagination.

    The world model is frozen. Trajectories are pre-generated,
    hidden states extracted, then the actor steps through the
    ImaginationEnv collecting rewards. Actor and critic are
    updated with policy gradient + lambda-returns.

    Args:
        world_model: Frozen TransformerWorldModel.
        actor: Actor network.
        critic: Critic network.
        env: ImaginationEnv for trade simulation.
        dataset: MorpheusDataset for sampling contexts.
        config: PolicyConfig.
        device: Torch device.
    """

    def __init__(
        self,
        world_model: TransformerWorldModel,
        actor: Actor,
        critic: Critic,
        env: ImaginationEnv,
        dataset: MorpheusDataset,
        config: PolicyConfig,
        device: torch.device,
    ) -> None:
        self._wm = world_model
        self._actor = actor
        self._critic = critic
        self._env = env
        self._dataset = dataset
        self._config = config
        self._device = device

        for p in self._wm.parameters():
            p.requires_grad_(False)
        self._wm.eval()

        self._actor_opt = Adam(self._actor.parameters(), lr=config.actor_lr)
        self._critic_opt = Adam(self._critic.parameters(), lr=config.critic_lr)

    def _sample_contexts(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample n context windows and their close prices."""
        indices = torch.randint(0, len(self._dataset), (n,))
        batch = torch.stack([self._dataset[int(i)] for i in indices])
        closes = torch.tensor(
            [self._dataset.get_close(int(i)) for i in indices],
            dtype=torch.float32,
        )
        return batch.to(self._device), closes.to(self._device)

    @torch.no_grad()
    def _generate_rollout_data(
        self, contexts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate imagined observations and hidden states.

        Args:
            contexts: (batch, context_len, obs_dim).

        Returns:
            imagined_obs: (batch, horizon, obs_dim).
            hidden_states: (batch, horizon, d_model).
        """
        imagined = self._wm.imagine(contexts, self._config.horizon)
        full_seq = torch.cat([contexts, imagined], dim=1)
        all_hidden = self._wm._run_backbone(full_seq)
        hidden = all_hidden[:, self._config.context_len :]
        return imagined, hidden

    def _rollout(
        self,
        contexts: torch.Tensor,
        start_prices: torch.Tensor,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor,
        dict[str, int | float],
    ]:
        """Run one batch of rollouts, collecting states, actions, rewards.

        Returns:
            states, actions, rewards, final_capital, final_margin,
            final_unrealized, rollout_stats.
        """
        imagined, hidden = self._generate_rollout_data(contexts)
        portfolio = self._env.reset(contexts.shape[0], start_prices)
        horizon = self._config.horizon
        rollout_stats: dict[str, int | float] = {}

        states_list: list[torch.Tensor] = []
        actions_list: list[torch.Tensor] = []
        rewards_list: list[torch.Tensor] = []

        for t in range(horizon):
            h_t = hidden[:, t]
            port_feat = self._env.get_features(portfolio)
            state = torch.cat([h_t, port_feat], dim=-1)
            states_list.append(state)

            with torch.no_grad():
                logits = self._actor(state)
                dist = Categorical(logits=logits)
                action: torch.Tensor = dist.sample()  # type: ignore[no-untyped-call]

            reward, portfolio, step_stats = self._env.step(
                action, imagined[:, t], portfolio,
            )
            actions_list.append(action)
            rewards_list.append(reward)
            for k, v in step_stats.items():
                rollout_stats[k] = rollout_stats.get(k, 0) + v

        states = torch.stack(states_list, dim=1)
        actions = torch.stack(actions_list, dim=1)
        rewards = torch.stack(rewards_list, dim=1)

        return (
            states, actions, rewards,
            portfolio["capital"], portfolio["cumulative_pnl"],
            torch.zeros_like(portfolio["capital"]),
            rollout_stats,
        )

    def train_epoch(self, epoch: int) -> PolicyEpochMetrics:
        """Run one training epoch over multiple rollout batches."""
        total_samples = self._config.rollouts_per_epoch * self._config.batch_size
        contexts, closes = self._sample_contexts(total_samples)

        all_actor_loss: list[float] = []
        all_critic_loss: list[float] = []
        all_rewards: list[float] = []
        all_returns: list[float] = []
        all_entropy: list[float] = []
        all_capital: list[float] = []
        total_stats: dict[str, int | float] = {}
        bs = self._config.batch_size

        for i in range(0, total_samples, bs):
            batch_ctx = contexts[i : i + bs]
            batch_closes = closes[i : i + bs]
            (
                states, actions, rewards,
                final_capital, final_margin, final_unrealized,
                batch_stats,
            ) = self._rollout(batch_ctx, batch_closes)

            # Compute MC returns in symlog space (no critic bootstrapping)
            sym_rewards = symlog(rewards)
            sym_returns = compute_mc_returns(sym_rewards, self._config.gamma)
            with torch.no_grad():
                flat_states = states.reshape(-1, states.shape[-1])
                sym_values = self._critic(flat_states).reshape(
                    states.shape[0], -1,
                )
            advantages = sym_returns - sym_values
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-8
            )

            # Re-run actor for gradient
            flat_states_grad = states.detach().reshape(-1, states.shape[-1])
            logits = self._actor(flat_states_grad)
            dist = Categorical(logits=logits)
            flat_actions = actions.reshape(-1)
            log_probs = dist.log_prob(flat_actions).reshape(states.shape[0], -1)  # type: ignore[no-untyped-call]
            entropies = dist.entropy().reshape(states.shape[0], -1)  # type: ignore[no-untyped-call]

            actor_loss = -(log_probs * advantages.detach()).mean()
            actor_loss = actor_loss - self._config.entropy_coef * entropies.mean()

            self._actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            clip_grad_norm_(self._actor.parameters(), self._config.grad_clip)
            self._actor_opt.step()

            # Re-run critic for gradient (predicts in symlog space)
            v_pred = self._critic(flat_states_grad)
            v_pred = v_pred.reshape(states.shape[0], -1)
            critic_loss = f.huber_loss(v_pred, sym_returns.detach())

            self._critic_opt.zero_grad(set_to_none=True)
            critic_loss.backward()  # type: ignore[no-untyped-call]
            clip_grad_norm_(self._critic.parameters(), self._config.grad_clip)
            self._critic_opt.step()

            all_actor_loss.append(actor_loss.item())
            all_critic_loss.append(critic_loss.item())
            all_rewards.append(rewards.sum(dim=1).mean().item())
            all_returns.append(sym_returns.mean().item())
            all_entropy.append(entropies.mean().item())
            effective_capital = (
                final_capital + final_margin + final_unrealized
            )
            all_capital.append(effective_capital.mean().item())
            for k, v in batch_stats.items():
                total_stats[k] = total_stats.get(k, 0) + v

        def _avg(xs: list[float]) -> float:
            return sum(xs) / len(xs) if xs else float("nan")

        return PolicyEpochMetrics(
            epoch=epoch + 1,
            actor_loss=_avg(all_actor_loss),
            critic_loss=_avg(all_critic_loss),
            mean_reward=_avg(all_rewards),
            mean_return=_avg(all_returns),
            entropy=_avg(all_entropy),
            mean_capital=_avg(all_capital),
            position_changes=int(total_stats.get("position_changes", 0)),
            long_steps=int(total_stats.get("long_steps", 0)),
            short_steps=int(total_stats.get("short_steps", 0)),
            flat_steps=int(total_stats.get("flat_steps", 0)),
        )

    def train(self, output_dir: Path) -> list[PolicyEpochMetrics]:
        """Full training loop with logging and checkpoint saving."""
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "policy_metrics.csv"
        history: list[PolicyEpochMetrics] = []

        fields = [
            "epoch", "actor_loss", "critic_loss",
            "mean_reward", "mean_return", "entropy", "mean_capital",
            "position_changes", "long_steps", "short_steps", "flat_steps",
        ]
        with metrics_path.open("w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=fields).writeheader()

        for epoch in range(self._config.epochs):
            m = self.train_epoch(epoch)
            history.append(m)

            with metrics_path.open("a", newline="") as fh:
                csv.DictWriter(fh, fieldnames=fields).writerow(asdict(m))

            logger.info(
                "policy_epoch",
                epoch=m.epoch,
                actor_loss=f"{m.actor_loss:.4f}",
                critic_loss=f"{m.critic_loss:.4f}",
                mean_reward=f"{m.mean_reward:.4f}",
                mean_capital=f"{m.mean_capital:.0f}",
                entropy=f"{m.entropy:.4f}",
                pos_chg=m.position_changes,
                long=m.long_steps,
                short=m.short_steps,
                flat=m.flat_steps,
            )

        _save_policy_checkpoint(
            output_dir / "policy_final.pt",
            actor=self._actor,
            critic=self._critic,
            actor_opt=self._actor_opt,
            critic_opt=self._critic_opt,
            epoch=self._config.epochs,
            config=self._config,
        )

        return history


def _save_policy_checkpoint(
    path: Path,
    *,
    actor: Actor,
    critic: Critic,
    actor_opt: Adam,
    critic_opt: Adam,
    epoch: int,
    config: PolicyConfig,
) -> None:
    """Save policy checkpoint atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "actor_state": actor.state_dict(),
            "critic_state": critic.state_dict(),
            "actor_optimizer_state": actor_opt.state_dict(),
            "critic_optimizer_state": critic_opt.state_dict(),
            "epoch": epoch,
            "policy_config": asdict(config),
        },
        tmp,
    )
    tmp.replace(path)
