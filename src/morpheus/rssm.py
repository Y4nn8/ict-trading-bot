"""Recurrent State-Space Model (RSSM) for Morpheus world model.

Dreamer-style RSSM with a GRU deterministic path and Gaussian
stochastic latent.  No action conditioning — the market evolves
independently of our trades.

State at each timestep:
    h_t  (deterministic) = GRU(h_{t-1}, z_{t-1})
    z_t  (stochastic)    ~ q(z_t | h_t, e_t)   [posterior, training]
                         ~ p(z_t | h_t)         [prior, imagination]
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as f
from torch import Tensor, nn


@dataclass(frozen=True)
class RSSMState:
    """Deterministic + stochastic state pair."""

    h: Tensor  # (batch, det_dim)
    z: Tensor  # (batch, stoch_dim)


@dataclass(frozen=True)
class GaussianParams:
    """Mean and standard deviation of a diagonal Gaussian."""

    mu: Tensor  # (batch, [seq_len,] stoch_dim)
    std: Tensor  # (batch, [seq_len,] stoch_dim)


@dataclass(frozen=True)
class RSSMOutput:
    """Full RSSM output for a sequence processed with observations."""

    h_seq: Tensor  # (batch, seq_len, det_dim)
    z_seq: Tensor  # (batch, seq_len, stoch_dim)
    prior: GaussianParams
    posterior: GaussianParams


MIN_STD = 0.1


class RSSM(nn.Module):
    """Recurrent State-Space Model.

    Args:
        embed_dim: Dimensionality of observation embeddings (encoder output).
        det_dim: GRU hidden state dimensionality (deterministic path).
        stoch_dim: Gaussian latent dimensionality (stochastic path).
        hidden_dim: Width of prior/posterior MLP hidden layers.
        min_std: Floor on Gaussian std to prevent collapse.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        det_dim: int = 256,
        stoch_dim: int = 64,
        hidden_dim: int = 128,
        min_std: float = MIN_STD,
    ) -> None:
        super().__init__()
        self.det_dim = det_dim
        self.stoch_dim = stoch_dim
        self.min_std = min_std

        self.gru = nn.GRUCell(stoch_dim, det_dim)

        self.prior_net = nn.Sequential(
            nn.Linear(det_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * stoch_dim),
        )

        self.posterior_net = nn.Sequential(
            nn.Linear(det_dim + embed_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * stoch_dim),
        )

    def initial_state(self, batch_size: int, *, device: torch.device | None = None) -> RSSMState:
        """Return zero-initialised state."""
        h = torch.zeros(batch_size, self.det_dim, device=device)
        z = torch.zeros(batch_size, self.stoch_dim, device=device)
        return RSSMState(h=h, z=z)

    def prior(self, h: Tensor) -> GaussianParams:
        """Compute prior p(z_t | h_t)."""
        out = self.prior_net(h)
        mu, raw_std = out.chunk(2, dim=-1)
        std = f.softplus(raw_std) + self.min_std
        return GaussianParams(mu=mu, std=std)

    def posterior(self, h: Tensor, embed: Tensor) -> GaussianParams:
        """Compute posterior q(z_t | h_t, e_t)."""
        out = self.posterior_net(torch.cat([h, embed], dim=-1))
        mu, raw_std = out.chunk(2, dim=-1)
        std = f.softplus(raw_std) + self.min_std
        return GaussianParams(mu=mu, std=std)

    def observe(
        self,
        embeds: Tensor,
        state: RSSMState,
    ) -> RSSMOutput:
        """Process a sequence of observation embeddings (training path).

        Uses the posterior at each step to sample z_t.

        Args:
            embeds: (batch, seq_len, embed_dim) — encoder output.
            state: Initial RSSM state.

        Returns:
            RSSMOutput with sequences of h, z, prior params, posterior params.
        """
        _, seq_len, _ = embeds.shape
        h, z = state.h, state.z

        h_list: list[Tensor] = []
        z_list: list[Tensor] = []
        prior_mu_list: list[Tensor] = []
        prior_std_list: list[Tensor] = []
        post_mu_list: list[Tensor] = []
        post_std_list: list[Tensor] = []

        for t in range(seq_len):
            h = self.gru(z, h)

            pri = self.prior(h)
            post = self.posterior(h, embeds[:, t])

            z = post.mu + post.std * torch.randn_like(post.std)

            h_list.append(h)
            z_list.append(z)
            prior_mu_list.append(pri.mu)
            prior_std_list.append(pri.std)
            post_mu_list.append(post.mu)
            post_std_list.append(post.std)

        return RSSMOutput(
            h_seq=torch.stack(h_list, dim=1),
            z_seq=torch.stack(z_list, dim=1),
            prior=GaussianParams(
                mu=torch.stack(prior_mu_list, dim=1),
                std=torch.stack(prior_std_list, dim=1),
            ),
            posterior=GaussianParams(
                mu=torch.stack(post_mu_list, dim=1),
                std=torch.stack(post_std_list, dim=1),
            ),
        )

    def imagine(self, state: RSSMState, horizon: int) -> tuple[Tensor, Tensor]:
        """Roll forward using prior only (no observations).

        Args:
            state: Starting RSSM state.
            horizon: Number of steps to imagine.

        Returns:
            h_seq: (batch, horizon, det_dim)
            z_seq: (batch, horizon, stoch_dim)
        """
        h, z = state.h, state.z
        h_list: list[Tensor] = []
        z_list: list[Tensor] = []

        for _ in range(horizon):
            h = self.gru(z, h)
            pri = self.prior(h)
            z = pri.mu + pri.std * torch.randn_like(pri.std)
            h_list.append(h)
            z_list.append(z)

        return torch.stack(h_list, dim=1), torch.stack(z_list, dim=1)
