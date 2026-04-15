"""Morpheus world model — Dreamer-style assembly with ELBO loss.

Connects encoder, RSSM, and decoder.  Provides:
  - forward():  training loss (reconstruction + KL divergence)
  - imagine():  generate future trajectories from encoded context
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.distributions import Normal, kl_divergence

from src.morpheus.decoder import ObservationDecoder
from src.morpheus.encoder import ObservationEncoder
from src.morpheus.rssm import RSSM, GaussianParams


@dataclass(frozen=True)
class WorldModelOutput:
    """Training output bundle."""

    loss: Tensor
    recon_loss: Tensor
    kl_loss: Tensor
    reconstructed: Tensor  # (batch, seq_len, obs_dim)


def kl_divergence_gaussian(
    post: GaussianParams,
    prior: GaussianParams,
) -> Tensor:
    """KL divergence between two diagonal Gaussians via torch.distributions.

    Args:
        post: Posterior distribution parameters.
        prior: Prior distribution parameters.

    Returns:
        KL divergence per element, same shape as mu/std tensors.
    """
    q = Normal(post.mu, post.std)
    p = Normal(prior.mu, prior.std)
    return kl_divergence(q, p)


class WorldModel(nn.Module):
    """Dreamer-style world model.

    Args:
        obs_dim: Observation dimensionality.
        embed_dim: Encoder output / RSSM embedding dimensionality.
        det_dim: RSSM deterministic state (GRU hidden) dimensionality.
        stoch_dim: RSSM stochastic latent dimensionality.
        hidden_dim: MLP hidden layer width (shared across components).
        kl_weight: Coefficient for the KL term in the ELBO loss (beta).
        free_nats: Minimum KL per timestep — prevents posterior collapse.
    """

    def __init__(
        self,
        obs_dim: int = 10,
        embed_dim: int = 64,
        det_dim: int = 256,
        stoch_dim: int = 64,
        hidden_dim: int = 128,
        kl_weight: float = 1.0,
        free_nats: float = 1.0,
    ) -> None:
        super().__init__()
        self.kl_weight = kl_weight
        self.free_nats = free_nats

        self.encoder = ObservationEncoder(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
        )
        self.rssm = RSSM(
            embed_dim=embed_dim,
            det_dim=det_dim,
            stoch_dim=stoch_dim,
            hidden_dim=hidden_dim,
        )
        self.decoder = ObservationDecoder(
            det_dim=det_dim,
            stoch_dim=stoch_dim,
            hidden_dim=hidden_dim,
            obs_dim=obs_dim,
        )

    def forward(self, obs_seq: Tensor) -> WorldModelOutput:
        """Compute ELBO loss on a batch of observation sequences.

        Args:
            obs_seq: (batch, seq_len, obs_dim).

        Returns:
            WorldModelOutput with loss components and reconstructed observations.
        """
        batch = obs_seq.shape[0]
        device = obs_seq.device

        embeds = self.encoder(obs_seq)

        state = self.rssm.initial_state(batch, device=device)
        rssm_out = self.rssm.observe(embeds, state)

        reconstructed = self.decoder(rssm_out.h_seq, rssm_out.z_seq)

        recon_loss = (reconstructed - obs_seq).pow(2).mean()

        kl_per_element = kl_divergence_gaussian(rssm_out.posterior, rssm_out.prior)
        kl_per_step = kl_per_element.sum(dim=-1)
        kl_clamped = torch.clamp(kl_per_step, min=self.free_nats)
        kl_loss = kl_clamped.mean()

        loss = recon_loss + self.kl_weight * kl_loss

        return WorldModelOutput(
            loss=loss,
            recon_loss=recon_loss,
            kl_loss=kl_loss,
            reconstructed=reconstructed,
        )

    @torch.no_grad()
    def imagine(
        self,
        obs_context: Tensor,
        horizon: int,
    ) -> Tensor:
        """Generate imagined future trajectory from context observations.

        Encodes the context through the posterior, then rolls forward
        using only the prior (no observations).

        Args:
            obs_context: (batch, context_len, obs_dim) — real observations.
            horizon: Number of future steps to generate.

        Returns:
            Predicted observations (batch, horizon, obs_dim).
        """
        batch = obs_context.shape[0]
        device = obs_context.device

        embeds = self.encoder(obs_context)
        state = self.rssm.initial_state(batch, device=device)
        final_state = self.rssm.observe_final(embeds, state)

        h_imag, z_imag = self.rssm.imagine(final_state, horizon)

        return self.decoder(h_imag, z_imag)  # type: ignore[no-any-return]

    def param_count(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
