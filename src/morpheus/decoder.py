"""Observation decoder for Morpheus world model.

Reconstructs the predicted observation from the RSSM latent state
(deterministic h + stochastic z).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ObservationDecoder(nn.Module):
    """MLP decoder: concat(h, z) -> predicted observation.

    Args:
        det_dim: Dimensionality of the deterministic state (GRU hidden).
        stoch_dim: Dimensionality of the stochastic latent.
        hidden_dim: Width of the hidden layer.
        obs_dim: Output observation dimensionality.
    """

    def __init__(
        self,
        det_dim: int = 256,
        stoch_dim: int = 64,
        hidden_dim: int = 128,
        obs_dim: int = 10,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(det_dim + stoch_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def forward(self, h: Tensor, z: Tensor) -> Tensor:
        """Decode latent state to predicted observation.

        Args:
            h: Deterministic state (..., det_dim).
            z: Stochastic latent (..., stoch_dim).

        Returns:
            Predicted observation (..., obs_dim).
        """
        return self.net(torch.cat([h, z], dim=-1))  # type: ignore[no-any-return]
