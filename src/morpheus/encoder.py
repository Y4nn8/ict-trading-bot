"""Observation encoder for Morpheus world model.

Maps raw observations (10 dims) to a dense embedding used by
the RSSM posterior to infer the stochastic latent state.
"""

from __future__ import annotations

from torch import Tensor, nn


class ObservationEncoder(nn.Module):
    """MLP encoder: observation vector -> dense embedding.

    Args:
        obs_dim: Dimensionality of the observation vector.
        hidden_dim: Width of the hidden layer.
        embed_dim: Output embedding dimensionality.
    """

    def __init__(
        self,
        obs_dim: int = 10,
        hidden_dim: int = 128,
        embed_dim: int = 64,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, obs: Tensor) -> Tensor:
        """Encode observations.

        Args:
            obs: Tensor of shape (..., obs_dim).

        Returns:
            Embedding of shape (..., embed_dim).
        """
        return self.net(obs)  # type: ignore[no-any-return]
