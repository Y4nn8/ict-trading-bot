"""Transformer world model — decoder-only, no latent, RoPE.

GPT-style causal transformer for next-step prediction on financial
time series.  Outputs a Gaussian distribution (mean + log_var per dim)
at each position.  No VAE/KL — uncertainty comes from sampling the
output distribution during imagination.

Provides the same interface as :class:`WorldModel`:
  - forward():  training loss (Gaussian NLL, kl_loss always 0)
  - imagine():  autoregressive generation from context
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as f
from torch import Tensor, nn

from src.morpheus.dataset import BASE_OBS_COLUMNS
from src.morpheus.world_model import WorldModelOutput

_RET_CLOSE_IDX = BASE_OBS_COLUMNS.index("ret_close")


def _apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary positional embedding to a tensor.

    Args:
        x: (batch, seq_len, n_heads, head_dim).
        cos: (seq_len, head_dim // 2).
        sin: (seq_len, head_dim // 2).
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    c = cos.unsqueeze(0).unsqueeze(2)
    s = sin.unsqueeze(0).unsqueeze(2)
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


class RotaryPositionalEncoding(nn.Module):
    """Rotary Positional Encoding (RoPE).

    Precomputes sin/cos tables up to ``max_seq_len`` and applies
    rotary embedding to query and key tensors.

    Args:
        dim: Head dimension (must be even).
        max_seq_len: Maximum sequence length to precompute.
        base: Frequency base for the positional encoding.
    """

    def __init__(
        self, dim: int, max_seq_len: int = 1024, base: float = 10000.0,
    ) -> None:
        super().__init__()
        if dim % 2 != 0:
            msg = f"RotaryPositionalEncoding dim must be even, got {dim}"
            raise ValueError(msg)
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())

    cos_cached: Tensor
    sin_cached: Tensor

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        """Apply RoPE to query and key.

        Args:
            q: (batch, seq_len, n_heads, head_dim).
            k: (batch, seq_len, n_heads, head_dim).

        Returns:
            Tuple of rotated (q, k) with same shape.
        """
        seq_len = q.shape[1]
        if seq_len > self.max_seq_len:
            msg = (
                f"Sequence length {seq_len} exceeds max_seq_len "
                f"{self.max_seq_len}. Increase max_seq_len."
            )
            raise ValueError(msg)
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        return _apply_rotary(q, cos, sin), _apply_rotary(k, cos, sin)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        dropout: Attention dropout probability.
        max_seq_len: Maximum sequence length for RoPE precomputation.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            msg = f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            raise ValueError(msg)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = dropout
        self.rope = RotaryPositionalEncoding(self.head_dim, max_seq_len)

    def forward(self, x: Tensor) -> Tensor:
        """Apply causal self-attention.

        Args:
            x: (batch, seq_len, d_model).

        Returns:
            (batch, seq_len, d_model).
        """
        b, t, c = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k = self.rope(q, k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        drop = self.attn_dropout if self.training else 0.0
        out = f.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop)
        return self.out_proj(out.transpose(1, 2).reshape(b, t, c))  # type: ignore[no-any-return]


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LN -> Attention -> LN -> FFN.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        d_ff: Feed-forward hidden dimension.
        dropout: Dropout probability.
        max_seq_len: Maximum sequence length for RoPE.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply transformer block.

        Args:
            x: (batch, seq_len, d_model).

        Returns:
            (batch, seq_len, d_model).
        """
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerWorldModel(nn.Module):
    """Decoder-only transformer world model.

    Predicts the next observation from all previous observations using
    causal attention.  Output is a Gaussian distribution (mean + log_var)
    per dimension at each position.  No stochastic latent variable.

    Args:
        obs_dim: Observation dimensionality.
        d_model: Transformer hidden dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer blocks.
        d_ff: Feed-forward hidden dimension.
        dropout: Dropout probability.
        max_seq_len: Maximum sequence length for RoPE precomputation.
    """

    def __init__(
        self,
        obs_dim: int = 16,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        aux_horizon: int = 0,
        aux_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.max_seq_len = max_seq_len
        self.aux_horizon = aux_horizon
        self.aux_weight = aux_weight
        self.input_proj = nn.Linear(obs_dim, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, max_seq_len)
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, obs_dim * 2)
        self.dir_head = nn.Linear(d_model, 1) if aux_horizon > 0 else None

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _run_backbone(self, obs_seq: Tensor) -> Tensor:
        """Run input through projection + transformer blocks + final LN.

        Args:
            obs_seq: (batch, seq_len, obs_dim).

        Returns:
            (batch, seq_len, d_model).
        """
        x = self.input_proj(obs_seq)
        for block in self.blocks:
            x = block(x)
        return self.ln_final(x)  # type: ignore[no-any-return]

    def forward(self, obs_seq: Tensor) -> WorldModelOutput:
        """Compute next-step prediction loss (Gaussian NLL).

        Uses teacher forcing: position t predicts observation at t+1.
        Loss is computed over positions 0..T-2 predicting targets 1..T-1.

        Args:
            obs_seq: (batch, seq_len, obs_dim).

        Returns:
            WorldModelOutput with NLL as loss, kl_loss=0.
        """
        if obs_seq.shape[1] < 2:
            msg = "obs_seq must have seq_len >= 2 for next-step prediction"
            raise ValueError(msg)

        h = self._run_backbone(obs_seq)
        params = self.output_head(h)
        mean, log_var = params.chunk(2, dim=-1)
        log_var = torch.clamp(log_var, min=-4.0, max=2.0)

        target = obs_seq[:, 1:]
        pred_mean = mean[:, :-1]
        pred_log_var = log_var[:, :-1]

        nll = 0.5 * (
            pred_log_var + (target - pred_mean).pow(2) / pred_log_var.exp()
            + math.log(2 * math.pi)
        )
        recon_loss = nll.mean()

        # Auxiliary directional loss
        aux_loss = torch.zeros(1, device=obs_seq.device, dtype=obs_seq.dtype).squeeze()
        kl_loss = torch.zeros(1, device=obs_seq.device, dtype=obs_seq.dtype).squeeze()
        seq_len = obs_seq.shape[1]
        if self.dir_head is not None and self.aux_horizon > 0 and seq_len > self.aux_horizon + 1:
            ret_close = obs_seq[:, :, _RET_CLOSE_IDX]
            cumsum = ret_close.cumsum(dim=1)
            future_sum = cumsum[:, self.aux_horizon :] - cumsum[:, : -self.aux_horizon]
            h_for_dir = h[:, : seq_len - self.aux_horizon]
            dir_logits = self.dir_head(h_for_dir).squeeze(-1)
            dir_targets = (future_sum > 0).float()
            aux_loss = f.binary_cross_entropy_with_logits(dir_logits, dir_targets)

        loss = recon_loss + self.aux_weight * aux_loss

        return WorldModelOutput(
            loss=loss,
            recon_loss=recon_loss,
            kl_loss=kl_loss,
            reconstructed=mean,
            aux_loss=aux_loss,
        )

    @torch.no_grad()
    def imagine(
        self,
        obs_context: Tensor,
        horizon: int,
    ) -> Tensor:
        """Generate imagined future trajectory from context.

        Autoregressively extends the context: at each step, feeds the
        full sequence through the model, samples from the output
        distribution at the last position, and appends.

        Args:
            obs_context: (batch, context_len, obs_dim).
            horizon: Number of future steps to generate.

        Returns:
            Predicted observations (batch, horizon, obs_dim).
        """
        if horizon == 0:
            return obs_context[:, :0]

        total_len = obs_context.shape[1] + horizon
        if total_len > self.max_seq_len:
            msg = (
                f"context_len ({obs_context.shape[1]}) + horizon ({horizon}) "
                f"= {total_len} exceeds max_seq_len ({self.max_seq_len})"
            )
            raise ValueError(msg)

        generated: list[Tensor] = []
        current = obs_context

        for _ in range(horizon):
            h = self._run_backbone(current)
            params = self.output_head(h[:, -1:])
            mean, log_var = params.chunk(2, dim=-1)
            log_var = torch.clamp(log_var, min=-4.0, max=2.0)
            std = (0.5 * log_var).exp()
            next_obs = mean + std * torch.randn_like(std)
            generated.append(next_obs[:, 0])
            current = torch.cat([current, next_obs], dim=1)

        return torch.stack(generated, dim=1)

    def param_count(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
