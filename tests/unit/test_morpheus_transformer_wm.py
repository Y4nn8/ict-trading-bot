"""Tests for Morpheus transformer world model."""

from __future__ import annotations

import pytest
import torch

from src.morpheus.transformer_wm import (
    CausalSelfAttention,
    RotaryPositionalEncoding,
    TransformerBlock,
    TransformerWorldModel,
)
from src.morpheus.world_model import WorldModelOutput

BATCH = 4
SEQ_LEN = 16
OBS_DIM = 16
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
D_FF = 128
MAX_SEQ_LEN = 64


@pytest.fixture
def obs_seq() -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(BATCH, SEQ_LEN, OBS_DIM)


@pytest.fixture
def model() -> TransformerWorldModel:
    torch.manual_seed(0)
    return TransformerWorldModel(
        obs_dim=OBS_DIM, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, d_ff=D_FF, dropout=0.0,
        max_seq_len=MAX_SEQ_LEN,
    )


class TestRotaryPositionalEncoding:
    def test_output_shape(self) -> None:
        rope = RotaryPositionalEncoding(dim=16, max_seq_len=32)
        q = torch.randn(BATCH, 10, N_HEADS, 16)
        k = torch.randn(BATCH, 10, N_HEADS, 16)
        q_rot, k_rot = rope(q, k)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_different_positions_get_different_embeddings(self) -> None:
        rope = RotaryPositionalEncoding(dim=16, max_seq_len=32)
        q = torch.ones(1, 2, 1, 16)
        k = torch.ones(1, 2, 1, 16)
        q_rot, _ = rope(q, k)
        assert not torch.allclose(q_rot[0, 0], q_rot[0, 1])

    def test_deterministic(self) -> None:
        rope = RotaryPositionalEncoding(dim=16, max_seq_len=32)
        q = torch.randn(BATCH, 5, N_HEADS, 16)
        k = torch.randn(BATCH, 5, N_HEADS, 16)
        q1, k1 = rope(q, k)
        q2, k2 = rope(q, k)
        torch.testing.assert_close(q1, q2)
        torch.testing.assert_close(k1, k2)


class TestCausalSelfAttention:
    def test_output_shape(self) -> None:
        attn = CausalSelfAttention(
            d_model=D_MODEL, n_heads=N_HEADS, dropout=0.0,
            max_seq_len=MAX_SEQ_LEN,
        )
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        out = attn(x)
        assert out.shape == x.shape

    def test_causality(self) -> None:
        """Changing a future token should not affect earlier outputs."""
        attn = CausalSelfAttention(
            d_model=D_MODEL, n_heads=N_HEADS, dropout=0.0,
            max_seq_len=MAX_SEQ_LEN,
        )
        attn.eval()
        x = torch.randn(1, 8, D_MODEL)
        out1 = attn(x)

        x_modified = x.clone()
        x_modified[0, 5:] = torch.randn(3, D_MODEL)
        out2 = attn(x_modified)

        torch.testing.assert_close(out1[0, :5], out2[0, :5])

    def test_rejects_bad_dims(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            CausalSelfAttention(d_model=10, n_heads=3)

    def test_gradient_flow(self) -> None:
        attn = CausalSelfAttention(
            d_model=D_MODEL, n_heads=N_HEADS, dropout=0.0,
            max_seq_len=MAX_SEQ_LEN,
        )
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None


class TestTransformerBlock:
    def test_output_shape(self) -> None:
        block = TransformerBlock(
            d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF,
            dropout=0.0, max_seq_len=MAX_SEQ_LEN,
        )
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        out = block(x)
        assert out.shape == x.shape

    def test_residual_connection(self) -> None:
        block = TransformerBlock(
            d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF,
            dropout=0.0, max_seq_len=MAX_SEQ_LEN,
        )
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        out = block(x)
        assert not torch.allclose(out, x)
        assert not torch.allclose(out, torch.zeros_like(out))


class TestTransformerWorldModel:
    def test_forward_output_type(
        self, model: TransformerWorldModel, obs_seq: torch.Tensor,
    ) -> None:
        out = model(obs_seq)
        assert isinstance(out, WorldModelOutput)

    def test_forward_loss_scalar(
        self, model: TransformerWorldModel, obs_seq: torch.Tensor,
    ) -> None:
        out = model(obs_seq)
        assert out.loss.shape == ()
        assert out.recon_loss.shape == ()
        assert out.kl_loss.shape == ()

    def test_forward_loss_positive(
        self, model: TransformerWorldModel, obs_seq: torch.Tensor,
    ) -> None:
        out = model(obs_seq)
        assert out.loss.item() > 0
        assert out.recon_loss.item() > 0

    def test_forward_kl_zero(
        self, model: TransformerWorldModel, obs_seq: torch.Tensor,
    ) -> None:
        out = model(obs_seq)
        assert out.kl_loss.item() == 0.0

    def test_forward_reconstructed_shape(
        self, model: TransformerWorldModel, obs_seq: torch.Tensor,
    ) -> None:
        out = model(obs_seq)
        assert out.reconstructed.shape == obs_seq.shape

    def test_forward_gradient_flow(
        self, model: TransformerWorldModel, obs_seq: torch.Tensor,
    ) -> None:
        out = model(obs_seq)
        out.loss.backward()
        grad_count = sum(
            1 for p in model.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        total = sum(1 for p in model.parameters())
        assert grad_count == total

    def test_forward_no_nan(
        self, model: TransformerWorldModel, obs_seq: torch.Tensor,
    ) -> None:
        out = model(obs_seq)
        assert not torch.isnan(out.loss)
        assert not torch.any(torch.isnan(out.reconstructed))

    def test_imagine_shape(
        self, model: TransformerWorldModel, obs_seq: torch.Tensor,
    ) -> None:
        horizon = 8
        pred = model.imagine(obs_seq, horizon)
        assert pred.shape == (BATCH, horizon, OBS_DIM)

    def test_imagine_no_nan(
        self, model: TransformerWorldModel, obs_seq: torch.Tensor,
    ) -> None:
        pred = model.imagine(obs_seq, 5)
        assert not torch.any(torch.isnan(pred))

    def test_imagine_zero_horizon(
        self, model: TransformerWorldModel, obs_seq: torch.Tensor,
    ) -> None:
        pred = model.imagine(obs_seq, 0)
        assert pred.shape[1] == 0

    def test_imagine_deterministic_with_seed(
        self, model: TransformerWorldModel, obs_seq: torch.Tensor,
    ) -> None:
        torch.manual_seed(123)
        pred1 = model.imagine(obs_seq, 5)
        torch.manual_seed(123)
        pred2 = model.imagine(obs_seq, 5)
        torch.testing.assert_close(pred1, pred2)

    def test_imagine_stochastic_without_seed(
        self, model: TransformerWorldModel, obs_seq: torch.Tensor,
    ) -> None:
        pred1 = model.imagine(obs_seq, 5)
        pred2 = model.imagine(obs_seq, 5)
        assert not torch.allclose(pred1, pred2)

    def test_param_count_in_range(
        self, model: TransformerWorldModel,
    ) -> None:
        count = model.param_count()
        assert 10_000 < count < 10_000_000, f"Unexpected param count: {count}"

    def test_default_construction(self) -> None:
        m = TransformerWorldModel()
        obs = torch.randn(2, 8, 16)
        out = m(obs)
        assert out.loss.shape == ()

    def test_causality_in_forward(
        self, model: TransformerWorldModel,
    ) -> None:
        """Changing future inputs should not affect earlier reconstructions."""
        model.eval()
        x = torch.randn(1, 10, OBS_DIM)
        out1 = model(x)

        x_mod = x.clone()
        x_mod[0, 7:] = torch.randn(3, OBS_DIM)
        out2 = model(x_mod)

        torch.testing.assert_close(
            out1.reconstructed[0, :7], out2.reconstructed[0, :7],
        )


class TestTransformerTrainingCompat:
    """Verify transformer works with the training loop."""

    def test_train_epoch_runs(self, model: TransformerWorldModel) -> None:
        from torch.optim import Adam
        from torch.utils.data import DataLoader, TensorDataset

        from src.morpheus.training import train_epoch

        data = torch.randn(32, SEQ_LEN, OBS_DIM)
        loader: DataLoader[torch.Tensor] = DataLoader(
            TensorDataset(data), batch_size=8, drop_last=True,
        )
        # TensorDataset returns tuples; adapt loader
        optimizer = Adam(model.parameters(), lr=1e-3)

        class _Wrapper:
            """Wrap TensorDataset loader to yield plain tensors."""

            def __init__(self, dl: DataLoader[torch.Tensor]) -> None:
                self._dl = dl

            def __len__(self) -> int:
                return len(self._dl)

            def __iter__(self):
                for (batch,) in self._dl:
                    yield batch

        avg_loss, _avg_recon, avg_kl = train_epoch(
            model, _Wrapper(loader), optimizer,
            grad_clip=1.0, device=torch.device("cpu"),
        )
        assert avg_loss > 0
        assert avg_kl == 0.0

    def test_evaluate_runs(self, model: TransformerWorldModel) -> None:
        from torch.utils.data import DataLoader, TensorDataset

        from src.morpheus.training import evaluate

        data = torch.randn(16, SEQ_LEN, OBS_DIM)
        loader: DataLoader[torch.Tensor] = DataLoader(
            TensorDataset(data), batch_size=8,
        )

        class _Wrapper:
            def __init__(self, dl: DataLoader[torch.Tensor]) -> None:
                self._dl = dl

            def __len__(self) -> int:
                return len(self._dl)

            def __iter__(self):
                for (batch,) in self._dl:
                    yield batch

        avg_loss, _, avg_kl = evaluate(
            model, _Wrapper(loader), device=torch.device("cpu"),
        )
        assert avg_loss > 0
        assert avg_kl == 0.0


class TestBuildModel:
    def test_build_rssm(self) -> None:
        from src.morpheus.training import TrainConfig, build_model
        from src.morpheus.world_model import WorldModel

        config = TrainConfig(model_type="rssm", obs_dim=10)
        m = build_model(config)
        assert isinstance(m, WorldModel)

    def test_build_transformer(self) -> None:
        from src.morpheus.training import TrainConfig, build_model

        config = TrainConfig(
            model_type="transformer", obs_dim=16,
            d_model=64, n_heads=4, n_layers=2, d_ff=128,
        )
        m = build_model(config)
        assert isinstance(m, TransformerWorldModel)
