"""Tests for Morpheus world model components (encoder, RSSM, decoder, assembly)."""

from __future__ import annotations

import pytest
import torch

from src.morpheus.decoder import ObservationDecoder
from src.morpheus.encoder import ObservationEncoder
from src.morpheus.rssm import RSSM, GaussianParams
from src.morpheus.world_model import WorldModel, WorldModelOutput, kl_divergence_gaussian

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BATCH = 4
SEQ_LEN = 16
OBS_DIM = 10
EMBED_DIM = 64
DET_DIM = 256
STOCH_DIM = 64
HIDDEN_DIM = 128


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def obs_seq() -> torch.Tensor:
    """Random observation sequence."""
    torch.manual_seed(42)
    return torch.randn(BATCH, SEQ_LEN, OBS_DIM)


@pytest.fixture
def encoder() -> ObservationEncoder:
    return ObservationEncoder(
        obs_dim=OBS_DIM, hidden_dim=HIDDEN_DIM, embed_dim=EMBED_DIM,
    )


@pytest.fixture
def rssm() -> RSSM:
    return RSSM(
        embed_dim=EMBED_DIM, det_dim=DET_DIM,
        stoch_dim=STOCH_DIM, hidden_dim=HIDDEN_DIM,
    )


@pytest.fixture
def decoder() -> ObservationDecoder:
    return ObservationDecoder(
        det_dim=DET_DIM, stoch_dim=STOCH_DIM,
        hidden_dim=HIDDEN_DIM, obs_dim=OBS_DIM,
    )


@pytest.fixture
def world_model() -> WorldModel:
    return WorldModel(
        obs_dim=OBS_DIM, embed_dim=EMBED_DIM, det_dim=DET_DIM,
        stoch_dim=STOCH_DIM, hidden_dim=HIDDEN_DIM,
    )


# ---------------------------------------------------------------------------
# ObservationEncoder tests
# ---------------------------------------------------------------------------

class TestObservationEncoder:
    def test_output_shape(self, encoder: ObservationEncoder) -> None:
        obs = torch.randn(BATCH, OBS_DIM)
        out = encoder(obs)
        assert out.shape == (BATCH, EMBED_DIM)

    def test_batch_sequence_shape(self, encoder: ObservationEncoder) -> None:
        obs = torch.randn(BATCH, SEQ_LEN, OBS_DIM)
        out = encoder(obs)
        assert out.shape == (BATCH, SEQ_LEN, EMBED_DIM)

    def test_gradient_flow(self, encoder: ObservationEncoder) -> None:
        obs = torch.randn(BATCH, OBS_DIM, requires_grad=True)
        out = encoder(obs)
        out.sum().backward()
        assert obs.grad is not None
        assert obs.grad.shape == (BATCH, OBS_DIM)

    def test_default_dims(self) -> None:
        enc = ObservationEncoder()
        obs = torch.randn(2, 10)
        out = enc(obs)
        assert out.shape == (2, 64)


# ---------------------------------------------------------------------------
# RSSM tests
# ---------------------------------------------------------------------------

class TestRSSM:
    def test_initial_state(self, rssm: RSSM) -> None:
        state = rssm.initial_state(BATCH)
        assert state.h.shape == (BATCH, DET_DIM)
        assert state.z.shape == (BATCH, STOCH_DIM)
        assert torch.all(state.h == 0)
        assert torch.all(state.z == 0)

    def test_prior_shape_and_positivity(self, rssm: RSSM) -> None:
        h = torch.randn(BATCH, DET_DIM)
        params = rssm.prior(h)
        assert params.mu.shape == (BATCH, STOCH_DIM)
        assert params.std.shape == (BATCH, STOCH_DIM)
        assert torch.all(params.std > 0)

    def test_posterior_shape_and_positivity(self, rssm: RSSM) -> None:
        h = torch.randn(BATCH, DET_DIM)
        embed = torch.randn(BATCH, EMBED_DIM)
        params = rssm.posterior(h, embed)
        assert params.mu.shape == (BATCH, STOCH_DIM)
        assert params.std.shape == (BATCH, STOCH_DIM)
        assert torch.all(params.std > 0)

    def test_min_std_enforced(self, rssm: RSSM) -> None:
        h = torch.zeros(BATCH, DET_DIM)
        params = rssm.prior(h)
        assert torch.all(params.std >= rssm.min_std)

    def test_observe_shapes(self, rssm: RSSM) -> None:
        embeds = torch.randn(BATCH, SEQ_LEN, EMBED_DIM)
        state = rssm.initial_state(BATCH)
        out = rssm.observe(embeds, state)

        assert out.h_seq.shape == (BATCH, SEQ_LEN, DET_DIM)
        assert out.z_seq.shape == (BATCH, SEQ_LEN, STOCH_DIM)
        assert out.prior.mu.shape == (BATCH, SEQ_LEN, STOCH_DIM)
        assert out.prior.std.shape == (BATCH, SEQ_LEN, STOCH_DIM)
        assert out.posterior.mu.shape == (BATCH, SEQ_LEN, STOCH_DIM)
        assert out.posterior.std.shape == (BATCH, SEQ_LEN, STOCH_DIM)

    def test_observe_gradient_flow(self, rssm: RSSM) -> None:
        embeds = torch.randn(BATCH, SEQ_LEN, EMBED_DIM, requires_grad=True)
        state = rssm.initial_state(BATCH)
        out = rssm.observe(embeds, state)
        out.z_seq.sum().backward()
        assert embeds.grad is not None

    def test_observe_no_nan(self, rssm: RSSM) -> None:
        embeds = torch.randn(BATCH, SEQ_LEN, EMBED_DIM)
        state = rssm.initial_state(BATCH)
        out = rssm.observe(embeds, state)
        assert not torch.any(torch.isnan(out.h_seq))
        assert not torch.any(torch.isnan(out.z_seq))

    def test_imagine_shapes(self, rssm: RSSM) -> None:
        state = rssm.initial_state(BATCH)
        horizon = 20
        h_imag, z_imag = rssm.imagine(state, horizon)
        assert h_imag.shape == (BATCH, horizon, DET_DIM)
        assert z_imag.shape == (BATCH, horizon, STOCH_DIM)

    def test_imagine_no_nan(self, rssm: RSSM) -> None:
        state = rssm.initial_state(BATCH)
        h_imag, z_imag = rssm.imagine(state, 10)
        assert not torch.any(torch.isnan(h_imag))
        assert not torch.any(torch.isnan(z_imag))

    def test_imagine_deterministic_with_seed(self, rssm: RSSM) -> None:
        state = rssm.initial_state(BATCH)
        torch.manual_seed(123)
        h1, z1 = rssm.imagine(state, 10)
        torch.manual_seed(123)
        h2, z2 = rssm.imagine(state, 10)
        torch.testing.assert_close(h1, h2)
        torch.testing.assert_close(z1, z2)


# ---------------------------------------------------------------------------
# ObservationDecoder tests
# ---------------------------------------------------------------------------

class TestObservationDecoder:
    def test_output_shape(self, decoder: ObservationDecoder) -> None:
        h = torch.randn(BATCH, DET_DIM)
        z = torch.randn(BATCH, STOCH_DIM)
        out = decoder(h, z)
        assert out.shape == (BATCH, OBS_DIM)

    def test_sequence_shape(self, decoder: ObservationDecoder) -> None:
        h = torch.randn(BATCH, SEQ_LEN, DET_DIM)
        z = torch.randn(BATCH, SEQ_LEN, STOCH_DIM)
        out = decoder(h, z)
        assert out.shape == (BATCH, SEQ_LEN, OBS_DIM)

    def test_gradient_flow(self, decoder: ObservationDecoder) -> None:
        h = torch.randn(BATCH, DET_DIM, requires_grad=True)
        z = torch.randn(BATCH, STOCH_DIM, requires_grad=True)
        out = decoder(h, z)
        out.sum().backward()
        assert h.grad is not None
        assert z.grad is not None

    def test_default_dims(self) -> None:
        dec = ObservationDecoder()
        h = torch.randn(2, 256)
        z = torch.randn(2, 64)
        out = dec(h, z)
        assert out.shape == (2, 10)


# ---------------------------------------------------------------------------
# KL divergence tests
# ---------------------------------------------------------------------------

class TestKLDivergence:
    def test_identical_distributions_zero_kl(self) -> None:
        params = GaussianParams(
            mu=torch.zeros(BATCH, STOCH_DIM),
            std=torch.ones(BATCH, STOCH_DIM),
        )
        kl = kl_divergence_gaussian(params, params)
        torch.testing.assert_close(kl, torch.zeros_like(kl), atol=1e-6, rtol=0)

    def test_known_kl_value(self) -> None:
        post = GaussianParams(
            mu=torch.tensor([[1.0]]),
            std=torch.tensor([[1.0]]),
        )
        prior = GaussianParams(
            mu=torch.tensor([[0.0]]),
            std=torch.tensor([[1.0]]),
        )
        kl = kl_divergence_gaussian(post, prior)
        expected = torch.tensor([[0.5]])
        torch.testing.assert_close(kl, expected, atol=1e-6, rtol=0)

    def test_kl_non_negative(self) -> None:
        torch.manual_seed(42)
        post = GaussianParams(
            mu=torch.randn(BATCH, STOCH_DIM),
            std=torch.rand(BATCH, STOCH_DIM) + 0.1,
        )
        prior = GaussianParams(
            mu=torch.randn(BATCH, STOCH_DIM),
            std=torch.rand(BATCH, STOCH_DIM) + 0.1,
        )
        kl = kl_divergence_gaussian(post, prior)
        assert torch.all(kl >= -1e-6)

    def test_kl_shape(self) -> None:
        post = GaussianParams(
            mu=torch.randn(BATCH, SEQ_LEN, STOCH_DIM),
            std=torch.rand(BATCH, SEQ_LEN, STOCH_DIM) + 0.1,
        )
        prior = GaussianParams(
            mu=torch.randn(BATCH, SEQ_LEN, STOCH_DIM),
            std=torch.rand(BATCH, SEQ_LEN, STOCH_DIM) + 0.1,
        )
        kl = kl_divergence_gaussian(post, prior)
        assert kl.shape == (BATCH, SEQ_LEN, STOCH_DIM)


# ---------------------------------------------------------------------------
# WorldModel tests
# ---------------------------------------------------------------------------

class TestWorldModel:
    def test_forward_output_type(
        self, world_model: WorldModel, obs_seq: torch.Tensor,
    ) -> None:
        out = world_model(obs_seq)
        assert isinstance(out, WorldModelOutput)

    def test_forward_loss_scalar(
        self, world_model: WorldModel, obs_seq: torch.Tensor,
    ) -> None:
        out = world_model(obs_seq)
        assert out.loss.shape == ()
        assert out.recon_loss.shape == ()
        assert out.kl_loss.shape == ()

    def test_forward_loss_positive(
        self, world_model: WorldModel, obs_seq: torch.Tensor,
    ) -> None:
        out = world_model(obs_seq)
        assert out.loss.item() > 0
        assert out.recon_loss.item() >= 0
        assert out.kl_loss.item() >= 0

    def test_forward_reconstructed_shape(
        self, world_model: WorldModel, obs_seq: torch.Tensor,
    ) -> None:
        out = world_model(obs_seq)
        assert out.reconstructed.shape == obs_seq.shape

    def test_forward_gradient_flow(
        self, world_model: WorldModel, obs_seq: torch.Tensor,
    ) -> None:
        out = world_model(obs_seq)
        out.loss.backward()
        grad_count = sum(
            1 for p in world_model.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        total = sum(1 for p in world_model.parameters())
        assert grad_count == total

    def test_forward_no_nan(
        self, world_model: WorldModel, obs_seq: torch.Tensor,
    ) -> None:
        out = world_model(obs_seq)
        assert not torch.isnan(out.loss)
        assert not torch.any(torch.isnan(out.reconstructed))

    def test_imagine_shape(
        self, world_model: WorldModel, obs_seq: torch.Tensor,
    ) -> None:
        horizon = 32
        pred = world_model.imagine(obs_seq, horizon)
        assert pred.shape == (BATCH, horizon, OBS_DIM)

    def test_imagine_no_nan(
        self, world_model: WorldModel, obs_seq: torch.Tensor,
    ) -> None:
        pred = world_model.imagine(obs_seq, 10)
        assert not torch.any(torch.isnan(pred))

    def test_free_nats_clamps_kl(self, obs_seq: torch.Tensor) -> None:
        model = WorldModel(
            obs_dim=OBS_DIM, embed_dim=EMBED_DIM, det_dim=DET_DIM,
            stoch_dim=STOCH_DIM, hidden_dim=HIDDEN_DIM,
            free_nats=100.0,
        )
        out = model(obs_seq)
        assert out.kl_loss.item() >= 100.0

    def test_kl_weight_scales_loss(self, obs_seq: torch.Tensor) -> None:
        model_low = WorldModel(
            obs_dim=OBS_DIM, embed_dim=EMBED_DIM, det_dim=DET_DIM,
            stoch_dim=STOCH_DIM, hidden_dim=HIDDEN_DIM,
            kl_weight=0.0, free_nats=0.0,
        )
        model_high = WorldModel(
            obs_dim=OBS_DIM, embed_dim=EMBED_DIM, det_dim=DET_DIM,
            stoch_dim=STOCH_DIM, hidden_dim=HIDDEN_DIM,
            kl_weight=10.0, free_nats=0.0,
        )
        # Same weights for deterministic comparison
        model_high.load_state_dict(model_low.state_dict())

        torch.manual_seed(0)
        out_low = model_low(obs_seq)
        torch.manual_seed(0)
        out_high = model_high(obs_seq)

        torch.testing.assert_close(out_low.recon_loss, out_high.recon_loss)
        assert out_high.loss.item() >= out_low.loss.item()

    def test_param_count_in_range(self, world_model: WorldModel) -> None:
        count = world_model.param_count()
        assert 100_000 < count < 1_000_000, f"Unexpected param count: {count}"

    def test_default_construction(self) -> None:
        model = WorldModel()
        obs = torch.randn(2, 8, 10)
        out = model(obs)
        assert out.loss.shape == ()


# ---------------------------------------------------------------------------
# Integration: encoder -> RSSM -> decoder round-trip
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_components_chain(
        self,
        encoder: ObservationEncoder,
        rssm: RSSM,
        decoder: ObservationDecoder,
        obs_seq: torch.Tensor,
    ) -> None:
        embeds = encoder(obs_seq)
        assert embeds.shape == (BATCH, SEQ_LEN, EMBED_DIM)

        state = rssm.initial_state(BATCH)
        rssm_out = rssm.observe(embeds, state)
        assert rssm_out.h_seq.shape == (BATCH, SEQ_LEN, DET_DIM)

        recon = decoder(rssm_out.h_seq, rssm_out.z_seq)
        assert recon.shape == (BATCH, SEQ_LEN, OBS_DIM)

    def test_single_step_matches_sequence_first_step(
        self, rssm: RSSM,
    ) -> None:
        torch.manual_seed(99)
        embeds = torch.randn(1, 5, EMBED_DIM)
        state = rssm.initial_state(1)

        torch.manual_seed(0)
        full_out = rssm.observe(embeds, state)

        torch.manual_seed(0)
        h = rssm.gru(state.z, state.h)
        post = rssm.posterior(h, embeds[:, 0])
        z = post.mu + post.std * torch.randn_like(post.std)

        torch.testing.assert_close(full_out.h_seq[:, 0], h)
        torch.testing.assert_close(full_out.z_seq[:, 0], z)
