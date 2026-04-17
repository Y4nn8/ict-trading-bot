"""Tests for Morpheus actor-critic networks."""

from __future__ import annotations

import torch

from src.morpheus.actor_critic import (
    PORTFOLIO_DIM,
    Actor,
    Critic,
    PolicyConfig,
    build_actor_critic,
    compute_state_dim,
    portfolio_features,
)

BATCH = 4
D_MODEL = 64
STATE_DIM = D_MODEL + PORTFOLIO_DIM


class TestActor:
    def test_output_shape(self) -> None:
        actor = Actor(STATE_DIM)
        x = torch.randn(BATCH, STATE_DIM)
        out = actor(x)
        assert out.shape == (BATCH, 3)

    def test_gradient_flow(self) -> None:
        actor = Actor(STATE_DIM)
        x = torch.randn(BATCH, STATE_DIM, requires_grad=True)
        out = actor(x)
        out.sum().backward()
        assert x.grad is not None

    def test_batched_sequence(self) -> None:
        actor = Actor(STATE_DIM)
        x = torch.randn(BATCH, 10, STATE_DIM)
        out = actor(x)
        assert out.shape == (BATCH, 10, 3)


class TestCritic:
    def test_output_shape(self) -> None:
        critic = Critic(STATE_DIM)
        x = torch.randn(BATCH, STATE_DIM)
        out = critic(x)
        assert out.shape == (BATCH,)

    def test_gradient_flow(self) -> None:
        critic = Critic(STATE_DIM)
        x = torch.randn(BATCH, STATE_DIM, requires_grad=True)
        out = critic(x)
        out.sum().backward()
        assert x.grad is not None


class TestPortfolioFeatures:
    def test_shape(self) -> None:
        feat = portfolio_features(
            position=torch.zeros(BATCH, dtype=torch.long),
            unrealized_pnl=torch.zeros(BATCH),
            step_in_pos=torch.zeros(BATCH),
            capital=torch.full((BATCH,), 5000.0),
            margin_used=torch.zeros(BATCH),
            initial_capital=5000.0,
            horizon=64,
        )
        assert feat.shape == (BATCH, PORTFOLIO_DIM)

    def test_flat_position_encoding(self) -> None:
        feat = portfolio_features(
            position=torch.zeros(1, dtype=torch.long),
            unrealized_pnl=torch.zeros(1),
            step_in_pos=torch.zeros(1),
            capital=torch.full((1,), 5000.0),
            margin_used=torch.zeros(1),
            initial_capital=5000.0,
            horizon=64,
        )
        assert feat[0, 0].item() == 1.0  # is_flat
        assert feat[0, 1].item() == 0.0  # is_long
        assert feat[0, 2].item() == 0.0  # is_short

    def test_long_position_encoding(self) -> None:
        feat = portfolio_features(
            position=torch.ones(1, dtype=torch.long),
            unrealized_pnl=torch.zeros(1),
            step_in_pos=torch.zeros(1),
            capital=torch.full((1,), 5000.0),
            margin_used=torch.zeros(1),
            initial_capital=5000.0,
            horizon=64,
        )
        assert feat[0, 0].item() == 0.0
        assert feat[0, 1].item() == 1.0
        assert feat[0, 2].item() == 0.0

    def test_capital_normalization(self) -> None:
        feat = portfolio_features(
            position=torch.zeros(1, dtype=torch.long),
            unrealized_pnl=torch.zeros(1),
            step_in_pos=torch.zeros(1),
            capital=torch.full((1,), 2500.0),
            margin_used=torch.zeros(1),
            initial_capital=5000.0,
            horizon=64,
        )
        assert feat[0, 5].item() == 0.5  # cap_norm


class TestBuildActorCritic:
    def test_returns_pair(self) -> None:
        config = PolicyConfig()
        actor, critic = build_actor_critic(D_MODEL, config)
        assert isinstance(actor, Actor)
        assert isinstance(critic, Critic)

    def test_state_dim(self) -> None:
        assert compute_state_dim(128) == 128 + PORTFOLIO_DIM
