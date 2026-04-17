"""Tests for Morpheus imagination environment."""

from __future__ import annotations

import numpy as np
import torch

from src.morpheus.actor_critic import BUY, HOLD, SELL, PolicyConfig
from src.morpheus.dataset import NormStats
from src.morpheus.imagination import ImaginationEnv

BATCH = 4
OBS_DIM = 16
START_PRICE = 2000.0


def _make_env(
    initial_capital: float = 5000.0,
    spread_points: float = 0.0,
) -> ImaginationEnv:
    config = PolicyConfig(
        spread_points=spread_points,
        margin_rate=0.05,
        initial_capital=initial_capital,
    )
    norm_stats = NormStats(
        mean=np.zeros(OBS_DIM, dtype=np.float32),
        std=np.ones(OBS_DIM, dtype=np.float32),
    )
    return ImaginationEnv(config, norm_stats, torch.device("cpu"))


def _prices(batch: int = BATCH) -> torch.Tensor:
    return torch.full((batch,), START_PRICE)


def _zero_obs(batch: int = BATCH) -> torch.Tensor:
    return torch.zeros(batch, OBS_DIM)


def _obs_with_ret(ret: float, batch: int = BATCH) -> torch.Tensor:
    obs = torch.zeros(batch, OBS_DIM)
    obs[:, 3] = ret
    return obs


class TestReset:
    def test_initial_state(self) -> None:
        env = _make_env()
        port = env.reset(BATCH, _prices())
        assert port["position"].shape == (BATCH,)
        assert (port["position"] == HOLD).all()
        assert (port["capital"] == 5000.0).all()
        assert (port["price"] == START_PRICE).all()


class TestPositionReward:
    def test_long_positive_return_gives_positive_reward(self) -> None:
        env = _make_env()
        port = env.reset(1, torch.tensor([2000.0]))
        _, port, _ = env.step(torch.tensor([BUY]), _zero_obs(1), port)
        obs = _obs_with_ret(0.001, 1)
        reward, _, _ = env.step(torch.tensor([HOLD]), obs, port)
        assert reward[0].item() > 0

    def test_long_negative_return_gives_negative_reward(self) -> None:
        env = _make_env()
        port = env.reset(1, torch.tensor([2000.0]))
        _, port, _ = env.step(torch.tensor([BUY]), _zero_obs(1), port)
        obs = _obs_with_ret(-0.001, 1)
        reward, _, _ = env.step(torch.tensor([HOLD]), obs, port)
        assert reward[0].item() < 0

    def test_short_negative_return_gives_positive_reward(self) -> None:
        env = _make_env()
        port = env.reset(1, torch.tensor([2000.0]))
        _, port, _ = env.step(torch.tensor([SELL]), _zero_obs(1), port)
        obs = _obs_with_ret(-0.001, 1)
        reward, _, _ = env.step(torch.tensor([HOLD]), obs, port)
        assert reward[0].item() > 0

    def test_flat_zero_return_gives_zero_reward(self) -> None:
        env = _make_env(spread_points=0.0)
        env._config = PolicyConfig(idle_penalty=0.0, spread_points=0.0)
        port = env.reset(1, torch.tensor([2000.0]))
        reward, _, _ = env.step(torch.tensor([HOLD]), _zero_obs(1), port)
        assert reward[0].item() == 0.0


class TestHoldKeepsPosition:
    def test_hold_keeps_long(self) -> None:
        env = _make_env()
        port = env.reset(1, torch.tensor([2000.0]))
        _, port, _ = env.step(torch.tensor([BUY]), _zero_obs(1), port)
        assert port["position"][0].item() == BUY
        _, port, _ = env.step(torch.tensor([HOLD]), _zero_obs(1), port)
        assert port["position"][0].item() == BUY

    def test_hold_keeps_short(self) -> None:
        env = _make_env()
        port = env.reset(1, torch.tensor([2000.0]))
        _, port, _ = env.step(torch.tensor([SELL]), _zero_obs(1), port)
        assert port["position"][0].item() == SELL
        _, port, _ = env.step(torch.tensor([HOLD]), _zero_obs(1), port)
        assert port["position"][0].item() == SELL

    def test_hold_keeps_flat(self) -> None:
        env = _make_env()
        port = env.reset(1, torch.tensor([2000.0]))
        _, port, _ = env.step(torch.tensor([HOLD]), _zero_obs(1), port)
        assert port["position"][0].item() == HOLD


class TestSpread:
    def test_opening_costs_spread(self) -> None:
        env = _make_env(spread_points=1.0)
        port = env.reset(1, torch.tensor([2000.0]))
        reward, _, _ = env.step(torch.tensor([BUY]), _zero_obs(1), port)
        assert reward[0].item() < 0

    def test_holding_position_no_spread(self) -> None:
        env = _make_env(spread_points=1.0)
        port = env.reset(1, torch.tensor([2000.0]))
        _, port, _ = env.step(torch.tensor([BUY]), _zero_obs(1), port)
        reward, _, _ = env.step(torch.tensor([HOLD]), _zero_obs(1), port)
        # No spread cost, no price change, only step_penalty
        assert abs(reward[0].item()) < 0.01

    def test_flipping_costs_double_spread(self) -> None:
        env = _make_env(spread_points=1.0)
        port = env.reset(1, torch.tensor([2000.0]))
        _, port, _ = env.step(torch.tensor([BUY]), _zero_obs(1), port)
        r_flip, _, _ = env.step(torch.tensor([SELL]), _zero_obs(1), port)
        # Flip costs 2x spread
        port2 = env.reset(1, torch.tensor([2000.0]))
        r_open, _, _ = env.step(torch.tensor([BUY]), _zero_obs(1), port2)
        assert r_flip[0].item() < r_open[0].item()


class TestStats:
    def test_position_changes_counted(self) -> None:
        env = _make_env()
        port = env.reset(1, torch.tensor([2000.0]))
        _, port, stats = env.step(torch.tensor([BUY]), _zero_obs(1), port)
        assert stats["position_changes"] == 1
        _, port, stats = env.step(torch.tensor([HOLD]), _zero_obs(1), port)
        assert stats["position_changes"] == 0

    def test_long_short_flat_counted(self) -> None:
        env = _make_env()
        port = env.reset(2, torch.tensor([2000.0, 2000.0]))
        actions = torch.tensor([BUY, SELL])
        _, port, stats = env.step(actions, _zero_obs(2), port)
        assert stats["long_steps"] == 1
        assert stats["short_steps"] == 1
        assert stats["flat_steps"] == 0


class TestPriceReconstruction:
    def test_price_updates(self) -> None:
        env = _make_env()
        port = env.reset(1, torch.tensor([2000.0]))
        obs = _obs_with_ret(0.01, 1)
        _, port, _ = env.step(torch.tensor([HOLD]), obs, port)
        expected = 2000.0 * 1.01
        assert abs(port["price"][0].item() - expected) < 0.1
