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
    sl_points: float = 6.0,
    tp_points: float = 5.0,
    initial_capital: float = 5000.0,
) -> ImaginationEnv:
    config = PolicyConfig(
        sl_points=sl_points,
        tp_points=tp_points,
        spread_points=0.0,
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

    def test_start_prices_used(self) -> None:
        env = _make_env()
        prices = torch.tensor([1000.0, 2000.0, 3000.0, 4000.0])
        port = env.reset(4, prices)
        torch.testing.assert_close(port["price"], prices)


class TestOpenPosition:
    def test_buy_opens_long(self) -> None:
        env = _make_env()
        port = env.reset(BATCH, _prices())
        actions = torch.full((BATCH,), BUY, dtype=torch.long)
        _, port = env.step(actions, _zero_obs(), port)
        assert (port["position"] == BUY).all()
        assert (port["margin_used"] > 0).all()
        assert (port["capital"] == 0.0).all()

    def test_sell_opens_short(self) -> None:
        env = _make_env()
        port = env.reset(BATCH, _prices())
        actions = torch.full((BATCH,), SELL, dtype=torch.long)
        _, port = env.step(actions, _zero_obs(), port)
        assert (port["position"] == SELL).all()

    def test_size_correct(self) -> None:
        env = _make_env(initial_capital=5000.0)
        port = env.reset(1, torch.tensor([2000.0]))
        _, port = env.step(torch.tensor([BUY]), _zero_obs(1), port)
        expected_size = 5000.0 / (0.05 * 2000.0)
        assert abs(port["size"][0].item() - expected_size) < 0.01


class TestSLTP:
    def test_tp_hit_closes_long(self) -> None:
        env = _make_env(tp_points=5.0, sl_points=100.0)
        port = env.reset(1, torch.tensor([2000.0]))
        _, port = env.step(torch.tensor([BUY]), _zero_obs(1), port)
        # Return that moves price by > 5 pts: 5/2000 = 0.0025
        obs = _obs_with_ret(0.003, 1)
        reward, port = env.step(torch.tensor([HOLD]), obs, port)
        assert (port["position"] == HOLD).all()
        assert reward[0].item() > 0

    def test_sl_hit_closes_long(self) -> None:
        env = _make_env(sl_points=6.0, tp_points=100.0)
        port = env.reset(1, torch.tensor([2000.0]))
        _, port = env.step(torch.tensor([BUY]), _zero_obs(1), port)
        # Return that moves price by > 6 pts: 6/2000 = 0.003
        obs = _obs_with_ret(-0.004, 1)
        reward, port = env.step(torch.tensor([HOLD]), obs, port)
        assert (port["position"] == HOLD).all()
        assert reward[0].item() < 0

    def test_tp_hit_closes_short(self) -> None:
        env = _make_env(tp_points=5.0, sl_points=100.0)
        port = env.reset(1, torch.tensor([2000.0]))
        _, port = env.step(torch.tensor([SELL]), _zero_obs(1), port)
        obs = _obs_with_ret(-0.003, 1)
        reward, port = env.step(torch.tensor([HOLD]), obs, port)
        assert (port["position"] == HOLD).all()
        assert reward[0].item() > 0

    def test_sl_tp_independent_of_price_level(self) -> None:
        """6 pts SL should trigger at same point change regardless of price."""
        env = _make_env(sl_points=6.0, tp_points=100.0)

        # At price 2000: 6 pts = 0.3% return
        port1 = env.reset(1, torch.tensor([2000.0]))
        _, port1 = env.step(torch.tensor([BUY]), _zero_obs(1), port1)
        obs1 = _obs_with_ret(-6.0 / 2000.0 - 0.001, 1)
        r1, port1 = env.step(torch.tensor([HOLD]), obs1, port1)
        assert port1["position"][0].item() == HOLD

        # At price 4000: 6 pts = 0.15% return
        port2 = env.reset(1, torch.tensor([4000.0]))
        _, port2 = env.step(torch.tensor([BUY]), _zero_obs(1), port2)
        obs2 = _obs_with_ret(-6.0 / 4000.0 - 0.001, 1)
        r2, port2 = env.step(torch.tensor([HOLD]), obs2, port2)
        assert port2["position"][0].item() == HOLD


class TestInvalidActions:
    def test_buy_while_long_penalized(self) -> None:
        env = _make_env()
        port = env.reset(1, _prices(1))
        _, port = env.step(torch.tensor([BUY]), _zero_obs(1), port)
        reward, _ = env.step(torch.tensor([BUY]), _zero_obs(1), port)
        assert reward[0].item() < 0

    def test_hold_while_flat_no_penalty(self) -> None:
        env = _make_env()
        port = env.reset(1, _prices(1))
        reward, _ = env.step(torch.tensor([HOLD]), _zero_obs(1), port)
        assert reward[0].item() == 0.0


class TestForceClose:
    def test_force_close_with_unrealized_pnl(self) -> None:
        env = _make_env(sl_points=100.0, tp_points=100.0)
        port = env.reset(1, torch.tensor([2000.0]))
        _, port = env.step(torch.tensor([BUY]), _zero_obs(1), port)
        obs = _obs_with_ret(0.001, 1)
        _, port = env.step(torch.tensor([HOLD]), obs, port)
        reward = env.force_close(port)
        assert reward[0].item() > 0

    def test_force_close_flat_no_reward(self) -> None:
        env = _make_env()
        port = env.reset(1, _prices(1))
        reward = env.force_close(port)
        assert reward[0].item() == 0.0


class TestCapitalTracking:
    def test_capital_decreases_after_sl(self) -> None:
        env = _make_env(sl_points=6.0, tp_points=100.0)
        port = env.reset(1, torch.tensor([2000.0]))
        _, port = env.step(torch.tensor([BUY]), _zero_obs(1), port)
        obs = _obs_with_ret(-0.004, 1)
        _, port = env.step(torch.tensor([HOLD]), obs, port)
        assert port["capital"][0].item() < 5000.0

    def test_capital_increases_after_tp(self) -> None:
        env = _make_env(tp_points=5.0, sl_points=100.0)
        port = env.reset(1, torch.tensor([2000.0]))
        _, port = env.step(torch.tensor([BUY]), _zero_obs(1), port)
        obs = _obs_with_ret(0.003, 1)
        _, port = env.step(torch.tensor([HOLD]), obs, port)
        assert port["capital"][0].item() > 5000.0


class TestPriceReconstruction:
    def test_price_updates_with_returns(self) -> None:
        env = _make_env()
        port = env.reset(1, torch.tensor([2000.0]))
        obs = _obs_with_ret(0.01, 1)  # 1% return
        _, port = env.step(torch.tensor([HOLD]), obs, port)
        expected = 2000.0 * 1.01
        assert abs(port["price"][0].item() - expected) < 0.1

    def test_price_compounds_over_steps(self) -> None:
        env = _make_env()
        port = env.reset(1, torch.tensor([1000.0]))
        for _ in range(3):
            obs = _obs_with_ret(0.01, 1)
            _, port = env.step(torch.tensor([HOLD]), obs, port)
        expected = 1000.0 * 1.01 ** 3
        assert abs(port["price"][0].item() - expected) < 0.1
