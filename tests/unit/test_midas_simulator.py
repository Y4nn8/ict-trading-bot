"""Tests for Midas TradeSimulator."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from src.midas.trade_simulator import SimConfig, TradeSimulator
from src.midas.types import Tick


def _tick(seconds: float, bid: float, ask: float) -> Tick:
    base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    return Tick(time=base + timedelta(seconds=seconds), bid=bid, ask=ask)



class TestTradeSimulator:
    """Tests for TradeSimulator."""

    def test_buy_entry_at_ask(self) -> None:
        sim = TradeSimulator(SimConfig(sl_points=2.0, tp_points=2.0))
        tick = _tick(0, bid=100.0, ask=101.0)
        sim.on_signal(tick, signal=1)  # BUY

        assert sim.open_count == 1
        pos = sim._positions[0]
        assert pos.entry_price == 101.0  # entered at ask
        assert pos.sl_price == 99.0  # ask - SL
        assert pos.tp_price == 103.0  # ask + TP

    def test_sell_entry_at_bid(self) -> None:
        sim = TradeSimulator(SimConfig(sl_points=2.0, tp_points=2.0))
        tick = _tick(0, bid=100.0, ask=101.0)
        sim.on_signal(tick, signal=2)  # SELL

        pos = sim._positions[0]
        assert pos.entry_price == 100.0  # entered at bid
        assert pos.sl_price == 102.0  # bid + SL
        assert pos.tp_price == 98.0  # bid - TP

    def test_buy_tp_hit(self) -> None:
        sim = TradeSimulator(SimConfig(
            sl_points=2.0, tp_points=2.0,
            size=0.1, value_per_point=10.0,
        ))
        sim.on_signal(_tick(0, bid=100.0, ask=101.0), signal=1)

        # TP at 103, bid reaches 103
        trades = sim.on_tick(_tick(1, bid=103.0, ask=104.0))

        assert len(trades) == 1
        assert trades[0].is_win is True
        assert trades[0].pnl_points == pytest.approx(2.0)  # 103 - 101
        assert trades[0].pnl == pytest.approx(2.0)  # 2pts * 0.1 * 10

    def test_buy_sl_hit(self) -> None:
        sim = TradeSimulator(SimConfig(
            sl_points=2.0, tp_points=2.0,
            size=0.1, value_per_point=10.0,
        ))
        sim.on_signal(_tick(0, bid=100.0, ask=101.0), signal=1)

        # SL at 99, bid drops to 99
        trades = sim.on_tick(_tick(1, bid=99.0, ask=100.0))

        assert len(trades) == 1
        assert trades[0].is_win is False
        assert trades[0].pnl_points == pytest.approx(-2.0)  # 99 - 101
        assert trades[0].pnl == pytest.approx(-2.0)

    def test_sell_tp_hit(self) -> None:
        sim = TradeSimulator(SimConfig(
            sl_points=2.0, tp_points=2.0,
            size=0.1, value_per_point=10.0,
        ))
        sim.on_signal(_tick(0, bid=100.0, ask=101.0), signal=2)

        # SELL TP at 98, ask drops to 98
        trades = sim.on_tick(_tick(1, bid=97.0, ask=98.0))

        assert len(trades) == 1
        assert trades[0].is_win is True
        assert trades[0].pnl_points == pytest.approx(2.0)  # 100 - 98

    def test_max_positions_enforced(self) -> None:
        sim = TradeSimulator(SimConfig(max_open_positions=1))
        sim.on_signal(_tick(0, bid=100.0, ask=101.0), signal=1)
        sim.on_signal(_tick(1, bid=100.0, ask=101.0), signal=2)

        assert sim.open_count == 1  # second entry rejected

    def test_pass_signal_no_entry(self) -> None:
        sim = TradeSimulator()
        sim.on_signal(_tick(0, bid=100.0, ask=101.0), signal=0)
        assert sim.open_count == 0

    def test_spread_filter(self) -> None:
        sim = TradeSimulator(SimConfig(max_spread=1.0))
        # Spread = 2.0, exceeds max
        sim.on_signal(_tick(0, bid=100.0, ask=102.0), signal=1)
        assert sim.open_count == 0

    def test_close_all(self) -> None:
        sim = TradeSimulator(SimConfig(
            max_open_positions=3,
            sl_points=10.0, tp_points=10.0,
        ))
        sim.on_signal(_tick(0, bid=100.0, ask=101.0), signal=1)

        closed = sim.close_all(_tick(5, bid=102.0, ask=103.0))
        assert len(closed) == 1
        assert closed[0].exit_price == 102.0  # BUY exits at bid
        assert sim.open_count == 0

    def test_capital_tracking(self) -> None:
        sim = TradeSimulator(SimConfig(
            initial_capital=10000.0,
            sl_points=2.0, tp_points=2.0,
            size=1.0, value_per_point=10.0,
        ))
        sim.on_signal(_tick(0, bid=100.0, ask=101.0), signal=1)
        sim.on_tick(_tick(1, bid=103.0, ask=104.0))  # TP hit, +20

        assert sim.capital == pytest.approx(10020.0)

    def test_reset(self) -> None:
        sim = TradeSimulator(SimConfig(initial_capital=5000.0))
        sim.on_signal(_tick(0, bid=100.0, ask=101.0), signal=1)
        sim.reset()
        assert sim.open_count == 0
        assert len(sim.closed_trades) == 0
        assert sim.capital == 5000.0

    def test_early_close(self) -> None:
        sim = TradeSimulator(SimConfig(
            sl_points=5.0, tp_points=5.0,
            size=0.1, value_per_point=10.0,
        ))
        sim.on_signal(_tick(0, bid=100.0, ask=101.0), signal=1)

        # Close early at +1.0 points (bid=102)
        trade = sim.early_close(_tick(5, bid=102.0, ask=103.0))

        assert trade is not None
        assert trade.pnl_points == pytest.approx(1.0)  # 102 - 101
        assert trade.is_win is True
        assert sim.open_count == 0

    def test_get_position_context(self) -> None:
        sim = TradeSimulator(SimConfig(sl_points=5.0, tp_points=5.0))
        sim.on_signal(_tick(0, bid=100.0, ask=101.0), signal=1)

        ctx = sim.get_position_context(_tick(30, bid=102.0, ask=103.0))

        assert ctx is not None
        assert ctx["pos_unrealized_pnl"] == pytest.approx(1.0)  # bid - entry
        assert ctx["pos_duration_sec"] == pytest.approx(30.0)
        assert ctx["pos_direction"] == 1.0  # BUY

    def test_get_position_context_sell(self) -> None:
        sim = TradeSimulator(SimConfig(sl_points=5.0, tp_points=5.0))
        sim.on_signal(_tick(0, bid=100.0, ask=101.0), signal=2)

        ctx = sim.get_position_context(_tick(10, bid=98.0, ask=99.0))

        assert ctx is not None
        assert ctx["pos_unrealized_pnl"] == pytest.approx(1.0)  # entry - ask
        assert ctx["pos_direction"] == -1.0  # SELL

    def test_get_position_context_empty(self) -> None:
        sim = TradeSimulator()
        assert sim.get_position_context(_tick(0, bid=100.0, ask=101.0)) is None


class TestATRBasedSLTP:
    """Tests for ATR-based dynamic SL/TP."""

    def test_atr_mode_buy_tp_hit(self) -> None:
        """ATR mode: SL = k_sl * ATR, TP = k_tp * ATR."""
        sim = TradeSimulator(SimConfig(
            sl_points=10.0, tp_points=10.0,  # fallback (unused)
            k_sl=1.0, k_tp=2.0,
            size=0.1, value_per_point=10.0,
        ))
        # ATR=1.5 → SL=1.5, TP=3.0
        # BUY at ask=101, TP=104
        sim.on_signal(_tick(0, bid=100.0, ask=101.0), signal=1, atr=1.5)

        pos = sim._positions[0]
        assert pos.sl_price == pytest.approx(99.5)   # 101 - 1.5
        assert pos.tp_price == pytest.approx(104.0)   # 101 + 3.0

        # TP hit
        trades = sim.on_tick(_tick(1, bid=104.0, ask=105.0))
        assert len(trades) == 1
        assert trades[0].is_win is True
        assert trades[0].pnl_points == pytest.approx(3.0)

    def test_atr_mode_sell_sl_hit(self) -> None:
        sim = TradeSimulator(SimConfig(
            sl_points=10.0, tp_points=10.0,
            k_sl=2.0, k_tp=1.0,
        ))
        # ATR=1.0 → SL=2.0, TP=1.0
        # SELL at bid=100, SL=102
        sim.on_signal(_tick(0, bid=100.0, ask=101.0), signal=2, atr=1.0)

        pos = sim._positions[0]
        assert pos.sl_price == pytest.approx(102.0)
        assert pos.tp_price == pytest.approx(99.0)

        # SL hit
        trades = sim.on_tick(_tick(1, bid=101.0, ask=102.0))
        assert len(trades) == 1
        assert trades[0].is_win is False

    def test_atr_zero_falls_back_to_fixed(self) -> None:
        """When ATR=0, falls back to fixed sl_points/tp_points."""
        sim = TradeSimulator(SimConfig(
            sl_points=2.0, tp_points=3.0,
            k_sl=1.0, k_tp=1.0,
        ))
        sim.on_signal(_tick(0, bid=100.0, ask=101.0), signal=1, atr=0.0)

        pos = sim._positions[0]
        assert pos.sl_price == pytest.approx(99.0)   # 101 - 2.0 (fixed)
        assert pos.tp_price == pytest.approx(104.0)   # 101 + 3.0 (fixed)

    def test_no_atr_param_uses_fixed(self) -> None:
        """Without atr= kwarg, uses fixed SL/TP even with k_sl/k_tp set."""
        sim = TradeSimulator(SimConfig(
            sl_points=2.0, tp_points=3.0,
            k_sl=1.0, k_tp=1.0,
        ))
        sim.on_signal(_tick(0, bid=100.0, ask=101.0), signal=1)

        pos = sim._positions[0]
        assert pos.sl_price == pytest.approx(99.0)   # fixed fallback
        assert pos.tp_price == pytest.approx(104.0)


class TestDynamicSizing:
    """Tests for dynamic position sizing with gamma ramp."""

    def _dynamic_config(self, **overrides: Any) -> SimConfig:
        """Build a SimConfig with dynamic sizing enabled."""
        defaults: dict[str, Any] = {
            "sl_points": 5.0,
            "tp_points": 5.0,
            "initial_capital": 5_000.0,
            "value_per_point": 1.0,
            "max_open_positions": 3,
            "max_spread": 2.0,
            "gamma": 1.0,
            "max_margin_proba": 0.85,
            "margin_pct": 0.05,
            "min_lot_size": 0.1,
        }
        defaults.update(overrides)
        return SimConfig(**defaults)  # type: ignore[arg-type]

    def test_compute_dynamic_size_saturation(self) -> None:
        """At max_margin_proba, use full available margin."""
        # available=5000, price=1000 → margin_per_lot=50, size_max=100
        size = TradeSimulator.compute_dynamic_size(
            proba=0.90, threshold=0.33, gamma=1.0,
            max_margin_proba=0.85, available_margin=5000.0,
            margin_per_lot=50.0, min_lot_size=0.1,
        )
        # floor(5000 / 50 / 0.1) * 0.1 = 100.0
        assert size == pytest.approx(100.0)

    def test_compute_dynamic_size_linear_ramp(self) -> None:
        """gamma=1 → linear mapping from confidence to size."""
        # Use exact fractions to avoid IEEE 754 drift
        size = TradeSimulator.compute_dynamic_size(
            proba=0.75, threshold=0.50, gamma=1.0,
            max_margin_proba=1.0, available_margin=5000.0,
            margin_per_lot=50.0, min_lot_size=0.1,
        )
        # confidence = (0.75 - 0.50) / (1.0 - 0.50) = 0.5
        # size_max = floor(5000 / 50 / 0.1) * 0.1 = 100.0
        # size = floor(0.5 * 100 / 0.1) * 0.1 = 50.0
        assert size == pytest.approx(50.0)

    def test_compute_dynamic_size_conservative_gamma(self) -> None:
        """gamma=2 → quadratic ramp, smaller size at low confidence."""
        size = TradeSimulator.compute_dynamic_size(
            proba=0.75, threshold=0.50, gamma=2.0,
            max_margin_proba=1.0, available_margin=5000.0,
            margin_per_lot=50.0, min_lot_size=0.1,
        )
        # confidence = 0.5, gamma=2 → 0.25
        # size = floor(0.25 * 100 / 0.1) * 0.1 = 25.0
        assert size == pytest.approx(25.0)

    def test_compute_dynamic_size_aggressive_gamma(self) -> None:
        """gamma=0.5 → sqrt ramp, larger size at low confidence."""
        # confidence = 0.5, gamma=0.5 → sqrt(0.5) ≈ 0.7071
        # size = floor(0.7071 * 100 / 0.1) * 0.1 = 70.7
        size = TradeSimulator.compute_dynamic_size(
            proba=0.59, threshold=0.33, gamma=0.5,
            max_margin_proba=0.85, available_margin=5000.0,
            margin_per_lot=50.0, min_lot_size=0.1,
        )
        expected = math.floor(0.5**0.5 * 100.0 / 0.1) * 0.1
        assert size == pytest.approx(expected)

    def test_compute_dynamic_size_below_min_lot(self) -> None:
        """Low proba with high gamma → size below min_lot → None."""
        # confidence ~0.04, gamma=3 → 0.04^3 ≈ 6e-5, * 100 → 0.006
        # floor(0.006 / 0.1) * 0.1 = 0.0 < 0.1 → None
        size = TradeSimulator.compute_dynamic_size(
            proba=0.35, threshold=0.33, gamma=3.0,
            max_margin_proba=0.85, available_margin=5000.0,
            margin_per_lot=50.0, min_lot_size=0.1,
        )
        assert size is None

    def test_compute_dynamic_size_no_available_margin(self) -> None:
        """Zero available margin → None."""
        size = TradeSimulator.compute_dynamic_size(
            proba=0.90, threshold=0.33, gamma=1.0,
            max_margin_proba=0.85, available_margin=0.0,
            margin_per_lot=50.0, min_lot_size=0.1,
        )
        assert size is None

    def test_dynamic_sizing_entry_uses_proba(self) -> None:
        """on_signal with proba → dynamic size, not fixed."""
        cfg = self._dynamic_config()
        sim = TradeSimulator(cfg)
        # price=1000, threshold=1/3 (the default SimConfig.sizing_threshold)
        tick = _tick(0, bid=999.0, ask=1000.0)
        sim.on_signal(tick, signal=1, proba=0.59)

        assert sim.open_count == 1
        pos = sim._positions[0]
        # Compute expected size with same formula
        margin_per_lot = 1000.0 * 0.05
        size_max = math.floor(5000.0 / margin_per_lot / 0.1) * 0.1
        conf = (0.59 - 1.0 / 3.0) / (0.85 - 1.0 / 3.0)
        expected = math.floor(conf * size_max / 0.1) * 0.1
        assert pos.size == pytest.approx(expected)
        assert pos.margin == pytest.approx(expected * margin_per_lot)

    def test_dynamic_sizing_margin_tracking(self) -> None:
        """Margin is tracked across open positions and released on close."""
        cfg = self._dynamic_config(max_open_positions=3)
        sim = TradeSimulator(cfg)

        tick = _tick(0, bid=999.0, ask=1000.0)
        sim.on_signal(tick, signal=1, proba=0.85)  # saturation → 100 lots
        assert sim.margin_used == pytest.approx(100.0 * 50.0)  # 5000

        # Second trade: no available margin → skipped
        sim.on_signal(_tick(1, bid=999.0, ask=1000.0), signal=2, proba=0.85)
        assert sim.open_count == 1  # second entry rejected

    def test_dynamic_sizing_margin_released_on_sl(self) -> None:
        """Margin is released when position hits SL."""
        cfg = self._dynamic_config()
        sim = TradeSimulator(cfg)

        sim.on_signal(_tick(0, bid=999.0, ask=1000.0), signal=1, proba=0.59)
        margin_before = sim.margin_used
        assert margin_before > 0

        # SL hit (entry=1000, sl=995, bid drops to 995)
        sim.on_tick(_tick(1, bid=995.0, ask=996.0))
        assert sim.open_count == 0
        assert sim.margin_used == pytest.approx(0.0)

    def test_dynamic_sizing_margin_released_on_close_all(self) -> None:
        """close_all releases all margin."""
        cfg = self._dynamic_config()
        sim = TradeSimulator(cfg)

        sim.on_signal(_tick(0, bid=999.0, ask=1000.0), signal=1, proba=0.70)
        assert sim.margin_used > 0

        sim.close_all(_tick(5, bid=1001.0, ask=1002.0))
        assert sim.margin_used == pytest.approx(0.0)

    def test_dynamic_sizing_margin_released_on_early_close(self) -> None:
        """early_close releases margin."""
        cfg = self._dynamic_config()
        sim = TradeSimulator(cfg)

        sim.on_signal(_tick(0, bid=999.0, ask=1000.0), signal=1, proba=0.70)
        assert sim.margin_used > 0

        sim.early_close(_tick(5, bid=1001.0, ask=1002.0))
        assert sim.margin_used == pytest.approx(0.0)

    def test_dynamic_sizing_pnl_uses_dynamic_size(self) -> None:
        """PnL computation uses the dynamic size, not fixed."""
        cfg = self._dynamic_config(value_per_point=1.0)
        sim = TradeSimulator(cfg)

        sim.on_signal(_tick(0, bid=999.0, ask=1000.0), signal=1, proba=0.59)
        actual_size = sim._positions[0].size
        trades = sim.on_tick(_tick(1, bid=1005.0, ask=1006.0))  # TP hit

        assert len(trades) == 1
        # pnl = 5pts * dynamic_size * 1.0
        assert trades[0].pnl == pytest.approx(5.0 * actual_size)
        assert trades[0].size == pytest.approx(actual_size)

    def test_dynamic_sizing_reset_clears_margin(self) -> None:
        """reset() clears margin_used."""
        cfg = self._dynamic_config()
        sim = TradeSimulator(cfg)

        sim.on_signal(_tick(0, bid=999.0, ask=1000.0), signal=1, proba=0.70)
        sim.reset()
        assert sim.margin_used == pytest.approx(0.0)

    def test_min_risk_floor_bumps_size(self) -> None:
        """min_risk_pct bumps size when gamma ramp produces too little risk."""
        cfg = self._dynamic_config(
            min_risk_pct=0.01,  # risk at least 1% of 5000 = 50€
        )
        sim = TradeSimulator(cfg)
        # price=1000, sl=5pts, vpp=1 → risk_per_lot = 5*1 = 5€/lot
        # min_risk_size = ceil(50 / 5 / 0.1) * 0.1 = 10.0 lots
        # Low proba → gamma ramp gives tiny size, floor kicks in
        tick = _tick(0, bid=999.0, ask=1000.0)
        sim.on_signal(tick, signal=1, proba=0.35)
        assert sim.open_count == 1
        pos = sim._positions[0]
        # confidence = (0.35 - 1/3) / (0.85 - 1/3) ≈ 0.032
        # gamma_size = None (below min lot) → floor rescues to 10.0
        assert pos.size == pytest.approx(10.0)

    def test_min_risk_floor_skips_when_margin_insufficient(self) -> None:
        """Skip trade if min risk floor exceeds available margin."""
        cfg = self._dynamic_config(
            initial_capital=100.0,
            min_risk_pct=0.10,  # risk at least 10% of 100 = 10€
        )
        sim = TradeSimulator(cfg)
        # price=1000, margin_per_lot=50, available=100 → size_max=2.0
        # sl=5, vpp=1 → risk_per_lot=5 → min_size = ceil(10/5/0.1)*0.1 = 2.0
        # margin for 2.0 lots = 100 → exactly fits
        tick = _tick(0, bid=999.0, ask=1000.0)
        sim.on_signal(tick, signal=1, proba=0.35)
        assert sim.open_count == 1

        # Now with higher min_risk_pct that forces more margin than available
        sim.reset()
        cfg2 = self._dynamic_config(
            initial_capital=100.0,
            min_risk_pct=0.20,  # risk at least 20% of 100 = 20€
        )
        sim2 = TradeSimulator(cfg2)
        # min_size = ceil(20/5/0.1)*0.1 = 4.0 → margin = 200 > 100
        sim2.on_signal(tick, signal=1, proba=0.35)
        assert sim2.open_count == 0  # skipped

    def test_min_risk_floor_no_effect_when_gamma_larger(self) -> None:
        """Floor has no effect when gamma ramp already exceeds it."""
        cfg = self._dynamic_config(min_risk_pct=0.001)  # very small floor
        sim = TradeSimulator(cfg)
        tick = _tick(0, bid=999.0, ask=1000.0)
        sim.on_signal(tick, signal=1, proba=0.70)

        # Without floor, compute expected gamma size
        margin_per_lot = 1000.0 * 0.05
        size_max = math.floor(5000.0 / margin_per_lot / 0.1) * 0.1
        conf = (0.70 - 1.0 / 3.0) / (0.85 - 1.0 / 3.0)
        gamma_size = math.floor(conf * size_max / 0.1) * 0.1

        pos = sim._positions[0]
        assert pos.size == pytest.approx(gamma_size)

    def test_min_risk_floor_disabled_by_default(self) -> None:
        """min_risk_pct=None (default) → no floor applied."""
        cfg = self._dynamic_config()
        assert cfg.min_risk_pct is None
        sim = TradeSimulator(cfg)
        tick = _tick(0, bid=999.0, ask=1000.0)
        sim.on_signal(tick, signal=1, proba=0.35)

        # Low proba → small size, no floor
        pos = sim._positions[0]
        conf = (0.35 - 1.0 / 3.0) / (0.85 - 1.0 / 3.0)
        gamma_size = math.floor((conf ** 1.0) * 100.0 / 0.1) * 0.1
        assert pos.size == pytest.approx(gamma_size)

    def test_fixed_sizing_unchanged(self) -> None:
        """Without gamma, sizing is still fixed (backward compat)."""
        sim = TradeSimulator(SimConfig(
            sl_points=5.0, tp_points=5.0, size=0.5,
        ))
        sim.on_signal(_tick(0, bid=999.0, ask=1000.0), signal=1, proba=0.90)

        pos = sim._positions[0]
        assert pos.size == pytest.approx(0.5)  # fixed
        assert pos.margin == pytest.approx(0.0)  # no margin tracking

    def test_can_open_rejects_insufficient_margin(self) -> None:
        """_can_open returns False when margin is exhausted."""
        cfg = self._dynamic_config(initial_capital=100.0)
        sim = TradeSimulator(cfg)

        # price=1000, margin_per_lot=50, min_lot=0.1 → min_margin=5
        # capital=100, first trade at saturation uses ~100 margin
        sim.on_signal(_tick(0, bid=999.0, ask=1000.0), signal=1, proba=0.85)
        # Second entry: no margin left
        sim.on_signal(_tick(1, bid=999.0, ask=1000.0), signal=2, proba=0.85)
        assert sim.open_count == 1

    def test_dynamic_sizing_sell_uses_bid_for_margin(self) -> None:
        """SELL uses bid price for margin computation."""
        cfg = self._dynamic_config()
        sim = TradeSimulator(cfg)

        # SELL: entry at bid=999, margin based on bid=999
        sim.on_signal(_tick(0, bid=999.0, ask=1000.0), signal=2, proba=0.59)

        pos = sim._positions[0]
        expected_margin_per_lot = 999.0 * 0.05
        expected_size_max = (
            math.floor(5000.0 / expected_margin_per_lot / 0.1) * 0.1
        )
        conf = (0.59 - 1.0 / 3.0) / (0.85 - 1.0 / 3.0)
        expected_size = math.floor(
            conf * expected_size_max / 0.1,
        ) * 0.1
        assert pos.size == pytest.approx(expected_size)
        assert pos.margin == pytest.approx(expected_size * expected_margin_per_lot)

    def test_proba_propagated_to_closed_trade(self) -> None:
        """Proba from entry signal is recorded on the closed trade."""
        cfg = self._dynamic_config()
        sim = TradeSimulator(cfg)

        sim.on_signal(_tick(0, bid=999.0, ask=1000.0), signal=1, proba=0.72)
        trades = sim.on_tick(_tick(1, bid=1005.0, ask=1006.0))  # TP hit

        assert len(trades) == 1
        assert trades[0].proba == pytest.approx(0.72)

    def test_proba_propagated_on_early_close(self) -> None:
        """Proba is preserved through early_close."""
        cfg = self._dynamic_config()
        sim = TradeSimulator(cfg)

        sim.on_signal(_tick(0, bid=999.0, ask=1000.0), signal=1, proba=0.65)
        trade = sim.early_close(_tick(5, bid=1001.0, ask=1002.0))

        assert trade is not None
        assert trade.proba == pytest.approx(0.65)


class TestSlippageSimulation:
    """Tests for random slippage on market orders."""

    def test_no_slippage_by_default(self) -> None:
        """Default slippage_max_pts=0 → no slippage."""
        sim = TradeSimulator(SimConfig(sl_points=5.0, tp_points=5.0))
        sim.on_signal(_tick(0, bid=100.0, ask=101.0), signal=1)

        pos = sim._positions[0]
        assert pos.entry_price == 101.0  # exact ask

    def test_buy_entry_slippage_adverse(self) -> None:
        """BUY entry slippage moves price UP (worse fill)."""
        sim = TradeSimulator(SimConfig(
            sl_points=5.0, tp_points=5.0,
            slippage_max_pts=1.0, slippage_seed=42,
        ))
        sim.on_signal(_tick(0, bid=100.0, ask=101.0), signal=1)

        pos = sim._positions[0]
        assert pos.entry_price > 101.0  # slipped higher
        assert pos.entry_price <= 102.0  # at most ask + max

    def test_slippage_min_floor(self) -> None:
        """slippage_min_pts guarantees a minimum slippage per order."""
        entries: list[float] = []
        for seed in range(50):
            sim = TradeSimulator(SimConfig(
                sl_points=5.0, tp_points=5.0,
                slippage_min_pts=0.2, slippage_max_pts=0.5,
                slippage_seed=seed,
            ))
            sim.on_signal(_tick(0, bid=100.0, ask=101.0), signal=1)
            entries.append(sim._positions[0].entry_price)

        # Every entry must be >= ask + min slippage
        assert all(e >= 101.0 + 0.2 - 1e-9 for e in entries)
        # And <= ask + max slippage
        assert all(e <= 101.0 + 0.5 + 1e-9 for e in entries)

    def test_sell_entry_slippage_adverse(self) -> None:
        """SELL entry slippage moves price DOWN (worse fill)."""
        sim = TradeSimulator(SimConfig(
            sl_points=5.0, tp_points=5.0,
            slippage_max_pts=1.0, slippage_seed=42,
        ))
        sim.on_signal(_tick(0, bid=100.0, ask=101.0), signal=2)

        pos = sim._positions[0]
        assert pos.entry_price < 100.0  # slipped lower
        assert pos.entry_price >= 99.0  # at most bid - max

    def test_sl_tp_recalculated_from_slipped_entry(self) -> None:
        """SL/TP are set relative to the slipped entry price."""
        sim = TradeSimulator(SimConfig(
            sl_points=2.0, tp_points=3.0,
            slippage_max_pts=1.0, slippage_seed=42,
        ))
        sim.on_signal(_tick(0, bid=100.0, ask=101.0), signal=1)

        pos = sim._positions[0]
        assert pos.sl_price == pytest.approx(pos.entry_price - 2.0)
        assert pos.tp_price == pytest.approx(pos.entry_price + 3.0)

    def test_sl_tp_exit_no_slippage(self) -> None:
        """SL/TP exits use exact stop/limit price — no slippage."""
        sim = TradeSimulator(SimConfig(
            sl_points=2.0, tp_points=2.0,
            size=0.1, value_per_point=10.0,
            slippage_max_pts=1.0, slippage_seed=42,
        ))
        sim.on_signal(_tick(0, bid=100.0, ask=101.0), signal=1)
        sl = sim._positions[0].sl_price

        # SL hit — exit at exact sl_price, not slipped
        trades = sim.on_tick(_tick(1, bid=sl - 1.0, ask=sl))
        assert len(trades) == 1
        assert trades[0].exit_price == pytest.approx(sl)

    def test_early_close_buy_slippage_adverse(self) -> None:
        """BUY early_close slippage moves bid DOWN (worse exit)."""
        sim = TradeSimulator(SimConfig(
            sl_points=10.0, tp_points=10.0,
            size=0.1, value_per_point=10.0,
            slippage_max_pts=0.5, slippage_seed=99,
        ))
        sim.on_signal(_tick(0, bid=100.0, ask=101.0), signal=1)
        trade = sim.early_close(_tick(5, bid=102.0, ask=103.0))

        assert trade is not None
        assert trade.exit_price < 102.0  # slipped below bid

    def test_early_close_sell_slippage_adverse(self) -> None:
        """SELL early_close slippage moves ask UP (worse exit)."""
        sim = TradeSimulator(SimConfig(
            sl_points=10.0, tp_points=10.0,
            size=0.1, value_per_point=10.0,
            slippage_max_pts=0.5, slippage_seed=99,
        ))
        sim.on_signal(_tick(0, bid=100.0, ask=101.0), signal=2)
        trade = sim.early_close(_tick(5, bid=98.0, ask=99.0))

        assert trade is not None
        assert trade.exit_price > 99.0  # slipped above ask

    def test_close_all_applies_slippage(self) -> None:
        """close_all applies adverse slippage per position."""
        sim = TradeSimulator(SimConfig(
            sl_points=10.0, tp_points=10.0,
            size=0.1, value_per_point=10.0,
            slippage_max_pts=0.5, slippage_seed=7,
        ))
        sim.on_signal(_tick(0, bid=100.0, ask=101.0), signal=2)  # SELL

        trades = sim.close_all(_tick(3, bid=99.0, ask=100.0))
        assert len(trades) == 1
        # SELL exit at ask + slip → worse than 100.0
        assert trades[0].exit_price > 100.0

    def test_slippage_seed_reproducible(self) -> None:
        """Same seed → same slippage sequence."""
        def run(seed: int) -> float:
            sim = TradeSimulator(SimConfig(
                sl_points=5.0, tp_points=5.0,
                slippage_max_pts=1.0, slippage_seed=seed,
            ))
            sim.on_signal(_tick(0, bid=100.0, ask=101.0), signal=1)
            return sim._positions[0].entry_price

        assert run(42) == run(42)
        # Different seeds should (almost certainly) differ
        assert run(42) != run(99)

    def test_reset_resets_rng(self) -> None:
        """After reset, slippage sequence restarts from seed."""
        sim = TradeSimulator(SimConfig(
            sl_points=5.0, tp_points=5.0,
            slippage_max_pts=1.0, slippage_seed=42,
        ))
        sim.on_signal(_tick(0, bid=100.0, ask=101.0), signal=1)
        entry_1 = sim._positions[0].entry_price

        sim.reset()
        sim.on_signal(_tick(0, bid=100.0, ask=101.0), signal=1)
        entry_2 = sim._positions[0].entry_price

        assert entry_1 == entry_2

    def test_slippage_pnl_impact(self) -> None:
        """Slippage reduces PnL compared to no-slippage run."""
        # No slippage
        sim_clean = TradeSimulator(SimConfig(
            sl_points=5.0, tp_points=5.0,
            size=1.0, value_per_point=1.0,
        ))
        sim_clean.on_signal(_tick(0, bid=100.0, ask=101.0), signal=1)
        trade_clean = sim_clean.early_close(_tick(5, bid=104.0, ask=105.0))

        # With slippage
        sim_slip = TradeSimulator(SimConfig(
            sl_points=5.0, tp_points=5.0,
            size=1.0, value_per_point=1.0,
            slippage_max_pts=0.5, slippage_seed=42,
        ))
        sim_slip.on_signal(_tick(0, bid=100.0, ask=101.0), signal=1)
        trade_slip = sim_slip.early_close(_tick(5, bid=104.0, ask=105.0))

        assert trade_clean is not None
        assert trade_slip is not None
        # Slippage always hurts: entry higher + exit lower → less PnL
        assert trade_slip.pnl < trade_clean.pnl
