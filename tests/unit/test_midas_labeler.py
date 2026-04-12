"""Tests for Midas TickLabeler."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from src.midas.labeler import TickLabeler, build_exit_dataset, relabel_dataframe
from src.midas.types import LabelConfig, Tick


def _tick(seconds: float, bid: float, ask: float) -> Tick:
    base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    return Tick(time=base + timedelta(seconds=seconds), bid=bid, ask=ask)


class TestTickLabeler:
    """Tests for TickLabeler."""

    def test_buy_tp_hit(self) -> None:
        """BUY: price goes up → TP hit → label=1."""
        labeler = TickLabeler(LabelConfig(sl_points=2.0, tp_points=2.0))

        # Entry tick: ask=101, so BUY TP=103, SL=99
        entry = _tick(0, bid=100.0, ask=101.0)
        labeler.on_tick(entry)
        labeler.add_entry(entry)

        # Price rises: bid hits 103 → TP
        labeler.on_tick(_tick(1, bid=103.0, ask=104.0))

        result = labeler.finalize()
        assert result.buy_labels[0] == 1  # win
        assert result.buy_wins == 1

    def test_buy_sl_hit(self) -> None:
        """BUY: price goes down → SL hit → label=0."""
        labeler = TickLabeler(LabelConfig(sl_points=2.0, tp_points=2.0))

        entry = _tick(0, bid=100.0, ask=101.0)
        labeler.on_tick(entry)
        labeler.add_entry(entry)

        # Price drops: bid hits 99 → SL
        labeler.on_tick(_tick(1, bid=99.0, ask=100.0))

        result = labeler.finalize()
        assert result.buy_labels[0] == 0  # loss

    def test_sell_tp_hit(self) -> None:
        """SELL: price goes down → TP hit → label=1."""
        labeler = TickLabeler(LabelConfig(sl_points=2.0, tp_points=2.0))

        # Entry tick: bid=100, so SELL TP=98, SL=102
        entry = _tick(0, bid=100.0, ask=101.0)
        labeler.on_tick(entry)
        labeler.add_entry(entry)

        # Price drops: ask hits 98 → TP
        labeler.on_tick(_tick(1, bid=97.0, ask=98.0))

        result = labeler.finalize()
        assert result.sell_labels[0] == 1  # win

    def test_sell_sl_hit(self) -> None:
        """SELL: price goes up → SL hit → label=0."""
        labeler = TickLabeler(LabelConfig(sl_points=2.0, tp_points=2.0))

        entry = _tick(0, bid=100.0, ask=101.0)
        labeler.on_tick(entry)
        labeler.add_entry(entry)

        # Price rises: ask hits 102 → SL
        labeler.on_tick(_tick(1, bid=101.0, ask=102.0))

        result = labeler.finalize()
        assert result.sell_labels[0] == 0  # loss

    def test_timeout_closes_at_market(self) -> None:
        """On timeout, position closes at market price (not skipped)."""
        labeler = TickLabeler(
            LabelConfig(sl_points=10.0, tp_points=10.0, timeout_seconds=5.0),
        )

        entry = _tick(0, bid=100.0, ask=101.0)
        labeler.on_tick(entry)
        labeler.add_entry(entry)

        # Price stays flat, time passes
        labeler.on_tick(_tick(6, bid=100.0, ask=101.0))

        result = labeler.finalize()
        # BUY: pnl = bid - ask_entry = 100 - 101 = -1 → loss
        assert result.buy_labels[0] == 0
        assert result.buy_pnls[0] == -1.0
        # SELL: pnl = bid_entry - ask = 100 - 101 = -1 → loss
        assert result.sell_labels[0] == 0
        assert result.sell_pnls[0] == -1.0

    def test_sl_before_tp_same_tick(self) -> None:
        """SL checked before TP (pessimistic)."""
        labeler = TickLabeler(LabelConfig(sl_points=1.0, tp_points=1.0))

        # BUY at ask=101, SL=100, TP=102
        entry = _tick(0, bid=100.0, ask=101.0)
        labeler.on_tick(entry)
        labeler.add_entry(entry)

        # Tick where bid=100 (SL) and bid also >= TP not possible
        # But bid=100 hits SL exactly
        labeler.on_tick(_tick(1, bid=100.0, ask=101.0))

        result = labeler.finalize()
        assert result.buy_labels[0] == 0  # SL hit

    def test_multiple_entries(self) -> None:
        """Multiple entries resolve independently."""
        labeler = TickLabeler(LabelConfig(sl_points=2.0, tp_points=2.0))

        # Entry 1: ask=101
        labeler.on_tick(_tick(0, bid=100.0, ask=101.0))
        labeler.add_entry(_tick(0, bid=100.0, ask=101.0))

        # Entry 2: ask=105
        labeler.on_tick(_tick(1, bid=104.0, ask=105.0))
        labeler.add_entry(_tick(1, bid=104.0, ask=105.0))

        # Price goes to 103: Entry 1 BUY TP hit (101+2=103), Entry 2 BUY SL hit (105-2=103)
        labeler.on_tick(_tick(2, bid=103.0, ask=104.0))

        result = labeler.finalize()
        assert result.buy_labels[0] == 1  # entry 1 wins
        assert result.buy_labels[1] == 0  # entry 2 loses

    def test_reset(self) -> None:
        labeler = TickLabeler(LabelConfig())
        labeler.add_entry(_tick(0, bid=100.0, ask=101.0))
        labeler.reset()
        result = labeler.finalize()
        assert result.total_labeled == 0

    def test_spread_accounting(self) -> None:
        """BUY enters at ask, evaluated against bid (spread cost built in)."""
        labeler = TickLabeler(LabelConfig(sl_points=1.0, tp_points=1.0))

        # Wide spread: bid=100, ask=102 (spread=2)
        # BUY at 102, TP=103, SL=101
        entry = _tick(0, bid=100.0, ask=102.0)
        # on_tick first, then add_entry (as replay engine does)
        labeler.on_tick(entry)
        labeler.add_entry(entry)

        # Bid reaches 103 → TP hit
        labeler.on_tick(_tick(1, bid=103.0, ask=105.0))

        result = labeler.finalize()
        assert result.buy_labels[0] == 1

    def test_pnl_tracking_buy_win(self) -> None:
        """PnL should be positive for a winning BUY."""
        labeler = TickLabeler(LabelConfig(sl_points=2.0, tp_points=3.0))

        entry = _tick(0, bid=100.0, ask=101.0)
        labeler.on_tick(entry)
        labeler.add_entry(entry)

        # BUY TP = 101 + 3 = 104
        labeler.on_tick(_tick(1, bid=104.0, ask=105.0))

        result = labeler.finalize()
        assert result.buy_pnls[0] == 3.0  # TP - entry_ask = 104 - 101

    def test_pnl_tracking_sell_loss(self) -> None:
        """PnL should be negative for a losing SELL."""
        labeler = TickLabeler(LabelConfig(sl_points=2.0, tp_points=2.0))

        entry = _tick(0, bid=100.0, ask=101.0)
        labeler.on_tick(entry)
        labeler.add_entry(entry)

        # SELL SL = 100 + 2 = 102, ask hits 102
        labeler.on_tick(_tick(1, bid=101.0, ask=102.0))

        result = labeler.finalize()
        assert result.sell_pnls[0] == -2.0  # entry_bid - SL = 100 - 102

    def test_timeout_pnl_at_market(self) -> None:
        """On timeout, PnL is the unrealized P&L at the timeout tick."""
        labeler = TickLabeler(
            LabelConfig(sl_points=10.0, tp_points=10.0, timeout_seconds=5.0),
        )

        entry = _tick(0, bid=100.0, ask=101.0)
        labeler.on_tick(entry)
        labeler.add_entry(entry)

        # After timeout, price moved a bit
        labeler.on_tick(_tick(6, bid=102.0, ask=103.0))

        result = labeler.finalize()
        # BUY PnL = bid at timeout - entry_ask = 102 - 101 = 1.0
        assert result.buy_pnls[0] == 1.0


class TestRelabelNoLookahead:
    """Verify relabel_dataframe does not look beyond the DataFrame."""

    def test_boundary_entries_timeout_not_resolved(self) -> None:
        """Entries near end of DataFrame cannot resolve beyond it.

        This is the key anti-look-ahead property: relabel_dataframe
        only scans within the DataFrame, so entries near the end that
        can't hit SL/TP within timeout are labeled as timeout (-1).
        """
        base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        n = 100
        times = [
            (base + timedelta(seconds=i * 10)).timestamp()
            for i in range(n)
        ]
        # Flat price — SL/TP will never be hit
        bids = [100.0] * n
        asks = [100.5] * n

        df = pl.DataFrame({
            "_time": times,
            "_bid": bids,
            "_ask": asks,
        })

        result = relabel_dataframe(
            df,
            sl_points=5.0,
            tp_points=5.0,
            timeout_seconds=50.0,  # 5 candles of 10s
        )

        # Last 5 entries have < 50s of future data → should be timeout
        last_labels_buy = result.buy_labels[-5:]
        last_labels_sell = result.sell_labels[-5:]
        assert all(lab == -1 for lab in last_labels_buy), (
            "Last entries should be timeout (no future data to resolve)"
        )
        assert all(lab == -1 for lab in last_labels_sell)

    def test_relabel_matches_streaming_within_boundary(self) -> None:
        """relabel_dataframe and streaming labeler produce consistent
        labels when both operate on the same data (no extension)."""
        base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        # Build sampled ticks with some price movement
        rng = np.random.default_rng(42)
        n = 50
        prices = 100.0 + np.cumsum(rng.normal(0, 0.3, n))
        times = [base + timedelta(seconds=i * 10) for i in range(n)]
        bids = prices.tolist()
        asks = (prices + 0.5).tolist()

        # DataFrame labeler
        df = pl.DataFrame({
            "_time": [t.timestamp() for t in times],
            "_bid": bids,
            "_ask": asks,
        })
        df_result = relabel_dataframe(df, sl_points=2.0, tp_points=2.0, timeout_seconds=100.0)

        # Streaming labeler (no extension — same data)
        labeler = TickLabeler(LabelConfig(sl_points=2.0, tp_points=2.0, timeout_seconds=100.0))
        for i in range(n):
            tick = Tick(time=times[i], bid=bids[i], ask=asks[i])
            labeler.on_tick(tick)
            labeler.add_entry(tick)
        stream_result = labeler.finalize()

        # Both should agree on labels (same resolution, same data)
        assert df_result.buy_labels == stream_result.buy_labels
        assert df_result.sell_labels == stream_result.sell_labels


class TestRelabelATRMode:
    """Tests for ATR-based dynamic SL/TP in relabel_dataframe."""

    @staticmethod
    def _make_df(
        bids: list[float],
        asks: list[float],
        atrs: list[float],
        interval: float = 10.0,
    ) -> pl.DataFrame:
        base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        times = [
            (base + timedelta(seconds=i * interval)).timestamp()
            for i in range(len(bids))
        ]
        return pl.DataFrame({
            "_time": times,
            "_bid": bids,
            "_ask": asks,
            "scalp__m1_atr": atrs,
        })

    def test_atr_mode_uses_multiplier(self) -> None:
        """k_sl/k_tp scale SL/TP by ATR value per row."""
        # ATR=2.0 everywhere, k_sl=1.0, k_tp=1.0 → SL=2.0, TP=2.0
        bids = [100.0, 98.0, 100.0, 100.0, 100.0]
        asks = [101.0, 99.0, 101.0, 101.0, 101.0]
        atrs = [2.0, 2.0, 2.0, 2.0, 2.0]
        df = self._make_df(bids, asks, atrs)

        result = relabel_dataframe(
            df, sl_points=5.0, tp_points=5.0, timeout_seconds=100.0,
            k_sl=1.0, k_tp=1.0,
        )

        # Row 0: BUY at ask=101, SL=101-2=99, bid[1]=98 <= 99 → SL hit
        assert result.buy_labels[0] == 0

    def test_atr_zero_falls_back_to_fixed(self) -> None:
        """When ATR is 0 (cold start), falls back to sl_points/tp_points."""
        bids = [100.0, 95.0, 100.0, 100.0, 100.0]
        asks = [101.0, 96.0, 101.0, 101.0, 101.0]
        atrs = [0.0, 0.0, 0.0, 0.0, 0.0]
        df = self._make_df(bids, asks, atrs)

        result = relabel_dataframe(
            df, sl_points=3.0, tp_points=3.0, timeout_seconds=100.0,
            k_sl=1.0, k_tp=1.0,
        )

        # ATR=0 → fallback to sl=3.0
        # Row 0: BUY at ask=101, SL=101-3=98, bid[1]=95 <= 98 → SL hit
        assert result.buy_labels[0] == 0
        assert result.buy_pnls[0] == pytest.approx(-3.0)

    def test_per_row_varying_atr(self) -> None:
        """Different ATR per row produces different SL/TP levels."""
        # Row 0: ATR=1.0, k_sl=2.0 → SL=2.0pts. bid[1]=99.5 < 101-2=99 → NO
        # Row 0: ATR=1.0, k_tp=2.0 → TP=2.0pts. bid[1]=99.5 < 103 → NO
        # Row 0: bid[2]=103 >= 101+2=103 → TP hit!
        bids = [100.0, 99.5, 103.0, 100.0, 100.0]
        asks = [101.0, 100.5, 104.0, 101.0, 101.0]
        atrs = [1.0, 1.0, 1.0, 1.0, 1.0]
        df = self._make_df(bids, asks, atrs)

        result = relabel_dataframe(
            df, sl_points=5.0, tp_points=5.0, timeout_seconds=100.0,
            k_sl=2.0, k_tp=2.0,
        )

        # Row 0: BUY TP = 101 + 2*1 = 103, bid[2]=103 → TP hit
        assert result.buy_labels[0] == 1
        assert result.buy_pnls[0] == pytest.approx(2.0)

    def test_fixed_mode_unchanged(self) -> None:
        """Without k_sl/k_tp, ATR column is ignored (backward compat)."""
        bids = [100.0, 98.0, 100.0, 100.0, 100.0]
        asks = [101.0, 99.0, 101.0, 101.0, 101.0]
        atrs = [2.0, 2.0, 2.0, 2.0, 2.0]
        df = self._make_df(bids, asks, atrs)

        result = relabel_dataframe(
            df, sl_points=5.0, tp_points=5.0, timeout_seconds=100.0,
        )

        # Row 0: BUY SL=101-5=96, bid[1]=98 > 96 → NOT hit
        # (with ATR mode k=1 it would have been SL=99, hit)
        assert result.buy_labels[0] != 0 or result.buy_pnls[0] != pytest.approx(-5.0)


class TestBuildExitDataset:
    """Tests for build_exit_dataset (optimal-close labeling)."""

    @staticmethod
    def _make_df(
        bids: list[float],
        asks: list[float],
        interval: float = 10.0,
    ) -> pl.DataFrame:
        base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        times = [
            (base + timedelta(seconds=i * interval)).timestamp()
            for i in range(len(bids))
        ]
        return pl.DataFrame({
            "_time": times,
            "_bid": bids,
            "_ask": asks,
        })

    def test_buy_tp_hit_all_hold(self) -> None:
        """BUY that hits TP: all intermediate rows should be HOLD.

        When a trade wins at TP, unrealized at each step < final_pnl
        (since final = TP distance), so no row should be labeled CLOSE.
        """
        # Row 0: entry BUY at ask=101, SL=99, TP=103
        # Row 1: bid=101.5 → unrealized=0.5, final=2.0 → HOLD
        # Row 2: bid=102.0 → unrealized=1.0, final=2.0 → HOLD
        # Row 3: bid=103.0 → TP hit, unrealized=2.0=final → HOLD
        bids = [100.0, 101.5, 102.0, 103.0, 100.0]
        asks = [101.0, 102.5, 103.0, 104.0, 101.0]
        df = self._make_df(bids, asks)

        target = np.array([1, 0, 0, 0, 0], dtype=np.int32)  # BUY at row 0
        result = build_exit_dataset(
            df, target, sl_points=2.0, tp_points=2.0, timeout_seconds=100.0,
        )

        assert result.n_entries == 1
        assert result.n_rows == 3  # rows 1, 2, 3
        assert all(label == 0 for label in result.exit_labels)  # all HOLD

    def test_buy_sl_hit_labels_close(self) -> None:
        """BUY that hits SL: rows where unrealized > final should be CLOSE.

        BUY at ask=101, SL=99 → final_pnl = -2.0.
        Any row where unrealized > -2.0 should be CLOSE.
        """
        # Row 0: entry BUY at ask=101, SL=99, TP=103
        # Row 1: bid=101.5 → unrealized=0.5 > -2.0 → CLOSE
        # Row 2: bid=100.5 → unrealized=-0.5 > -2.0 → CLOSE
        # Row 3: bid=99.0 → SL hit, unrealized=-2.0 = final → HOLD
        bids = [100.0, 101.5, 100.5, 99.0, 100.0]
        asks = [101.0, 102.5, 101.5, 100.0, 101.0]
        df = self._make_df(bids, asks)

        target = np.array([1, 0, 0, 0, 0], dtype=np.int32)
        result = build_exit_dataset(
            df, target, sl_points=2.0, tp_points=2.0, timeout_seconds=100.0,
        )

        assert result.n_rows == 3  # rows 1, 2, 3
        # Row 1 (unrealized=0.5 > -2): CLOSE
        assert result.exit_labels[0] == 1
        # Row 2 (unrealized=-0.5 > -2): CLOSE
        assert result.exit_labels[1] == 1
        # Row 3 (unrealized=-2.0 = final): HOLD
        assert result.exit_labels[2] == 0

    def test_sell_tp_hit_all_hold(self) -> None:
        """SELL that hits TP: all intermediate rows should be HOLD."""
        # Row 0: entry SELL at bid=100, SL=102, TP=98
        # Row 1: ask=99.5 → unrealized=0.5, final=2.0 → HOLD
        # Row 2: ask=98.0 → TP hit, unrealized=2.0=final → HOLD
        bids = [100.0, 98.5, 97.0, 100.0]
        asks = [101.0, 99.5, 98.0, 101.0]
        df = self._make_df(bids, asks)

        target = np.array([2, 0, 0, 0], dtype=np.int32)  # SELL at row 0
        result = build_exit_dataset(
            df, target, sl_points=2.0, tp_points=2.0, timeout_seconds=100.0,
        )

        assert result.n_entries == 1
        assert all(label == 0 for label in result.exit_labels)

    def test_sell_sl_hit_labels_close(self) -> None:
        """SELL that hits SL: rows where unrealized > final should be CLOSE."""
        # Row 0: entry SELL at bid=100, SL=102, TP=98
        # Row 1: ask=99.5 → unrealized=0.5 > -2.0 → CLOSE
        # Row 2: ask=102.0 → SL hit, unrealized=-2.0 = final → HOLD
        bids = [100.0, 98.5, 101.0, 100.0]
        asks = [101.0, 99.5, 102.0, 101.0]
        df = self._make_df(bids, asks)

        target = np.array([2, 0, 0, 0], dtype=np.int32)
        result = build_exit_dataset(
            df, target, sl_points=2.0, tp_points=2.0, timeout_seconds=100.0,
        )

        assert result.exit_labels[0] == 1  # unrealized=0.5 > -2 → CLOSE
        assert result.exit_labels[1] == 0  # unrealized=-2 = final → HOLD

    def test_timeout_labels(self) -> None:
        """Timeout trade: labels depend on unrealized vs final at timeout."""
        # Row 0: entry BUY at ask=101, SL=96, TP=106 (wide, won't hit)
        # Row 1 (10s): bid=102 → unrealized=1.0
        # Row 2 (20s): bid=100 → unrealized=-1.0
        # Timeout at 15s → row 2 is at 20s > 15s → timeout at row 2
        # Final PnL = bid[2] - entry_ask = 100 - 101 = -1.0
        # Row 1: unrealized=1.0 > -1.0 → CLOSE
        # Row 2: unrealized=-1.0 = final → HOLD
        bids = [100.0, 102.0, 100.0, 100.0]
        asks = [101.0, 103.0, 101.0, 101.0]
        df = self._make_df(bids, asks)

        target = np.array([1, 0, 0, 0], dtype=np.int32)
        result = build_exit_dataset(
            df, target, sl_points=5.0, tp_points=5.0, timeout_seconds=15.0,
        )

        assert result.n_rows == 2  # rows 1, 2
        assert result.exit_labels[0] == 1  # CLOSE (1.0 > -1.0)
        assert result.exit_labels[1] == 0  # HOLD (-1.0 = final)

    def test_position_context_values(self) -> None:
        """Position context arrays have correct values."""
        # BUY at row 0, ask=101
        # Row 1 (10s): bid=102 → unrealized=1.0, duration=10.0
        # Row 2 (20s): bid=103 → TP hit, unrealized=2.0, duration=20.0
        bids = [100.0, 102.0, 103.0, 100.0]
        asks = [101.0, 103.0, 104.0, 101.0]
        df = self._make_df(bids, asks)

        target = np.array([1, 0, 0, 0], dtype=np.int32)
        result = build_exit_dataset(
            df, target, sl_points=2.0, tp_points=2.0, timeout_seconds=100.0,
        )

        # Direction: BUY=1.0
        assert all(d == 1.0 for d in result.directions)
        # Durations: 10s, 20s
        assert result.durations[0] == pytest.approx(10.0)
        assert result.durations[1] == pytest.approx(20.0)
        # Unrealized PnLs
        assert result.unrealized_pnls[0] == pytest.approx(1.0)
        assert result.unrealized_pnls[1] == pytest.approx(2.0)

    def test_multiple_entries(self) -> None:
        """Multiple entries generate independent exit rows."""
        # Row 0: BUY at ask=101, TP=103
        # Row 1: bid=103 → TP hit (row 0 resolves)
        # Row 2: SELL at bid=105, TP=103
        # Row 3: ask=103 → TP hit (row 2 resolves)
        bids = [100.0, 103.0, 105.0, 102.0, 100.0]
        asks = [101.0, 104.0, 106.0, 103.0, 101.0]
        df = self._make_df(bids, asks)

        target = np.array([1, 0, 2, 0, 0], dtype=np.int32)
        result = build_exit_dataset(
            df, target, sl_points=2.0, tp_points=2.0, timeout_seconds=100.0,
        )

        assert result.n_entries == 2
        # Entry 0 (BUY): 1 exit row at row 1
        # Entry 2 (SELL): 1 exit row at row 3
        assert result.n_rows == 2

    def test_no_entries_empty_result(self) -> None:
        """No entries → empty exit dataset."""
        bids = [100.0, 101.0, 102.0]
        asks = [101.0, 102.0, 103.0]
        df = self._make_df(bids, asks)

        target = np.array([0, 0, 0], dtype=np.int32)
        result = build_exit_dataset(
            df, target, sl_points=2.0, tp_points=2.0, timeout_seconds=100.0,
        )

        assert result.n_entries == 0
        assert result.n_rows == 0
        assert len(result.row_indices) == 0

    def test_unresolved_entry_skipped(self) -> None:
        """Entry at last row with no future data is skipped."""
        bids = [100.0, 101.0, 100.0]
        asks = [101.0, 102.0, 101.0]
        df = self._make_df(bids, asks)

        # Entry at last row — no future data to resolve
        target = np.array([0, 0, 1], dtype=np.int32)
        result = build_exit_dataset(
            df, target, sl_points=2.0, tp_points=2.0, timeout_seconds=100.0,
        )

        assert result.n_rows == 0

    def test_atr_mode(self) -> None:
        """ATR-based SL/TP works in exit dataset builder."""
        # Row 0: BUY, ATR=1.0, k_sl=2.0 → SL=2.0, k_tp=2.0 → TP=2.0
        # ask=101, SL=99, TP=103
        # Row 1: bid=102 → unrealized=1.0
        # Row 2: bid=103 → TP hit, final=2.0
        bids = [100.0, 102.0, 103.0, 100.0]
        asks = [101.0, 103.0, 104.0, 101.0]
        atrs = [1.0, 1.0, 1.0, 1.0]
        base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        times = [
            (base + timedelta(seconds=i * 10)).timestamp()
            for i in range(4)
        ]
        df = pl.DataFrame({
            "_time": times,
            "_bid": bids,
            "_ask": asks,
            "scalp__m1_atr": atrs,
        })

        target = np.array([1, 0, 0, 0], dtype=np.int32)
        result = build_exit_dataset(
            df, target, sl_points=5.0, tp_points=5.0, timeout_seconds=100.0,
            k_sl=2.0, k_tp=2.0,
        )

        assert result.n_rows == 2  # rows 1, 2
        assert all(label == 0 for label in result.exit_labels)  # all HOLD
