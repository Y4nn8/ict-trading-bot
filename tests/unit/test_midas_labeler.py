"""Tests for Midas TickLabeler."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.midas.labeler import TickLabeler
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

    def test_timeout(self) -> None:
        """Neither SL nor TP hit within timeout → label=-1."""
        labeler = TickLabeler(
            LabelConfig(sl_points=10.0, tp_points=10.0, timeout_seconds=5.0),
        )

        entry = _tick(0, bid=100.0, ask=101.0)
        labeler.on_tick(entry)
        labeler.add_entry(entry)

        # Price stays flat, time passes
        labeler.on_tick(_tick(6, bid=100.0, ask=101.0))

        result = labeler.finalize()
        assert result.buy_labels[0] == -1
        assert result.sell_labels[0] == -1
        assert result.timeouts == 1

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
