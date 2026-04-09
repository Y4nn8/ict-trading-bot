"""Tests for Midas feature extractors."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from src.midas.candle_builder import CandleBuilder
from src.midas.extractors.ict_features import ICTFeatureExtractor
from src.midas.extractors.scalping_features import ScalpingFeatureExtractor
from src.midas.extractors.tick_features import TickFeatureExtractor
from src.midas.feature_extractor import FeatureRegistry
from src.midas.types import PartialCandle, Tick


def _make_partial(
    price: float = 100.0,
    elapsed: float = 5.0,
    tick_count: int = 10,
) -> PartialCandle:
    return PartialCandle(
        bucket_start=datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC),
        open=price - 0.5,
        high=price + 1.0,
        low=price - 1.0,
        close=price,
        tick_count=tick_count,
        bid=price - 0.25,
        ask=price + 0.25,
        elapsed_seconds=elapsed,
    )


def _make_tick(
    price: float = 100.0,
    spread: float = 0.5,
    hour: int = 14,
) -> Tick:
    return Tick(
        time=datetime(2025, 1, 1, hour, 0, 0, tzinfo=UTC),
        bid=price - spread / 2,
        ask=price + spread / 2,
    )


def _make_candle(
    index: int,
    open_: float = 100.0,
    close: float = 101.0,
    high: float = 102.0,
    low: float = 99.0,
) -> dict[str, Any]:
    base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    return {
        "time": base + timedelta(seconds=index * 10),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "tick_count": 30,
    }


class TestTickFeatureExtractor:
    """Tests for TickFeatureExtractor."""

    def test_name(self) -> None:
        ext = TickFeatureExtractor()
        assert ext.name == "tick"

    def test_extract_returns_expected_keys(self) -> None:
        ext = TickFeatureExtractor()
        ext.configure({"spread_avg_period": 30})

        features = ext.extract(
            _make_tick(), _make_partial(), candle_index=0,
        )

        expected_keys = {
            "tick__spread",
            "tick__spread_z",
            "tick__partial_range",
            "tick__position_in_range",
            "tick__elapsed_pct",
            "tick__tick_count",
            "tick__mid",
        }
        assert set(features.keys()) == expected_keys

    def test_spread_value(self) -> None:
        ext = TickFeatureExtractor()
        ext.configure({})
        tick = _make_tick(spread=0.6)
        features = ext.extract(tick, _make_partial(), 0)
        assert features["tick__spread"] == pytest.approx(0.6)

    def test_elapsed_pct(self) -> None:
        ext = TickFeatureExtractor()
        ext.configure({})
        partial = _make_partial(elapsed=7.5)
        features = ext.extract(_make_tick(), partial, 0)
        assert features["tick__elapsed_pct"] == pytest.approx(0.75)

    def test_reset_clears_state(self) -> None:
        ext = TickFeatureExtractor()
        ext.configure({})
        ext._record_spread(0.5)
        ext.reset()
        assert len(ext._candle_spreads) == 0


class TestScalpingFeatureExtractor:
    """Tests for ScalpingFeatureExtractor."""

    def test_name(self) -> None:
        ext = ScalpingFeatureExtractor()
        assert ext.name == "scalp"

    def test_extract_cold_start(self) -> None:
        """With no candle history, all features should be 0."""
        ext = ScalpingFeatureExtractor()
        ext.configure({})
        features = ext.extract(_make_tick(), _make_partial(), 0)
        assert features["scalp__10s_roc_fast"] == 0.0
        assert features["scalp__10s_atr"] == 0.0
        assert features["scalp__10s_direction_streak"] == 0.0
        assert "scalp__m1_roc_fast" in features

    def test_extract_has_all_timeframes(self) -> None:
        ext = ScalpingFeatureExtractor()
        ext.configure({})
        features = ext.extract(_make_tick(), _make_partial(), 0)
        for tf in ("10s", "m1"):
            assert f"scalp__{tf}_roc_fast" in features
            assert f"scalp__{tf}_roc_slow" in features
            assert f"scalp__{tf}_mean_rev_z" in features
            assert f"scalp__{tf}_atr" in features
            assert f"scalp__{tf}_range_ratio" in features
            assert f"scalp__{tf}_body_ratio" in features
            assert f"scalp__{tf}_direction_streak" in features
        # M5/H1 not produced (covered by htf__ extractor)
        assert "scalp__m5_roc_fast" not in features
        assert "scalp__h1_roc_fast" not in features

    def test_roc_after_candles(self) -> None:
        ext = ScalpingFeatureExtractor()
        ext.configure({"roc_fast": 3, "roc_slow": 5,
                       "mean_rev_period": 5, "atr_period": 5})

        # Feed 6 candles with increasing price
        for i in range(6):
            ext.on_candle_close(
                _make_candle(i, open_=100 + i, close=101 + i,
                             high=102 + i, low=99 + i),
                i,
            )

        features = ext.extract(_make_tick(price=106.0), _make_partial(), 6)
        # 10s ROC should be positive (price went up)
        assert features["scalp__10s_roc_fast"] > 0
        assert features["scalp__10s_roc_slow"] > 0
        # M1 should have 1 closed candle (6 10s = 1 M1) but ATR needs
        # atr_period candles, so just check it exists (will be 0 with 1 candle)
        assert "scalp__m1_atr" in features

    def test_direction_streak(self) -> None:
        ext = ScalpingFeatureExtractor()
        ext.configure({})

        # 3 bullish candles
        for i in range(3):
            ext.on_candle_close(
                _make_candle(i, open_=100, close=101),
                i,
            )

        features = ext.extract(_make_tick(), _make_partial(), 3)
        assert features["scalp__10s_direction_streak"] == 3.0

    def test_direction_streak_bearish(self) -> None:
        ext = ScalpingFeatureExtractor()
        ext.configure({})

        # 2 bearish candles
        for i in range(2):
            ext.on_candle_close(
                _make_candle(i, open_=101, close=100),
                i,
            )

        features = ext.extract(_make_tick(), _make_partial(), 2)
        assert features["scalp__10s_direction_streak"] == -2.0

    def test_m1_populated_after_6_candles(self) -> None:
        ext = ScalpingFeatureExtractor()
        ext.configure({"atr_period": 1, "roc_fast": 1,
                       "roc_slow": 1, "mean_rev_period": 1})

        # 6 candles → 1 M1 candle
        for i in range(6):
            ext.on_candle_close(
                _make_candle(i, open_=100 + i, close=101 + i,
                             high=102 + i, low=99 + i),
                i,
            )

        features = ext.extract(_make_tick(price=106.0), _make_partial(), 6)
        assert features["scalp__m1_atr"] > 0


class TestICTFeatureExtractor:
    """Tests for ICTFeatureExtractor."""

    def test_name(self) -> None:
        ext = ICTFeatureExtractor(instrument="XAUUSD")
        assert ext.name == "ict"

    def test_extract_returns_expected_keys(self) -> None:
        ext = ICTFeatureExtractor()
        ext.configure({})
        features = ext.extract(
            _make_tick(hour=14), _make_partial(), 0,
        )
        # 10 features per TF (m5, h1) + 1 killzone = 21
        for tf in ("m5", "h1"):
            assert f"ict__{tf}_fvg_distance" in features
            assert f"ict__{tf}_fvg_direction" in features
            assert f"ict__{tf}_ob_distance" in features
            assert f"ict__{tf}_ob_direction" in features
            assert f"ict__{tf}_bos_recent" in features
            assert f"ict__{tf}_choch_recent" in features
            assert f"ict__{tf}_trend" in features
            assert f"ict__{tf}_displacement_recent" in features
            assert f"ict__{tf}_liq_sweep_distance" in features
            assert f"ict__{tf}_premium_discount" in features
        assert "ict__killzone" in features
        assert len(features) == 21

    def test_killzone_detection(self) -> None:
        ext = ICTFeatureExtractor()
        ext.configure({})

        # 14:00 UTC is NY killzone
        features_kz = ext.extract(
            _make_tick(hour=14), _make_partial(), 0,
        )
        assert features_kz["ict__killzone"] == 1.0

        # 22:00 UTC is no killzone
        features_no_kz = ext.extract(
            _make_tick(hour=22), _make_partial(), 0,
        )
        assert features_no_kz["ict__killzone"] == 0.0

    def test_cold_start_no_crash(self) -> None:
        """Extracting features with no candle history should not crash."""
        ext = ICTFeatureExtractor()
        ext.configure({})
        features = ext.extract(_make_tick(), _make_partial(), 0)
        assert features["ict__m5_fvg_distance"] == 0.0
        assert features["ict__m5_trend"] == 0.0
        assert features["ict__h1_trend"] == 0.0

    def test_m5_events_after_30_candles(self) -> None:
        """After 30 10s candles, one M5 candle forms and ICT runs on it."""
        ext = ICTFeatureExtractor()
        ext.configure({})

        for i in range(30):
            ext.on_candle_close(
                _make_candle(i, open_=100 + i * 0.1, close=100.1 + i * 0.1,
                             high=100.5 + i * 0.1, low=99.5 + i * 0.1),
                i,
            )

        features = ext.extract(_make_tick(price=103.0), _make_partial(), 30)
        assert "ict__m5_trend" in features


class TestFeatureRegistry:
    """Tests for FeatureRegistry composition."""

    def test_register_and_extract(self) -> None:
        registry = FeatureRegistry()
        registry.register(TickFeatureExtractor())
        registry.register(ScalpingFeatureExtractor())
        registry.configure_all({})

        features = registry.extract_all(
            _make_tick(), _make_partial(), 0,
        )

        # Should have features from both extractors
        tick_keys = [k for k in features if k.startswith("tick__")]
        scalp_keys = [k for k in features if k.startswith("scalp__")]
        assert len(tick_keys) > 0
        assert len(scalp_keys) > 0

    def test_all_tunable_params_namespaced(self) -> None:
        registry = FeatureRegistry()
        registry.register(TickFeatureExtractor())
        registry.register(ScalpingFeatureExtractor())

        params = registry.all_tunable_params()
        names = [p.name for p in params]
        assert "tick__spread_avg_period" in names
        assert "scalp__roc_fast" in names
        assert "scalp__atr_period" in names

    def test_configure_all_applies_params(self) -> None:
        registry = FeatureRegistry()
        ext = ScalpingFeatureExtractor()
        registry.register(ext)

        registry.configure_all({"scalp__roc_fast": 10})
        assert ext._roc_fast == 10

    def test_reset_all(self) -> None:
        registry = FeatureRegistry()
        ext = ScalpingFeatureExtractor()
        registry.register(ext)
        registry.configure_all({})
        ext.on_candle_close(_make_candle(0), 0)
        assert len(ext._buffers["10s"].closes) == 1

        registry.reset_all()
        assert len(ext._buffers["10s"].closes) == 0


class TestEndToEndPipeline:
    """End-to-end test: ticks → candle builder → extractors → features."""

    def test_tick_stream_produces_features(self) -> None:
        """Feed a stream of ticks through the full pipeline."""
        builder = CandleBuilder(bucket_seconds=10)
        registry = FeatureRegistry()
        registry.register(TickFeatureExtractor())
        registry.register(ScalpingFeatureExtractor())
        registry.configure_all({})

        base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        feature_rows: list[dict[str, float]] = []

        # Generate 50 ticks (~5 candles)
        for i in range(50):
            tick = Tick(
                time=base + timedelta(seconds=i),
                bid=1400.0 + i * 0.1,
                ask=1400.6 + i * 0.1,
            )

            closed = builder.process_tick(tick)
            if closed is not None:
                registry.on_candle_close(closed, builder.candle_index - 1)

            partial = builder.partial
            if partial is not None:
                features = registry.extract_all(
                    tick, partial, builder.candle_index,
                )
                feature_rows.append(features)

        assert len(feature_rows) == 50
        # After enough candles, scalping features should be non-zero
        last = feature_rows[-1]
        assert "tick__spread" in last
        assert "scalp__10s_roc_fast" in last
        assert last["tick__spread"] == pytest.approx(0.6)
