"""Tests for Midas feature extractors."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from src.midas.candle_builder import CandleBuilder
from src.midas.extractors.ict_features import ICTFeatureExtractor
from src.midas.extractors.macd_features import MACDFeatureExtractor
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

    def test_spread_z_nonzero_after_candle_history(self) -> None:
        ext = TickFeatureExtractor()
        ext.configure({"spread_avg_period": 5})

        # Feed 5 candles with varying spreads
        for i in range(5):
            spread = 0.5 + i * 0.1
            ext.on_candle_close(
                {"time": None, "open": 100, "high": 101, "low": 99,
                 "close": 100, "tick_count": 10,
                 "bid": 100 - spread / 2, "ask": 100 + spread / 2},
                i,
            )

        # Tick with a very wide spread should have positive z-score
        features = ext.extract(
            _make_tick(spread=2.0), _make_partial(), 5,
        )
        assert features["tick__spread_z"] > 1.0  # well above average

        # Tick with a tight spread should have negative z-score
        features_tight = ext.extract(
            _make_tick(spread=0.1), _make_partial(), 5,
        )
        assert features_tight["tick__spread_z"] < -1.0

    def test_reset_clears_state(self) -> None:
        ext = TickFeatureExtractor()
        ext.configure({})
        ext.on_candle_close(
            {"time": None, "open": 100, "high": 101, "low": 99,
             "close": 100, "tick_count": 10, "bid": 99.75, "ask": 100.25},
            0,
        )
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
            assert f"scalp__{tf}_vol_regime" in features
            assert f"scalp__{tf}_trend_regime" in features
        # M5/H1 momentum not produced here (covered by ict__ extractor)
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

    def test_vol_regime_detects_expansion(self) -> None:
        """vol_regime > 1 when recent volatility exceeds long-term avg."""
        ext = ScalpingFeatureExtractor()
        ext.configure({"atr_period": 5, "regime_period": 20})

        # 20 quiet candles (range ~0.5) + 5 loud ones (range ~5)
        for i in range(20):
            ext.on_candle_close(
                _make_candle(i, open_=100, close=100.1,
                             high=100.3, low=99.8),
                i,
            )
        for i in range(20, 25):
            ext.on_candle_close(
                _make_candle(i, open_=100, close=102,
                             high=105, low=100),
                i,
            )

        features = ext.extract(_make_tick(price=102.0), _make_partial(), 25)
        assert features["scalp__10s_vol_regime"] > 1.0

    def test_trend_regime_detects_bullish(self) -> None:
        """trend_regime close to +1 when all recent candles bullish."""
        ext = ScalpingFeatureExtractor()
        ext.configure({"regime_period": 20})

        for i in range(25):
            ext.on_candle_close(
                _make_candle(i, open_=100 + i * 0.5, close=100.5 + i * 0.5,
                             high=101 + i * 0.5, low=99.5 + i * 0.5),
                i,
            )

        features = ext.extract(_make_tick(price=120.0), _make_partial(), 25)
        assert features["scalp__10s_trend_regime"] == 1.0

    def test_trend_regime_detects_chop(self) -> None:
        """trend_regime near 0 when direction alternates."""
        ext = ScalpingFeatureExtractor()
        ext.configure({"regime_period": 20})

        for i in range(25):
            if i % 2 == 0:
                ext.on_candle_close(
                    _make_candle(i, open_=100, close=101,
                                 high=102, low=99),
                    i,
                )
            else:
                ext.on_candle_close(
                    _make_candle(i, open_=100, close=99,
                                 high=101, low=98),
                    i,
                )

        features = ext.extract(_make_tick(price=100.0), _make_partial(), 25)
        assert abs(features["scalp__10s_trend_regime"]) < 0.2

    def test_regime_zero_on_cold_start(self) -> None:
        ext = ScalpingFeatureExtractor()
        ext.configure({"regime_period": 50})
        features = ext.extract(_make_tick(), _make_partial(), 0)
        assert features["scalp__10s_vol_regime"] == 0.0
        assert features["scalp__10s_trend_regime"] == 0.0


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

    def test_ict_with_price_movement(self) -> None:
        """Feed enough candles with a trend to exercise detection paths."""
        ext = ICTFeatureExtractor()
        ext.configure({"lookback": 5, "level_max_age": 50})

        # Feed 90 candles (3 M5 candles) with uptrend
        for i in range(90):
            price = 100.0 + i * 0.5
            ext.on_candle_close(
                _make_candle(i, open_=price, close=price + 0.3,
                             high=price + 1.0, low=price - 0.5),
                i,
            )

        features = ext.extract(
            _make_tick(price=145.0, hour=8), _make_partial(price=145.0), 90,
        )
        # With 3 M5 candles of uptrend, some features should be non-zero
        assert features["ict__killzone"] == 1.0  # 8:00 = London KZ
        # ATR-normalized distances are >= 0
        assert features["ict__m5_fvg_distance"] >= 0.0
        assert features["ict__m5_ob_distance"] >= 0.0

    def test_ict_reset(self) -> None:
        ext = ICTFeatureExtractor()
        ext.configure({})
        for i in range(30):
            ext.on_candle_close(
                _make_candle(i, open_=100, close=101, high=102, low=99),
                i,
            )
        ext.reset()
        features = ext.extract(_make_tick(), _make_partial(), 0)
        assert features["ict__m5_trend"] == 0.0


class TestMACDFeatureExtractor:
    """Tests for MACDFeatureExtractor (M1 + M5)."""

    def test_name(self) -> None:
        ext = MACDFeatureExtractor()
        assert ext.name == "macd"

    def test_no_tunable_params(self) -> None:
        ext = MACDFeatureExtractor()
        assert ext.tunable_params() == []

    def test_extract_cold_start_all_zero(self) -> None:
        """Before any candles, all MACD features return 0.0."""
        ext = MACDFeatureExtractor()
        ext.configure({})
        features = ext.extract(_make_tick(), _make_partial(), 0)
        for tf in ("m1", "m5"):
            assert features[f"macd__{tf}_macd"] == 0.0
            assert features[f"macd__{tf}_signal"] == 0.0
            assert features[f"macd__{tf}_hist"] == 0.0
            for lag in range(1, 6):
                assert features[f"macd__{tf}_hist_lag{lag}"] == 0.0

    def test_expected_feature_keys(self) -> None:
        """Exactly 16 features: 8 per TF x 2 TFs."""
        ext = MACDFeatureExtractor()
        ext.configure({})
        features = ext.extract(_make_tick(), _make_partial(), 0)
        assert len(features) == 16
        expected = set()
        for tf in ("m1", "m5"):
            expected.add(f"macd__{tf}_macd")
            expected.add(f"macd__{tf}_signal")
            expected.add(f"macd__{tf}_hist")
            for lag in range(1, 6):
                expected.add(f"macd__{tf}_hist_lag{lag}")
        assert set(features.keys()) == expected

    def test_m1_activates_after_6_candles(self) -> None:
        """M1 state starts updating after the first 6 10s candles."""
        ext = MACDFeatureExtractor()
        ext.configure({})

        # No M1 candle yet
        for i in range(5):
            ext.on_candle_close(_make_candle(i, open_=100, close=100.1), i)
        features = ext.extract(_make_tick(), _make_partial(), 5)
        assert features["macd__m1_macd"] == 0.0

        # 6th 10s candle closes → 1 M1 candle
        ext.on_candle_close(_make_candle(5, open_=100, close=100.5), 5)
        features = ext.extract(_make_tick(), _make_partial(), 6)
        # With a single M1 close, fast and slow EMAs both = same close,
        # so macd == 0. Still, the state has been touched.
        assert "macd__m1_macd" in features

    def test_m5_activates_only_after_30_candles(self) -> None:
        """M5 state requires 30 10s candles to produce any update."""
        ext = MACDFeatureExtractor()
        ext.configure({})

        # 29 candles: no M5 yet. M1 is already updating (4 M1 closes).
        for i in range(29):
            ext.on_candle_close(
                _make_candle(i, open_=100 + i * 0.1, close=100.1 + i * 0.1),
                i,
            )
        features = ext.extract(_make_tick(), _make_partial(), 29)
        # M5: still cold (no M5 candle closed)
        assert features["macd__m5_macd"] == 0.0

        # 30th closes → 1 M5 candle triggers state update
        ext.on_candle_close(
            _make_candle(29, open_=102.9, close=103.0), 29,
        )
        features = ext.extract(_make_tick(), _make_partial(), 30)
        # Still 0 at this point: first M5 close → fast==slow EMA seed.
        # But hist_history now has 1 entry, so hist exists (= 0).
        assert "macd__m5_hist" in features

    def test_macd_nonzero_on_uptrend(self) -> None:
        """A clean uptrend should produce a positive MACD on M1."""
        ext = MACDFeatureExtractor()
        ext.configure({})

        # Feed 50 M1 candles worth (300x 10s) with a steady uptrend,
        # so EMAs diverge and MACD becomes positive.
        for i in range(300):
            price = 100.0 + i * 0.05
            ext.on_candle_close(
                _make_candle(
                    i, open_=price, close=price + 0.02,
                    high=price + 0.1, low=price - 0.05,
                ),
                i,
            )
        features = ext.extract(_make_tick(price=115.0), _make_partial(), 300)
        # Fast EMA > slow EMA after a sustained rise → MACD > 0
        assert features["macd__m1_macd"] > 0
        assert features["macd__m5_macd"] > 0

    def test_macd_negative_on_downtrend(self) -> None:
        """Sustained downtrend gives a negative MACD on M1 and M5."""
        ext = MACDFeatureExtractor()
        ext.configure({})

        for i in range(300):
            price = 150.0 - i * 0.05
            ext.on_candle_close(
                _make_candle(
                    i, open_=price, close=price - 0.02,
                    high=price + 0.05, low=price - 0.1,
                ),
                i,
            )
        features = ext.extract(_make_tick(price=135.0), _make_partial(), 300)
        assert features["macd__m1_macd"] < 0
        assert features["macd__m5_macd"] < 0

    def test_hist_lag_shifts_correctly(self) -> None:
        """hist_lag1..5 should report the previous N histograms.

        After 8 M1 closes we have 8 histogram values in the history deque
        (maxlen=6 → only the last 6 are retained). hist_lag1 must equal
        the histogram produced one M1 close earlier, etc.
        """
        ext = MACDFeatureExtractor()
        ext.configure({})

        # Feed 8 M1 candles (48x 10s) with varying closes to produce
        # distinct histogram values.
        closes_m1 = [100.0, 101.0, 100.5, 102.0, 101.0, 103.0, 102.5, 104.0]
        for m1_idx, m1_close in enumerate(closes_m1):
            for j in range(6):
                ext.on_candle_close(
                    _make_candle(
                        m1_idx * 6 + j,
                        open_=m1_close, close=m1_close,
                        high=m1_close + 0.1, low=m1_close - 0.1,
                    ),
                    m1_idx * 6 + j,
                )

        features = ext.extract(_make_tick(), _make_partial(), 48)
        current_hist = features["macd__m1_hist"]
        lag1 = features["macd__m1_hist_lag1"]
        lag2 = features["macd__m1_hist_lag2"]
        # Values should differ (price pattern is oscillating upward).
        assert current_hist != lag1
        assert lag1 != lag2
        # All 5 lags must be populated after 8 M1 closes (deque full).
        for lag in range(1, 6):
            assert features[f"macd__m1_hist_lag{lag}"] != 0.0

    def test_no_lookahead_extract_uses_only_closed_candles(self) -> None:
        """extract() with an unrealistically favourable partial candle
        must not change MACD values — the in-progress candle must be
        ignored. We verify by comparing extract output before and after
        we change the partial candle's price.
        """
        ext = MACDFeatureExtractor()
        ext.configure({})

        for i in range(30):
            ext.on_candle_close(
                _make_candle(i, open_=100 + i * 0.1, close=100.1 + i * 0.1),
                i,
            )

        features_cheap = ext.extract(
            _make_tick(price=103.0), _make_partial(price=103.0), 30,
        )
        features_spike = ext.extract(
            _make_tick(price=1000.0), _make_partial(price=1000.0), 30,
        )
        # MACD features must be identical regardless of current tick
        # price — they only depend on closed HTF candles.
        for key in features_cheap:
            assert features_cheap[key] == features_spike[key], key

    def test_reset_clears_state(self) -> None:
        ext = MACDFeatureExtractor()
        ext.configure({})

        for i in range(30):
            ext.on_candle_close(
                _make_candle(i, open_=100 + i, close=101 + i), i,
            )
        features_before = ext.extract(_make_tick(), _make_partial(), 30)
        assert features_before["macd__m1_macd"] != 0.0

        ext.reset()
        features_after = ext.extract(_make_tick(), _make_partial(), 0)
        for tf in ("m1", "m5"):
            assert features_after[f"macd__{tf}_macd"] == 0.0
            assert features_after[f"macd__{tf}_signal"] == 0.0
            assert features_after[f"macd__{tf}_hist"] == 0.0


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
