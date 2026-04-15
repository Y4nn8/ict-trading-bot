"""Tests for Morpheus validation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from src.morpheus.validation import (
    autocorrelation,
    autocorrelation_batched,
    hourly_volatility,
    ks_2samp,
    validate_trajectories,
)

# ---------------------------------------------------------------------------
# ks_2samp
# ---------------------------------------------------------------------------

class TestKS2Samp:
    def test_identical_samples_near_zero(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 500)
        stat, p = ks_2samp(x, x)
        assert stat == 0.0
        assert p == pytest.approx(1.0, abs=1e-6)

    def test_shifted_distributions_detected(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 1000)
        y = rng.normal(3, 1, 1000)
        stat, p = ks_2samp(x, y)
        assert stat > 0.5
        assert p < 0.05

    def test_same_distribution_large_sample_non_significant(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 1000)
        y = rng.normal(0, 1, 1000)
        stat, p = ks_2samp(x, y)
        assert stat < 0.2
        assert p > 0.05

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            ks_2samp(np.array([]), np.array([1.0]))

    def test_pvalue_in_unit_interval(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 100)
        y = rng.normal(0, 1, 100)
        _, p = ks_2samp(x, y)
        assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# autocorrelation
# ---------------------------------------------------------------------------

class TestAutocorrelation:
    def test_white_noise_acf_near_zero(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 10_000)
        acf = autocorrelation(x, max_lag=5)
        assert acf.shape == (5,)
        assert np.all(np.abs(acf) < 0.05)

    def test_ar1_positive_acf(self) -> None:
        # Generate AR(1) with phi=0.7 — lag-1 ACF should be ~0.7.
        rng = np.random.default_rng(0)
        n = 5000
        phi = 0.7
        x = np.zeros(n)
        eps = rng.normal(0, 1, n)
        for i in range(1, n):
            x[i] = phi * x[i - 1] + eps[i]
        acf = autocorrelation(x, max_lag=3)
        assert acf[0] == pytest.approx(phi, abs=0.05)
        assert acf[0] > acf[1] > acf[2]

    def test_constant_series_returns_zeros(self) -> None:
        x = np.full(100, 3.0)
        acf = autocorrelation(x, max_lag=3)
        np.testing.assert_array_equal(acf, np.zeros(3))

    def test_rejects_invalid_max_lag(self) -> None:
        with pytest.raises(ValueError, match="max_lag"):
            autocorrelation(np.zeros(10), max_lag=0)
        with pytest.raises(ValueError, match="max_lag"):
            autocorrelation(np.zeros(10), max_lag=20)

    def test_batched_matches_mean_of_individual(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, (5, 200))
        manual = np.mean(
            [autocorrelation(row, max_lag=3) for row in x], axis=0,
        )
        batched = autocorrelation_batched(x, max_lag=3)
        np.testing.assert_allclose(batched, manual)

    def test_batched_rejects_non_2d(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            autocorrelation_batched(np.zeros(10), max_lag=1)


# ---------------------------------------------------------------------------
# hourly_volatility
# ---------------------------------------------------------------------------

class TestHourlyVolatility:
    def test_shape(self) -> None:
        returns = np.random.default_rng(0).normal(0, 1, 200)
        hours = (np.arange(200) % 24).astype(np.int32)
        hv = hourly_volatility(returns, hours)
        assert hv.shape == (24,)
        assert np.all(np.isfinite(hv))

    def test_missing_hours_nan(self) -> None:
        returns = np.array([1.0, -1.0, 2.0])
        hours = np.array([0, 1, 2], dtype=np.int32)
        hv = hourly_volatility(returns, hours)
        assert hv[0] == 1.0
        assert hv[1] == 1.0
        assert hv[2] == 2.0
        assert np.isnan(hv[3])

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="shape mismatch"):
            hourly_volatility(np.zeros(3), np.zeros(4, dtype=np.int32))


# ---------------------------------------------------------------------------
# validate_trajectories integration
# ---------------------------------------------------------------------------

class TestValidateTrajectories:
    def test_matching_data_has_low_divergence(self) -> None:
        rng = np.random.default_rng(0)
        n_samples, horizon = 32, 64
        real = rng.normal(0, 1, (n_samples, horizon))
        imag = rng.normal(0, 1, (n_samples, horizon))
        hours = np.tile(np.arange(horizon) % 24, (n_samples, 1)).astype(np.int32)
        report = validate_trajectories(real, imag, hours, max_lag=5)
        assert 0.0 <= report.ks_statistic <= 1.0
        assert report.ks_statistic < 0.2
        assert report.acf_real.shape == (5,)
        assert report.acf_imag.shape == (5,)
        assert report.hourly_vol_real.shape == (24,)
        assert report.hourly_vol_imag.shape == (24,)

    def test_divergent_data_high_ks(self) -> None:
        rng = np.random.default_rng(0)
        n_samples, horizon = 32, 64
        real = rng.normal(0, 1, (n_samples, horizon))
        imag = rng.normal(5, 1, (n_samples, horizon))
        hours = np.tile(np.arange(horizon) % 24, (n_samples, 1)).astype(np.int32)
        report = validate_trajectories(real, imag, hours, max_lag=5)
        assert report.ks_statistic > 0.8
        assert report.ks_pvalue < 0.05

    def test_shape_mismatch_raises(self) -> None:
        rng = np.random.default_rng(0)
        real = rng.normal(0, 1, (4, 10))
        imag = rng.normal(0, 1, (4, 11))
        hours = np.zeros((4, 10), dtype=np.int32)
        with pytest.raises(ValueError, match="shapes must match"):
            validate_trajectories(real, imag, hours)
