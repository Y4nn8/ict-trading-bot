"""Validation metrics for Morpheus imagined trajectories.

Compares statistics of imagined trajectories (produced by
``WorldModel.imagine``) to those of the corresponding real
trajectories.  A world model that captures true market dynamics
should match these statistics; one that only captures noise will
diverge on at least one of them.

Metrics:
  - Return distribution (two-sample Kolmogorov-Smirnov test)
  - Autocorrelation of returns (lead/lag structure)
  - Autocorrelation of squared returns (volatility clustering / ARCH)
  - Intraday (hourly) volatility profile
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class ValidationReport:
    """Summary of trajectory validation metrics (real vs imagined)."""

    ks_statistic: float
    ks_pvalue: float
    acf_real: NDArray[np.float64]
    acf_imag: NDArray[np.float64]
    acf_max_abs_diff: float
    acf_sq_real: NDArray[np.float64]
    acf_sq_imag: NDArray[np.float64]
    acf_sq_max_abs_diff: float
    hourly_vol_real: NDArray[np.float64]
    hourly_vol_imag: NDArray[np.float64]
    hourly_vol_max_abs_diff: float


def ks_2samp(x: NDArray[np.floating], y: NDArray[np.floating]) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test.

    Args:
        x: First sample (1D).
        y: Second sample (1D).

    Returns:
        (statistic, p_value).  statistic is in [0, 1]; lower means
        distributions are closer.  p_value is the asymptotic two-sided
        p-value (Smirnov series).
    """
    if x.size == 0 or y.size == 0:
        msg = "ks_2samp requires non-empty samples"
        raise ValueError(msg)

    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    all_sorted = np.sort(np.concatenate([x, y]))
    cdf_x = np.searchsorted(x_sorted, all_sorted, side="right") / x.size
    cdf_y = np.searchsorted(y_sorted, all_sorted, side="right") / y.size
    d = float(np.max(np.abs(cdf_x - cdf_y)))

    if d == 0.0:
        return 0.0, 1.0

    n_eff = (x.size * y.size) / (x.size + y.size)
    en = math.sqrt(n_eff)
    lam = (en + 0.12 + 0.11 / en) * d

    q = 0.0
    for j in range(1, 101):
        term = 2.0 * ((-1.0) ** (j - 1)) * math.exp(-2.0 * j * j * lam * lam)
        q += term
        if abs(term) < 1e-10:
            break
    p_value = float(max(0.0, min(1.0, q)))
    return d, p_value


def autocorrelation(series: NDArray[np.floating], max_lag: int) -> NDArray[np.float64]:
    """Sample autocorrelation function up to ``max_lag`` (1..max_lag inclusive).

    Uses the biased estimator (divides by total variance, not the
    variance of the truncated subseries) so values at all lags share a
    consistent scale.
    """
    if max_lag < 1:
        msg = f"max_lag must be >= 1, got {max_lag}"
        raise ValueError(msg)
    if series.size <= max_lag:
        msg = f"series length {series.size} must exceed max_lag {max_lag}"
        raise ValueError(msg)

    centered = series.astype(np.float64) - series.mean()
    variance = float((centered * centered).mean())
    if variance == 0.0:
        return np.zeros(max_lag, dtype=np.float64)

    acf = np.zeros(max_lag, dtype=np.float64)
    n = centered.size
    for lag in range(1, max_lag + 1):
        acf[lag - 1] = float((centered[:-lag] * centered[lag:]).sum() / (n * variance))
    return acf


def autocorrelation_batched(
    x: NDArray[np.floating], max_lag: int,
) -> NDArray[np.float64]:
    """Per-row ACF averaged across rows — avoids discontinuities between rows."""
    if x.ndim != 2:
        msg = f"expected 2D array, got shape {x.shape}"
        raise ValueError(msg)
    acfs = np.stack([autocorrelation(row, max_lag) for row in x])
    result: NDArray[np.float64] = acfs.mean(axis=0).astype(np.float64)
    return result


def hourly_volatility(
    returns: NDArray[np.floating], hours: NDArray[np.integer],
) -> NDArray[np.float64]:
    """Mean absolute return per hour-of-day (0..23).

    Args:
        returns: Flat array of returns.
        hours: Same shape; hour-of-day (0..23) for each return.

    Returns:
        Array of length 24.  Hours with no samples are filled with NaN.
    """
    if returns.shape != hours.shape:
        msg = f"shape mismatch: returns {returns.shape} vs hours {hours.shape}"
        raise ValueError(msg)
    abs_ret = np.abs(returns.astype(np.float64))
    out = np.full(24, np.nan, dtype=np.float64)
    for h in range(24):
        mask = hours == h
        if mask.any():
            out[h] = float(abs_ret[mask].mean())
    return out


def _max_abs_diff_ignore_nan(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """``max(|a - b|)`` ignoring positions where either side is NaN."""
    diff = np.abs(a - b)
    finite = np.isfinite(diff)
    if not finite.any():
        return float("nan")
    return float(diff[finite].max())


def validate_trajectories(
    real_returns: NDArray[np.floating],
    imag_returns: NDArray[np.floating],
    hours: NDArray[np.integer],
    *,
    max_lag: int = 20,
) -> ValidationReport:
    """Compare imagined to real trajectories across four metric families.

    Args:
        real_returns: (n_samples, horizon) — realised returns.
        imag_returns: (n_samples, horizon) — imagined returns.
        hours: (n_samples, horizon) — hour-of-day (0..23) per step.
        max_lag: Number of lags for ACF comparison.

    Returns:
        ValidationReport with paired real/imag statistics and the max
        absolute differences for the ACF / hourly-vol series.
    """
    if real_returns.shape != imag_returns.shape:
        msg = (
            f"real and imag shapes must match: "
            f"{real_returns.shape} vs {imag_returns.shape}"
        )
        raise ValueError(msg)
    if hours.shape != real_returns.shape:
        msg = f"hours shape {hours.shape} must match returns {real_returns.shape}"
        raise ValueError(msg)

    real_flat = real_returns.reshape(-1)
    imag_flat = imag_returns.reshape(-1)
    hours_flat = hours.reshape(-1)

    ks_stat, ks_p = ks_2samp(real_flat, imag_flat)

    acf_real = autocorrelation_batched(real_returns, max_lag)
    acf_imag = autocorrelation_batched(imag_returns, max_lag)

    acf_sq_real = autocorrelation_batched(real_returns ** 2, max_lag)
    acf_sq_imag = autocorrelation_batched(imag_returns ** 2, max_lag)

    hv_real = hourly_volatility(real_flat, hours_flat)
    hv_imag = hourly_volatility(imag_flat, hours_flat)

    return ValidationReport(
        ks_statistic=ks_stat,
        ks_pvalue=ks_p,
        acf_real=acf_real,
        acf_imag=acf_imag,
        acf_max_abs_diff=float(np.max(np.abs(acf_real - acf_imag))),
        acf_sq_real=acf_sq_real,
        acf_sq_imag=acf_sq_imag,
        acf_sq_max_abs_diff=float(np.max(np.abs(acf_sq_real - acf_sq_imag))),
        hourly_vol_real=hv_real,
        hourly_vol_imag=hv_imag,
        hourly_vol_max_abs_diff=_max_abs_diff_ignore_nan(hv_real, hv_imag),
    )
