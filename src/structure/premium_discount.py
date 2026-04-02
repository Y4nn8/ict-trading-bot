"""Premium/Discount zone detection using Fibonacci levels.

Based on the current swing range (latest swing high to latest swing low),
divides price into:
- Premium zone: above 50% (equilibrium) — overpriced, look to sell
- Discount zone: below 50% (equilibrium) — underpriced, look to buy
- Optimal Trade Entry (OTE): 62-79% retracement zone
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import polars as pl


class PriceZone(StrEnum):
    """Price zone relative to the swing range."""

    PREMIUM = "premium"
    DISCOUNT = "discount"
    EQUILIBRIUM = "equilibrium"


@dataclass(frozen=True, slots=True)
class PremiumDiscountLevels:
    """Computed premium/discount levels for a swing range."""

    swing_high: float
    swing_low: float
    equilibrium: float
    ote_high: float  # 62% retracement
    ote_low: float  # 79% retracement


def compute_pd_levels(swing_high: float, swing_low: float) -> PremiumDiscountLevels:
    """Compute premium/discount and OTE levels for a swing range.

    Args:
        swing_high: The swing high price.
        swing_low: The swing low price.

    Returns:
        PremiumDiscountLevels with equilibrium and OTE boundaries.
    """
    range_size = swing_high - swing_low
    return PremiumDiscountLevels(
        swing_high=swing_high,
        swing_low=swing_low,
        equilibrium=swing_low + range_size * 0.5,
        ote_high=swing_low + range_size * 0.62,
        ote_low=swing_low + range_size * 0.79,
    )


def classify_price_zone(price: float, levels: PremiumDiscountLevels) -> PriceZone:
    """Classify a price as premium, discount, or equilibrium.

    Args:
        price: Current price to classify.
        levels: The premium/discount levels.

    Returns:
        The PriceZone classification.
    """
    if price > levels.equilibrium:
        return PriceZone.PREMIUM
    if price < levels.equilibrium:
        return PriceZone.DISCOUNT
    return PriceZone.EQUILIBRIUM


def is_in_ote(price: float, levels: PremiumDiscountLevels) -> bool:
    """Check if price is within the Optimal Trade Entry zone.

    Args:
        price: Current price.
        levels: The premium/discount levels.

    Returns:
        True if price is between OTE high and OTE low.
    """
    lower = min(levels.ote_high, levels.ote_low)
    upper = max(levels.ote_high, levels.ote_low)
    return lower <= price <= upper


def detect_pd_zones_vectorized(
    df: pl.DataFrame,
    swing_high: float,
    swing_low: float,
) -> pl.DataFrame:
    """Classify each candle's close into premium/discount zones.

    Args:
        df: DataFrame with column: close.
        swing_high: Current swing high.
        swing_low: Current swing low.

    Returns:
        DataFrame with columns: time, close, zone, in_ote.
    """
    levels = compute_pd_levels(swing_high, swing_low)

    result = df.select("time", "close").with_columns(
        pl.when(pl.col("close") > levels.equilibrium)
        .then(pl.lit(PriceZone.PREMIUM))
        .when(pl.col("close") < levels.equilibrium)
        .then(pl.lit(PriceZone.DISCOUNT))
        .otherwise(pl.lit(PriceZone.EQUILIBRIUM))
        .alias("zone"),
        (
            (pl.col("close") >= levels.ote_low) & (pl.col("close") <= levels.ote_high)
        ).alias("in_ote"),
    )

    return result
