"""Trading session and killzone time masks.

Defines major forex session times and ICT killzones.
All times in UTC.

Sessions:
- Asian:   00:00 - 09:00 UTC
- London:  07:00 - 16:00 UTC
- New York: 12:00 - 21:00 UTC

ICT Killzones (high-probability trading windows):
- Asian KZ:      00:00 - 04:00 UTC
- London KZ:     07:00 - 10:00 UTC
- NY KZ:         12:00 - 15:00 UTC
- London Close:  15:00 - 16:00 UTC
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import polars as pl


class Session(StrEnum):
    """Major trading session."""

    ASIAN = "asian"
    LONDON = "london"
    NEW_YORK = "new_york"
    OFF_HOURS = "off_hours"


class Killzone(StrEnum):
    """ICT killzone windows."""

    ASIAN = "asian_kz"
    LONDON_OPEN = "london_open_kz"
    NEW_YORK_OPEN = "ny_open_kz"
    LONDON_CLOSE = "london_close_kz"
    NONE = "no_killzone"


@dataclass(frozen=True, slots=True)
class SessionWindow:
    """A time window defined by start/end hours (UTC)."""

    start_hour: int
    end_hour: int


SESSION_WINDOWS: dict[Session, SessionWindow] = {
    Session.ASIAN: SessionWindow(0, 7),
    Session.LONDON: SessionWindow(7, 12),
    Session.NEW_YORK: SessionWindow(12, 21),
}

KILLZONE_WINDOWS: dict[Killzone, SessionWindow] = {
    Killzone.ASIAN: SessionWindow(0, 4),
    Killzone.LONDON_OPEN: SessionWindow(7, 10),
    Killzone.NEW_YORK_OPEN: SessionWindow(12, 15),
    Killzone.LONDON_CLOSE: SessionWindow(15, 16),
}


def get_session(hour: int) -> Session:
    """Classify an hour (UTC) into a trading session.

    Args:
        hour: Hour of day (0-23) in UTC.

    Returns:
        The active Session.
    """
    for session, window in SESSION_WINDOWS.items():
        if window.start_hour <= hour < window.end_hour:
            return session
    return Session.OFF_HOURS


def get_killzone(hour: int) -> Killzone:
    """Classify an hour (UTC) into a killzone.

    Args:
        hour: Hour of day (0-23) in UTC.

    Returns:
        The active Killzone, or Killzone.NONE.
    """
    for kz, window in KILLZONE_WINDOWS.items():
        if window.start_hour <= hour < window.end_hour:
            return kz
    return Killzone.NONE


def add_session_columns_vectorized(df: pl.DataFrame) -> pl.DataFrame:
    """Add session and killzone columns to a DataFrame.

    Args:
        df: DataFrame with a time column (datetime with timezone).

    Returns:
        DataFrame with added columns: session, killzone, in_killzone.
    """
    hour_col = pl.col("time").dt.hour()

    # Session classification
    session_expr = (
        pl.when((hour_col >= 0) & (hour_col < 7))
        .then(pl.lit(Session.ASIAN))
        .when((hour_col >= 7) & (hour_col < 12))
        .then(pl.lit(Session.LONDON))
        .when((hour_col >= 12) & (hour_col < 21))
        .then(pl.lit(Session.NEW_YORK))
        .otherwise(pl.lit(Session.OFF_HOURS))
    )

    # Killzone classification
    kz_expr = (
        pl.when((hour_col >= 0) & (hour_col < 4))
        .then(pl.lit(Killzone.ASIAN))
        .when((hour_col >= 7) & (hour_col < 10))
        .then(pl.lit(Killzone.LONDON_OPEN))
        .when((hour_col >= 12) & (hour_col < 15))
        .then(pl.lit(Killzone.NEW_YORK_OPEN))
        .when((hour_col >= 15) & (hour_col < 16))
        .then(pl.lit(Killzone.LONDON_CLOSE))
        .otherwise(pl.lit(Killzone.NONE))
    )

    return df.with_columns(
        session_expr.alias("session"),
        kz_expr.alias("killzone"),
        (kz_expr != pl.lit(Killzone.NONE)).alias("in_killzone"),
    )
