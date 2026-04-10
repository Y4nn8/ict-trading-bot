"""CandleBuilder: aggregate ticks into fixed-duration candles on the fly.

Maintains a running partial candle. When a tick arrives in a new time
bucket, the current candle is closed and emitted.

Also provides CandleAggregator for building higher-TF candles from
lower-TF candles (e.g. 10s → M1/M5/H1).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from src.midas.types import PartialCandle, Tick


class CandleAggregator:
    """Aggregates lower-TF candles into a higher-TF candle.

    Counts incoming candles. When ``candles_per_bucket`` candles have
    been consumed, emits a closed HTF candle.

    Args:
        candles_per_bucket: Number of source candles per output candle.
            M1 = 6 (60s / 10s), M5 = 30, H1 = 360.
    """

    def __init__(self, candles_per_bucket: int) -> None:
        self._candles_per_bucket = candles_per_bucket
        self._count = 0
        self._open: float = 0.0
        self._high: float = 0.0
        self._low: float = 0.0
        self._close: float = 0.0
        self._time: Any = None
        self._active = False

    def add(self, candle: dict[str, Any]) -> dict[str, Any] | None:
        """Add a source candle, returning a HTF candle if one closes."""
        o = float(candle["open"])
        h = float(candle["high"])
        lo = float(candle["low"])
        c = float(candle["close"])

        if not self._active:
            self._time = candle["time"]
            self._open = o
            self._high = h
            self._low = lo
            self._close = c
            self._count = 1
            self._active = True
        else:
            self._high = max(self._high, h)
            self._low = min(self._low, lo)
            self._close = c
            self._count += 1

        if self._count >= self._candles_per_bucket:
            closed = {
                "time": self._time,
                "open": self._open,
                "high": self._high,
                "low": self._low,
                "close": self._close,
            }
            self._active = False
            self._count = 0
            return closed
        return None

    def reset(self) -> None:
        """Reset aggregator state."""
        self._count = 0
        self._active = False


class CandleBuilder:
    """Builds fixed-duration candles from a stream of ticks.

    Uses mid price (bid+ask)/2 for OHLC. Stores raw bid/ask in the
    partial candle for spread features.

    Args:
        bucket_seconds: Duration of each candle bucket in seconds.
    """

    def __init__(self, bucket_seconds: int = 10) -> None:
        self._bucket_seconds = bucket_seconds
        self._bucket_td = timedelta(seconds=bucket_seconds)
        self._partial: PartialCandle | None = None
        self._candle_index: int = 0

    @property
    def partial(self) -> PartialCandle | None:
        """Current partial (in-progress) candle."""
        return self._partial

    @property
    def candle_index(self) -> int:
        """Index of the current candle being built."""
        return self._candle_index

    def _bucket_start(self, time: datetime) -> datetime:
        """Truncate a timestamp to its bucket boundary."""
        ts = time.timestamp()
        bucket_ts = (int(ts) // self._bucket_seconds) * self._bucket_seconds
        return datetime.fromtimestamp(bucket_ts, tz=time.tzinfo)

    def process_tick(self, tick: Tick) -> dict[str, Any] | None:
        """Process a tick, returning a closed candle dict if one completed.

        Args:
            tick: The incoming tick.

        Returns:
            A closed candle dict (time, open, high, low, close, tick_count,
            bid, ask) if a candle just closed, else None.
        """
        mid = tick.mid
        bucket = self._bucket_start(tick.time)

        if self._partial is None:
            # First tick ever
            self._partial = PartialCandle(
                bucket_start=bucket,
                open=mid,
                high=mid,
                low=mid,
                close=mid,
                tick_count=1,
                bid=tick.bid,
                ask=tick.ask,
                elapsed_seconds=0.0,
            )
            return None

        if bucket == self._partial.bucket_start:
            # Same bucket — update partial candle
            self._partial.high = max(self._partial.high, mid)
            self._partial.low = min(self._partial.low, mid)
            self._partial.close = mid
            self._partial.tick_count += 1
            self._partial.bid = tick.bid
            self._partial.ask = tick.ask
            elapsed = (tick.time - self._partial.bucket_start).total_seconds()
            self._partial.elapsed_seconds = min(
                elapsed, float(self._bucket_seconds),
            )
            return None

        # New bucket — close current candle and start fresh
        closed = self._close_candle()
        self._candle_index += 1
        self._partial = PartialCandle(
            bucket_start=bucket,
            open=mid,
            high=mid,
            low=mid,
            close=mid,
            tick_count=1,
            bid=tick.bid,
            ask=tick.ask,
            elapsed_seconds=0.0,
        )
        return closed

    def _close_candle(self) -> dict[str, Any]:
        """Convert the current partial candle to a closed candle dict."""
        p = self._partial
        if p is None:
            msg = "Cannot close candle: no partial candle in progress"
            raise RuntimeError(msg)
        return {
            "time": p.bucket_start,
            "open": p.open,
            "high": p.high,
            "low": p.low,
            "close": p.close,
            "tick_count": p.tick_count,
            "bid": p.bid,
            "ask": p.ask,
        }

    def flush(self) -> dict[str, Any] | None:
        """Force-close the current partial candle (end of replay).

        Returns:
            The closed candle dict, or None if no partial candle exists.
        """
        if self._partial is None:
            return None
        closed = self._close_candle()
        self._candle_index += 1
        self._partial = None
        return closed

    def reset(self) -> None:
        """Reset builder state for a new replay run."""
        self._partial = None
        self._candle_index = 0
