"""Backtest engine: M5 event loop consuming pre-computed data.

Processes candles one by one, evaluates strategy signals,
manages simulated positions, and records trades.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from src.common.logging import get_logger
from src.common.models import Direction, Trade
from src.news.event_manager import EventManager
from src.news.interpreter import NewsAction

if TYPE_CHECKING:
    from datetime import datetime

    from src.backtest.simulator import SimulationConfig
    from src.backtest.vectorized import PrecomputedData
    from src.execution.position_sizer import PositionSizer
    from src.execution.risk_manager import RiskManager
    from src.strategy.confluence import ConfluenceScorer
    from src.strategy.entry import EntryEvaluator
    from src.strategy.exit import ExitEvaluator
    from src.strategy.filters import TradeFilter

logger = get_logger(__name__)


@dataclass
class OpenPosition:
    """A currently open backtest position."""

    trade_id: str
    instrument: str
    direction: Direction
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    size: float
    confluence_score: float
    setup_context: dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Result of a backtest run."""

    trades: list[Trade]
    open_positions: list[OpenPosition]
    start_time: datetime | None = None
    end_time: datetime | None = None


class BacktestEngine:
    """M5 event loop backtest engine.

    Consumes pre-computed detector data and evaluates strategy rules
    candle by candle.

    Args:
        precomputed: Pre-computed structure data.
        confluence_scorer: Strategy confluence scoring.
        entry_evaluator: Entry condition evaluator.
        exit_evaluator: Exit condition evaluator.
        trade_filter: Trade filters (spread, session, etc.).
        position_sizer: Dynamic position sizing.
        risk_manager: Circuit breakers.
        sim_config: Simulation configuration.
        initial_capital: Starting capital.
    """

    def __init__(
        self,
        precomputed: PrecomputedData,
        confluence_scorer: ConfluenceScorer,
        entry_evaluator: EntryEvaluator,
        exit_evaluator: ExitEvaluator,
        trade_filter: TradeFilter,
        position_sizer: PositionSizer,
        risk_manager: RiskManager,
        sim_config: SimulationConfig,
        initial_capital: float = 10000.0,
        news_events: list[dict[str, Any]] | None = None,
        news_pause_minutes: int = 30,
        news_resume_minutes: int = 15,
    ) -> None:
        self._precomputed = precomputed
        self._confluence = confluence_scorer
        self._entry = entry_evaluator
        self._exit = exit_evaluator
        self._filter = trade_filter
        self._sizer = position_sizer
        self._risk = risk_manager
        self._sim_config = sim_config
        self._initial_capital = initial_capital
        self._capital = initial_capital
        self._open_positions: list[OpenPosition] = []
        self._closed_trades: list[Trade] = []
        self._daily_pnl: float = 0.0
        self._current_day: int = -1
        self._event_manager = EventManager(
            pre_event_pause_minutes=news_pause_minutes,
            post_event_resume_minutes=news_resume_minutes,
        )
        self._news_by_time: dict[str, list[dict[str, Any]]] = {}
        if news_events:
            for event in news_events:
                # Index by minute-rounded time for quick lookup
                t = event.get("time")
                if t is not None:
                    key = str(t)[:16]  # "2026-03-15 13:30"
                    self._news_by_time.setdefault(key, []).append(event)

    def run(self) -> BacktestResult:
        """Execute the backtest over all candles.

        Returns:
            BacktestResult with all closed and open trades.
        """
        candles = self._precomputed.candles
        n = len(candles)

        if n == 0:
            return BacktestResult(trades=[], open_positions=[])

        start_time = candles["time"][0]
        end_time = candles["time"][n - 1]

        # Build lookup indices for pre-computed data
        fvg_by_idx = self._build_index(self._precomputed.fvgs)
        ob_by_idx = self._build_index(self._precomputed.order_blocks)
        ms_by_idx = self._build_index(self._precomputed.market_structure)
        disp_by_idx = self._build_index(self._precomputed.displacements)

        for i in range(n):
            candle = candles.row(i, named=True)
            time = candle["time"]

            # Reset daily PnL tracking
            day = time.timetuple().tm_yday if hasattr(time, "timetuple") else -1
            if day != self._current_day:
                self._daily_pnl = 0.0
                self._current_day = day

            # Replay news events at this timestamp
            self._replay_news(time)

            # Close positions opposing a strong news sentiment
            inst_sentiments = self._event_manager.get_instrument_sentiments(time)
            if inst_sentiments:
                self._close_opposing_positions(candle, inst_sentiments)

            # Check circuit breakers or news pause
            if self._risk.is_circuit_broken(
                self._daily_pnl, self._capital, self._initial_capital
            ) or self._event_manager.is_paused(time):
                self._check_exits(candle, i)
                continue

            # Check exits on open positions
            self._check_exits(candle, i)

            # Build context
            context: dict[str, Any] = {
                "fvgs": fvg_by_idx.get(i, []),
                "order_blocks": ob_by_idx.get(i, []),
                "ms_breaks": ms_by_idx.get(i, []),
                "displacements": disp_by_idx.get(i, []),
                "session": candle.get("session", ""),
                "killzone": candle.get("killzone", ""),
                "in_killzone": candle.get("in_killzone", False),
            }

            # Check for news-triggered entries (per-instrument)
            news_triggers = self._event_manager.pop_triggers()
            for trigger in news_triggers:
                inst_sents = trigger.get("instrument_sentiments", {})
                if inst_sents:
                    inst_sent = inst_sents.get(
                        self._precomputed.instrument,
                        inst_sents.get("__all__", "none"),
                    )
                else:
                    inst_sent = trigger.get("sentiment", "neutral")

                if inst_sent == "bullish":
                    context["ms_breaks"] = [{"direction": "bullish"}]
                elif inst_sent == "bearish":
                    context["ms_breaks"] = [{"direction": "bearish"}]

            # Score confluence
            score = self._confluence.score(candle, context)

            # Apply filters
            if not self._filter.passes(
                candle,
                context,
                len(self._open_positions),
            ):
                continue

            # Evaluate entry
            entry_signal = self._entry.evaluate(candle, context, score)
            if entry_signal is None:
                continue

            # Size the position
            size = self._sizer.compute_size(
                capital=self._capital,
                confluence_score=score,
                entry_price=entry_signal.entry_price,
                stop_loss=entry_signal.stop_loss,
            )

            if size <= 0:
                continue

            # Open position
            position = OpenPosition(
                trade_id=str(uuid4()),
                instrument=self._precomputed.instrument,
                direction=entry_signal.direction,
                entry_price=entry_signal.entry_price,
                entry_time=time,
                stop_loss=entry_signal.stop_loss,
                take_profit=entry_signal.take_profit,
                size=size,
                confluence_score=score,
                setup_context=context,
            )
            self._open_positions.append(position)

        return BacktestResult(
            trades=self._closed_trades,
            open_positions=self._open_positions,
            start_time=start_time,
            end_time=end_time,
        )

    def _check_exits(self, candle: dict[str, Any], index: int) -> None:
        """Check exit conditions for all open positions."""
        remaining: list[OpenPosition] = []
        high = float(candle["high"])
        low = float(candle["low"])
        time = candle["time"]

        for pos in self._open_positions:
            exit_result = self._exit.evaluate(pos, candle)

            if exit_result is not None:
                exit_price = exit_result.exit_price
            elif pos.direction == Direction.LONG:
                if low <= pos.stop_loss:
                    exit_price = pos.stop_loss
                elif high >= pos.take_profit:
                    exit_price = pos.take_profit
                else:
                    remaining.append(pos)
                    continue
            elif pos.direction == Direction.SHORT:
                if high >= pos.stop_loss:
                    exit_price = pos.stop_loss
                elif low <= pos.take_profit:
                    exit_price = pos.take_profit
                else:
                    remaining.append(pos)
                    continue
            else:
                remaining.append(pos)
                continue

            # Close the trade
            pnl = self._compute_pnl(pos, exit_price)
            pnl_pct = (pnl / (pos.entry_price * pos.size)) * 100 if pos.size > 0 else 0
            risk = abs(pos.entry_price - pos.stop_loss)
            r_multiple = pnl / (risk * pos.size) if risk > 0 and pos.size > 0 else 0

            trade = Trade(
                opened_at=pos.entry_time,
                closed_at=time,
                instrument=pos.instrument,
                direction=pos.direction,
                entry_price=pos.entry_price,
                exit_price=exit_price,
                stop_loss=pos.stop_loss,
                take_profit=pos.take_profit,
                size=pos.size,
                pnl=pnl,
                pnl_percent=pnl_pct,
                r_multiple=r_multiple,
                confluence_score=pos.confluence_score,
                is_backtest=True,
            )
            self._closed_trades.append(trade)
            self._capital += pnl
            self._daily_pnl += pnl

        self._open_positions = remaining

    def _compute_pnl(self, pos: OpenPosition, exit_price: float) -> float:
        """Compute PnL for a closed position."""
        if pos.direction == Direction.LONG:
            return (exit_price - pos.entry_price) * pos.size
        return (pos.entry_price - exit_price) * pos.size

    def _close_opposing_positions(
        self, candle: dict[str, Any], inst_sentiments: dict[str, str]
    ) -> None:
        """Close positions that oppose the news sentiment per instrument.

        Uses per-instrument sentiments. If "__all__" key exists,
        applies to all instruments.
        """
        close_price = float(candle["close"])
        time = candle["time"]
        remaining: list[OpenPosition] = []

        for pos in self._open_positions:
            sentiment = inst_sentiments.get(
                pos.instrument, inst_sentiments.get("__all__", "")
            )
            should_close = (
                (sentiment == "bullish" and pos.direction == Direction.SHORT)
                or (sentiment == "bearish" and pos.direction == Direction.LONG)
            )
            if should_close:
                pnl = self._compute_pnl(pos, close_price)
                pnl_pct = (
                    (pnl / (pos.entry_price * pos.size)) * 100
                    if pos.size > 0
                    else 0
                )
                risk = abs(pos.entry_price - pos.stop_loss)
                r_multiple = (
                    pnl / (risk * pos.size) if risk > 0 and pos.size > 0 else 0
                )

                self._closed_trades.append(Trade(
                    opened_at=pos.entry_time,
                    closed_at=time,
                    instrument=pos.instrument,
                    direction=pos.direction,
                    entry_price=pos.entry_price,
                    exit_price=close_price,
                    stop_loss=pos.stop_loss,
                    take_profit=pos.take_profit,
                    size=pos.size,
                    pnl=pnl,
                    pnl_percent=pnl_pct,
                    r_multiple=r_multiple,
                    confluence_score=pos.confluence_score,
                    is_backtest=True,
                    news_context={"close_reason": "news_opposing", "sentiment": sentiment},
                ))
                self._capital += pnl
                self._daily_pnl += pnl
            else:
                remaining.append(pos)

        self._open_positions = remaining

    def _replay_news(self, time: Any) -> None:
        """Check for news events at this timestamp and apply actions."""
        if not self._news_by_time:
            return
        key = str(time)[:16]
        events = self._news_by_time.get(key, [])
        for event in events:
            analysis = event.get("llm_analysis") or {}
            action_str = analysis.get("action", "none")
            # Map legacy actions to new ones
            if action_str in ("trigger_entry", "close_opposing"):
                action_str = "directional"
            try:
                action = NewsAction(action_str)
            except ValueError:
                action = NewsAction.NONE
            self._event_manager.apply_action(action, time, analysis)

    @staticmethod
    def _build_index(df: Any) -> dict[int, list[dict[str, Any]]]:
        """Build a lookup dict from pre-computed DataFrame by index column."""
        if df is None or (hasattr(df, "is_empty") and df.is_empty()):
            return {}
        if "index" not in df.columns:
            return {}
        result: dict[int, list[dict[str, Any]]] = {}
        for row in df.to_dicts():
            idx = int(row["index"])
            result.setdefault(idx, []).append(row)
        return result
