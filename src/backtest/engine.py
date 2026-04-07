"""Backtest engine: M5 event loop consuming pre-computed data.

Processes candles one by one, evaluates strategy signals,
manages simulated positions, and records trades.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from src.backtest.simulator import SimulationConfig, simulate_fill
from src.common.logging import get_logger
from src.common.models import Direction, Trade
from src.news.event_manager import EventManager
from src.news.interpreter import NewsAction

if TYPE_CHECKING:
    from datetime import datetime

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
    initial_stop_loss: float = 0.0  # Immutable SL at entry for R-multiple
    trigger_source: str = "ict"  # "ict" or "news"
    setup_context: dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Result of a backtest run."""

    trades: list[Trade]
    open_positions: list[OpenPosition]
    start_time: datetime | None = None
    end_time: datetime | None = None
    margin_rejected: int = 0
    margin_capped: int = 0
    peak_margin_usage_pct: float = 0.0


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
        leverage: ESMA leverage for this instrument (e.g. 30 for forex).
        news_events: Pre-loaded news events for replay.
        news_pause_minutes: Minutes to pause before a news event.
        news_resume_minutes: Minutes to resume after a news event.
        be_trigger_pct: Move SL to BE when price reaches this % of TP distance (0=off).
        be_offset_pct: Place new SL at entry + this % of TP distance (0=exact BE).
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
        sim_config: SimulationConfig | None = None,
        initial_capital: float = 10000.0,
        leverage: float = 30.0,
        value_per_point: float = 1.0,
        min_size: float = 0.5,
        size_step: float = 0.5,
        avg_spread: float = 0.0,
        pip_size: float = 0.0001,
        news_events: list[dict[str, Any]] | None = None,
        news_pause_minutes: int = 30,
        news_resume_minutes: int = 15,
        ms_lookback_candles: int = 1,
        be_trigger_pct: float = 0.0,
        be_offset_pct: float = 0.0,
        min_stop_distance: float = 0.0,
    ) -> None:
        if leverage <= 0:
            raise ValueError(f"leverage must be positive, got {leverage}")
        self._precomputed = precomputed
        self._confluence = confluence_scorer
        self._entry = entry_evaluator
        self._exit = exit_evaluator
        self._filter = trade_filter
        self._sizer = position_sizer
        self._risk = risk_manager
        self._sim_config = sim_config or SimulationConfig()
        self._initial_capital = initial_capital
        self._capital = initial_capital
        self._leverage = leverage
        # Convert from "value per pip" (IG terminology) to "value per 1.0 price unit"
        # For forex: 1.0 / 0.0001 = 10,000; for indices/gold (pip_size=1.0): no change
        self._value_per_price_unit = value_per_point / pip_size if pip_size > 0 else value_per_point
        self._min_size = min_size
        self._size_step = size_step
        self._avg_spread = avg_spread
        self._pip_size = pip_size
        self._open_positions: list[OpenPosition] = []
        self._closed_trades: list[Trade] = []
        self._daily_pnl: float = 0.0
        self._current_day: int = -1
        self._margin_rejected: int = 0
        self._margin_capped: int = 0
        self._peak_margin_usage: float = 0.0
        self._be_trigger_pct = be_trigger_pct
        self._be_offset_pct = be_offset_pct
        self._min_stop_distance = min_stop_distance
        self._be_applied: set[str] = set()  # trade_ids already moved to BE
        self._ms_lookback = max(1, ms_lookback_candles)
        self._event_manager = EventManager(
            pre_event_pause_minutes=news_pause_minutes,
            post_event_resume_minutes=news_resume_minutes,
        )
        self._news_by_minute: dict[tuple[int, int, int, int, int], list[dict[str, Any]]] = {}
        if news_events:
            for event in news_events:
                t = event.get("time")
                if t is not None and hasattr(t, "timetuple"):
                    tt = t.timetuple()
                    key = (tt.tm_year, tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min)
                    self._news_by_minute.setdefault(key, []).append(event)

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

            # Collect MS breaks within lookback window
            recent_ms: list[dict[str, Any]] = []
            for j in range(max(0, i - self._ms_lookback + 1), i + 1):
                recent_ms.extend(ms_by_idx.get(j, []))

            # Build context
            context: dict[str, Any] = {
                "fvgs": fvg_by_idx.get(i, []),
                "order_blocks": ob_by_idx.get(i, []),
                "ms_breaks": recent_ms,
                "displacements": disp_by_idx.get(i, []),
                "session": candle.get("session", ""),
                "killzone": candle.get("killzone", ""),
                "in_killzone": candle.get("in_killzone", False),
            }
            if self._precomputed.htf_trend is not None:
                context["htf_trend"] = str(self._precomputed.htf_trend[i])

            # Check for news-triggered entries (per-instrument)
            news_triggered = False
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
                    news_triggered = True
                elif inst_sent == "bearish":
                    context["ms_breaks"] = [{"direction": "bearish"}]
                    news_triggered = True

            trigger_source = "news" if news_triggered else "ict"

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

            # Simulate realistic fill (spread + slippage + rejection)
            is_buy = entry_signal.direction == Direction.LONG
            fill = simulate_fill(
                target_price=entry_signal.entry_price,
                is_buy=is_buy,
                spread=self._avg_spread,
                config=self._sim_config,
                pip_size=self._pip_size,
            )
            if not fill.filled:
                continue

            # Adjust SL/TP by slippage only (not spread).
            # LONG entries fill on ask (bid+spread) but SL/TP exit on bid,
            # so the spread component must not shift exit levels.
            spread_component = self._avg_spread if is_buy else 0.0
            slippage_offset = fill.fill_price - entry_signal.entry_price - spread_component
            actual_entry = fill.fill_price
            actual_sl = entry_signal.stop_loss + slippage_offset
            actual_tp = entry_signal.take_profit + slippage_offset

            # Size the position using the actual fill price
            size = self._sizer.compute_size(
                capital=self._capital,
                confluence_score=score,
                entry_price=actual_entry,
                stop_loss=actual_sl,
                value_per_price_unit=self._value_per_price_unit,
                min_size=self._min_size,
                size_step=self._size_step,
            )

            if size <= 0:
                continue

            # Cap size to available margin
            current_price = float(candle["close"])
            original_size = size
            size, used_margin, equity = self._cap_size_to_margin(
                actual_entry, size, current_price,
            )
            if size <= 0:
                self._margin_rejected += 1
                continue
            if size < original_size:
                self._margin_capped += 1

            # Open position
            position = OpenPosition(
                trade_id=str(uuid4()),
                instrument=self._precomputed.instrument,
                direction=entry_signal.direction,
                entry_price=actual_entry,
                entry_time=time,
                stop_loss=actual_sl,
                take_profit=actual_tp,
                size=size,
                confluence_score=score,
                initial_stop_loss=actual_sl,
                trigger_source=trigger_source,
                setup_context=context,
            )
            self._open_positions.append(position)

            # Track peak margin usage (reuse values, add new position's margin)
            new_margin = actual_entry * size * self._value_per_price_unit / self._leverage
            total_used = used_margin + new_margin
            if equity > 0:
                usage_pct = total_used / equity * 100
                self._peak_margin_usage = max(self._peak_margin_usage, usage_pct)

        return BacktestResult(
            trades=self._closed_trades,
            open_positions=self._open_positions,
            start_time=start_time,
            end_time=end_time,
            margin_rejected=self._margin_rejected,
            margin_capped=self._margin_capped,
            peak_margin_usage_pct=self._peak_margin_usage,
        )

    def _compute_used_margin(self) -> float:
        """Compute total margin used by open positions."""
        vpp = self._value_per_price_unit
        return sum(
            pos.entry_price * pos.size * vpp / self._leverage
            for pos in self._open_positions
        )

    def _compute_unrealized_pnl(self, current_price: float) -> float:
        """Compute total unrealized PnL at current bid price."""
        pnl = 0.0
        vpp = self._value_per_price_unit
        for pos in self._open_positions:
            if pos.direction == Direction.LONG:
                # LONG sells at bid
                pnl += (current_price - pos.entry_price) * pos.size * vpp
            else:
                # SHORT buys at ask (bid + spread)
                ask = current_price + self._avg_spread
                pnl += (pos.entry_price - ask) * pos.size * vpp
        return pnl

    def _compute_equity(self, current_price: float) -> float:
        """Compute equity = capital + unrealized PnL."""
        return self._capital + self._compute_unrealized_pnl(current_price)

    def _cap_size_to_margin(
        self, entry_price: float, size: float, current_price: float
    ) -> tuple[float, float, float]:
        """Reduce position size if needed to fit within available margin.

        Args:
            entry_price: Entry price of the new position.
            size: Desired position size.
            current_price: Current market price for equity calculation.

        Returns:
            Tuple of (capped_size, used_margin, equity).
        """
        if entry_price <= 0:
            return 0.0, 0.0, 0.0
        used_margin = self._compute_used_margin()
        equity = self._compute_equity(current_price)
        available_margin = equity - used_margin
        if available_margin <= 0:
            return 0.0, used_margin, equity
        vpp = self._value_per_price_unit
        max_size = available_margin * self._leverage / (entry_price * vpp)
        capped = min(size, max_size)
        return capped, used_margin, equity

    def _check_exits(self, candle: dict[str, Any], index: int) -> None:
        """Check exit conditions for all open positions."""
        remaining: list[OpenPosition] = []
        high = float(candle["high"])
        low = float(candle["low"])
        time = candle["time"]
        spread = self._avg_spread
        # Bid-based candles: ask = bid + spread
        ask_high = high + spread
        ask_low = low + spread

        for pos in self._open_positions:
            # Move SL to breakeven if trigger reached
            if (
                self._be_trigger_pct > 0
                and pos.trade_id not in self._be_applied
            ):
                tp_dist = abs(pos.take_profit - pos.entry_price)
                if pos.direction == Direction.LONG:
                    progress = high - pos.entry_price
                else:
                    progress = pos.entry_price - ask_low
                if tp_dist > 0 and progress / tp_dist >= self._be_trigger_pct:
                    if pos.direction == Direction.LONG:
                        new_sl = pos.entry_price + tp_dist * self._be_offset_pct
                        # Enforce min stop distance from current price
                        max_sl = float(candle["close"]) - self._min_stop_distance
                        new_sl = min(new_sl, max_sl)
                    else:
                        new_sl = pos.entry_price - tp_dist * self._be_offset_pct
                        min_sl = float(candle["close"]) + spread + self._min_stop_distance
                        new_sl = max(new_sl, min_sl)
                    # Only move SL if it improves the position (tighter)
                    improves = (
                        (pos.direction == Direction.LONG and new_sl > pos.stop_loss)
                        or (pos.direction == Direction.SHORT and new_sl < pos.stop_loss)
                    )
                    if improves:
                        pos.stop_loss = new_sl
                        self._be_applied.add(pos.trade_id)

            exit_result = self._exit.evaluate(pos, candle)

            if exit_result is not None:
                # Time-based exit: LONG sells at bid, SHORT buys at ask
                if pos.direction == Direction.LONG:
                    exit_price = exit_result.exit_price
                else:
                    exit_price = exit_result.exit_price + spread
            elif pos.direction == Direction.LONG:
                # LONG exits sell at bid (candle prices are already bid)
                if low <= pos.stop_loss:
                    exit_price = pos.stop_loss
                elif high >= pos.take_profit:
                    exit_price = pos.take_profit
                else:
                    remaining.append(pos)
                    continue
            elif pos.direction == Direction.SHORT:
                # SHORT exits buy at ask; trigger on ask prices
                if ask_high >= pos.stop_loss:
                    exit_price = pos.stop_loss
                elif ask_low <= pos.take_profit:
                    exit_price = pos.take_profit
                else:
                    remaining.append(pos)
                    continue
            else:
                remaining.append(pos)
                continue

            # Close the trade
            pnl = self._compute_pnl(pos, exit_price)
            price_delta = abs(exit_price - pos.entry_price)
            pnl_pct = (price_delta / pos.entry_price) * 100 if pos.entry_price > 0 else 0
            if pnl < 0:
                pnl_pct = -pnl_pct
            risk = abs(pos.entry_price - pos.initial_stop_loss)
            r_multiple = (price_delta / risk) if risk > 0 else 0
            if pnl < 0:
                r_multiple = -r_multiple

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
                setup_type={"trigger_source": pos.trigger_source},
                context={"value_per_price_unit": self._value_per_price_unit},
                is_backtest=True,
            )
            self._closed_trades.append(trade)
            self._capital += pnl
            self._daily_pnl += pnl
            self._be_applied.discard(pos.trade_id)

        self._open_positions = remaining

    def _compute_pnl(self, pos: OpenPosition, exit_price: float) -> float:
        """Compute PnL for a closed position."""
        if pos.direction == Direction.LONG:
            return (exit_price - pos.entry_price) * pos.size * self._value_per_price_unit
        return (pos.entry_price - exit_price) * pos.size * self._value_per_price_unit

    def _close_opposing_positions(
        self, candle: dict[str, Any], inst_sentiments: dict[str, str]
    ) -> None:
        """Close positions that oppose the news sentiment per instrument.

        Uses per-instrument sentiments. If "__all__" key exists,
        applies to all instruments.
        """
        raw_close = float(candle["close"])
        spread = self._avg_spread
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
                # Bid-based candles: LONG sells at bid, SHORT buys at ask
                close_price = raw_close if pos.direction == Direction.LONG else raw_close + spread
                pnl = self._compute_pnl(pos, close_price)
                price_delta = abs(close_price - pos.entry_price)
                pnl_pct = (price_delta / pos.entry_price) * 100 if pos.entry_price > 0 else 0
                if pnl < 0:
                    pnl_pct = -pnl_pct
                risk = abs(pos.entry_price - pos.initial_stop_loss)
                r_multiple = (price_delta / risk) if risk > 0 else 0
                if pnl < 0:
                    r_multiple = -r_multiple

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
                    setup_type={"trigger_source": pos.trigger_source},
                    context={"value_per_price_unit": self._value_per_price_unit},
                    is_backtest=True,
                    news_context={"close_reason": "news_opposing", "sentiment": sentiment},
                ))
                self._capital += pnl
                self._daily_pnl += pnl
                self._be_applied.discard(pos.trade_id)
            else:
                remaining.append(pos)

        self._open_positions = remaining

    def _replay_news(self, time: Any) -> None:
        """Check for news events at this timestamp and apply actions."""
        if not self._news_by_minute:
            return
        if not hasattr(time, "timetuple"):
            return
        tt = time.timetuple()
        key = (tt.tm_year, tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min)
        events = self._news_by_minute.get(key, [])
        for event in events:
            raw_analysis = event.get("llm_analysis") or {}
            if isinstance(raw_analysis, str):
                import json
                try:
                    analysis: dict[str, Any] = json.loads(raw_analysis)
                except json.JSONDecodeError:
                    analysis = {}
            else:
                analysis = raw_analysis
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
