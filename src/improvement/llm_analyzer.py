"""Weekly LLM-based structural analysis using Claude Sonnet.

Analyzes recent trades and proposes structural code changes
(new filters, rule modifications, parameter adjustments).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.common.logging import get_logger

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic

    from src.improvement.trade_logger import TradeContext

logger = get_logger(__name__)


@dataclass
class LLMProposal:
    """A structural improvement proposed by the LLM."""

    description: str
    category: str  # "filter", "rule", "parameter", "exit_logic"
    changes: list[dict[str, Any]]
    expected_impact: str
    confidence: float


class LLMAnalyzer:
    """Weekly structural analysis using Claude Sonnet.

    Args:
        client: Anthropic async client.
        model: Model to use.
        max_iterations: Max improvement iterations per cycle.
    """

    def __init__(
        self,
        client: AsyncAnthropic,
        model: str = "claude-sonnet-4-6-20250514",
        max_iterations: int = 5,
    ) -> None:
        self._client = client
        self._model = model
        self._max_iterations = max_iterations

    async def analyze_trades(
        self,
        trades: list[TradeContext],
        current_config: dict[str, Any],
    ) -> list[LLMProposal]:
        """Analyze recent trades and propose improvements.

        Args:
            trades: Recent trade contexts to analyze.
            current_config: Current strategy configuration.

        Returns:
            List of improvement proposals.
        """
        if not trades:
            return []

        prompt = self._build_analysis_prompt(trades, current_config)

        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )

            block = response.content[0] if response.content else None
            text = block.text if block and hasattr(block, "text") else ""
            proposals = self._parse_proposals(text)

            await logger.ainfo(
                "llm_analysis_complete",
                n_trades_analyzed=len(trades),
                n_proposals=len(proposals),
            )
            return proposals[:self._max_iterations]

        except Exception as e:
            await logger.awarning("llm_analysis_failed", error=str(e))
            return []

    def _build_analysis_prompt(
        self,
        trades: list[TradeContext],
        config: dict[str, Any],
    ) -> str:
        """Build the analysis prompt from trade data."""
        # Summarize trades
        winners = [t for t in trades if t.pnl and t.pnl > 0]
        losers = [t for t in trades if t.pnl and t.pnl < 0]
        total_pnl = sum(t.pnl or 0 for t in trades)

        trade_summary = f"""
Total trades: {len(trades)}
Winners: {len(winners)}, Losers: {len(losers)}
Total PnL: {total_pnl:.2f}
Win rate: {len(winners)/len(trades)*100:.1f}%

Common losing patterns:
"""
        for t in losers[:10]:
            trade_summary += (
                f"- {t.instrument} {t.direction} at {t.entry_price}, "
                f"SL hit, confluence={t.confluence_score:.2f}, "
                f"session={t.session}, killzone={t.killzone}\n"
            )

        return f"""You are analyzing trading performance for an ICT/SMC strategy bot.

{trade_summary}

Current config: {config}

Propose up to 3 specific, actionable improvements. For each:
PROPOSAL: [one-line description]
CATEGORY: [filter|rule|parameter|exit_logic]
EXPECTED_IMPACT: [one-line expected effect]
CONFIDENCE: [0.0-1.0]
---"""

    def _parse_proposals(self, text: str) -> list[LLMProposal]:
        """Parse LLM response into proposals."""
        proposals: list[LLMProposal] = []
        current: dict[str, Any] = {}

        for line in text.strip().split("\n"):
            line = line.strip()
            if line.startswith("PROPOSAL:"):
                if current.get("description"):
                    proposals.append(self._make_proposal(current))
                current = {"description": line.split(":", 1)[1].strip()}
            elif line.startswith("CATEGORY:"):
                current["category"] = line.split(":", 1)[1].strip().lower()
            elif line.startswith("EXPECTED_IMPACT:"):
                current["expected_impact"] = line.split(":", 1)[1].strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    current["confidence"] = float(line.split(":", 1)[1].strip())
                except ValueError:
                    current["confidence"] = 0.5

        if current.get("description"):
            proposals.append(self._make_proposal(current))

        return proposals

    @staticmethod
    def _make_proposal(data: dict[str, Any]) -> LLMProposal:
        return LLMProposal(
            description=data.get("description", ""),
            category=data.get("category", "parameter"),
            changes=[],
            expected_impact=data.get("expected_impact", ""),
            confidence=data.get("confidence", 0.5),
        )
