"""IG Markets order execution manager.

Handles opening and closing positions via the IG REST API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.common.exceptions import ExecutionError
from src.common.logging import get_logger
from src.common.models import Direction

if TYPE_CHECKING:
    from src.market_data.ig_client import IGClient

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class OrderResult:
    """Result of an order execution."""

    deal_id: str
    deal_reference: str
    status: str
    reason: str = ""


class OrderManager:
    """Manages order execution via IG Markets API.

    Args:
        ig_client: Authenticated IG client.
        currency_code: Account currency (e.g. "GBP", "EUR").
    """

    def __init__(self, ig_client: IGClient, currency_code: str = "EUR") -> None:
        self._ig = ig_client
        self._currency = currency_code

    def open_position(
        self,
        epic: str,
        direction: Direction,
        size: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        guaranteed_stop: bool = False,
    ) -> OrderResult:
        """Open a new position on IG.

        Args:
            epic: IG instrument epic.
            direction: LONG or SHORT.
            size: Position size.
            stop_loss: Stop loss price level.
            take_profit: Take profit price level.
            guaranteed_stop: Use guaranteed stop (additional cost).

        Returns:
            OrderResult with deal details.

        Raises:
            ExecutionError: If the order fails.
        """
        ig_direction = "BUY" if direction == Direction.LONG else "SELL"

        try:
            result = self._ig.service.create_open_position(
                epic=epic,
                direction=ig_direction,
                size=size,
                currency_code=self._currency,
                order_type="MARKET",
                expiry="-",
                force_open=True,
                guaranteed_stop=guaranteed_stop,
                stop_level=stop_loss,
                limit_level=take_profit,
            )

            deal_ref = result.get("dealReference", "")
            confirmation = self._ig.service.fetch_deal_by_deal_reference(deal_ref)

            deal_id = confirmation.get("dealId", "")
            status = confirmation.get("dealStatus", "UNKNOWN")
            reason = confirmation.get("reason", "")

            if status != "ACCEPTED":
                logger.warning(
                    "order_rejected",
                    epic=epic,
                    reason=reason,
                    deal_ref=deal_ref,
                )

            logger.info(
                "position_opened",
                epic=epic,
                direction=ig_direction,
                size=size,
                deal_id=deal_id,
                status=status,
            )

            return OrderResult(
                deal_id=deal_id,
                deal_reference=deal_ref,
                status=status,
                reason=reason,
            )

        except Exception as e:
            raise ExecutionError(f"Failed to open position: {e}") from e

    def close_position(
        self,
        deal_id: str,
        direction: Direction,
        size: float,
        epic: str,
    ) -> OrderResult:
        """Close an existing position.

        Args:
            deal_id: The deal ID to close.
            direction: Original position direction.
            size: Size to close.
            epic: Instrument epic.

        Returns:
            OrderResult with close details.

        Raises:
            ExecutionError: If the close fails.
        """
        close_direction = "SELL" if direction == Direction.LONG else "BUY"

        try:
            result = self._ig.service.close_open_position(
                deal_id=deal_id,
                direction=close_direction,
                size=size,
                order_type="MARKET",
                expiry="-",
            )

            deal_ref = result.get("dealReference", "")
            confirmation = self._ig.service.fetch_deal_by_deal_reference(deal_ref)

            return OrderResult(
                deal_id=confirmation.get("dealId", ""),
                deal_reference=deal_ref,
                status=confirmation.get("dealStatus", "UNKNOWN"),
                reason=confirmation.get("reason", ""),
            )

        except Exception as e:
            raise ExecutionError(f"Failed to close position: {e}") from e

    def update_stop_loss(
        self, deal_id: str, new_stop: float
    ) -> OrderResult:
        """Update the stop loss on an open position.

        Args:
            deal_id: The deal ID to update.
            new_stop: New stop loss price.

        Returns:
            OrderResult with update details.
        """
        try:
            result = self._ig.service.update_open_position(
                stop_level=new_stop,
                limit_level=None,
                deal_id=deal_id,
            )
            deal_ref = result.get("dealReference", "")
            confirmation = self._ig.service.fetch_deal_by_deal_reference(deal_ref)

            return OrderResult(
                deal_id=confirmation.get("dealId", ""),
                deal_reference=deal_ref,
                status=confirmation.get("dealStatus", "UNKNOWN"),
            )
        except Exception as e:
            raise ExecutionError(f"Failed to update stop loss: {e}") from e
