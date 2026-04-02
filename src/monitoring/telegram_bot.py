"""Telegram bot for trade alerts and bot control.

Sends notifications on trade events and provides
commands: /status, /pause, /resume, /stats.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.common.logging import get_logger

if TYPE_CHECKING:
    from telegram import Bot

logger = get_logger(__name__)


class TelegramNotifier:
    """Sends trading notifications via Telegram.

    Args:
        bot_token: Telegram bot token.
        chat_id: Target chat/group ID.
    """

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._token = bot_token
        self._chat_id = chat_id
        self._bot: Bot | None = None

    async def connect(self) -> None:
        """Initialize the Telegram bot."""
        from telegram import Bot

        self._bot = Bot(token=self._token)
        await logger.ainfo("telegram_bot_connected")

    async def send_message(self, text: str) -> None:
        """Send a text message to the configured chat.

        Args:
            text: Message text (supports Markdown).
        """
        if not self._bot:
            await self.connect()
        assert self._bot is not None

        try:
            await self._bot.send_message(
                chat_id=self._chat_id,
                text=text,
                parse_mode="Markdown",
            )
        except Exception as e:
            await logger.awarning("telegram_send_failed", error=str(e))

    async def notify_trade_opened(self, trade_info: dict[str, Any]) -> None:
        """Send notification for a new trade.

        Args:
            trade_info: Dict with trade details.
        """
        msg = (
            f"🟢 *Trade Opened*\n"
            f"Instrument: `{trade_info.get('instrument', 'N/A')}`\n"
            f"Direction: `{trade_info.get('direction', 'N/A')}`\n"
            f"Entry: `{trade_info.get('entry_price', 'N/A')}`\n"
            f"SL: `{trade_info.get('stop_loss', 'N/A')}`\n"
            f"TP: `{trade_info.get('take_profit', 'N/A')}`\n"
            f"Size: `{trade_info.get('size', 'N/A')}`"
        )
        await self.send_message(msg)

    async def notify_trade_closed(self, trade_info: dict[str, Any]) -> None:
        """Send notification for a closed trade.

        Args:
            trade_info: Dict with trade details including PnL.
        """
        pnl = trade_info.get("pnl", 0)
        emoji = "🟢" if pnl >= 0 else "🔴"
        msg = (
            f"{emoji} *Trade Closed*\n"
            f"Instrument: `{trade_info.get('instrument', 'N/A')}`\n"
            f"PnL: `{pnl}`\n"
            f"R:R: `{trade_info.get('r_multiple', 'N/A')}`"
        )
        await self.send_message(msg)

    async def notify_alert(self, title: str, message: str) -> None:
        """Send a general alert.

        Args:
            title: Alert title.
            message: Alert body.
        """
        await self.send_message(f"⚠️ *{title}*\n{message}")

    async def send_status(self, portfolio_summary: dict[str, Any]) -> None:
        """Send portfolio status summary.

        Args:
            portfolio_summary: Dict from Portfolio.get_summary().
        """
        msg = (
            f"📊 *Portfolio Status*\n"
            f"Capital: `{portfolio_summary.get('capital', 'N/A')}`\n"
            f"Equity: `{portfolio_summary.get('equity', 'N/A')}`\n"
            f"Open positions: `{portfolio_summary.get('open_positions', 0)}`\n"
            f"Realized PnL: `{portfolio_summary.get('realized_pnl', 'N/A')}`"
        )
        await self.send_message(msg)
