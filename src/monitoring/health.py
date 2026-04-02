"""Health check endpoints and system monitoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from src.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ComponentHealth:
    """Health status of a single component."""

    name: str
    healthy: bool
    last_check: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    details: str = ""


class HealthChecker:
    """Monitors health of all bot components.

    Tracks database, broker, news sources, and processing pipeline.
    """

    def __init__(self) -> None:
        self._components: dict[str, ComponentHealth] = {}
        self._last_candle_time: datetime | None = None
        self._last_heartbeat: datetime = datetime.now(tz=UTC)

    def register_component(self, name: str) -> None:
        """Register a component for health tracking.

        Args:
            name: Component name.
        """
        self._components[name] = ComponentHealth(name=name, healthy=False)

    def update_health(
        self, name: str, healthy: bool, details: str = ""
    ) -> None:
        """Update the health status of a component.

        Args:
            name: Component name.
            healthy: Whether the component is healthy.
            details: Optional details about the status.
        """
        self._components[name] = ComponentHealth(
            name=name,
            healthy=healthy,
            details=details,
        )

    def record_heartbeat(self) -> None:
        """Record a processing heartbeat."""
        self._last_heartbeat = datetime.now(tz=UTC)

    def record_candle_processed(self, candle_time: datetime) -> None:
        """Record that a candle was processed.

        Args:
            candle_time: The timestamp of the processed candle.
        """
        self._last_candle_time = candle_time
        self.record_heartbeat()

    def is_healthy(self) -> bool:
        """Check if all components are healthy.

        Returns:
            True if all registered components are healthy.
        """
        if not self._components:
            return True
        return all(c.healthy for c in self._components.values())

    def get_status(self) -> dict[str, Any]:
        """Get full health status report.

        Returns:
            Dict with overall status and per-component details.
        """
        return {
            "healthy": self.is_healthy(),
            "last_heartbeat": self._last_heartbeat.isoformat(),
            "last_candle_time": (
                self._last_candle_time.isoformat()
                if self._last_candle_time
                else None
            ),
            "components": {
                name: {
                    "healthy": comp.healthy,
                    "last_check": comp.last_check.isoformat(),
                    "details": comp.details,
                }
                for name, comp in self._components.items()
            },
        }
