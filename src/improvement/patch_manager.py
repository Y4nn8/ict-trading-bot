"""Apply and rollback strategy parameter patches."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Patch:
    """A parameter patch to apply."""

    patch_id: str
    params: dict[str, Any]
    previous_params: dict[str, Any] = field(default_factory=dict)
    applied: bool = False


class PatchManager:
    """Manages applying and rolling back parameter changes.

    Keeps a history of patches for rollback capability.
    """

    def __init__(self) -> None:
        self._history: list[Patch] = []
        self._current_params: dict[str, Any] = {}

    def set_baseline(self, params: dict[str, Any]) -> None:
        """Set the baseline parameters.

        Args:
            params: Current parameter values.
        """
        self._current_params = dict(params)

    def apply_patch(self, patch: Patch) -> dict[str, Any]:
        """Apply a parameter patch.

        Args:
            patch: The patch to apply.

        Returns:
            New combined parameters after patch.
        """
        patch.previous_params = dict(self._current_params)
        self._current_params.update(patch.params)
        patch.applied = True
        self._history.append(patch)

        logger.info("patch_applied", patch_id=patch.patch_id, params=patch.params)
        return dict(self._current_params)

    def rollback_last(self) -> dict[str, Any]:
        """Rollback the last applied patch.

        Returns:
            Parameters after rollback.
        """
        if not self._history:
            return dict(self._current_params)

        last = self._history.pop()
        self._current_params = dict(last.previous_params)
        logger.info("patch_rolled_back", patch_id=last.patch_id)
        return dict(self._current_params)

    @property
    def current_params(self) -> dict[str, Any]:
        """Get current parameters."""
        return dict(self._current_params)

    @property
    def history(self) -> list[Patch]:
        """Get patch history."""
        return list(self._history)
