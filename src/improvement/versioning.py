"""Automatic git tagging for improvement iterations."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime

from src.common.logging import get_logger

logger = get_logger(__name__)


def create_version_tag(
    improvement_type: str,
    iteration: int,
    accepted: bool,
) -> str:
    """Create a git tag for an improvement iteration.

    Args:
        improvement_type: "optuna" or "llm".
        iteration: Iteration number.
        accepted: Whether the improvement was accepted.

    Returns:
        The tag name created.
    """
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
    status = "accepted" if accepted else "rejected"
    tag_name = f"improvement/{improvement_type}/iter-{iteration}/{status}/{timestamp}"

    try:
        subprocess.run(
            ["git", "tag", tag_name],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("version_tagged", tag=tag_name)
    except subprocess.CalledProcessError as e:
        logger.warning("version_tag_failed", tag=tag_name, error=e.stderr)

    return tag_name


def get_current_tag() -> str | None:
    """Get the current git tag if HEAD is tagged.

    Returns:
        Tag name or None.
    """
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--exact-match"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None
