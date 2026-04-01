"""Tests for structured logging setup."""

from __future__ import annotations

from src.common.logging import get_logger, setup_logging


class TestLogging:
    """Tests for logging configuration."""

    def test_setup_logging_json(self) -> None:
        setup_logging(log_level="INFO", json_format=True)
        logger = get_logger("test")
        # After setup, logger should be usable (bound to stdlib)
        assert hasattr(logger, "info")
        assert hasattr(logger, "warning")

    def test_setup_logging_console(self) -> None:
        setup_logging(log_level="DEBUG", json_format=False)
        logger = get_logger("test")
        assert hasattr(logger, "debug")

    def test_get_logger_has_async_methods(self) -> None:
        setup_logging()
        logger = get_logger("test.module")
        assert hasattr(logger, "ainfo")
        assert hasattr(logger, "awarning")
