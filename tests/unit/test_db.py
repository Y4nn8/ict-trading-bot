"""Tests for the Database connection manager."""

from __future__ import annotations

import pytest

from src.common.config import DatabaseConfig
from src.common.db import Database
from src.common.exceptions import DatabaseError


class TestDatabase:
    """Tests for Database class (unit tests, no real DB connection)."""

    def test_pool_raises_when_not_connected(self) -> None:
        db = Database(DatabaseConfig(url="postgresql://test:test@localhost/test"))
        with pytest.raises(DatabaseError, match="not connected"):
            _ = db.pool

    @pytest.mark.integration
    async def test_connect_and_disconnect(self, db_config: DatabaseConfig) -> None:
        db = Database(db_config)
        await db.connect()
        assert db._pool is not None
        await db.disconnect()
        assert db._pool is None

    @pytest.mark.integration
    async def test_execute_query(self, db_config: DatabaseConfig) -> None:
        db = Database(db_config)
        await db.connect()
        try:
            result = await db.fetchval("SELECT 1")
            assert result == 1
        finally:
            await db.disconnect()
