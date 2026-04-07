"""Tests for the Database connection manager."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.common.config import DatabaseConfig
from src.common.db import Database
from src.common.exceptions import DatabaseError


@pytest.fixture
def db_config() -> DatabaseConfig:
    return DatabaseConfig(
        url="postgresql://test:test@localhost:5432/test_trading_bot",
        min_connections=1,
        max_connections=5,
    )


def _make_mock_pool() -> AsyncMock:
    """Create a mock pool that behaves like asyncpg.Pool."""
    pool = AsyncMock()
    pool.close = AsyncMock()
    pool.execute = AsyncMock(return_value="INSERT 0 1")
    pool.fetch = AsyncMock(return_value=[])
    pool.fetchrow = AsyncMock(return_value=None)
    pool.fetchval = AsyncMock(return_value=1)
    pool.executemany = AsyncMock()
    return pool


class TestDatabase:
    """Tests for Database class."""

    def test_pool_raises_when_not_connected(self, db_config: DatabaseConfig) -> None:
        db = Database(db_config)
        with pytest.raises(DatabaseError, match="not connected"):
            _ = db.pool

    async def test_connect_and_disconnect(self, db_config: DatabaseConfig) -> None:
        mock_pool = _make_mock_pool()
        with patch(
            "src.common.db.asyncpg.create_pool",
            new_callable=AsyncMock,
            return_value=mock_pool,
        ):
            db = Database(db_config)
            await db.connect()
            assert db._pool is not None
            await db.disconnect()
            assert db._pool is None
            mock_pool.close.assert_awaited_once()

    async def test_execute_query(self, db_config: DatabaseConfig) -> None:
        mock_pool = _make_mock_pool()
        mock_pool.fetchval = AsyncMock(return_value=42)
        with patch(
            "src.common.db.asyncpg.create_pool",
            new_callable=AsyncMock,
            return_value=mock_pool,
        ):
            db = Database(db_config)
            await db.connect()
            result = await db.fetchval("SELECT 42")
            assert result == 42
            await db.disconnect()

    async def test_connect_failure(self, db_config: DatabaseConfig) -> None:
        with patch(
            "src.common.db.asyncpg.create_pool",
            new_callable=AsyncMock,
            side_effect=Exception("connection refused"),
        ):
            db = Database(db_config)
            with pytest.raises(DatabaseError, match="Failed to connect"):
                await db.connect()

    async def test_fetch(self, db_config: DatabaseConfig) -> None:
        mock_pool = _make_mock_pool()
        mock_pool.fetch = AsyncMock(return_value=[{"id": 1}])
        with patch(
            "src.common.db.asyncpg.create_pool",
            new_callable=AsyncMock,
            return_value=mock_pool,
        ):
            db = Database(db_config)
            await db.connect()
            result = await db.fetch("SELECT * FROM t")
            assert result == [{"id": 1}]

    async def test_execute(self, db_config: DatabaseConfig) -> None:
        mock_pool = _make_mock_pool()
        mock_pool.execute = AsyncMock(return_value="INSERT 0 1")
        with patch(
            "src.common.db.asyncpg.create_pool",
            new_callable=AsyncMock,
            return_value=mock_pool,
        ):
            db = Database(db_config)
            await db.connect()
            result = await db.execute("INSERT INTO t VALUES (1)")
            assert result == "INSERT 0 1"

    async def test_query_failure_raises_database_error(
        self, db_config: DatabaseConfig
    ) -> None:
        mock_pool = _make_mock_pool()
        mock_pool.fetchval = AsyncMock(side_effect=Exception("syntax error"))
        with patch(
            "src.common.db.asyncpg.create_pool",
            new_callable=AsyncMock,
            return_value=mock_pool,
        ):
            db = Database(db_config)
            await db.connect()
            with pytest.raises(DatabaseError, match="Query fetchval failed"):
                await db.fetchval("BAD SQL")
