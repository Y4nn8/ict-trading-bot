"""Database connection management using asyncpg."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import asyncpg

from src.common.exceptions import DatabaseError
from src.common.logging import get_logger

if TYPE_CHECKING:
    from src.common.config import DatabaseConfig

logger = get_logger(__name__)


class Database:
    """Async PostgreSQL connection pool manager.

    Args:
        config: Database configuration.
    """

    def __init__(self, config: DatabaseConfig) -> None:
        self._config = config
        self._pool: asyncpg.Pool[asyncpg.Record] | None = None

    @property
    def pool(self) -> asyncpg.Pool[asyncpg.Record]:
        """Get the connection pool, raising if not connected."""
        if self._pool is None:
            raise DatabaseError("Database not connected. Call connect() first.")
        return self._pool

    async def connect(self) -> None:
        """Establish the connection pool."""
        try:
            self._pool = await asyncpg.create_pool(
                self._config.url,
                min_size=self._config.min_connections,
                max_size=self._config.max_connections,
            )
            await logger.ainfo("database_connected", url=self._config.url.split("@")[-1])
        except Exception as e:
            raise DatabaseError(f"Failed to connect to database: {e}") from e

    async def disconnect(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            await logger.ainfo("database_disconnected")

    async def execute(self, query: str, *args: Any) -> str:
        """Execute a query and return the status string."""
        try:
            result: str = await self.pool.execute(query, *args)
            return result
        except Exception as e:
            raise DatabaseError(f"Query execution failed: {e}") from e

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        """Execute a query and return all rows."""
        try:
            result: list[Any] = await self.pool.fetch(query, *args)
            return result
        except Exception as e:
            raise DatabaseError(f"Query fetch failed: {e}") from e

    async def fetchrow(self, query: str, *args: Any) -> asyncpg.Record | None:
        """Execute a query and return a single row."""
        try:
            return await self.pool.fetchrow(query, *args)
        except Exception as e:
            raise DatabaseError(f"Query fetchrow failed: {e}") from e

    async def fetchval(self, query: str, *args: Any) -> Any:
        """Execute a query and return a single value."""
        try:
            return await self.pool.fetchval(query, *args)
        except Exception as e:
            raise DatabaseError(f"Query fetchval failed: {e}") from e
