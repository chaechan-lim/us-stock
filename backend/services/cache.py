"""Redis cache service for KIS token storage and market data caching."""

import json
import logging
from typing import Any

import redis.asyncio as redis

logger = logging.getLogger(__name__)


class CacheService:
    """Async Redis cache with typed helpers."""

    def __init__(self, url: str = "redis://localhost:6379/1"):
        self._url = url
        self._redis: redis.Redis | None = None

    async def initialize(self) -> None:
        self._redis = redis.from_url(self._url, decode_responses=True)
        try:
            await self._redis.ping()
            logger.info("Redis connected: %s", self._url)
        except Exception as e:
            logger.warning("Redis unavailable (%s), running without cache", e)
            self._redis = None

    async def close(self) -> None:
        if self._redis:
            await self._redis.close()

    @property
    def available(self) -> bool:
        return self._redis is not None

    async def get(self, key: str) -> str | None:
        if not self._redis:
            return None
        try:
            return await self._redis.get(key)
        except Exception as e:
            logger.warning("Redis GET failed for %s: %s", key, e)
            return None

    async def set(self, key: str, value: str, ex: int | None = None) -> bool:
        if not self._redis:
            return False
        try:
            await self._redis.set(key, value, ex=ex)
            return True
        except Exception as e:
            logger.warning("Redis SET failed for %s: %s", key, e)
            return False

    async def delete(self, key: str) -> bool:
        if not self._redis:
            return False
        try:
            await self._redis.delete(key)
            return True
        except Exception as e:
            logger.warning("Redis DELETE failed for %s: %s", key, e)
            return False

    async def get_json(self, key: str) -> dict | None:
        raw = await self.get(key)
        if raw:
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                logger.warning("Invalid JSON in cache key %s, ignoring", key)
                return None
        return None

    async def set_json(self, key: str, value: dict, ex: int | None = None) -> bool:
        return await self.set(key, json.dumps(value), ex=ex)
