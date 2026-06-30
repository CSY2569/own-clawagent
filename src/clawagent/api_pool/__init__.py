"""API Key pool — automatic key failover with exponential backoff."""

from __future__ import annotations

from clawagent.api_pool.loader import load_pools_from_env
from clawagent.api_pool.models import KeyRecord, KeyStatus, PoolConfig, PoolStrategy
from clawagent.api_pool.pool import ApiKeyPool
from clawagent.api_pool.wrapper import KeyPoolChatModel

__all__ = [
    "ApiKeyPool",
    "KeyPoolChatModel",
    "KeyRecord",
    "KeyStatus",
    "PoolConfig",
    "PoolStrategy",
    "get_global_pool",
    "init_global_pool",
]

_GLOBAL_POOL: ApiKeyPool | None = None


def init_global_pool() -> ApiKeyPool:
    """Initialize the global key pool from environment variables."""
    global _GLOBAL_POOL
    pool = ApiKeyPool()
    configs = load_pools_from_env()
    for cfg in configs:
        pool.add_pool(cfg)
    _GLOBAL_POOL = pool
    return pool


def get_global_pool() -> ApiKeyPool:
    """Get the global key pool, initializing it on first call if needed."""
    if _GLOBAL_POOL is None:
        return init_global_pool()
    return _GLOBAL_POOL
