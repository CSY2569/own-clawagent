"""ApiKeyPool manager — thread-safe key selection with exponential backoff."""

from __future__ import annotations

import random
import threading
import time
from collections import defaultdict
from typing import Any

from clawagent.api_pool.models import KeyRecord, KeyStatus, PoolConfig, PoolStrategy


class ApiKeyPool:
    """Thread-safe API key pool with fault transfer and exponential backoff."""

    def __init__(self) -> None:
        self._pools: dict[str, PoolConfig] = {}
        self._rr_index: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def add_pool(self, config: PoolConfig) -> None:
        with self._lock:
            self._pools[config.name] = config

    def get_key(self, pool_name: str = "default") -> KeyRecord | None:
        """Select an active key from the named pool. Thread-safe."""
        with self._lock:
            pool = self._pools.get(pool_name)
            if not pool:
                return None

            now = time.time()
            for k in pool.keys:
                if k.status == KeyStatus.DEGRADED and (now - k.last_error_at) >= k.cooldown_seconds:
                    k.status = KeyStatus.ACTIVE
                    k.error_count = 0

            active = [k for k in pool.keys if k.status == KeyStatus.ACTIVE]
            if not active:
                return None

            if pool.strategy == PoolStrategy.ROUND_ROBIN:
                idx = self._rr_index[pool_name] % len(active)
                self._rr_index[pool_name] += 1
                return active[idx]
            elif pool.strategy == PoolStrategy.RANDOM:
                return random.choice(active)
            else:  # LEAST_ERRORS
                return min(active, key=lambda k: k.error_count)

    def mark_error(self, key: KeyRecord, status_code: int) -> None:
        """Mark a key as errored with exponential backoff cooldown."""
        with self._lock:
            key.error_count += 1
            key.last_error_at = time.time()

            pool = self._pools.get(key.pool_name)
            max_cd = pool.max_cooldown if pool else 600.0

            if status_code == 401:
                key.status = KeyStatus.DISABLED
            elif status_code == 429:
                key.cooldown_seconds = min(key.cooldown_seconds * 2, max_cd)
                key.status = KeyStatus.DEGRADED
            else:
                key.cooldown_seconds = min(key.cooldown_seconds * 2, 300)
                key.status = KeyStatus.DEGRADED

    def mark_success(self, key: KeyRecord) -> None:
        """Reset error state after a successful call."""
        with self._lock:
            if key.error_count > 0:
                key.error_count = 0
            key.cooldown_seconds = 30.0

    def record_usage(self, key: KeyRecord, input_tokens: int, output_tokens: int) -> None:
        """Record token usage for a key."""
        with self._lock:
            key.total_input_tokens += input_tokens
            key.total_output_tokens += output_tokens

    def get_pool_stats(self, pool_name: str) -> dict[str, Any]:
        """Return statistics for a named pool."""
        with self._lock:
            pool = self._pools.get(pool_name)
            if not pool:
                return {}
            return {
                "name": pool_name,
                "strategy": pool.strategy.value,
                "total": len(pool.keys),
                "active": sum(1 for k in pool.keys if k.status == KeyStatus.ACTIVE),
                "degraded": sum(1 for k in pool.keys if k.status == KeyStatus.DEGRADED),
                "disabled": sum(1 for k in pool.keys if k.status in (KeyStatus.DISABLED, KeyStatus.EXHAUSTED)),
                "total_input_tokens": sum(k.total_input_tokens for k in pool.keys),
                "total_output_tokens": sum(k.total_output_tokens for k in pool.keys),
            }

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Return statistics for all pools."""
        return {name: self.get_pool_stats(name) for name in self._pools}
