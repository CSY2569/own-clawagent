"""Load API key pool configurations from environment variables.

.env format:
    API_POOL_DEEPSEEK_STRATEGY=round_robin
    API_POOL_DEEPSEEK_KEY_1=sk-ds-xxx1
    API_POOL_DEEPSEEK_KEY_2=sk-ds-xxx2
    API_POOL_DEEPSEEK_KEY_1_BASE=https://api.deepseek.com/anthropic
"""

from __future__ import annotations

import os
import re

from clawagent.api_pool.models import KeyRecord, PoolConfig, PoolStrategy


def load_pools_from_env() -> list[PoolConfig]:
    """Discover and load all key pools from environment variables."""
    pool_names: set[str] = set()
    for key in os.environ:
        m = re.match(r"^API_POOL_(\w+)_(STRATEGY|KEY_\d+)(?:_BASE)?$", key)
        if m:
            pool_names.add(m.group(1).lower())

    pools: list[PoolConfig] = []
    for name in sorted(pool_names):
        prefix = f"API_POOL_{name.upper()}"
        strategy_str = os.getenv(f"{prefix}_STRATEGY", "round_robin")
        try:
            strategy = PoolStrategy(strategy_str)
        except ValueError:
            strategy = PoolStrategy.ROUND_ROBIN

        keys: list[KeyRecord] = []
        for env_key, env_val in os.environ.items():
            m = re.match(rf"^{prefix}_KEY_(\d+)$", env_key)
            if m and env_val.strip():
                idx = int(m.group(1))
                base_key = f"{prefix}_KEY_{idx}"
                api_base = os.getenv(f"{base_key}_BASE", "")
                keys.append(KeyRecord(
                    name=f"{name}-key{idx}",
                    api_key=env_val.strip(),
                    api_base=api_base,
                    pool_name=name,
                ))

        if keys:
            pools.append(PoolConfig(name=name, strategy=strategy, keys=keys))

    return pools
