"""API Key pool data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class KeyStatus(Enum):
    ACTIVE = "active"
    DEGRADED = "degraded"
    EXHAUSTED = "exhausted"
    DISABLED = "disabled"


class PoolStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LEAST_ERRORS = "least_errors"


@dataclass
class KeyRecord:
    """Single API key in the pool."""
    name: str
    api_key: str
    api_base: str = ""
    provider: str = "anthropic"
    pool_name: str = "default"
    status: KeyStatus = KeyStatus.ACTIVE
    error_count: int = 0
    last_error_at: float = 0.0
    cooldown_seconds: float = 30.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0


@dataclass
class PoolConfig:
    """Configuration for a named key pool."""
    name: str
    strategy: PoolStrategy = PoolStrategy.ROUND_ROBIN
    keys: list[KeyRecord] = field(default_factory=list)
    max_cooldown: float = 600.0
