"""Public data types for clawagent agents — Usage, AgentResponse."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Usage:
    """Token usage for a single agent invocation."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    prompt_cache_hit_tokens: int = 0
    prompt_cache_miss_tokens: int = 0

    @classmethod
    def from_response_metadata(cls, metadata: dict[str, Any]) -> Usage:
        usage = metadata.get("usage", {})
        if not usage:
            return cls()
        return cls(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
            cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
            prompt_cache_hit_tokens=usage.get("prompt_cache_hit_tokens", 0),
            prompt_cache_miss_tokens=usage.get("prompt_cache_miss_tokens", 0),
        )


@dataclass
class AgentResponse:
    """Result of a single agent invocation."""

    text: str
    usage: Usage
