"""Worker configuration — one instance per worker role."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkerConfig:
    """Configuration for a single worker role.

    Model creation uses LangChain's init_chat_model(), supporting any provider.
    Leave model_provider empty for auto-detection.

    Attributes:
        role:            Role identifier (e.g. "coder", "researcher")
        model:           LLM model name
        model_provider:  Model provider ("anthropic", "openai", "ollama", etc.)
                         Empty = auto-detect via init_chat_model
        max_tokens:      Max output tokens
        temperature:     Sampling temperature
        memory_db:       SQLite memory database path
        prompts_dir:     Prompt directory (empty = project default)
        tools:           Tool functions available to this worker
        api_key:         Optional independent API key for different providers
        api_base:        Optional API base URL for OpenAI-compatible providers
    """

    role: str
    model: str = "deepseek-v4-flash"
    model_provider: str = ""
    max_tokens: int = 4096
    temperature: float = 0.0
    request_timeout: int = 120
    memory_db: str = ""
    prompts_dir: str = ""
    tools: list[Any] = field(default_factory=list)
    api_key: str = ""
    api_base: str = ""


def _env(key: str, default: str = "") -> str:
    """Read an environment variable, returning default if unset or empty."""
    val = os.getenv(key, "")
    return val.strip() or default


def load_worker_configs() -> dict[str, WorkerConfig]:
    """Load all worker configurations from environment variables.

    Each worker reads WORKER_{ROLE}_{KEY}, falling back to WORKER_COMMON_{KEY}.
    Automatically discovers roles from WORKER_{ROLE}_MODEL vars.
    """
    # Discover worker roles from env vars matching WORKER_*_MODEL
    roles: set[str] = set()
    for key in os.environ:
        if key.startswith("WORKER_") and not key.startswith("WORKER_COMMON_"):
            for suffix in ("_MODEL", "_API_KEY", "_API_BASE"):
                if key.endswith(suffix):
                    role = key.removeprefix("WORKER_").removesuffix(suffix).lower()
                    if role:
                        roles.add(role)
                    break

    if not roles:
        return {}

    configs: dict[str, WorkerConfig] = {}
    for role in sorted(roles):
        prefix = f"WORKER_{role.upper()}"
        configs[role] = WorkerConfig(
            role=role,
            model=_env(f"{prefix}_MODEL", _env("WORKER_COMMON_MODEL", "deepseek-v4-flash")),
            model_provider=_env(f"{prefix}_MODEL_PROVIDER", _env("WORKER_COMMON_MODEL_PROVIDER")),
            max_tokens=int(_env(f"{prefix}_MAX_TOKENS", _env("WORKER_COMMON_MAX_TOKENS", "4096"))),
            temperature=float(_env(f"{prefix}_TEMPERATURE", _env("WORKER_COMMON_TEMPERATURE", "0.0"))),
            memory_db=_env(f"{prefix}_MEMORY_DB", f"memories/{role}.db"),
            prompts_dir=_env(f"{prefix}_PROMPTS_DIR") or "prompts/",
            request_timeout=int(_env(f"{prefix}_REQUEST_TIMEOUT", _env("WORKER_COMMON_REQUEST_TIMEOUT", "120"))),
            api_key=_env(f"{prefix}_API_KEY", _env("WORKER_COMMON_API_KEY")),
            api_base=_env(f"{prefix}_API_BASE", _env("WORKER_COMMON_API_BASE")),
        )
    return configs
