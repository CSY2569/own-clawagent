"""Privacy level constants and filtering for memory storage.

Three levels:
    - public:   project/technical content, safe to vectorize and share
    - sensitive: personal preferences, work background -- store in SQLite only
    - private:  passwords, API keys, personal identity -- never stored
"""

from __future__ import annotations

from typing import Any

PUBLIC = "public"
SENSITIVE = "sensitive"
PRIVATE = "private"

VALID_LEVELS = frozenset({PUBLIC, SENSITIVE, PRIVATE})

DEFAULT_LEVEL = PUBLIC


def is_storeable(level: str) -> bool:
    """Return True if content at this privacy level may be stored in SQLite."""
    return level != PRIVATE


def is_vectorizable(level: str) -> bool:
    """Return True if content at this privacy level may be vectorized into Chroma."""
    return level == PUBLIC


def normalize_level(level: str | None) -> str:
    """Normalize a privacy level string, falling back to DEFAULT_LEVEL."""
    if level and level in VALID_LEVELS:
        return level
    return DEFAULT_LEVEL


def filter_storeable(items: list[dict[str, Any]], key: str = "privacy_level") -> list[dict[str, Any]]:
    """Filter a list of dicts, dropping any with private privacy level."""
    return [item for item in items if is_storeable(normalize_level(item.get(key)))]
