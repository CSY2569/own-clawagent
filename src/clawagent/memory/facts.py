"""Key facts - specific factual statements extracted from conversations.

Facts are concrete things the user explicitly stated, like
"my project is called clawagent" or "I use Rust for performance-critical code".
Each fact has a privacy_level that controls whether it can be vectorized.
"""

from __future__ import annotations

from typing import Any

from clawagent.memory.privacy import is_storeable, normalize_level
from clawagent.memory.summarizer import _get_cached_conn


def save_fact(
    db_path: str,
    content: str,
    category: str = "general",
    privacy_level: str = "public",
    confidence: float = 0.7,
    session_id: str = "",
) -> bool:
    """Save a fact. Returns True if stored, False if filtered by privacy."""
    level = normalize_level(privacy_level)
    if not is_storeable(level):
        return False
    conn = _get_cached_conn(db_path)
    conn.execute(
        """INSERT INTO facts (content, category, privacy_level, confidence, session_id)
           VALUES (?, ?, ?, ?, ?)""",
        (content, category, level, confidence, session_id),
    )
    conn.commit()
    return True


def save_facts_batch(
    db_path: str, entries: list[dict[str, Any]], session_id: str = ""
) -> int:
    """Save a batch of facts. Returns count of stored entries (private filtered)."""
    count = 0
    for entry in entries:
        content = str(entry.get("content", "")).strip()
        if not content:
            continue
        level = normalize_level(entry.get("privacy_level"))
        if not is_storeable(level):
            continue
        category = str(entry.get("category", "general"))
        confidence = float(entry.get("confidence", 0.7))
        save_fact(db_path, content, category, level, confidence, session_id)
        count += 1
    return count


def load_facts(db_path: str, category: str | None = None) -> list[dict[str, Any]]:
    """Load facts, optionally filtered by category. Private facts are excluded."""
    from pathlib import Path

    if not Path(db_path).exists():
        return []
    conn = _get_cached_conn(db_path)
    if category:
        rows = conn.execute(
            "SELECT content, category, confidence FROM facts WHERE category = ? ORDER BY confidence DESC",
            (category,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT content, category, confidence FROM facts ORDER BY confidence DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def load_vectorizable_facts(db_path: str) -> list[str]:
    """Return fact contents that are safe to vectorize (public only)."""
    from pathlib import Path

    if not Path(db_path).exists():
        return []
    conn = _get_cached_conn(db_path)
    rows = conn.execute(
        "SELECT content FROM facts WHERE privacy_level = ? ORDER BY confidence DESC",
        ("public",),
    ).fetchall()
    return [r["content"] for r in rows]


def clear_facts(db_path: str) -> None:
    """Delete all facts."""
    conn = _get_cached_conn(db_path)
    conn.execute("DELETE FROM facts")
    conn.commit()
