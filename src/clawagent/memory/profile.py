"""User profile - macro-level user attributes extracted from conversations.

Stores structured info like role, tech_stack, language, timezone.
Distinct from preferences (fine-grained interaction style) and facts
(specific stated facts).
"""

from __future__ import annotations

from typing import Any

from clawagent.memory.summarizer import _get_cached_conn


def save_profile_entry(db_path: str, key: str, value: str, confidence: float = 0.5) -> None:
    """Insert or update a profile entry (upsert by key, keep max confidence)."""
    conn = _get_cached_conn(db_path)
    existing = conn.execute(
        "SELECT confidence FROM user_profile WHERE key = ?", (key,)
    ).fetchone()
    if existing and existing["confidence"] >= confidence:
        return
    conn.execute(
        """INSERT OR REPLACE INTO user_profile (key, value, confidence, updated_at)
           VALUES (?, ?, ?, datetime('now'))""",
        (key, value, confidence),
    )
    conn.commit()


def load_profile(db_path: str) -> dict[str, str]:
    """Return all profile entries as a {key: value} dict."""
    from pathlib import Path

    if not Path(db_path).exists():
        return {}
    conn = _get_cached_conn(db_path)
    rows = conn.execute(
        "SELECT key, value FROM user_profile ORDER BY confidence DESC"
    ).fetchall()
    return {r["key"]: r["value"] for r in rows}


def save_profile_batch(db_path: str, entries: list[dict[str, Any]]) -> int:
    """Save a batch of profile entries. Returns count of stored entries.

    Each entry should have: key, value, confidence (optional).
    Entries without a 'key' or 'value' field are skipped.
    """
    count = 0
    for entry in entries:
        key = str(entry.get("key", "")).strip()
        value = str(entry.get("value", "")).strip()
        if not key or not value:
            continue
        confidence = float(entry.get("confidence", 0.5))
        save_profile_entry(db_path, key, value, confidence)
        count += 1
    return count


def clear_profile(db_path: str) -> None:
    """Delete all profile entries."""
    conn = _get_cached_conn(db_path)
    conn.execute("DELETE FROM user_profile")
    conn.commit()
