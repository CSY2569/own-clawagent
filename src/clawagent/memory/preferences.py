"""Preference extraction and querying from conversation history.

Extracts three types of memory in one LLM call:
  1. Preferences (interaction style)
  2. Profile (macro user attributes)
  3. Facts (specific stated facts)

Each item carries a privacy_level that controls storage and vectorization.
"""

import json
import re
from pathlib import Path
from typing import Any

from clawagent.memory.privacy import normalize_level
from clawagent.memory.summarizer import _get_cached_conn


def save_preference(
    db_path: str,
    key: str,
    value: str,
    session_id: str,
    evidence: str = "",
    confidence: float = 0.5,
    privacy_level: str = "sensitive",
) -> None:
    """Save or update a preference entry."""
    conn = _get_cached_conn(db_path)
    level = normalize_level(privacy_level)
    conn.execute(
        """INSERT INTO preferences (key, value, confidence, session_id, evidence, privacy_level)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (key, value, confidence, session_id, evidence, level),
    )
    conn.commit()


def load_top_preferences(db_path: str, limit: int = 5) -> list[dict[str, str]]:
    """Load the top-N most confident preferences, grouped by key-value."""
    path = Path(db_path)
    if not path.exists():
        return []

    conn = _get_cached_conn(db_path)
    rows = conn.execute(
        """SELECT key, value, MAX(confidence) as confidence
           FROM preferences
           GROUP BY key, value
           ORDER BY confidence DESC
           LIMIT ?""",
        (limit,),
    ).fetchall()
    return [{"key": r["key"], "value": r["value"]} for r in rows]


def extract_memories_from_messages(
    messages_text: str,
    session_id: str,
    db_path: str,
    model: Any | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Analyze conversation text and extract preferences, profile, and facts.

    Returns a dict with keys 'preferences', 'profile', 'facts'.
    Each value is a list of extracted items. Private items are filtered
    before storage but included in the return value for transparency.
    """
    result = _extract_basic(messages_text) if model is None else _extract_llm(messages_text, model)

    from clawagent.memory.facts import save_facts_batch
    from clawagent.memory.privacy import is_storeable
    from clawagent.memory.profile import save_profile_batch

    prefs_stored: list[dict[str, Any]] = []
    for p in result.get("preferences", []):
        level = normalize_level(p.get("privacy_level"))
        if not is_storeable(level):
            continue
        save_preference(
            db_path=db_path,
            key=str(p.get("key", "")),
            value=str(p.get("value", "")),
            session_id=session_id,
            evidence=str(p.get("evidence", "")),
            confidence=float(p.get("confidence", 0.5)),
            privacy_level=level,
        )
        prefs_stored.append(p)

    profile_count = save_profile_batch(db_path, result.get("profile", []))
    facts_count = save_facts_batch(db_path, result.get("facts", []), session_id)

    return {
        "preferences": prefs_stored,
        "profile": result.get("profile", [])[:profile_count] if profile_count else [],
        "facts": result.get("facts", [])[:facts_count] if facts_count else [],
    }


def clear_preferences(db_path: str) -> None:
    """Delete all preferences."""
    conn = _get_cached_conn(db_path)
    conn.execute("DELETE FROM preferences")
    conn.commit()


def extract_preferences_from_messages(
    messages_text: str,
    session_id: str,
    db_path: str,
    model: Any | None = None,
) -> list[dict[str, Any]]:
    """Backward-compatible wrapper -- calls extract_memories_from_messages."""
    result = extract_memories_from_messages(messages_text, session_id, db_path, model)
    return result.get("preferences", [])


def _extract_patterns_basic(text: str) -> list[dict[str, Any]]:
    """Backward-compatible wrapper -- returns preferences list from _extract_basic."""
    return _extract_basic(text).get("preferences", [])


def _extract_patterns_llm(text: str, model: Any) -> list[dict[str, Any]]:
    """Backward-compatible wrapper -- returns preferences list from _extract_llm."""
    return _extract_llm(text, model).get("preferences", [])


def _extract_basic(text: str) -> dict[str, list[dict[str, Any]]]:
    """Simple keyword-based extraction when LLM is not available."""
    preferences: list[dict[str, Any]] = []
    cn_chars = sum(1 for c in text if ord(c) > 0x4E00)
    if cn_chars > len(text) * 0.3:
        preferences.append({
            "key": "language",
            "value": "chinese_priority",
            "evidence": "text contains significant Chinese content",
            "confidence": 0.4,
            "privacy_level": "sensitive",
        })
    return {"preferences": preferences, "profile": [], "facts": []}


def _extract_llm(text: str, model: Any) -> dict[str, list[dict[str, Any]]]:
    """Use LLM to extract preferences, profile, and facts from conversation."""
    from langchain_core.messages import SystemMessage

    prompt = (
        "Analyze the following conversation excerpt and extract three types of memory.\n\n"
        "1. PREFERENCES: User's interaction style (response length, tone, format)\n"
        "2. PROFILE: Who the user is (role, tech_stack, language, timezone, expertise)\n"
        "3. FACTS: Important facts the user explicitly stated (project names, constraints, goals)\n\n"
        "For each item, assign a privacy_level:\n"
        '- "public": project/technical content, safe to share\n'
        '- "sensitive": personal preferences, work background\n'
        '- "private": passwords, API keys, personal identity (DO NOT include these in output)\n\n'
        "Return JSON:\n"
        '{\n'
        '  "preferences": [{"key": "...", "value": "...", "evidence": "...", "confidence": 0.8, "privacy_level": "sensitive"}],\n'
        '  "profile": [{"key": "role", "value": "backend_developer", "confidence": 0.7}],\n'
        '  "facts": [{"content": "...", "category": "project", "confidence": 0.9, "privacy_level": "public"}]\n'
        '}\n\n'
        "If nothing notable, return empty arrays. Do NOT include private-level items.\n\n"
        f"Conversation:\n{text}"
    )
    try:
        response = model.invoke([SystemMessage(content=prompt)])
        content = response.content.strip() if response.content else ""
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*\n?", "", content)
            content = re.sub(r"\n?```\s*$", "", content)
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return {
                "preferences": parsed.get("preferences", []) if isinstance(parsed.get("preferences"), list) else [],
                "profile": parsed.get("profile", []) if isinstance(parsed.get("profile"), list) else [],
                "facts": parsed.get("facts", []) if isinstance(parsed.get("facts"), list) else [],
            }
    except Exception:
        pass
    return {"preferences": [], "profile": [], "facts": []}
