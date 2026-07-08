"""Preference extraction and querying from conversation history."""

import json
import re
from pathlib import Path
from typing import Any

from clawagent.memory.summarizer import _get_cached_conn


def save_preference(
    db_path: str,
    key: str,
    value: str,
    session_id: str,
    evidence: str = "",
    confidence: float = 0.5,
) -> None:
    """Save or update a preference entry."""
    conn = _get_cached_conn(db_path)
    conn.execute(
        """INSERT INTO preferences (key, value, confidence, session_id, evidence)
           VALUES (?, ?, ?, ?, ?)""",
        (key, value, confidence, session_id, evidence),
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


def extract_preferences_from_messages(
    messages_text: str,
    session_id: str,
    db_path: str,
    model: Any | None = None,
) -> list[dict[str, Any]]:
    """Analyze conversation text for user preferences and save them.

    If model is None, uses basic pattern matching instead of LLM.
    """
    if model is None:
        patterns = _extract_patterns_basic(messages_text)
    else:
        patterns = _extract_patterns_llm(messages_text, model)

    for p in patterns:
        save_preference(
            db_path=db_path,
            key=str(p["key"]),
            value=str(p["value"]),
            session_id=session_id,
            evidence=str(p.get("evidence", "")),
            confidence=float(p.get("confidence", 0.5)),
        )
    return patterns


def _extract_patterns_basic(text: str) -> list[dict[str, Any]]:
    """Simple keyword-based preference extraction when LLM is not available."""
    patterns = []

    # Detect language preference
    cn_chars = sum(1 for c in text if ord(c) > 0x4E00)
    if cn_chars > len(text) * 0.3:
        patterns.append({
            "key": "language",
            "value": "chinese_priority",
            "evidence": "text contains significant Chinese content",
            "confidence": 0.4,
        })

    return patterns


def _extract_patterns_llm(text: str, model: Any) -> list[dict[str, Any]]:
    """Use LLM to extract preference patterns from conversation."""
    from langchain_core.messages import SystemMessage

    prompt = (
        "Analyze the following conversation for user preferences. "
        "Return a JSON array of objects with keys: 'key', 'value', 'evidence', 'confidence'.\n"
        "Example: [{\"key\": \"response_style\", \"value\": \"concise\", "
        "\"evidence\": \"user said 'keep it short'\", \"confidence\": 0.8}]\n\n"
        f"Conversation:\n{text}"
    )
    try:
        response = model.invoke([SystemMessage(content=prompt)])

        content = response.content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*\n?", "", content)
            content = re.sub(r"\n?```\s*$", "", content)
        patterns = json.loads(content)
        if isinstance(patterns, list):
            return patterns
    except Exception:
        pass
    return []
