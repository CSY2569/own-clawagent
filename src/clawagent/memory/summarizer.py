"""Conversation summarization — generate and store session summaries."""

import contextlib
import sqlite3
import threading
from pathlib import Path
from typing import Any

_conn_cache: dict[str, sqlite3.Connection] = {}
_cache_lock = threading.Lock()


def _get_conn(db_path: str) -> sqlite3.Connection:
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _ensure_table(conn)
    _ensure_messages_table(conn)
    _ensure_preferences_table(conn)
    _ensure_user_profile_table(conn)
    _ensure_facts_table(conn)
    return conn


def _get_cached_conn(db_path: str) -> sqlite3.Connection:
    """Return a cached connection for db_path, creating one if needed."""
    with _cache_lock:
        if db_path not in _conn_cache:
            conn = _get_conn(db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            _conn_cache[db_path] = conn
        return _conn_cache[db_path]


def close_all_cached() -> None:
    """Close all cached connections. Call on agent shutdown."""
    with _cache_lock:
        for conn in _conn_cache.values():
            conn.close()
        _conn_cache.clear()


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS session_summaries (
            thread_id TEXT PRIMARY KEY,
            title TEXT,
            summary TEXT,
            message_count INTEGER,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()


def _ensure_messages_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversation_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_messages_thread ON conversation_messages(thread_id)"
    )
    conn.commit()


def _ensure_preferences_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            confidence REAL DEFAULT 0.5,
            session_id TEXT,
            evidence TEXT,
            privacy_level TEXT DEFAULT 'sensitive',
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    with contextlib.suppress(sqlite3.OperationalError):
        conn.execute("ALTER TABLE preferences ADD COLUMN privacy_level TEXT DEFAULT 'sensitive'")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_preferences_key ON preferences(key)")
    conn.commit()


def _ensure_user_profile_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_profile (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            confidence REAL DEFAULT 0.5,
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()


def _ensure_facts_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            category TEXT DEFAULT 'general',
            privacy_level TEXT DEFAULT 'public',
            confidence REAL DEFAULT 0.7,
            session_id TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category)")
    conn.commit()


def save_summary(
    db_path: str,
    thread_id: str,
    title: str,
    summary: str,
    message_count: int,
) -> None:
    """Insert or update a session summary."""
    conn = _get_cached_conn(db_path)
    conn.execute(
        """INSERT OR REPLACE INTO session_summaries
           (thread_id, title, summary, message_count, updated_at)
           VALUES (?, ?, ?, ?, datetime('now'))""",
        (thread_id, title, summary, message_count),
    )
    conn.commit()


def get_summary(db_path: str, thread_id: str) -> dict[str, Any] | None:
    """Get summary for a specific session."""
    path = Path(db_path)
    if not path.exists():
        return None
    conn = _get_cached_conn(db_path)
    row = conn.execute(
        "SELECT * FROM session_summaries WHERE thread_id = ?", (thread_id,)
    ).fetchone()
    if row is None:
        return None
    return dict(row)


def list_summaries(db_path: str) -> list[dict[str, Any]]:
    """List all session summaries ordered by update time (newest first)."""
    path = Path(db_path)
    if not path.exists():
        return []
    conn = _get_cached_conn(db_path)
    rows = conn.execute(
        "SELECT * FROM session_summaries ORDER BY updated_at DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def list_thread_ids(db_path: str) -> list[dict[str, Any]]:
    """List all thread IDs that have messages, with first-user-message as title."""
    path = Path(db_path)
    if not path.exists():
        return []
    conn = _get_cached_conn(db_path)
    rows = conn.execute(
        """SELECT thread_id,
                  (SELECT content FROM conversation_messages cm2
                   WHERE cm2.thread_id = cm.thread_id AND cm2.role = 'user'
                   ORDER BY cm2.id LIMIT 1) as title,
                  COUNT(*) as message_count,
                  MAX(created_at) as updated_at
           FROM conversation_messages cm
           GROUP BY thread_id
           ORDER BY updated_at DESC"""
    ).fetchall()
    return [dict(r) for r in rows]


def ensure_session_entry(db_path: str, thread_id: str, first_message: str) -> None:
    """Ensure a basic session summary entry exists (INSERT OR IGNORE)."""
    conn = _get_cached_conn(db_path)
    conn.execute(
        """INSERT OR IGNORE INTO session_summaries
           (thread_id, title, summary, message_count)
           VALUES (?, ?, ?, 0)""",
        (thread_id, first_message[:60], f"Session started: {first_message[:80]}"),
    )
    conn.commit()


def save_messages(
    db_path: str,
    thread_id: str,
    messages: list[tuple[str, str]],
) -> None:
    """Save conversation messages for later recall.

    Args:
        db_path: Path to SQLite database.
        thread_id: Session identifier.
        messages: List of (role, content) tuples.
    """
    conn = _get_cached_conn(db_path)
    conn.executemany(
        "INSERT INTO conversation_messages (thread_id, role, content) VALUES (?, ?, ?)",
        [(thread_id, role, content) for role, content in messages],
    )
    conn.commit()


def load_messages(db_path: str, thread_id: str) -> list[dict[str, Any]]:
    """Load saved messages for a session."""
    path = Path(db_path)
    if not path.exists():
        return []
    conn = _get_cached_conn(db_path)
    rows = conn.execute(
        "SELECT role, content, created_at FROM conversation_messages "
        "WHERE thread_id = ? ORDER BY id",
        (thread_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def generate_session_summary(
    messages_text: str,
    model: Any | None = None,
) -> tuple[str, str]:
    """Generate a title and summary from conversation text.

    Returns (title, summary) tuple.
    If model is None, falls back to a simple heuristic.
    """
    if model is None:
        return _heuristic_summary(messages_text)

    from langchain_core.messages import HumanMessage

    prompt = (
        "Analyze the following conversation and return:\n"
        "TITLE: a short phrase (under 10 words) summarizing the main topic\n"
        "SUMMARY: a paragraph (under 150 words) covering: "
        "who the user is role-wise, main topics discussed, "
        "key conclusions reached, any action items\n\n"
        f"Conversation:\n{messages_text}"
    )
    try:
        response = model.invoke([HumanMessage(content=prompt)])
        text = response.content.strip() if response.content else ""

        title = "Conversation"
        summary = text
        if "TITLE:" in text and "SUMMARY:" in text:
            parts = text.split("SUMMARY:", 1)
            title = parts[0].replace("TITLE:", "").strip()
            summary = parts[1].strip()

        return title, summary
    except Exception:
        return _heuristic_summary(messages_text)


def _heuristic_summary(text: str) -> tuple[str, str]:
    lines = text.strip().split("\n")
    first_line = lines[0][:60] if lines and lines[0] else "Conversation"
    word_count = len(text.split())
    return first_line, f"Conversation with {len(lines)} exchanges ({word_count} words total)."


def get_recent_summaries(db_path: str, limit: int = 3) -> list[dict[str, Any]]:
    """Return the N most recently updated session summaries (excluding empty sessions)."""
    path = Path(db_path)
    if not path.exists():
        return []
    conn = _get_cached_conn(db_path)
    rows = conn.execute(
        """SELECT thread_id, title, summary, message_count, updated_at
           FROM session_summaries
           WHERE message_count > 0
           ORDER BY updated_at DESC
           LIMIT ?""",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]
