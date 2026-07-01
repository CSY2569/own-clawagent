"""Per-session JSONL conversation logging.

Writes one JSON object per line to logs/{thread_id}.jsonl.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from clawagent.agent import Usage
from clawagent.config import Settings


def _settings_dict(settings: Settings) -> dict[str, Any]:
    """Serialize Settings to a dict, excluding API keys."""
    return {
        "model_name": settings.model_name,
        "model_provider": settings.model_provider,
        "max_tokens": settings.max_tokens,
        "temperature": settings.temperature,
        "context_window": settings.context_window,
        "compression_strategy": settings.compression_strategy,
        "compression_max_messages": settings.compression_max_messages,
        "compression_max_tokens": settings.compression_max_tokens,
        "compression_keep_recent": settings.compression_keep_recent,
        "agent_id": settings.agent_id,
    }


def _usage_dict(usage: Usage) -> dict[str, int]:
    """Serialize Usage to a plain dict."""
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_input_tokens": usage.cache_read_input_tokens,
        "cache_creation_input_tokens": usage.cache_creation_input_tokens,
    }


class ConversationLogger:
    """Writes structured JSONL logs for each conversation session."""

    def __init__(self, log_dir: str = "logs") -> None:
        self._log_dir = Path(log_dir)

    def _write(self, thread_id: str, record: dict[str, Any]) -> None:
        """Append a JSON record to the session log file."""
        self._log_dir.mkdir(parents=True, exist_ok=True)
        path = self._log_dir / f"{thread_id}.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _now() -> str:
        return datetime.now(tz=timezone.utc).isoformat()  # noqa: UP017

    def log_session_start(self, thread_id: str, settings: Settings) -> None:
        self._write(
            thread_id,
            {
                "type": "session_start",
                "timestamp": self._now(),
                "thread_id": thread_id,
                "settings": _settings_dict(settings),
            },
        )

    def log_turn(
        self,
        thread_id: str,
        round_n: int,
        user_msg: str,
        response_text: str,
        usage: Usage,
        settings: Settings,
        error: str | None = None,
    ) -> None:
        self._write(
            thread_id,
            {
                "type": "turn",
                "timestamp": self._now(),
                "thread_id": thread_id,
                "round": round_n,
                "user_message": user_msg,
                "response_preview": response_text[:200],
                "usage": _usage_dict(usage),
                "model_name": settings.model_name,
                "temperature": settings.temperature,
                "max_tokens": settings.max_tokens,
                "compression_strategy": settings.compression_strategy,
                "error": error,
            },
        )

    def log_settings_change(
        self,
        thread_id: str,
        field: str,
        old_value: Any,
        new_value: Any,
    ) -> None:
        self._write(
            thread_id,
            {
                "type": "settings_change",
                "timestamp": self._now(),
                "thread_id": thread_id,
                "field": field,
                "old_value": str(old_value),
                "new_value": str(new_value),
            },
        )

    def log_session_end(
        self,
        thread_id: str,
        total_rounds: int,
        cumulative_usage: Usage,
    ) -> None:
        self._write(
            thread_id,
            {
                "type": "session_end",
                "timestamp": self._now(),
                "thread_id": thread_id,
                "total_rounds": total_rounds,
                "cumulative_usage": _usage_dict(cumulative_usage),
            },
        )
