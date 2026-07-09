"""Audit logger - JSONL audit trail for tool invocations."""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB


class AuditLogger:
    """Append-only JSONL audit logger with automatic rotation."""

    def __init__(self, log_path: str = "logs/audit.jsonl") -> None:
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        thread_id: str,
        tool: str,
        args: dict[str, Any],
        level: str,
        result: str,
        turn: int = 0,
    ) -> None:
        """Write a single audit entry. Silent on failure."""
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "thread_id": thread_id,
            "tool": tool,
            "args": _sanitize_args(args),
            "level": level,
            "result": result,
            "turn": turn,
        }
        try:
            self._rotate_if_needed()
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            logger.debug("Audit log write failed", exc_info=True)

    def _rotate_if_needed(self) -> None:
        """Rename current log to audit.{date}.jsonl if it exceeds max size."""
        try:
            if not self._path.exists():
                return
            if self._path.stat().st_size < _MAX_LOG_SIZE:
                return
            date_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            rotated = self._path.with_name(f"audit.{date_str}.jsonl")
            os.rename(self._path, rotated)
        except Exception:
            pass


def _sanitize_args(args: dict[str, Any]) -> dict[str, Any]:
    """Truncate long argument values to keep audit log readable."""
    sanitized: dict[str, Any] = {}
    for key, value in args.items():
        if isinstance(value, str) and len(value) > 200:
            sanitized[key] = value[:200] + f"... ({len(value)} chars)"
        else:
            sanitized[key] = value
    return sanitized
