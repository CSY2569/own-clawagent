"""Tool definitions for the clawagent."""

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

__all__ = [
    "ALL_TOOLS",
    "_PROJECT_ROOT",
    "_resolve_path",
    "get_current_time",
    "greet",
    "list_sessions",
    "read_file",
    "recall_session",
    "run_command",
    "summarize_session",
    "write_file",
]

# Project root for path-safe file operations
# __file__ = src/clawagent/tools/__init__.py → parent.parent.parent = project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _resolve_path(path: str) -> Path:
    """Resolve a path relative to project root and validate it's within bounds."""
    p = Path(path)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    p = p.resolve()
    if not str(p).startswith(str(_PROJECT_ROOT)):
        raise ValueError(f"Path is outside the project directory: {path}")
    return p


@tool
def get_current_time() -> str:
    """Return the current system time as an ISO 8601 string."""
    return datetime.now(tz=UTC).isoformat()


@tool
def greet(name: str) -> str:
    """Greet someone by name.

    Args:
        name: The name of the person to greet.
    """
    return f"Hello, {name}! Welcome to clawagent."


@tool
def read_file(path: str) -> str:
    """Read the content of a file.

    The path is relative to the project root. Use this when you need
    to inspect source code, configuration, or data files.

    Args:
        path: File path relative to project root (e.g. "README.md", "src/clawagent/main.py").
    """
    resolved = _resolve_path(path)
    if not resolved.exists():
        return f"File not found: {path}"
    if not resolved.is_file():
        return f"Not a file: {path}"
    return resolved.read_text(encoding="utf-8")


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file (create or overwrite).

    The path is relative to the project root. Creates parent directories
    if they don't exist. Use this to save generated code, data, or reports.

    Args:
        path: File path relative to project root (e.g. "output/result.txt").
        content: The content to write to the file.
    """
    resolved = _resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")
    return f"Written {len(content)} bytes to {path}"


@tool
def run_command(command: str) -> str:
    """Run a shell command and return its output.

    The command runs from the project root directory. Use this for
    development tasks like building, testing, linting, or git operations.
    Prefer read-only commands when possible.

    Args:
        command: Shell command to execute (e.g. "uv run ruff check .", "git status").
    """
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(_PROJECT_ROOT),
    )
    output = result.stdout
    if result.stderr:
        output += f"\n--- stderr ---\n{result.stderr}"
    if result.returncode != 0:
        output += f"\n--- exit code: {result.returncode} ---"
    return output


# Lazy imports to avoid circular dependencies
def _get_memory_tools() -> list[Any]:
    from clawagent.tools.memory_tools import (
        list_sessions,
        recall_session,
        summarize_session,
    )
    return [list_sessions, recall_session, summarize_session]


# Registry of all tools available to the agent
ALL_TOOLS = [
    *[get_current_time, greet, read_file, write_file, run_command],
    *_get_memory_tools(),
]
