"""Tool definitions for the clawagent."""

import subprocess
from pathlib import Path

from langchain_core.tools import BaseTool, tool

from clawagent.tools.rag_tool import configure_hybrid_search, search_documents
from clawagent.tools.web_search import web_search

__all__ = [
    "ALL_TOOLS",
    "PROJECT_ROOT",
    "_resolve_path",
    "configure_hybrid_search",
    "create_memory_tools",
    "read_file",
    "run_command",
    "search_documents",
    "web_search",
    "write_file",
]

# Project root for path-safe file operations
from clawagent.config import PROJECT_ROOT


def _resolve_path(path: str) -> Path:
    """Resolve a path relative to project root and validate it's within bounds."""
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p = p.resolve()
    if not str(p).startswith(str(PROJECT_ROOT)):
        raise ValueError(f"Path is outside the project directory: {path}")
    return p


@tool
def read_file(path: str) -> str:
    """Read the content of a file.

    The path is relative to the project root. Use this when you need
    to inspect source code, configuration, or data files.

    Args:
        path: File path relative to project root (e.g. "README.md", "src/clawagent/main.py").
    """
    try:
        resolved = _resolve_path(path)
    except ValueError as e:
        return f"Error: {e}"
    try:
        if not resolved.exists():
            return f"File not found: {path}"
        if not resolved.is_file():
            return f"Not a file: {path}"
        return resolved.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file (create or overwrite).

    The path is relative to the project root. Creates parent directories
    if they don't exist. Use this to save generated code, data, or reports.

    Args:
        path: File path relative to project root (e.g. "output/result.txt").
        content: The content to write to the file.
    """
    try:
        resolved = _resolve_path(path)
    except ValueError as e:
        return f"Error: {e}"
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return f"Written {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


@tool
def run_command(command: str) -> str:
    """Run a shell command and return its output.

    The command runs from the project root directory. Use this for
    development tasks like building, testing, linting, or git operations.
    Prefer read-only commands when possible.

    Args:
        command: Shell command to execute (e.g. "uv run ruff check .", "git status").
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(PROJECT_ROOT),
        )
    except subprocess.TimeoutExpired:
        return f"Command timed out after 120s: {command}"
    except Exception as e:
        return f"Error running command: {e}"
    output = result.stdout
    if result.stderr:
        output += f"\n--- stderr ---\n{result.stderr}"
    if result.returncode != 0:
        output += f"\n--- exit code: {result.returncode} ---"
    return output


# Registry of base tools (memory tools are created via create_memory_tools in agent.py)
ALL_TOOLS: list[BaseTool] = [
    *[read_file, write_file, run_command, search_documents],
]
