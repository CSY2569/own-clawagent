"""Tool definitions for the clawagent."""

import shlex
import subprocess
from pathlib import Path

from langchain_core.tools import BaseTool, tool

from clawagent.tools.browser import browser_extract_standalone, browser_navigate_standalone
from clawagent.tools.rag_tool import configure_hybrid_search, search_documents
from clawagent.tools.web_search import web_search

__all__ = [
    "ALL_TOOLS",
    "PROJECT_ROOT",
    "_resolve_path",
    "browser_extract_standalone",
    "browser_navigate_standalone",
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
    """Resolve a path relative to project root and validate it's within bounds.

    Uses Path.is_relative_to() for robust containment check — string prefix
    matching is vulnerable to sibling-directory bypass (e.g. /home/u/foo vs
    /home/u/fooevil).
    """
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p = p.resolve()
    if not p.is_relative_to(PROJECT_ROOT):
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


# Whititelist, not sandbox: python/docker can still run arbitrary code.
# True isolation requires container/seccomp.
_ALLOWED_COMMANDS: set[str] = {
    "git", "uv", "ruff", "mypy", "pytest",
    "python", "python3",
    "ls", "cat", "echo", "grep", "find", "pwd",
    "mkdir", "cp", "mv", "touch",
    "docker", "npm",
}


@tool
def run_command(command: str) -> str:
    """Run a command and return its output.

    Each invocation runs a single command with arguments. Pipes, redirects
    (>, >>), and command chaining (&&, ;, |) are not supported for safety.

    The command runs from the project root directory. Use this for
    development tasks like building, testing, linting, or git operations.

    Only commands in the whitelist are permitted; unknown commands are
    rejected. This is defense-in-depth, not a true sandbox.

    Args:
        command: Command with arguments (e.g. "uv run ruff check .",
                 "git status", "pytest tests/ -v"). No pipes or redirects.
    """
    try:
        args = shlex.split(command)
    except ValueError as e:
        return f"Command parsing error (shell metacharacters not allowed): {e}"

    if not args:
        return "Empty command"

    cmd_name = args[0]
    if cmd_name not in _ALLOWED_COMMANDS:
        allowed = ", ".join(sorted(_ALLOWED_COMMANDS))
        return (
            f"Blocked: '{cmd_name}' is not in the allowed command list. "
            f"Allowed: {allowed}."
        )

    try:
        result = subprocess.run(
            args,
            shell=False,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(PROJECT_ROOT),
        )
    except FileNotFoundError:
        return f"Command not found: {cmd_name}"
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
