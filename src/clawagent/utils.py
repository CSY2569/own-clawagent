"""Shared utility functions for clawagent."""

from __future__ import annotations

from typing import Any


def extract_text(content: Any) -> str:
    """Extract readable text from AI message content.

    Handles both plain strings and content blocks (e.g., DeepSeek's
    thinking/text blocks in Anthropic API format).

    Args:
        content: Message content — str, list[dict], or other type.

    Returns:
        Extracted text string. Never returns None; empty string for
        unrecognized formats.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(content) if content is not None else ""
