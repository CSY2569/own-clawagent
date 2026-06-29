"""Typed events for the agent execution stream."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

EventKind = Literal[
    "think_start", "think_end", "tool_call", "tool_result", "token", "error", "done"
]


@dataclass
class StreamEvent:
    """A single event in the agent's execution stream.

    Attributes:
        kind:    "think_start"  — LLM enters thinking block (DeepSeek)
                 "think_end"    — LLM exits thinking block
                 "tool_call"    — LLM initiates a tool call
                 "tool_result"  — tool returns result
                 "token"        — text fragment (typewriter effect)
                 "error"        — execution error
                 "done"         — stream finished
        node:    LangGraph node name ("agent" / "tools")
        content: Primary event content (text or tool name)
        metadata: Additional data (args, preview, usage, etc.)
    """

    kind: EventKind
    node: str = ""
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
