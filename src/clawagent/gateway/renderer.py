"""Event renderers — convert StreamEvent streams to platform-native messages.

Each platform has its own renderer inheriting from IEventRenderer.
The renderer accumulates token chunks, filters internal events
(think_start/think_end), and produces output in the platform's format.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from clawagent.stream_events import StreamEvent


class IEventRenderer(ABC):
    """Convert StreamEvent sequences to platform-sendable messages.

    Subclasses implement the platform-specific rendering logic for
    each event kind. Events that are irrelevant to a platform
    (e.g., think_start for WeChat) simply return an empty list.
    """

    @abstractmethod
    def on_token(self, text: str) -> list[Any]:
        """Handle a text token chunk."""
        ...

    @abstractmethod
    def on_tool_call(self, name: str, args: dict[str, Any]) -> list[Any]:
        """Handle the start of a tool call."""
        ...

    @abstractmethod
    def on_tool_result(self, name: str, preview: str) -> list[Any]:
        """Handle a tool result."""
        ...

    @abstractmethod
    def on_error(self, message: str) -> list[Any]:
        """Handle an execution error."""
        ...

    @abstractmethod
    def on_done(self, full_text: str, usage: dict[str, int]) -> list[Any]:
        """Handle stream completion — flush remaining buffered content."""
        ...

    def render(self, event: StreamEvent) -> list[Any]:
        """Dispatch a StreamEvent to the appropriate handler.

        Returns a list of platform-native messages (may be empty).
        """
        dispatch = {
            "think_start": lambda: [],
            "think_end": lambda: [],
            "token": lambda: self.on_token(event.content),
            "tool_call": lambda: self.on_tool_call(
                event.content, event.metadata.get("args", {})
            ),
            "tool_result": lambda: self.on_tool_result(
                event.content, event.metadata.get("preview", "")
            ),
            "error": lambda: self.on_error(event.content),
            "done": lambda: self.on_done(event.content, event.metadata),
        }
        handler = dispatch.get(event.kind)
        if handler:
            return handler()  # type: ignore[no-untyped-call]
        return []


class CliRenderer(IEventRenderer):
    """CLI renderer — pass-through for terminal display.

    The CLI already handles Rich rendering via ui_stream/stream_display().
    This renderer simply collects text and returns it for display.
    """

    def __init__(self) -> None:
        self._buffer: list[str] = []

    def on_token(self, text: str) -> list[str]:
        self._buffer.append(text)
        return []

    def on_tool_call(self, name: str, args: dict[str, Any]) -> list[str]:
        return []

    def on_tool_result(self, name: str, preview: str) -> list[str]:
        return []

    def on_error(self, message: str) -> list[str]:
        return [f"[Error] {message}"]

    def on_done(self, full_text: str, usage: dict[str, int]) -> list[str]:
        result = "".join(self._buffer)
        self._buffer.clear()
        return [result] if result else []
