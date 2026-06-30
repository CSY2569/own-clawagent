"""Real-time streaming display — compact, spinner-driven, no panel borders."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

from rich.console import Group
from rich.live import Live
from rich.rule import Rule
from rich.text import Text

from clawagent.stream_events import StreamEvent

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


@contextmanager
def stream_display() -> Iterator[StreamDisplay]:
    """Context manager wrapping Rich Live for streaming display.

    Usage:
        with stream_display() as display:
            for event in agent.stream_events(message):
                display.handle(event)
    """
    display = StreamDisplay()
    with Live(
        display._render(),
        refresh_per_second=15,
        vertical_overflow="crop",
        screen=False,
    ) as live:
        display._live = live
        yield display


class StreamDisplay:
    """Compact streaming display — spinner + inline tool status + final response.

    Layout (during tool calls):
      ⠋ Calling read_file("config.py")...
      ✓ read_file (152 lines)
      ⠙ Writing file...

    Layout (done):
      ✓ read_file (152 lines) · write_file (512 bytes)
      ──────────────────────────────────────
      Agent response text here...

    No Rich Layout or Panel — just Text lines with a spinner.
    """

    def __init__(self) -> None:
        self._live: Live | None = None
        self._spinner_frame: int = 0
        self._last_tick: float = 0.0
        self._status_text: str = ""
        self._tool_log: list[str] = []
        self._done: bool = False
        self._token_buffer: str = ""
        self._last_refresh: float = 0.0
        self._stats: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "cache_tokens": 0}

    def handle(self, event: StreamEvent) -> None:
        """Route a StreamEvent to the appropriate handler."""
        if event.kind == "think_start":
            self._on_think_start(event)
        elif event.kind == "think_end":
            self._on_think_end(event)
        elif event.kind == "tool_call":
            self._on_tool_call(event)
        elif event.kind == "tool_result":
            self._on_tool_result(event)
        elif event.kind == "token":
            self._on_token(event)
        elif event.kind == "error":
            self._on_error(event)
        elif event.kind == "done":
            self._on_done(event)

    def _spinner(self) -> str:
        """Return current spinner character, advancing on each render."""
        now = time.monotonic()
        if now - self._last_tick > 0.08:
            self._spinner_frame = (self._spinner_frame + 1) % len(_SPINNER)
            self._last_tick = now
        return _SPINNER[self._spinner_frame]

    def _on_think_start(self, event: StreamEvent) -> None:
        self._status_text = "[dim]深度思考中...[/dim]"
        self._refresh()

    def _on_think_end(self, event: StreamEvent) -> None:
        self._status_text = ""

    def _on_token(self, event: StreamEvent) -> None:
        """Append text chunk with 50ms throttle for typewriter effect."""
        self._token_buffer += event.content
        self._status_text = "[dim]输出中...[/dim]"
        now = time.monotonic()
        if now - self._last_refresh > 0.05:
            self._refresh()
            self._last_refresh = now

    def _on_tool_call(self, event: StreamEvent) -> None:
        args = event.metadata.get("args", {})
        args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
        self._status_text = f"[dim]Calling {event.content}({args_str})...[/dim]"
        self._refresh()

    def _on_tool_result(self, event: StreamEvent) -> None:
        preview = event.metadata.get("preview", "")
        result_line = f"  [green]✓[/green] [dim]{event.content}[/dim]"
        if preview:
            # Show brief preview — count lines, bytes, or first few words
            lines = preview.split("\n")
            if len(lines) > 1:
                result_line += f" [dim]({len(lines)} lines)[/dim]"
            elif len(preview) > 60:
                result_line += f" [dim]({len(preview)} chars)[/dim]"
            else:
                result_line += f" [dim]{preview}[/dim]"
        self._tool_log.append(result_line)
        self._status_text = ""
        self._refresh()

    def _on_error(self, event: StreamEvent) -> None:
        self._status_text = f"[red]✗ {event.content}[/red]"
        self._done = True
        self._refresh()

    def _on_done(self, event: StreamEvent) -> None:
        self._status_text = ""
        self._done = True
        if not self._token_buffer and event.content:
            self._token_buffer = event.content
        self._stats["input_tokens"] = event.metadata.get("input_tokens", 0)
        self._stats["output_tokens"] = event.metadata.get("output_tokens", 0)
        self._stats["cache_tokens"] = event.metadata.get("cache_read_input_tokens", 0) + event.metadata.get("cache_creation_input_tokens", 0)
        self._refresh()

    def _render(self) -> Group | Text:
        """Build the current display output."""
        parts: list[Text | Rule] = []

        # Status line: spinner + action, or empty when done/idle
        if self._done:
            pass
        elif self._status_text:
            parts.append(Text.from_markup(f"  {self._spinner()} {self._status_text}"))
        else:
            parts.append(Text.from_markup(f"  [dim]{self._spinner()} Thinking...[/dim]"))

        # Tool result log
        for line in self._tool_log:
            parts.append(Text.from_markup(line))

        # Token buffer (typewriter during streaming, final text after done)
        if self._token_buffer:
            if self._done and self._tool_log:
                parts.append(Rule(style="dim"))
            parts.append(Text(self._token_buffer))

        # Phase 3: per-request stats (only after done)
        if self._done and (self._stats["input_tokens"] or self._stats["output_tokens"]):
            stats_parts = []
            if self._stats["input_tokens"]:
                stats_parts.append(f"In [dim]{self._stats['input_tokens']}[/dim]")
            if self._stats["output_tokens"]:
                stats_parts.append(f"Out [dim]{self._stats['output_tokens']}[/dim]")
            if self._stats["cache_tokens"]:
                stats_parts.append(f"Cache [dim]{self._stats['cache_tokens']}[/dim]")
            parts.append(Text.from_markup("  [bold]·[/bold] ".join(stats_parts)))

        return Group(*parts)

    def _refresh(self) -> None:
        """Refresh the live display."""
        if self._live:
            self._live.update(self._render())
