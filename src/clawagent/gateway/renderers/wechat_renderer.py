"""WeChat message renderer — StreamEvent → batch text segments.

WeChat constraints:
- No Markdown support → plain text only.
- Single message ≤ 2048 bytes (UTF-8).
- Passive reply: return first segment within 5s.
- Subsequent segments: delivered via customer-service API.

Strategy:
- Accumulate tokens in a buffer.
- Flush every ~300 characters as a discrete message.
- Tool calls emit brief progress notifications.
- On stream completion, flush any remaining buffer.
"""

from __future__ import annotations

from typing import Any

from clawagent.gateway.renderer import IEventRenderer


class WechatRenderer(IEventRenderer):
    """Render StreamEvent stream into WeChat-friendly text batches.

    Each ``render()`` call may return 0 or 1 text segments.
    Segments exceeding the character threshold are flushed immediately.
    """

    # Flush the buffer when it reaches this length (characters).
    FLUSH_THRESHOLD: int = 300

    def __init__(self) -> None:
        self._buffer: list[str] = []
        self._sent_len: int = 0

    def on_token(self, text: str) -> list[str]:
        self._buffer.append(text)
        if self._buffer_len() >= self.FLUSH_THRESHOLD:
            return self._flush()
        return []

    def on_tool_call(self, name: str, args: dict[str, Any]) -> list[str]:
        # Flush any buffered text before showing tool status.
        flushed = self._flush() if self._buffer else []
        tool_msg = self._tool_message(name)
        return flushed + ([tool_msg] if tool_msg else [])

    def on_tool_result(self, name: str, preview: str) -> list[str]:
        # Tool results are internal — don't expose to WeChat user.
        return []

    def on_error(self, message: str) -> list[str]:
        flushed = self._flush() if self._buffer else []
        return [*flushed, "抱歉，处理请求时出错了，请稍后重试。"]

    def on_done(self, full_text: str, usage: dict[str, int]) -> list[str]:
        return self._flush()

    # ── Internal ────────────────────────────────────────────────

    def _flush(self) -> list[str]:
        """Emit the buffered text as a single message."""
        if not self._buffer:
            return []
        chunk = "".join(self._buffer)
        self._buffer.clear()
        self._sent_len += len(chunk)
        return [chunk]

    def _buffer_len(self) -> int:
        return sum(len(s) for s in self._buffer)

    @staticmethod
    def _tool_message(name: str) -> str | None:
        """Return a user-facing progress message for a tool call."""
        mapping: dict[str, str] = {
            "search_documents": "🔍 正在搜索知识库...",
            "web_search": "🌐 正在联网搜索...",
            "read_file": "📖 正在读取文件...",
            "write_file": "✏️ 正在写入文件...",
            "run_command": "⚙️ 正在执行命令...",
            "delegate_task": "🤖 正在委派子任务...",
        }
        return mapping.get(name)
