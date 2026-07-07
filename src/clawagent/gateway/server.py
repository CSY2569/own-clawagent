"""Gateway server — multi-channel event loop.

Launches all configured channels in parallel asyncio tasks.
Each channel feeds incoming messages through SessionManager
to the Agent core, and streams responses back through the
appropriate platform renderer.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from clawagent.gateway.channel import IChannel, IncomingMessage, OutgoingMessage
    from clawagent.gateway.renderer import IEventRenderer
    from clawagent.gateway.session_manager import SessionManager

logger = logging.getLogger("gateway")

# Pattern: [FILE:/absolute/path] — agent signals file sharing
_FILE_MARKER = re.compile(r"\[FILE:(.+?)\]")


# ── Renderer registry ─────────────────────────────────────────


def _get_renderer(channel_name: str) -> IEventRenderer:
    """Return the appropriate renderer for a channel type."""
    if channel_name == "wechat":
        from clawagent.gateway.renderers.wechat_renderer import WechatRenderer

        return WechatRenderer()
    from clawagent.gateway.renderer import CliRenderer

    return CliRenderer()


# ── Message handler ───────────────────────────────────────────


async def _handle_message(
    msg: IncomingMessage,
    session_mgr: SessionManager,
    renderer: IEventRenderer,
) -> AsyncIterator[str]:
    """Core message handler — shared by all channels."""
    channel = msg.channel.name.lower()
    t0 = time.monotonic()

    # Log incoming message (truncate long content)
    preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
    logger.info(
        "[%s] ← user=%s len=%d: %s",
        channel, msg.user_id[:12], len(msg.content), preview,
    )

    agent = session_mgr.get_or_create(
        channel_type=channel,
        user_id=msg.user_id,
        session_id=msg.session_id,
    )

    loop = asyncio.get_running_loop()
    tool_calls = 0
    total_tokens = 0

    try:
        events = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: list(agent.stream_events(msg.content)),
            ),
            timeout=120.0,
        )
    except TimeoutError:
        elapsed = time.monotonic() - t0
        logger.warning("[%s] timeout user=%s after %.1fs", channel, msg.user_id[:12], elapsed)
        yield "[Timeout] 请求超时，请重试。"
        return
    except Exception as exc:
        elapsed = time.monotonic() - t0
        logger.error("[%s] error user=%s after %.1fs: %s", channel, msg.user_id[:12], elapsed, exc)
        yield f"[Error] {exc}"
        return

    total_text_len = 0
    for event in events:
        if event.kind == "tool_call":
            tool_calls += 1
            logger.debug(
                "[%s] tool_call=%s args=%s", channel, event.content,
                str(event.metadata.get("args", {}))[:120],
            )
        elif event.kind == "done":
            usage = event.metadata
            total_tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)

        for rendered in renderer.render(event):
            if isinstance(rendered, str) and rendered:
                total_text_len += len(rendered)
                yield rendered

    elapsed = time.monotonic() - t0
    logger.info(
        "[%s] → user=%s time=%.1fs tools=%d tokens=%d out_len=%d",
        channel, msg.user_id[:12], elapsed, tool_calls, total_tokens, total_text_len,
    )


# ── Gateway runner ────────────────────────────────────────────


async def run_gateway(
    channels: list[IChannel],
    session_mgr: SessionManager,
) -> None:
    """Launch all channels and wait for them to complete."""

    async def _channel_handler(channel: IChannel) -> None:
        channel_name = channel.channel_type.name.lower()
        renderer = _get_renderer(channel_name)
        logger.info("[%s] channel started", channel_name)

        async def _on_message(
            msg: IncomingMessage,
        ) -> AsyncIterator[OutgoingMessage]:
            from clawagent.gateway.channel import OutgoingMessage

            # Collect all rendered text chunks
            chunks: list[str] = []
            async for text in _handle_message(msg, session_mgr, renderer):
                chunks.append(text)

            combined = "".join(chunks)

            # Detect [FILE:/path] marker — agent wants to share a file
            file_path: str | None = None
            match = _FILE_MARKER.search(combined)
            if match:
                raw_path = match.group(1).strip()
                resolved = Path(raw_path)
                if not resolved.is_absolute():
                    from clawagent.config import PROJECT_ROOT
                    resolved = PROJECT_ROOT / resolved
                file_path = str(resolved.resolve())
                combined = _FILE_MARKER.sub("", combined).strip()
                logger.info(
                    "[%s] file share detected: %s", channel_name, file_path,
                )

            if combined:
                yield OutgoingMessage(text=combined)
            if file_path:
                yield OutgoingMessage(file_url=file_path)

        try:
            await channel.start(_on_message)
        except Exception:
            logger.exception("[%s] channel crashed", channel_name)
        finally:
            logger.info("[%s] channel stopped", channel_name)

    tasks = [
        asyncio.create_task(_channel_handler(ch), name=ch.channel_type.name)
        for ch in channels
    ]
    await asyncio.gather(*tasks)
