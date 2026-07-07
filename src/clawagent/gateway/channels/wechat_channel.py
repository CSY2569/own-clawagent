"""WeChat iLink Bot channel — personal WeChat via official iLink protocol.

Uses the ``weixin-ilink`` Python SDK for long-polling message receive.
No public URL or webhook needed — the channel polls WeChat servers
and forwards messages to the Agent core.

Login: on first run, a QR code is displayed in the terminal.
Credentials are persisted to ``ilink_creds.json`` for subsequent runs.

Limitations (iLink protocol):
- Cannot initiate conversations — user must send first message.
- 24-hour inactivity window — replies after 24h are dropped.
- Tencent may terminate the iLink service at any time.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path

from weixin_ilink import WeixinBot  # type: ignore[import-untyped]

from clawagent.gateway.channel import (
    ChannelType,
    IChannel,
    IncomingMessage,
    MessageType,
    OutgoingMessage,
)

logger = logging.getLogger("gateway.wechat")


class WechatChannel(IChannel):
    """Personal WeChat channel via iLink Bot protocol."""

    def __init__(self, creds_file: str = "ilink_creds.json") -> None:
        self._creds_file = creds_file
        self._bot: WeixinBot | None = None
        self._poll_future: asyncio.Future[None] | None = None
        self._stop_event = asyncio.Event()
        self._msg_count: int = 0

    # ── IChannel interface ─────────────────────────────────────

    @property
    def channel_type(self) -> ChannelType:
        return ChannelType.WECHAT

    async def start(
        self,
        handler: Callable[[IncomingMessage], AsyncIterator[OutgoingMessage]],
    ) -> None:
        loop = asyncio.get_running_loop()

        creds_path = Path(self._creds_file)
        if creds_path.exists():
            logger.info("loading saved credentials from %s", self._creds_file)
            self._bot = WeixinBot(credentials_file=str(creds_path))
        else:
            logger.info("starting login — scan QR code in terminal")
            self._bot = WeixinBot.from_login(save_to=str(creds_path))
            logger.info("login successful, credentials saved to %s", self._creds_file)

        bot = self._bot
        logger.info("polling started (bot_id=%s)", getattr(bot, "account_id", "?"))

        def _poll_loop() -> None:
            try:
                for wx_msg in bot.messages():
                    if self._stop_event.is_set():
                        break

                    if not wx_msg.is_text or not wx_msg.text:
                        continue

                    self._msg_count += 1
                    incoming = IncomingMessage(
                        channel=ChannelType.WECHAT,
                        user_id=wx_msg.from_user or "unknown",
                        session_id="default",
                        content=wx_msg.text,
                        message_type=MessageType.TEXT,
                        raw={
                            "context_token": wx_msg.context_token,
                            "message_id": wx_msg.message_id,
                        },
                    )

                    future = asyncio.run_coroutine_threadsafe(
                        self._consume_and_reply(handler, incoming, bot, wx_msg),
                        loop,
                    )
                    try:
                        future.result(timeout=120)
                    except TimeoutError:
                        logger.warning(
                            "handler timed out for user=%s msg=%d",
                            wx_msg.from_user, self._msg_count,
                        )
                    except Exception:
                        logger.exception(
                            "handler error for user=%s msg=%d",
                            wx_msg.from_user, self._msg_count,
                        )

            except Exception:
                if not self._stop_event.is_set():
                    logger.exception("poll loop crashed after %d messages", self._msg_count)

        self._poll_future = asyncio.ensure_future(
            loop.run_in_executor(None, _poll_loop)
        )
        await self._poll_future

    async def stop(self) -> None:
        logger.info("stopping (processed %d messages)", self._msg_count)
        self._stop_event.set()
        if self._bot:
            self._bot.stop()
            self._bot = None
        if self._poll_future and not self._poll_future.done():
            self._poll_future.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._poll_future
        logger.info("stopped")

    # ── Message processing ─────────────────────────────────────

    async def _consume_and_reply(
        self,
        handler: Callable[[IncomingMessage], AsyncIterator[OutgoingMessage]],
        msg: IncomingMessage,
        bot: WeixinBot,
        wx_msg: object,
    ) -> None:
        ctx_token: str | None = getattr(wx_msg, "context_token", None)
        to_user: str = getattr(wx_msg, "from_user", "") or ""

        t0 = time.monotonic()
        reply_count = 0
        try:
            bot.send_typing(to_user, context_token=ctx_token)

            async for out in handler(msg):
                if out.text and to_user:
                    bot.send_text(to_user, out.text, context_token=ctx_token)
                    reply_count += 1

                if out.file_url and to_user:
                    file_path = Path(out.file_url)
                    if file_path.is_file():
                        logger.info(
                            "sending file user=%s name=%s size=%d",
                            to_user, file_path.name, file_path.stat().st_size,
                        )
                        bot.send_file(
                            to_user,
                            str(file_path),
                            file_name=file_path.name,
                            context_token=ctx_token,
                        )
                        reply_count += 1
                    else:
                        logger.warning(
                            "file not found user=%s path=%s", to_user, out.file_url,
                        )

        except Exception:
            logger.exception("reply failed user=%s chunks=%d", to_user, reply_count)
        else:
            elapsed = time.monotonic() - t0
            logger.debug(
                "reply sent to user=%s chunks=%d time=%.1fs",
                to_user, reply_count, elapsed,
            )
