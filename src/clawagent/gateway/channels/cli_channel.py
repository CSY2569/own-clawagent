"""CLI Channel — validates that the Channel abstraction works.

This wraps the existing CLI REPL as an IChannel implementation.
When ``uv run clawagent gateway`` is started with the CLI channel
enabled, the behavior should be identical to ``uv run clawagent``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable

from prompt_toolkit import PromptSession

from clawagent.gateway.channel import (
    ChannelType,
    IChannel,
    IncomingMessage,
    OutgoingMessage,
)


class CliChannel(IChannel):
    """CLI channel — interactive terminal via prompt_toolkit.

    Proves that the IChannel interface is sufficient to host the
    existing REPL experience as one channel among potentially many.
    """

    @property
    def channel_type(self) -> ChannelType:
        return ChannelType.CLI

    async def start(
        self,
        handler: Callable[[IncomingMessage], AsyncIterator[OutgoingMessage]],
    ) -> None:
        """Run the interactive REPL loop.

        Reads lines from stdin, builds IncomingMessage objects,
        calls handler, and prints OutgoingMessage text to stdout.
        """
        session: PromptSession[str] = PromptSession("› ")
        print("闻宝 Gateway (CLI Channel) — type 'quit' to exit")

        try:
            while True:
                user_input = await session.prompt_async()
                user_input = user_input.strip()

                if user_input.lower() in ("quit", "exit", "q"):
                    break
                if not user_input:
                    continue

                msg = IncomingMessage(
                    channel=ChannelType.CLI,
                    user_id="cli_user",
                    session_id="default",
                    content=user_input,
                )
                async for out in handler(msg):
                    print(out.text, end="", flush=True)
                print()
        except (EOFError, KeyboardInterrupt):
            pass

    async def stop(self) -> None:
        """No external resources to clean up."""
        pass
