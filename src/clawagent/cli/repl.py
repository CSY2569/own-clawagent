"""REPL interactive loop — input handling, command dispatch, streaming.

The functions in this module are called from ``clawagent.main`` after
``init_session()`` sets up the ``SessionContext``.
"""

from __future__ import annotations

from collections.abc import Iterable

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.rule import Rule

from clawagent.agent import Usage
from clawagent.cli.commands import handle_command
from clawagent.cli.session import SessionContext
from clawagent.cli.streaming import run_streaming_round
from clawagent.ui import render_status_line


class _SlashCommandCompleter(Completer):
    """Show slash command completions when input starts with /."""

    def __init__(self) -> None:
        from clawagent.cli import SLASH_COMMANDS
        self._commands = SLASH_COMMANDS

    def get_completions(self, document: Document, _complete_event: CompleteEvent) -> Iterable[Completion]:
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        for cmd, desc in self._commands:
            if cmd.startswith(text):
                yield Completion(cmd, start_position=-len(text), display_meta=desc)


def _handle_user_message(
    ctx: SessionContext,
    user_input: str,
    console: Console,
) -> None:
    """Process one user message through the streaming pipeline."""
    agent = ctx.agent_ref.agent

    try:
        result = run_streaming_round(agent, user_input)

        ctx.stats.update(result.usage)
        ctx.logger.log_turn(
            agent.thread_id,
            ctx.stats.message_count,
            user_input,
            result.response_text,
            result.usage,
            ctx.settings,
        )

    except KeyboardInterrupt:
        console.print("\n  [yellow]⏸ 已中断[/yellow]")

    except Exception:
        console.print("\n[red]Streaming error, falling back to non-streaming mode...[/red]")
        response = agent.run(user_input)
        ctx.stats.update(response.usage)
        console.print(f"\n[bold blue]Agent:[/bold blue] {response.text}")
        ctx.logger.log_turn(
            agent.thread_id,
            ctx.stats.message_count,
            user_input,
            response.text,
            response.usage,
            ctx.settings,
            error="streaming_error",
        )


def _check_bm25_ready(ctx: SessionContext, console: Console) -> None:
    """Notify when the BM25 index finishes building."""
    if ctx.bm25_ready_signal:
        console.print()
        console.print("[dim]BM25 索引构建完成，搜索已升级为混合检索[/dim]")
        ctx.bm25_ready_signal.clear()


def run_repl(ctx: SessionContext) -> None:
    """Run the interactive REPL loop.

    Reads user input, dispatches slash commands, streams agent responses,
    and handles graceful shutdown.
    """
    console = Console()

    pt_session: PromptSession[str] = PromptSession(
        history=InMemoryHistory(),
        style=Style.from_dict({"prompt": "bold green"}),
        completer=_SlashCommandCompleter(),
        complete_while_typing=True,
    )

    try:
        while True:
            _check_bm25_ready(ctx, console)
            console.print()
            render_status_line(ctx.stats, ctx.pricing, ctx.settings, console)
            console.print(Rule(style="dim"))
            user_input = pt_session.prompt(
                [("class:prompt", "› ")]
            ).strip()

            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue

            if user_input.startswith("/"):
                ctx.settings, ctx.pricing = handle_command(
                    user_input, ctx.agent_ref, ctx.settings, console,
                    ctx.stats, ctx.pricing, ctx.logger,
                )
                continue

            _handle_user_message(ctx, user_input, console)

    except (EOFError, KeyboardInterrupt):
        console.print("\nGoodbye!")
    finally:
        _cleanup(ctx, console)


def _cleanup(
    ctx: SessionContext,
    console: Console,
) -> None:
    """Release resources at REPL exit."""
    ctx.logger.log_session_end(
        ctx.agent_ref.agent.thread_id,
        ctx.stats.message_count,
        Usage(
            input_tokens=ctx.stats.cumulative_input_tokens,
            output_tokens=ctx.stats.cumulative_output_tokens,
            cache_read_input_tokens=ctx.stats.cumulative_cache_read_tokens,
            cache_creation_input_tokens=ctx.stats.cumulative_cache_creation_tokens,
            prompt_cache_hit_tokens=ctx.stats.cumulative_cache_hit_tokens,
            prompt_cache_miss_tokens=ctx.stats.cumulative_cache_miss_tokens,
        ),
    )
    ctx.conn.close()
