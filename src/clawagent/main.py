"""Entry point for the clawagent CLI."""

import queue
import sys
import threading
import time
from collections.abc import Iterable

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.rule import Rule

from clawagent.agent import Agent, Usage, create_agent
from clawagent.api_pool import init_global_pool
from clawagent.cancel_token import CancelToken
from clawagent.cli import SLASH_COMMANDS
from clawagent.cli.commands import handle_command
from clawagent.config import Settings, load_price_book
from clawagent.conversation_log import ConversationLogger
from clawagent.rag.bootstrap import bootstrap_rag
from clawagent.tools.rag_tool import configure_hybrid_search
from clawagent.ui import ConversationStats, render_splash, render_status_line
from clawagent.ui_stream import stream_display


class _SlashCommandCompleter(Completer):
    """Show slash command completions when input starts with /."""

    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Iterable[Completion]:
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        for cmd, desc in SLASH_COMMANDS:
            if cmd.startswith(text):
                yield Completion(cmd, start_position=-len(text), display_meta=desc)


def main() -> None:
    """Run the clawagent from the command line.

    Usage: uv run clawagent "Your question here"
    If no argument is given, runs an interactive REPL.
    """
    try:
        settings = Settings.from_env()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Initialize API key pool (no-op if no API_POOL_* vars set)
    pool = init_global_pool()

    graph, conn, factory, delegate_tool = create_agent(settings)
    agent_ref: dict[str, Agent] = {
        "agent": Agent(
            graph,
            db_path=settings.memory_db_path,
            conn=conn,
            factory=factory,
            delegate_tool=delegate_tool,
        )
    }

    logger = ConversationLogger()
    logger.log_session_start(agent_ref["agent"].thread_id, settings)

    # Initialize RAG (BM25 builds in background)
    _bm25_ready_signal: list[bool] = []
    rag_ctx = bootstrap_rag(settings, configure_hybrid_search)
    if rag_ctx:
        _bm25_ready_signal = rag_ctx.bm25_ready_signal

    # One-shot mode — plain output, no Rich
    if len(sys.argv) > 1:
        response = agent_ref["agent"].run(" ".join(sys.argv[1:]))
        print(response.text)
        conn.close()
        return

    # Interactive mode — Rich dashboard
    console = Console()
    pricing = load_price_book().get(settings.model_name)

    # Collect worker roles for splash display
    from clawagent.worker.config import load_worker_configs

    worker_configs = load_worker_configs()
    worker_roles = list(worker_configs.keys())

    all_stats = pool.get_all_stats()
    render_splash(settings, pricing, console, pool_stats=all_stats, worker_roles=worker_roles)
    if rag_ctx:
        console.print("  [dim]BM25 索引后台构建中，搜索将临时使用纯向量检索...[/dim]")

    stats = ConversationStats(start_time=time.monotonic())

    pt_session: PromptSession[str] = PromptSession(
        history=InMemoryHistory(),
        style=Style.from_dict({"prompt": "bold green"}),
        completer=_SlashCommandCompleter(),
        complete_while_typing=True,
    )

    _current_worker: threading.Thread | None = None
    _current_cancel: threading.Event | None = None

    try:
        while True:
            if _bm25_ready_signal:
                console.print()
                console.print("[dim]BM25 索引构建完成，搜索已升级为混合检索[/dim]")
                _bm25_ready_signal.clear()
            console.print()
            render_status_line(stats, pricing, settings, console)
            console.print(Rule(style="dim"))
            user_input = pt_session.prompt(
                [("class:prompt", "› ")]
            ).strip()
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue

            # Handle slash commands
            if user_input.startswith("/"):
                settings, pricing = handle_command(
                    user_input, agent_ref, settings, console, stats, pricing, logger
                )
                continue

            try:
                response_text = ""
                round_usage = Usage()
                event_queue: queue.Queue = queue.Queue(maxsize=64)  # type: ignore[type-arg]
                cancel_event = threading.Event()

                def _produce(
                    _user_input: str = user_input,
                    _cancel_event: threading.Event = cancel_event,
                    _event_queue: queue.Queue = event_queue,  # type: ignore[type-arg]
                ) -> None:
                    try:
                        for event in agent_ref["agent"].stream_events(_user_input):
                            if _cancel_event.is_set():
                                return
                            _event_queue.put(event, timeout=0.5)
                    except queue.Full:
                        pass
                    except Exception:
                        pass

                worker = threading.Thread(target=_produce, daemon=True)
                _current_worker = worker
                _current_cancel = cancel_event
                worker.start()

                with CancelToken() as cancel, stream_display() as display:
                    while True:
                        try:
                            event = event_queue.get(timeout=0.1)
                        except queue.Empty:
                            cancel.check()
                            if not worker.is_alive() and event_queue.empty():
                                break
                            continue
                        cancel.check()
                        display.handle(event)
                        if event.kind == "done":
                            response_text = event.content
                            round_usage = Usage(
                                input_tokens=event.metadata.get("input_tokens", 0),
                                output_tokens=event.metadata.get("output_tokens", 0),
                                cache_read_input_tokens=event.metadata.get("cache_read_input_tokens", 0),
                                cache_creation_input_tokens=event.metadata.get("cache_creation_input_tokens", 0),
                                prompt_cache_hit_tokens=event.metadata.get("prompt_cache_hit_tokens", 0),
                                prompt_cache_miss_tokens=event.metadata.get("prompt_cache_miss_tokens", 0),
                            )
                            stats.update(round_usage)
                            break
                if response_text:
                    logger.log_turn(
                        agent_ref["agent"].thread_id,
                        stats.message_count,
                        user_input,
                        response_text,
                        round_usage,
                        settings,
                    )
            except KeyboardInterrupt:
                cancel_event.set()
                console.print("\n  [yellow]⏸ 已中断[/yellow]")
                continue
            except Exception:
                cancel_event.set()
                console.print("\n[red]Streaming error, falling back to non-streaming mode...[/red]")
                response = agent_ref["agent"].run(user_input)
                stats.update(response.usage)
                console.print(f"\n[bold blue]Agent:[/bold blue] {response.text}")
                logger.log_turn(
                    agent_ref["agent"].thread_id,
                    stats.message_count,
                    user_input,
                    response.text,
                    response.usage,
                    settings,
                    error="streaming_error",
                )
    except (EOFError, KeyboardInterrupt):
        console.print("\nGoodbye!")
    finally:
        if _current_worker is not None and _current_worker.is_alive():
            if _current_cancel is not None:
                _current_cancel.set()
            _current_worker.join(timeout=2.0)
        logger.log_session_end(
            agent_ref["agent"].thread_id,
            stats.message_count,
            Usage(
                input_tokens=stats.cumulative_input_tokens,
                output_tokens=stats.cumulative_output_tokens,
                cache_read_input_tokens=stats.cumulative_cache_read_tokens,
                cache_creation_input_tokens=stats.cumulative_cache_creation_tokens,
                prompt_cache_hit_tokens=stats.cumulative_cache_hit_tokens,
                prompt_cache_miss_tokens=stats.cumulative_cache_miss_tokens,
            ),
        )
        conn.close()
