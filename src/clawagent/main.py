"""Entry point for the clawagent CLI."""

import sys
import time
from collections.abc import Iterable
from dataclasses import replace

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.rule import Rule
from rich.table import Table

from clawagent.agent import Agent, Usage, create_agent
from clawagent.api_pool import init_global_pool
from clawagent.config import _PROJECT_ROOT, PriceConfig, Settings, load_price_book
from clawagent.conversation_log import ConversationLogger
from clawagent.memory.summarizer import load_messages
from clawagent.orchestrator.delegator import update_worker_settings
from clawagent.tools.memory_tools import list_sessions as _list_sessions_tool
from clawagent.tools.rag_tool import configure_hybrid_search, search_rag
from clawagent.ui import ConversationStats, render_splash, render_status_line
from clawagent.ui_stream import stream_display

_SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/sessions", "列出所有历史会话"),
    ("/load", "加载指定会话（编号来自 /sessions）"),
    ("/new", "创建新会话"),
    ("/model", "切换模型（如 deepseek-v4-pro）"),
    ("/temp", "设置 temperature（如 0.7）"),
    ("/max-tokens", "设置最大输出 token 数（如 8192）"),
    ("/compress", "切换压缩策略（trim / token_trim / summarize）"),
    ("/settings", "显示当前设置"),
    ("/rag-search", "直接搜索 RAG 向量库（不经过 LLM）"),
    ("/help", "显示此帮助"),
]


class _SlashCommandCompleter(Completer):
    """Show slash command completions when input starts with /."""

    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Iterable[Completion]:
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        for cmd, desc in _SLASH_COMMANDS:
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

    graph, conn = create_agent(settings)
    agent_ref: dict[str, Agent] = {"agent": Agent(graph, db_path=settings.memory_db_path, conn=conn)}

    logger = ConversationLogger()
    logger.log_session_start(agent_ref["agent"].thread_id, settings)

    # Signal: BM25 background build completed (one-shot hint in REPL)
    _bm25_ready_signal: list[bool] = []

    # Initialize RAG if SILICONFLOW_API_KEY is configured
    if settings.siliconflow_api_key:
        import threading

        from clawagent.rag import BM25Retriever, HybridSearcher, RAGStore, SiliconFlowEmbedding

        embedding = SiliconFlowEmbedding(
            api_key=settings.siliconflow_api_key,
            model=settings.siliconflow_model,
            dimensions=settings.siliconflow_dimensions,
            base_url=settings.siliconflow_base_url,
        )
        rag_store = RAGStore(db_path=str(_PROJECT_ROOT / "chroma_db"), embedding=embedding)

        all_docs = rag_store.get_all_documents()
        corpus = [d["text"] for d in all_docs]

        # Create BM25 without building index — build happens in background
        bm25 = BM25Retriever()

        def _knn_retrieve(query: str, k: int) -> list[dict[str, str]]:
            return rag_store.retrieve(query, top_k=k)

        hybrid = HybridSearcher(
            knn_retriever=_knn_retrieve,
            bm25_retriever=bm25,
            all_docs=all_docs,
        )
        configure_hybrid_search(hybrid)

        def _build_bm25() -> None:
            bm25.build_async(corpus)
            _bm25_ready_signal.append(True)

        threading.Thread(target=_build_bm25, daemon=True).start()

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
    if settings.siliconflow_api_key:
        console.print("  [dim]BM25 索引后台构建中，搜索将临时使用纯向量检索...[/dim]")

    stats = ConversationStats(start_time=time.monotonic())

    pt_session: PromptSession[str] = PromptSession(
        history=InMemoryHistory(),
        style=Style.from_dict({"prompt": "bold green"}),
        completer=_SlashCommandCompleter(),
        complete_while_typing=True,
    )

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
                settings, pricing = _handle_command(user_input, agent_ref, settings, console, stats, pricing, logger)
                continue

            try:
                response_text = ""
                round_usage = Usage()
                with stream_display() as display:
                    for event in agent_ref["agent"].stream_events(user_input):
                        display.handle(event)
                        if event.kind == "done":
                            response_text = event.content
                            round_usage = Usage(
                                input_tokens=event.metadata.get("input_tokens", 0),
                                output_tokens=event.metadata.get("output_tokens", 0),
                                cache_read_input_tokens=event.metadata.get("cache_read_input_tokens", 0),
                                cache_creation_input_tokens=event.metadata.get("cache_creation_input_tokens", 0),
                            )
                            stats.update(round_usage)
                logger.log_turn(
                    agent_ref["agent"].thread_id,
                    stats.message_count,
                    user_input,
                    response_text,
                    round_usage,
                    settings,
                )
            except Exception:
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
        logger.log_session_end(
            agent_ref["agent"].thread_id,
            stats.message_count,
            Usage(
                input_tokens=stats.cumulative_input_tokens,
                output_tokens=stats.cumulative_output_tokens,
                cache_read_input_tokens=stats.cumulative_cache_read_tokens,
                cache_creation_input_tokens=stats.cumulative_cache_creation_tokens,
            ),
        )
        conn.close()


def _handle_command(
    cmd: str,
    agent_ref: dict[str, Agent],
    settings: Settings,
    console: Console,
    stats: ConversationStats,
    pricing: PriceConfig,
    logger: ConversationLogger,
) -> tuple[Settings, PriceConfig]:
    """Handle slash commands in interactive mode."""
    cmd = cmd.lower()
    agent = agent_ref["agent"]

    if cmd == "/sessions" or cmd == "/list":
        _show_sessions(agent, settings, console)
    elif cmd.startswith("/load "):
        _load_session(cmd[6:].strip(), agent, settings, console)
    elif cmd == "/new":
        _new_session(agent_ref, settings, console, stats, logger)
    elif cmd.startswith("/model "):
        model_name = cmd[7:].strip()
        try:
            new_settings = replace(settings, model_name=model_name)
            agent.reconfigure(new_settings)
            update_worker_settings(new_settings)
            new_pricing = load_price_book().get(model_name)
            logger.log_settings_change(agent.thread_id, "model_name", settings.model_name, model_name)
            console.print(f"[green]模型已切换至: {model_name}[/green]")
            return new_settings, new_pricing
        except Exception as e:
            console.print(f"[red]无效模型名 '{model_name}'：{e}[/red]")
            return settings, pricing
    elif cmd.startswith("/temp "):
        try:
            temp = float(cmd[6:].strip())
        except ValueError:
            console.print("[red]无效的温度值，请输入数字如 0.7[/red]")
            return settings, pricing
        new_settings = replace(settings, temperature=temp)
        agent.reconfigure(new_settings)
        update_worker_settings(new_settings)
        logger.log_settings_change(agent.thread_id, "temperature", settings.temperature, temp)
        console.print(f"[green]温度已设置为: {temp}[/green]")
        return new_settings, pricing
    elif cmd.startswith("/max-tokens "):
        try:
            max_tok = int(cmd[12:].strip())
        except ValueError:
            console.print("[red]无效的 token 数，请输入整数如 8192[/red]")
            return settings, pricing
        new_settings = replace(settings, max_tokens=max_tok)
        agent.reconfigure(new_settings)
        update_worker_settings(new_settings)
        logger.log_settings_change(agent.thread_id, "max_tokens", settings.max_tokens, max_tok)
        console.print(f"[green]最大输出 token 数已设置为: {max_tok}[/green]")
        return new_settings, pricing
    elif cmd == "/settings":
        from clawagent.ui import _format_tokens

        console.print(
            f"[bold]Model[/] {settings.model_name}  "
            f"[bold]T[/] {settings.temperature}  "
            f"[bold]Tok[/] {settings.max_tokens}  "
            f"[bold]Ctx[/] {_format_tokens(settings.context_window)}  "
            f"[bold]Compress[/] {settings.compression_strategy}"
        )
    elif cmd.startswith("/compress "):
        strategy = cmd[10:].strip()
        valid = {"trim", "token_trim", "summarize"}
        if strategy not in valid:
            console.print(f"[yellow]无效策略: {strategy}。可选: {', '.join(sorted(valid))}[/yellow]")
            return settings, pricing
        new_settings = replace(settings, compression_strategy=strategy)
        agent.reconfigure(new_settings)
        update_worker_settings(new_settings)
        logger.log_settings_change(agent.thread_id, "compression_strategy", settings.compression_strategy, strategy)
        console.print(f"[green]压缩策略已切换至: {strategy}[/green]")
        return new_settings, pricing
    elif cmd.startswith("/rag-search "):
        query = cmd[12:].strip()
        if not query:
            console.print("[yellow]用法: /rag-search <关键词>[/yellow]")
        else:
            _rag_search(query, console)
    elif cmd == "/help":
        table = Table(box=None, padding=(0, 2))
        table.add_column("Command", style="cyan", no_wrap=True)
        table.add_column("Description", style="dim")
        for cmd_name, desc in _SLASH_COMMANDS:
            table.add_row(cmd_name, desc)
        table.add_row("quit / q", "退出")
        console.print(table)
    else:
        console.print(f"[yellow]未知命令: {cmd}。输入 /help 查看可用命令。[/yellow]")

    return settings, pricing


def _rag_search(query: str, console: Console) -> None:
    """Search the RAG vector store directly and display results."""
    hits = search_rag(query, top_k=5)
    if not hits:
        console.print("[yellow]未在文档中找到相关内容。[/yellow]")
        return

    console.print(f"[bold]RAG 检索结果 — \"{query}\":[/bold]")
    for i, h in enumerate(hits, 1):
        score = h.get("score", "?")
        chapter = h.get("chapter", "")
        text = h.get("text", "")
        meta_parts = [f"相关度: {score}"]
        if chapter:
            meta_parts.append(chapter)
        label = f"[{i}] ({', '.join(meta_parts)}) — {text[:100]}"
        if len(text) > 100:
            label += "..."
        console.print(label)
    console.print()


def _show_sessions(
    agent: Agent,
    settings: Settings,
    console: Console,
) -> None:
    """Show all historical sessions."""
    result = _list_sessions_tool.invoke({})
    console.print(f"[bold]历史会话:[/bold]\n{result}")


def _load_session(
    session_id: str,
    agent: Agent,
    settings: Settings,
    console: Console,
) -> None:
    """Load a historical session and switch to it for continued conversation."""
    from clawagent.memory.summarizer import get_summary

    # Load messages from the session
    messages = load_messages(settings.memory_db_path, session_id)
    if not messages:
        # Try summary as fallback
        summary = get_summary(settings.memory_db_path, session_id)
        if summary:
            console.print(f"[bold]会话 {session_id}:[/bold]")
            console.print(f"标题: {summary['title']}")
            console.print(f"摘要: {summary['summary']}")
        else:
            console.print(f"[red]未找到会话: {session_id}[/red]")
            return

    if messages:
        console.print(f"[bold]会话 {session_id}:[/bold]")
        for m in messages:
            role = m.get("role", "?")
            content = m.get("content", "")
            style = "green" if role == "user" else "blue"
            console.print(f"[{style}]{role}:[/] {content[:200]}")
        console.print(f"\n[dim]共 {len(messages)} 条消息[/dim]")

    agent._thread_id = session_id
    console.print(f"\n[green]已切换到会话 {session_id}，可以继续对话。[/green]")


def _new_session(
    agent_ref: dict[str, Agent],
    settings: Settings,
    console: Console,
    stats: ConversationStats,
    logger: ConversationLogger,
) -> None:
    """Create a new session and switch to it immediately."""
    old_agent = agent_ref["agent"]
    logger.log_session_end(
        old_agent.thread_id,
        stats.message_count,
        Usage(
            input_tokens=stats.cumulative_input_tokens,
            output_tokens=stats.cumulative_output_tokens,
            cache_read_input_tokens=stats.cumulative_cache_read_tokens,
            cache_creation_input_tokens=stats.cumulative_cache_creation_tokens,
        ),
    )
    old_agent.close()

    graph, conn = create_agent(settings)
    agent_ref["agent"] = Agent(graph, db_path=settings.memory_db_path, conn=conn)
    logger.log_session_start(agent_ref["agent"].thread_id, settings)
    stats.reset()
    console.print(f"[bold green]已创建新会话: {agent_ref['agent'].thread_id}[/bold green]")
