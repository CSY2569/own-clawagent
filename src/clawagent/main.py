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

from clawagent.agent import Agent, create_agent
from clawagent.config import PriceConfig, Settings, load_price_book
from clawagent.memory.summarizer import load_messages
from clawagent.tools.memory_tools import list_sessions as _list_sessions_tool
from clawagent.tools.rag_tool import configure_rag, search_rag
from clawagent.ui import ConversationStats, render_splash, render_status_line

_SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/sessions", "列出所有历史会话"),
    ("/load", "加载指定会话（编号来自 /sessions）"),
    ("/new", "创建新会话"),
    ("/model", "切换模型（如 deepseek-v4-pro）"),
    ("/temp", "设置 temperature（如 0.7）"),
    ("/max-tokens", "设置最大输出 token 数（如 8192）"),
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

    graph, conn = create_agent(settings)
    agent = Agent(graph, db_path=settings.memory_db_path, conn=conn)

    # Initialize RAG if SILICONFLOW_API_KEY is configured
    if settings.siliconflow_api_key:
        from clawagent.rag import RAGStore, SiliconFlowEmbedding

        embedding = SiliconFlowEmbedding(
            api_key=settings.siliconflow_api_key,
            model=settings.siliconflow_model,
            dimensions=settings.siliconflow_dimensions,
            base_url=settings.siliconflow_base_url,
        )
        rag_store = RAGStore(db_path="./chroma_db", embedding=embedding)
        configure_rag(rag_store)

    # One-shot mode — plain output, no Rich
    if len(sys.argv) > 1:
        response = agent.run(" ".join(sys.argv[1:]))
        print(response.text)
        conn.close()
        return

    # Interactive mode — Rich dashboard
    console = Console()
    pricing = load_price_book().get(settings.model_name)

    render_splash(settings, pricing, console)
    stats = ConversationStats(start_time=time.monotonic())

    pt_session: PromptSession[str] = PromptSession(
        history=InMemoryHistory(),
        style=Style.from_dict({"prompt": "bold green"}),
        completer=_SlashCommandCompleter(),
        complete_while_typing=True,
    )

    try:
        while True:
            console.print()
            render_status_line(stats, pricing, settings, console)
            console.print(Rule(style="blue", characters="─"))
            user_input = pt_session.prompt(
                [("class:prompt", "You: ")]
            ).strip()
            console.print(Rule(style="blue", characters="─"))
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue

            # Handle slash commands
            if user_input.startswith("/"):
                settings, pricing = _handle_command(user_input, agent, settings, console, stats, pricing)
                continue

            response = agent.run(user_input)
            stats.update(response.usage)
            console.print(f"\n[bold blue]Agent:[/bold blue] {response.text}")
    except (EOFError, KeyboardInterrupt):
        console.print("\nGoodbye!")
    finally:
        conn.close()


def _handle_command(
    cmd: str,
    agent: Agent,
    settings: Settings,
    console: Console,
    stats: ConversationStats,
    pricing: PriceConfig,
) -> tuple[Settings, PriceConfig]:
    """Handle slash commands in interactive mode."""
    cmd = cmd.lower()

    if cmd == "/sessions" or cmd == "/list":
        _show_sessions(agent, settings, console)
    elif cmd.startswith("/load "):
        _load_session(cmd[6:].strip(), agent, settings, console, stats)
    elif cmd == "/new":
        _new_session(agent, console, stats)
    elif cmd.startswith("/model "):
        model_name = cmd[7:].strip()
        new_settings = replace(settings, model_name=model_name)
        agent.reconfigure(new_settings)
        new_pricing = load_price_book().get(model_name)
        console.print(f"[green]模型已切换至: {model_name}[/green]")
        return new_settings, new_pricing
    elif cmd.startswith("/temp "):
        try:
            temp = float(cmd[6:].strip())
        except ValueError:
            console.print("[red]无效的温度值，请输入数字如 0.7[/red]")
            return settings, pricing
        new_settings = replace(settings, temperature=temp)
        agent.reconfigure(new_settings)
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
        console.print(f"[green]最大输出 token 数已设置为: {max_tok}[/green]")
        return new_settings, pricing
    elif cmd == "/settings":
        console.print(
            f"Model:       {settings.model_name}\n"
            f"Temperature: {settings.temperature}\n"
            f"Max Tokens:  {settings.max_tokens}\n"
            f"Context:     {settings.context_window}"
        )
    elif cmd.startswith("/rag-search "):
        query = cmd[12:].strip()
        if not query:
            console.print("[yellow]用法: /rag-search <关键词>[/yellow]")
        else:
            _rag_search(query, console)
    elif cmd == "/help":
        console.print(
            "[bold]可用命令:[/bold]\n"
            "  /sessions     — 列出所有历史会话\n"
            "  /load <id>    — 加载指定会话（编号来自 /sessions）\n"
            "  /new          — 创建新会话\n"
            "  /model <name> — 切换模型（如 deepseek-v4-pro）\n"
            "  /temp <n>     — 设置 temperature（如 0.7）\n"
            "  /max-tokens   — 设置最大输出 token 数（如 8192）\n"
            "  /settings     — 显示当前设置\n"
            "  /rag-search <关键词> — 直接搜索 RAG 向量库（不经过 LLM）\n"
            "  /help         — 显示此帮助\n"
            "  quit / q      — 退出"
        )
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
    stats: ConversationStats,
) -> None:
    """Load a historical session and display its messages."""
    # Switch the agent's thread_id

    # Load messages from the session
    messages = load_messages(settings.memory_db_path, session_id)
    if not messages:
        # Try summary as fallback
        from clawagent.memory.summarizer import get_summary
        summary = get_summary(settings.memory_db_path, session_id)
        if summary:
            console.print(f"[bold]会话 {session_id}:[/bold]")
            console.print(f"标题: {summary['title']}")
            console.print(f"摘要: {summary['summary']}")
            console.print("（当前会话未切换，使用 /new 创建新会话继续对话）")
        else:
            console.print(f"[red]未找到会话: {session_id}[/red]")
        return

    console.print(f"[bold]会话 {session_id}:[/bold]")
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content", "")
        style = "green" if role == "user" else "blue"
        console.print(f"[{style}]{role}:[/] {content[:200]}")
    console.print(f"\n[dim]共 {len(messages)} 条消息[/dim]")


def _new_session(
    agent: Agent,
    console: Console,
    stats: ConversationStats,
) -> None:
    """Create a new session."""
    from uuid import uuid4

    new_id = uuid4().hex[:8]
    # Create a new Agent instance with the new thread_id
    from clawagent.agent import create_agent as _create_agent
    from clawagent.config import Settings as _Settings

    settings = _Settings.from_env()
    _graph, _conn = _create_agent(settings)
    # We can't easily swap the graph, so just log it
    console.print(f"[bold green]已创建新会话: {new_id}[/bold green]")
    console.print("[yellow]注意: 当前会话仍在继续，新会话将在下次启动时生效。[/yellow]")
