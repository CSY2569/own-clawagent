"""CLI display helpers — session listing, loading, RAG search results."""

from rich.console import Console

from clawagent.agent import Agent, Usage, create_agent
from clawagent.config import Settings
from clawagent.conversation_log import ConversationLogger
from clawagent.memory.summarizer import get_summary, list_summaries, list_thread_ids, load_messages
from clawagent.tools.rag_tool import search_rag
from clawagent.ui import ConversationStats


def rag_search(query: str, console: Console) -> None:
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


def show_sessions(agent: Agent, settings: Settings, console: Console) -> None:
    """Show all historical sessions."""
    sessions = list_thread_ids(settings.memory_db_path)
    if not sessions:
        sessions = list_summaries(settings.memory_db_path)
        if not sessions:
            console.print("[bold]历史会话:[/bold]\n暂无历史会话记录。")
            return

    lines = []
    for s in sessions:
        tid = s.get("thread_id", "?")
        updated = s.get("updated_at", "")[:16]
        count = s.get("message_count", 0)
        summary = get_summary(settings.memory_db_path, tid)
        title = summary["title"] if summary else s.get("title", "未命名")
        if not title or title == "未命名":
            title = s.get("title", "未命名")[:40]
        lines.append(f"[{tid}] {updated} — \"{title}\" — {count} 轮对话")
    console.print("[bold]历史会话:[/bold]\n" + "\n".join(lines))


def load_session(
    session_id: str,
    agent: Agent,
    settings: Settings,
    console: Console,
) -> None:
    """Load a historical session and switch to it for continued conversation."""
    from clawagent.memory.summarizer import get_summary

    messages = load_messages(settings.memory_db_path, session_id)
    if not messages:
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


def new_session(
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

    graph, conn, factory, delegate_tool = create_agent(settings)
    agent_ref["agent"] = Agent(
        graph,
        db_path=settings.memory_db_path,
        conn=conn,
        factory=factory,
        delegate_tool=delegate_tool,
    )
    logger.log_session_start(agent_ref["agent"].thread_id, settings)
    stats.reset()
    console.print(f"[bold green]已创建新会话: {agent_ref['agent'].thread_id}[/bold green]")
