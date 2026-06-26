"""Memory tools: list_sessions, recall_session, summarize_session.

These tools require _db_path to be set before use (done in agent.py during
create_agent). This module-level state pattern is standard for LangChain tools
that need access to shared resources.
"""

from langchain_core.tools import tool

from clawagent.memory.summarizer import (
    generate_session_summary,
    get_summary,
    list_summaries,
    list_thread_ids,
    load_messages,
    save_summary,
)

# Module-level state, set by agent.py on startup
_db_path: str = ""
_model: object | None = None


def configure(db_path: str, model: object | None = None) -> None:
    """Set the database path and model reference for memory tools.

    Called during agent initialization (see agent.py create_agent).
    """
    global _db_path, _model
    _db_path = db_path
    _model = model


@tool
def list_sessions() -> str:
    """列出所有历史会话。

    返回每个会话的编号、时间、标题和消息数量。
    当用户询问"我们之前聊过什么"时使用。
    """
    # Primary source: all thread_ids with messages
    sessions = list_thread_ids(_db_path)
    if not sessions:
        # Fallback: check session_summaries table
        ssummaries = list_summaries(_db_path)
        if not ssummaries:
            return "暂无历史会话记录。"
        sessions = ssummaries

    lines = []
    for s in sessions:
        tid = s.get("thread_id", "?")
        updated = s.get("updated_at", "")[:16]
        count = s.get("message_count", 0)

        # Try to get a richer title from session_summaries
        summary = get_summary(_db_path, tid)
        title = summary["title"] if summary else s.get("title", "未命名")
        if not title or title == "未命名":
            title = s.get("title", "未命名")[:40]

        lines.append(f"[{tid}] {updated} — \"{title}\" — {count} 轮对话")
    return "\n".join(lines)


@tool
def recall_session(session_id: str, summary_only: bool = True) -> str:
    """加载指定会话的详细内容。

    Args:
        session_id: 会话编号（如 list_sessions 返回的 [abc123]）。
        summary_only: 为 True 时返回摘要（省 token），为 False 时返回完整对话。
    """
    if summary_only:
        summary = get_summary(_db_path, session_id)
        if summary is None:
            return f"未找到会话: {session_id}"
        return (
            f"标题: {summary['title']}\n"
            f"摘要: {summary['summary']}\n"
            f"轮次: {summary.get('message_count', '?')} 轮\n"
            f"更新时间: {summary.get('updated_at', '?')}"
        )

    # Full recall: load saved messages
    messages = load_messages(_db_path, session_id)
    if not messages:
        return f"未找到会话详情: {session_id}"

    lines = []
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content", "")
        # Truncate very long content for display
        if len(content) > 500:
            content = content[:500] + "..."
        lines.append(f"[{role}] {content}")
    return "\n".join(lines)


@tool
def summarize_session(session_id: str = "") -> str:
    """为指定会话生成或刷新摘要。

    如果 session_id 为空则生成当前会话的摘要。
    摘要包含：标题（一句话主题）、内容概括（角色+话题+结论）。
    """
    if not _db_path:
        return "记忆系统未初始化。"

    if not session_id:
        return "请提供要总结的会话编号，如：summarize_session(\"abc123\")"

    # Load messages for the session
    messages = load_messages(_db_path, session_id)
    if not messages:
        return f"未找到会话 {session_id} 的消息记录。"

    # Build text for LLM summarization
    text = "\n".join(
        f"{m['role']}: {m['content']}" for m in messages
    )

    title, summary = generate_session_summary(text, _model)
    save_summary(_db_path, session_id, title, summary, len(messages))

    return f"摘要已生成: {title}"
