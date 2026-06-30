"""Compression strategies: trim_by_count, trim_by_tokens, summarize_by_llm."""

from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from clawagent.compression.counters import estimate_tokens

_SUMMARY_PREFIX = "[对话历史摘要]"


def _safe_extract_text(content: object) -> str:
    """Extract readable text from content that may be str, list[dict], or other."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(content) if content is not None else ""


def trim_by_count(
    messages: list[BaseMessage],
    max_messages: int = 40,
) -> list[BaseMessage]:
    """Level 1: trim by message count.

    Args:
        messages: Full message list.
        max_messages: Maximum messages to keep (including system messages).

    Returns:
        Trimmed message list.
    """
    if len(messages) <= max_messages:
        return messages

    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    non_system = [m for m in messages if not isinstance(m, SystemMessage)]

    keep_count = max_messages - len(system_msgs)
    if keep_count < 2:
        keep_count = 2

    return system_msgs + non_system[-keep_count:]


def trim_by_tokens(
    messages: list[BaseMessage],
    max_tokens: int = 80_000,
) -> list[BaseMessage]:
    """Level 2: trim by estimated token count.

    Deletes messages from the oldest until token count is within limit.
    """
    if estimate_tokens(messages) <= max_tokens:
        return messages

    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    non_system = [m for m in messages if not isinstance(m, SystemMessage)]

    kept = list(non_system)
    while estimate_tokens(system_msgs + kept) > max_tokens and len(kept) > 2:
        kept.pop(0)

    return system_msgs + kept


def summarize_by_llm(
    messages: list[BaseMessage],
    model: Any,
    max_messages: int = 40,
    keep_recent: int = 6,
) -> list[BaseMessage]:
    """Level 3: summarize overflow messages with LLM, keep recent ones intact.

    Args:
        messages: Full message list.
        model: LLM model instance (e.g. ChatAnthropic) for summary generation.
        max_messages: Threshold to trigger summarization.
        keep_recent: Number of recent non-system messages to keep as raw dialogue.

    Returns:
        Messages with a summary SystemMessage replacing overflow messages.
    """
    if len(messages) <= max_messages:
        return messages

    # Separate system messages, filtering out old summaries
    system_msgs = [
        m for m in messages
        if isinstance(m, SystemMessage) and not _is_summary(m)
    ]
    non_system = [m for m in messages if not isinstance(m, SystemMessage)]

    if len(non_system) <= keep_recent:
        return messages

    to_summarize = non_system[:-keep_recent]
    to_keep = non_system[-keep_recent:]

    # Extract old summary text to incorporate into the new one
    old_summary = _extract_old_summary(messages)

    summary_prompt = (
        "Summarize the following conversation in under 200 words. "
        "Include: user identity, discussion topics, decisions made. "
        "Respond in the same language as the conversation.\n\n"
        "以下是一段对话记录，请用 200 字以内概括关键信息："
        "用户身份、讨论话题、已做出的决策。请使用对话所用的语言回复。\n\n"
    )
    if old_summary:
        summary_prompt += f"之前的对话摘要：{old_summary}\n\n后续对话：\n"
    summary_prompt += "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {_safe_extract_text(m.content)[:500]}"
        for m in to_summarize
    )

    response = model.invoke([HumanMessage(content=summary_prompt)])
    content = response.content if hasattr(response, "content") else str(response)
    summary_text = _safe_extract_text(content).strip()

    summary_msg = SystemMessage(content=f"{_SUMMARY_PREFIX} {summary_text}")
    return [*system_msgs, summary_msg, *to_keep]


def _is_summary(msg: SystemMessage) -> bool:
    """Check if a SystemMessage is a previously generated summary."""
    content = msg.content
    if isinstance(content, list):
        content = " ".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    content = content or ""
    return isinstance(content, str) and content.startswith(_SUMMARY_PREFIX)


def _extract_old_summary(messages: list[BaseMessage]) -> str:
    """Extract text from an existing summary message, if present."""
    for m in messages:
        if isinstance(m, SystemMessage) and _is_summary(m):
            content = m.content or ""
            if not isinstance(content, str):
                continue
            return content.removeprefix(_SUMMARY_PREFIX).strip()
    return ""
