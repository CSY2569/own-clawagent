"""RAG retrieval tool — LLM calls search_documents when it needs document context."""

import sys
from typing import Any

from langchain_core.tools import tool

# Module-level state, set by main.py on startup
_rag_store: Any = None


def configure_rag(store: Any) -> None:
    """Set the module-level RAG store for use by search_documents."""
    global _rag_store
    _rag_store = store


def search_rag(query: str, top_k: int = 5) -> list[dict[str, str]]:
    """Search the RAG store and return raw hits (for CLI use)."""
    if _rag_store is None:
        return []
    return _rag_store.retrieve(query, top_k=top_k)  # type: ignore[no-any-return]


@tool
def search_documents(query: str) -> str:
    """搜索已入库的文档内容。当用户询问文档中的人物、情节、事件、数据或具体信息时使用。

    Args:
        query: 搜索关键词或问题描述（中文直接用原句，英文关键词用空格分隔）。
    """
    if _rag_store is None:
        return "RAG 未初始化。请在 .env 中配置 SILICONFLOW_API_KEY。"

    print(f"\033[2mRAG: 检索 \"{query}\"...\033[0m", file=sys.stderr)

    hits = _rag_store.retrieve(query, top_k=5)
    if not hits:
        return "未在文档中找到相关内容。"

    lines: list[str] = []
    for i, h in enumerate(hits, 1):
        source = h.get("source", h.get("id", "?"))
        score = h.get("score", "0")
        text = h.get("text", "")
        chapter = h.get("chapter", "")
        label = f"## 段落 {i} (来源: {source}, 相关度: {score}"
        if chapter:
            label += f", 章节: {chapter}"
        label += ")"
        lines.append(f"{label}\n{text}")
    return "\n\n".join(lines)
