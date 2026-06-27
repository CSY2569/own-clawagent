"""RAG retrieval tool — LLM calls search_documents when it needs document context."""

import sys
from typing import Any

from langchain_core.tools import tool

# Module-level state, set by main.py on startup
_hybrid_searcher: Any = None


def configure_hybrid_search(searcher: Any) -> None:
    """Set the module-level hybrid searcher for use by search_documents."""
    global _hybrid_searcher
    _hybrid_searcher = searcher


def search_rag(query: str, top_k: int = 5) -> list[dict[str, str]]:
    """Search the RAG store via hybrid search and return raw hits (for CLI use)."""
    if _hybrid_searcher is None:
        return []
    return _hybrid_searcher.search(query, top_k=top_k)  # type: ignore[no-any-return]


@tool
def search_documents(query: str, top_k: int = 5) -> str:
    """搜索已入库的文档内容。当用户询问文档中的人物、情节、事件、数据或具体信息时使用。

    使用规范:
    - 严格使用用户提到的专有名词原文搜索，不要擅自替换同义词
    - 搜不到时如实告知用户，不要用自己的知识编造答案
    - 低分结果要注明置信度

    Args:
        query: 搜索关键词或问题描述（中文直接用原句，英文关键词用空格分隔）。
        top_k: 返回条数（1-10，默认5）。
    """
    if _hybrid_searcher is None:
        return "RAG 未初始化。请在 .env 中配置 SILICONFLOW_API_KEY。"

    print(f"\033[2mRAG: 检索 \"{query}\" (top_k={top_k})...\033[0m", file=sys.stderr)

    hits = _hybrid_searcher.search(query, top_k=top_k)
    if not hits:
        return (
            f"【未找到】关键词 \"{query}\" 在知识库中没有匹配结果。\n"
            f"建议：请确认专有名词是否准确，或尝试其他关键词重新搜索。"
        )

    top_score = float(hits[0].get("score", 0))
    if top_score < 0.5:
        note = f"\n注意：以下结果相关度较低（最高分 {top_score:.2f}），可能不准确。"
    else:
        note = ""

    lines: list[str] = [f"找到 {len(hits)} 条结果：{note}"]
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
