"""ResearcherWorker — information retrieval and research specialist.

Uses the qwen model via SiliconFlow chat API by default.
Configure WORKER_RESEARCHER_API_KEY / WORKER_COMMON_API_KEY for provider access.
"""

from typing import Any

from langchain_core.tools import tool

from clawagent.worker.base import BaseWorker
from clawagent.worker.registry import register_worker


@register_worker("researcher")
class ResearcherWorker(BaseWorker):
    """Search local knowledge base AND the internet, collect and summarize information.

    Uses search_documents for RAG retrieval + web_search for internet access.
    """

    def _get_tools(self) -> list[Any]:
        from clawagent.tools import search_documents

        # ── DuckDuckGo 网页搜索 -- backend="bing", 避免 Google 超时 ──
        try:
            from langchain_community.tools import DuckDuckGoSearchResults
            from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

            wrapper = DuckDuckGoSearchAPIWrapper(
                max_results=3,
                time="y",
                safesearch="moderate",
                backend="bing",
                source="text",
            )
            ddg_tool: Any = DuckDuckGoSearchResults(  # type: ignore[call-arg]
                api_wrapper=wrapper,
                output_format="list",
                max_results=3,
            )
        except ImportError:
            ddg_tool = None

        @tool
        def web_search(query: str) -> str:
            """搜索互联网获取最新信息。返回结构化结果（标题+链接+摘要）。

            使用规范:
            - 严格使用用户提到的专有名词原文搜索，不要擅自替换同义词
            - 搜不到时如实告知，不要编造答案
            - 搜索结果包含来源链接，请在回答中标注来源

            Args:
                query: 搜索关键词（中文直接用原句，英文关键词用空格分隔）。
            """
            if ddg_tool is None:
                return (
                    "【错误】DuckDuckGo 搜索模块未安装。\n"
                    "请在项目目录执行: uv sync"
                )
            try:
                result = ddg_tool.invoke(query)

                if not result or len(result) == 0:
                    return f'【未找到】关键词 "{query}" 在网上没有匹配结果。\n建议：尝试更宽泛的关键词重试。'

                lines = [f"找到 {len(result)} 条网页结果："]
                for i, r in enumerate(result, 1):
                    title = r.get("title", "无标题")
                    link = r.get("link", "")
                    snippet = r.get("snippet", "")
                    lines.append(f"{i}. **{title}**")
                    lines.append(f"   链接: {link}")
                    lines.append(f"   摘要: {snippet}")
                return "\n".join(lines)

            except Exception as e:
                err_name = type(e).__name__
                if "Timeout" in err_name:
                    return (
                        f'【超时】搜索 "{query}" 请求超时。\n'
                        f"建议：尝试更短的关键词，或稍后重试。"
                    )
                if "Rate" in err_name:
                    return "【限流】搜索频率过高，请等待 10 秒后重试。"
                return f"【搜索失败】{err_name}: {e}"

        return [search_documents, web_search]
