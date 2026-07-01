"""ResearcherWorker — information retrieval and research specialist.

Uses the qwen model via SiliconFlow chat API by default.
Configure WORKER_RESEARCHER_API_KEY / WORKER_COMMON_API_KEY for provider access.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import httpx
import trafilatura
from bs4 import BeautifulSoup, Tag
from langchain_core.tools import tool

from clawagent.worker.base import BaseWorker
from clawagent.worker.registry import register_worker

# ── 常量 ──────────────────────────────────────────────

_MAX_SEARCH_RESULTS = 5         # Bing 搜索返回条数
_MAX_DEEP_PAGES = 3             # 抓取全文的条数(前 N 条)
_MAX_PAGE_CHARS = 2000          # 每页正文截断长度
_PAGE_TIMEOUT = 10              # 单页抓取超时(秒)
_SEARCH_TIMEOUT = 10            # 搜索请求超时(秒)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

# ── 类型别名 ─────────────────────────────────────────
_SearchResult = dict[str, str]  # keys: title, link, snippet, source


@register_worker("researcher")
class ResearcherWorker(BaseWorker):
    """Search local knowledge base AND the internet, collect and summarize information.

    Uses search_documents for RAG retrieval + web_search for Bing search
    with automatic full-text extraction of top results.
    """

    def _get_tools(self) -> list[Any]:
        from clawagent.tools import search_documents

        @tool
        def web_search(query: str) -> str:
            """搜索互联网并自动提取前 3 条结果的全文内容。

            返回两部分：
            - 搜索结果摘要：5 条结果的标题、链接和摘要片段
            - 全文提取：前 3 条结果的正文内容（自动去广告/导航/侧栏）

            使用规范：
            - 使用用户提到的专有名词原文搜索，不替换同义词
            - 在回答中标注来源链接
            - 搜不到时如实告知

            Args:
                query: 搜索关键词（中文直接用原句）。
            """
            # ═══════════════════════════════════════════
            # 阶段 1: Bing 搜索
            # ═══════════════════════════════════════════
            results, error = _do_search(query)
            if error:
                return error

            # ═══════════════════════════════════════════
            # 阶段 2 + 3: 格式化摘要 + 并行抓取正文
            # ═══════════════════════════════════════════
            return _format_summary(results) + _fetch_and_extract(results)

        # ── 辅助函数 ────────────────────────────────

        def _do_search(query: str) -> tuple[list[_SearchResult], str | None]:
            """执行 Bing 搜索。返回 (results, error)，error 为 None 表示成功。"""
            for attempt in range(3):
                if attempt > 0:
                    time.sleep(2)
                try:
                    resp = httpx.get(
                        "https://www.bing.com/search",
                        params={"q": query, "count": 10},
                        headers=_HEADERS,
                        timeout=_SEARCH_TIMEOUT,
                        follow_redirects=True,
                    )
                    resp.raise_for_status()
                    break
                except httpx.TimeoutException:
                    if attempt == 2:
                        return [], (
                            f'【超时】搜索 "{query}" 请求超时。\n'
                            "建议：换更短的关键词重试。"
                        )
                except httpx.HTTPStatusError as e:
                    return [], f"【搜索失败】Bing 返回 HTTP {e.response.status_code}"
                except Exception as e:
                    if attempt == 2:
                        err_msg = str(e)
                        if "ConnectError" in err_msg or "Network is unreachable" in err_msg:
                            return [], (
                                f'【网络不可达】搜索 "{query}" 失败。\n'
                                "请检查代理是否运行 (http_proxy/https_proxy)。"
                            )
                        return [], f"【搜索失败】{type(e).__name__}: {e}"

            soup = BeautifulSoup(resp.text, "lxml")
            raw = soup.select("li.b_algo")[:_MAX_SEARCH_RESULTS]

            if not raw:
                return [], (
                    f'【未找到】关键词 "{query}" 在网上没有匹配结果。\n'
                    "建议：尝试更宽泛的关键词重试。"
                )

            results = [
                {
                    "title": _get_title(r),
                    "link": _get_link(r),
                    "snippet": _get_text(r, "p"),
                    "source": _get_text(r, "cite"),
                }
                for r in raw
            ]
            return results, None


        def _get_title(tag: Any) -> str:
            """Extract title text from a Bing result tag."""
            h2 = tag.find("h2")
            if isinstance(h2, Tag):
                return h2.get_text(strip=True)
            a = tag.find("a")
            if isinstance(a, Tag):
                return a.get_text(strip=True)
            return "无标题"


        def _get_link(tag: Any) -> str:
            """Extract href from the anchor in a Bing result tag."""
            a = tag.find("a")
            if isinstance(a, Tag):
                href = a.get("href")
                if isinstance(href, str):
                    return href
            return ""


        def _get_text(tag: Any, selector: str) -> str:
            """Extract text from a child tag by selector."""
            child = tag.find(selector)
            if isinstance(child, Tag):
                return child.get_text(strip=True)
            return ""

        def _format_summary(results: list[_SearchResult]) -> str:
            """格式化搜索结果摘要。"""
            lines = [f"## 搜索结果（共 {len(results)} 条）"]
            for i, r in enumerate(results, 1):
                lines.append(f"{i}. **{r['title']}**")
                if r["link"]:
                    lines.append(f"   链接: {r['link']}")
                if r["snippet"]:
                    lines.append(f"   摘要: {r['snippet']}")
                if r["source"]:
                    lines.append(f"   来源: {r['source']}")
            return "\n".join(lines)

        def _fetch_and_extract(results: list[_SearchResult]) -> str:
            """并行抓取 top-3 页面正文。"""
            # 保留原始索引以防止 link 为空时错位
            targets: list[tuple[int, str]] = [
                (i, r["link"])
                for i, r in enumerate(results[:_MAX_DEEP_PAGES])
                if r["link"]
            ]
            if not targets:
                return ""

            extracted: dict[int, str] = {}  # original_index → text

            def _fetch_one(url: str) -> str | None:
                try:
                    resp = httpx.get(
                        url, headers=_HEADERS,
                        timeout=_PAGE_TIMEOUT, follow_redirects=True,
                    )
                    resp.raise_for_status()
                    text = trafilatura.extract(
                        resp.text,
                        include_links=False,
                        include_images=False,
                        include_formatting=False,
                        favor_precision=True,
                    )
                    if text and len(text.strip()) > 50:
                        return text
                    return None
                except Exception:
                    return None

            with ThreadPoolExecutor(max_workers=_MAX_DEEP_PAGES) as executor:
                future_map = {
                    executor.submit(_fetch_one, url): orig_idx
                    for orig_idx, url in targets
                }
                for future in as_completed(future_map):
                    orig_idx = future_map[future]
                    try:
                        content = future.result()
                        if content:
                            extracted[orig_idx] = content
                    except Exception:
                        pass

            if not extracted:
                return "\n\n（全文提取：所有页面均无法提取正文，请基于上述搜索摘要作答）"

            lines = [f"\n## 前 {_MAX_DEEP_PAGES} 条全文提取"]
            for i in range(min(len(results), _MAX_DEEP_PAGES)):
                r = results[i]
                content = extracted.get(i)
                lines.append(f"\n### {i + 1}. {r['title']}")
                lines.append(f"来源: {r['link']}")
                if content:
                    if len(content) > _MAX_PAGE_CHARS:
                        content = content[:_MAX_PAGE_CHARS] + (
                            f"\n\n...（已截断，原文共 {len(content)} 字符）"
                        )
                    lines.append(content)
                else:
                    lines.append("（无法提取正文，请参考上方搜索摘要中的片段）")

            return "\n".join(lines)

        return [search_documents, web_search]
