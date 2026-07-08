"""Web search tool using Bing + trafilatura full-text extraction."""

import ipaddress
import socket
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.parse import urlparse

import httpx
import trafilatura
from bs4 import BeautifulSoup, Tag
from langchain_core.tools import tool

_MAX_SEARCH_RESULTS = 5
_MAX_DEEP_PAGES = 3
_MAX_PAGE_CHARS = 2000
_PAGE_TIMEOUT = 10
_SEARCH_TIMEOUT = 10

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

_SearchResult = dict[str, str]

# Allowed URL schemes for fetching page content
_ALLOWED_SCHEMES = frozenset({"http", "https"})

# Private / reserved IP ranges blocked for SSRF prevention
_PRIVATE_NETS: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = [
    ipaddress.IPv4Network("10.0.0.0/8"),
    ipaddress.IPv4Network("172.16.0.0/12"),
    ipaddress.IPv4Network("192.168.0.0/16"),
    ipaddress.IPv4Network("127.0.0.0/8"),
    ipaddress.IPv4Network("169.254.0.0/16"),
    ipaddress.IPv4Network("0.0.0.0/8"),
    ipaddress.IPv4Network("224.0.0.0/4"),
    ipaddress.IPv6Network("::1/128"),
    ipaddress.IPv6Network("fc00::/7"),
    ipaddress.IPv6Network("fe80::/10"),
]


_DNS_TIMEOUT = 2.0


def _resolve_hostname(hostname: str) -> list[str]:
    """Resolve hostname to IP strings, with timeout to avoid blocking."""
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(socket.getaddrinfo, hostname, None)
            infos = future.result(timeout=_DNS_TIMEOUT)
        return [str(info[4][0]) for info in infos]
    except (socket.gaierror, TimeoutError, OSError):
        return []


def _is_safe_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return not any(ip in net for net in _PRIVATE_NETS)


def _is_safe_url(url: str) -> bool:
    """Check whether a URL points to a public internet address.

    Blocks private / loopback / link-local / multicast IPs. For domain
    hostnames, resolves DNS and rejects if any resolved IP is private
    (prevents DNS-rebinding SSRF).
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return False

    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        return False

    hostname = parsed.hostname
    if not hostname:
        return False

    try:
        ip = ipaddress.ip_address(hostname)
        return _is_safe_ip(ip)
    except ValueError:
        pass

    ips = _resolve_hostname(hostname)
    if not ips:
        return False
    for ip_str in ips:
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if not _is_safe_ip(ip):
            return False
    return True


def _do_search(query: str) -> tuple[list[_SearchResult], str | None]:
    """Execute Bing search. Returns (results, error) — error is None on success."""
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
                    f'[Search Error] Request timed out for query: "{query}". '
                    "Suggestion: try shorter or more specific keywords."
                )
        except httpx.HTTPStatusError as e:
            return [], (
                f"[Search Error] Bing returned HTTP {e.response.status_code} "
                f'for query: "{query}"'
            )
        except Exception as e:
            if attempt == 2:
                err_msg = str(e)
                if "ConnectError" in err_msg or "Network is unreachable" in err_msg:
                    return [], (
                        f'[Search Error] Network unreachable for query: "{query}". '
                        "Check proxy settings (http_proxy/https_proxy)."
                    )
                return [], (
                    f"[Search Error] {type(e).__name__}: {e} "
                    f'for query: "{query}"'
                )

    soup = BeautifulSoup(resp.text, "lxml")
    raw = soup.select("li.b_algo")[:_MAX_SEARCH_RESULTS]

    if not raw:
        return [], (
            f'[No Results] No matches found for "{query}". '
            "Try broader or different keywords."
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
    h2 = tag.find("h2")
    if isinstance(h2, Tag):
        return h2.get_text(strip=True)
    a = tag.find("a")
    if isinstance(a, Tag):
        return a.get_text(strip=True)
    return "No Title"


def _get_link(tag: Any) -> str:
    a = tag.find("a")
    if isinstance(a, Tag):
        href = a.get("href")
        if isinstance(href, str):
            return href
    return ""


def _get_text(tag: Any, selector: str) -> str:
    child = tag.find(selector)
    if isinstance(child, Tag):
        return child.get_text(strip=True)
    return ""


def _format_summary(results: list[_SearchResult]) -> str:
    lines = [f"## Search Results ({len(results)} found)"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. **{r['title']}**")
        if r["link"]:
            lines.append(f"   URL: {r['link']}")
        if r["snippet"]:
            lines.append(f"   Snippet: {r['snippet']}")
        if r["source"]:
            lines.append(f"   Source: {r['source']}")
    return "\n".join(lines)


def _fetch_and_extract(results: list[_SearchResult]) -> str:
    """Parallel fetch and extract full text from top-3 pages (safe URLs only)."""
    targets: list[tuple[int, str]] = [
        (i, r["link"])
        for i, r in enumerate(results[:_MAX_DEEP_PAGES])
        if r["link"] and _is_safe_url(r["link"])
    ]
    if not targets:
        return ""

    extracted: dict[int, str] = {}

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
        return (
            "\n\n(Full-text extraction: no pages could be extracted. "
            "Use the search snippets above instead.)"
        )

    lines = [f"\n## Full-Text Extraction (top {_MAX_DEEP_PAGES} pages)"]
    for i in range(min(len(results), _MAX_DEEP_PAGES)):
        r = results[i]
        content = extracted.get(i)
        lines.append(f"\n### {i + 1}. {r['title']}")
        lines.append(f"Source: {r['link']}")
        if content:
            if len(content) > _MAX_PAGE_CHARS:
                content = content[:_MAX_PAGE_CHARS] + (
                    f"\n\n... (truncated, original length: {len(content)} chars)"
                )
            lines.append(content)
        else:
            lines.append("(Could not extract text — see search snippet above)")

    return "\n".join(lines)


# Rate limiting — prevent IP bans from excessive Bing requests
_last_search_time: float = 0.0
_SEARCH_COOLDOWN: float = 2.0
_MAX_CONSECUTIVE_ERRORS: int = 5
_consecutive_errors: int = 0


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
    global _last_search_time, _consecutive_errors

    if _consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
        return (
            "[Search Error] Too many consecutive failures — "
            "web search temporarily disabled. Please wait before retrying."
        )

    elapsed = time.time() - _last_search_time
    if elapsed < _SEARCH_COOLDOWN:
        time.sleep(_SEARCH_COOLDOWN - elapsed)

    results, error = _do_search(query)
    _last_search_time = time.time()

    if error:
        _consecutive_errors += 1
        return error

    _consecutive_errors = 0
    return _format_summary(results) + _fetch_and_extract(results)
