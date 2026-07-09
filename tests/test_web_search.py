"""Tests for web_search HTML parsing, trafilatura extraction, and error handling."""

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import trafilatura
from bs4 import BeautifulSoup, Tag

SAMPLE_BING_HTML = """
<ol id="b_results">
<li class="b_algo">
  <h2><a href="https://example.com/1">Title 1</a></h2>
  <cite>example.com</cite>
  <p>Snippet 1 with date 2025年6月</p>
</li>
<li class="b_algo">
  <h2><a href="https://example.com/2">Title 2</a></h2>
  <cite>example.org</cite>
  <p>Snippet 2</p>
</li>
</ol>
"""

SAMPLE_PAGE_HTML = "<html><body><article><p>Full article text here.</p></article></body></html>"


def _get_text(tag: Any, selector: str) -> str:
    child = tag.find(selector)
    if isinstance(child, Tag):
        return child.get_text(strip=True)
    return ""


def _get_href(tag: Any) -> str:
    a = tag.find("a")
    if isinstance(a, Tag):
        href = a.get("href")
        if isinstance(href, str):
            return href
    return ""


class TestBingParsing:
    """Bing HTML parsing -- component-level tests."""

    def test_extracts_title_link_snippet(self) -> None:
        soup = BeautifulSoup(SAMPLE_BING_HTML, "lxml")
        results = soup.select("li.b_algo")
        assert len(results) == 2
        assert _get_text(results[0], "h2") == "Title 1"
        assert _get_href(results[0]) == "https://example.com/1"
        assert "2025年6月" in _get_text(results[0], "p")

    def test_empty_page_returns_empty(self) -> None:
        soup = BeautifulSoup("<html></html>", "lxml")
        assert len(soup.select("li.b_algo")) == 0


class TestTrafilatura:
    """trafilatura text extraction -- component-level tests."""

    def test_extracts_article_text(self) -> None:
        text = trafilatura.extract(SAMPLE_PAGE_HTML)
        assert text is not None
        assert "Full article text" in text

    def test_empty_html_returns_none(self) -> None:
        assert trafilatura.extract("<html></html>") is None


class TestWebSearchTool:
    """web_search tool integration -- requires mocked HTTP."""

    def test_tool_defined(self) -> None:
        from clawagent.worker.config import WorkerConfig
        from clawagent.worker.researcher import ResearcherWorker

        worker = ResearcherWorker(WorkerConfig(role="researcher"))
        tools = worker._get_tools()
        names = {t.name for t in tools}
        assert "web_search" in names
        assert "search_documents" in names

    @patch("httpx.get")
    def test_timeout_returns_friendly_message(self, mock_get: MagicMock) -> None:
        from clawagent.worker.config import WorkerConfig
        from clawagent.worker.researcher import ResearcherWorker

        mock_get.side_effect = httpx.TimeoutException("timed out")
        worker = ResearcherWorker(WorkerConfig(role="researcher"))
        tools = {t.name: t for t in worker._get_tools()}
        result = tools["web_search"].invoke("test query")
        assert "[Search Error]" in result

    @patch("httpx.get")
    def test_http_error_returns_status_code(self, mock_get: MagicMock) -> None:
        from clawagent.worker.config import WorkerConfig
        from clawagent.worker.researcher import ResearcherWorker

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=MagicMock(status_code=429)
        )
        mock_get.return_value = mock_resp
        worker = ResearcherWorker(WorkerConfig(role="researcher"))
        tools = {t.name: t for t in worker._get_tools()}
        result = tools["web_search"].invoke("test query")
        assert "429" in result


class TestSSRFProtection:
    """_is_safe_url — blocks private IPs and DNS-rebinding attempts."""

    def test_blocks_literal_private_ip(self) -> None:
        from clawagent.tools.web_search import _is_safe_url

        assert not _is_safe_url("http://127.0.0.1/secret")
        assert not _is_safe_url("http://10.0.0.1/secret")
        assert not _is_safe_url("http://192.168.1.1/secret")

    def test_blocks_non_http_scheme(self) -> None:
        from clawagent.tools.web_search import _is_safe_url

        assert not _is_safe_url("file:///etc/passwd")
        assert not _is_safe_url("ftp://example.com/x")

    @patch("clawagent.tools.web_search._resolve_hostname")
    def test_blocks_domain_resolving_to_private(self, mock_resolve: MagicMock) -> None:
        from clawagent.tools.web_search import _is_safe_url

        mock_resolve.return_value = ["127.0.0.1"]
        assert not _is_safe_url("http://evil.example.com/secret")

    @patch("clawagent.tools.web_search._resolve_hostname")
    def test_blocks_domain_with_mixed_ips_any_private(self, mock_resolve: MagicMock) -> None:
        from clawagent.tools.web_search import _is_safe_url

        mock_resolve.return_value = ["8.8.8.8", "10.0.0.1"]
        assert not _is_safe_url("http://mixed.example.com/x")

    @patch("clawagent.tools.web_search._resolve_hostname")
    def test_allows_domain_resolving_to_public(self, mock_resolve: MagicMock) -> None:
        from clawagent.tools.web_search import _is_safe_url

        mock_resolve.return_value = ["93.184.216.34"]
        assert _is_safe_url("https://example.com/page")

    @patch("clawagent.tools.web_search._resolve_hostname")
    def test_blocks_unresolvable_domain(self, mock_resolve: MagicMock) -> None:
        from clawagent.tools.web_search import _is_safe_url

        mock_resolve.return_value = []
        assert not _is_safe_url("http://nonexistent.invalid/x")
