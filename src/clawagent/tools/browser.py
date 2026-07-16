"""Browser tools - Playwright-based web interaction.

Uses a dedicated daemon thread for all Playwright operations to avoid
Playwright sync API cross-thread access violations. Tool functions send
commands via queue and wait for results.

All URL navigation goes through SSRF validation (reuses web_search._is_safe_url).
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, tool

from clawagent.tools.web_search import _is_safe_url

logger = logging.getLogger(__name__)

_MAX_CONTENT_CHARS = 8000
_CMD_TIMEOUT = 120  # seconds to wait for browser operation result


class _BrowserWorker:
    """Runs Playwright in a dedicated thread for thread safety."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._cmd_q: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()
        self._result_q: queue.Queue[Any] = queue.Queue()

    def _ensure_thread(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        """Browser thread main loop - owns the Playwright session."""
        from playwright.sync_api import sync_playwright

        pw = sync_playwright().start()
        browser = pw.chromium.launch(headless=False, channel="msedge")
        page = browser.new_page()
        page.set_default_timeout(30000)

        _closed_errors = (
            "target closed",
            "browser has been closed",
            "page has been closed",
            "target page has been closed",
            "connection closed",
        )

        try:
            while True:
                cmd, params = self._cmd_q.get()
                if cmd == "_shutdown":
                    break
                try:
                    result = self._execute(page, cmd, params)
                    self._result_q.put(("ok", result))
                except Exception as e:
                    error_msg = str(e).lower()
                    if not any(s in error_msg for s in _closed_errors):
                        self._result_q.put(("error", str(e)))
                        continue
                    logger.info("Browser closed externally, recreating session")
                    with contextlib.suppress(Exception):
                        page.close()
                    with contextlib.suppress(Exception):
                        browser.close()
                    try:
                        browser = pw.chromium.launch(headless=False, channel="msedge")
                        page = browser.new_page()
                        page.set_default_timeout(30000)
                        result = self._execute(page, cmd, params)
                        self._result_q.put(("ok", result))
                    except Exception as e2:
                        self._result_q.put(("error", f"Session recreation failed: {e2}"))
        finally:
            with contextlib.suppress(Exception):
                page.close()
            with contextlib.suppress(Exception):
                browser.close()
            with contextlib.suppress(Exception):
                pw.stop()

    def _execute(self, page: Any, cmd: str, params: dict[str, Any]) -> str:
        if cmd == "navigate":
            page.goto(params["url"], wait_until="domcontentloaded")
            with contextlib.suppress(Exception):
                page.wait_for_load_state("networkidle", timeout=10000)
            title = page.title()
            text = page.inner_text("body")[:_MAX_CONTENT_CHARS]
            return f"Title: {title}\nURL: {page.url}\n\n{text}"
        if cmd == "click":
            page.click(params["selector"], timeout=10000)
            import time

            time.sleep(1.5)
            with contextlib.suppress(Exception):
                page.wait_for_load_state("networkidle", timeout=5000)
            title = page.title()
            url = page.url
            snippet = ""
            with contextlib.suppress(Exception):
                snippet = page.inner_text("body")[:500]
            return f"Clicked: {params['selector']}\nCurrent page: {title}\nURL: {url}\n\n{snippet}"
        if cmd == "extract":
            if params.get("selector"):
                return str(page.inner_text(params["selector"])[:_MAX_CONTENT_CHARS])
            return str(page.inner_text("body")[:_MAX_CONTENT_CHARS])
        if cmd == "fill":
            page.fill(params["selector"], params["value"], timeout=10000)
            return f"Filled {params['selector']} with {len(params['value'])} chars"
        if cmd == "screenshot":
            out_dir = Path("output")
            out_dir.mkdir(parents=True, exist_ok=True)
            from datetime import UTC, datetime

            ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            path = out_dir / f"screenshot_{ts}.png"
            page.screenshot(path=str(path), full_page=True)
            return f"Screenshot saved: {path}"
        return f"Unknown command: {cmd}"

    def execute(self, cmd: str, **params: Any) -> str:
        """Send a command to the browser thread and wait for result."""
        self._ensure_thread()
        self._cmd_q.put((cmd, params))
        status, result = self._result_q.get(timeout=_CMD_TIMEOUT)
        if status == "error":
            return f"Browser {cmd} failed: {result}"
        return str(result)

    def shutdown(self) -> None:
        """Shut down the browser thread."""
        if self._thread is not None and self._thread.is_alive():
            self._cmd_q.put(("_shutdown", {}))
            self._thread.join(timeout=10)
        self._thread = None


_worker: _BrowserWorker | None = None


def _get_worker() -> _BrowserWorker:
    global _worker
    if _worker is None:
        _worker = _BrowserWorker()
    return _worker


def close_browser() -> None:
    """Shut down the browser worker thread."""
    global _worker
    if _worker is not None:
        _worker.shutdown()
        _worker = None


def create_browser_tools() -> tuple[list[BaseTool], Callable[[], None]]:
    """Create browser tools backed by a dedicated browser thread."""

    @tool
    def browser_navigate(url: str) -> str:
        """Navigate to a URL and return the page title and text summary.

        Args:
            url: Full URL (must start with http:// or https://)
        """
        if not _is_safe_url(url):
            return f"URL blocked by SSRF protection: {url}"
        return _get_worker().execute("navigate", url=url)

    @tool
    def browser_click(selector: str) -> str:
        """Click an element on the current page.

        Args:
            selector: CSS selector (e.g. "button#submit", "a[href='/about']")
        """
        return _get_worker().execute("click", selector=selector)

    @tool
    def browser_extract(selector: str = "") -> str:
        """Extract text content from the current page.

        Args:
            selector: CSS selector to extract from. Empty = full page text.
        """
        return _get_worker().execute("extract", selector=selector)

    @tool
    def browser_fill(selector: str, value: str) -> str:
        """Fill a text input on the current page.

        Args:
            selector: CSS selector targeting the input element.
            value: Text to enter into the field.
        """
        return _get_worker().execute("fill", selector=selector, value=value)

    @tool
    def browser_screenshot() -> str:
        """Take a screenshot of the current page and save to output/.

        Returns the saved file path.
        """
        return _get_worker().execute("screenshot")

    tools = [browser_navigate, browser_click, browser_extract, browser_fill, browser_screenshot]
    return tools, close_browser


@tool
def browser_navigate_standalone(url: str) -> str:
    """Navigate to a URL, extract page text, then close browser. Worker-safe.

    Args:
        url: Full URL (must start with http:// or https://)
    """
    if not _is_safe_url(url):
        return f"URL blocked by SSRF protection: {url}"
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, channel="msedge")
            page = browser.new_page()
            page.set_default_timeout(30000)
            page.goto(url, wait_until="domcontentloaded")
            title = page.title()
            text = page.inner_text("body")[:_MAX_CONTENT_CHARS]
            browser.close()
        return f"Title: {title}\nURL: {url}\n\n{text}"
    except Exception as e:
        return f"Navigation failed: {e}"


@tool
def browser_extract_standalone(url: str, selector: str = "") -> str:
    """Navigate to a URL, extract specific element text, then close browser.

    Args:
        url: Full URL to navigate to.
        selector: CSS selector for element to extract. Empty = full page.
    """
    if not _is_safe_url(url):
        return f"URL blocked by SSRF protection: {url}"
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, channel="msedge")
            page = browser.new_page()
            page.set_default_timeout(30000)
            page.goto(url, wait_until="domcontentloaded")
            if selector:
                content = page.inner_text(selector)[:_MAX_CONTENT_CHARS]
            else:
                content = page.inner_text("body")[:_MAX_CONTENT_CHARS]
            browser.close()
        return str(content)
    except Exception as e:
        return f"Extract failed: {e}"
