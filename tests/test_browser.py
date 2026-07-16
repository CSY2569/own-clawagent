"""Tests for clawagent.tools.browser."""

# mypy: disable-error-code="no-untyped-def"

from unittest.mock import MagicMock, patch

import pytest

from clawagent.tools import browser
from clawagent.tools.browser import (
    browser_extract_standalone,
    browser_navigate_standalone,
    close_browser,
    create_browser_tools,
)


@pytest.fixture(autouse=True)
def _reset_browser_worker():
    """Ensure the module-level _worker is None before and after each test."""
    browser._worker = None
    yield
    if browser._worker is not None:
        browser._worker.shutdown()
        browser._worker = None


class TestSSRFProtection:
    """SSRF validation is enforced in all browser tools."""

    def test_navigate_standalone_blocks_localhost(self):
        result = browser_navigate_standalone.invoke({"url": "http://127.0.0.1:8080"})
        assert "SSRF" in result

    def test_navigate_standalone_blocks_internal(self):
        result = browser_navigate_standalone.invoke({"url": "http://192.168.1.1"})
        assert "SSRF" in result

    def test_navigate_standalone_blocks_file_scheme(self):
        result = browser_navigate_standalone.invoke({"url": "file:///etc/passwd"})
        assert "SSRF" in result

    def test_extract_standalone_blocks_localhost(self):
        result = browser_extract_standalone.invoke({"url": "http://localhost/admin"})
        assert "SSRF" in result


class TestCreateBrowserTools:
    """Shared-session browser tools factory."""

    def test_returns_five_tools_and_cleanup(self):
        tools, cleanup = create_browser_tools()
        assert len(tools) == 5
        names = {t.name for t in tools}
        assert names == {
            "browser_navigate",
            "browser_click",
            "browser_extract",
            "browser_fill",
            "browser_screenshot",
        }
        assert callable(cleanup)
        # Should not have spun up a worker yet (no operations performed).
        assert browser._worker is None
        cleanup()

    def test_navigate_blocks_ssrf(self):
        tools, cleanup = create_browser_tools()
        navigate = tools[0]
        with patch.object(browser, "_get_worker") as mock_get:
            result = navigate.invoke({"url": "http://10.0.0.1"})
        assert "SSRF" in result
        mock_get.assert_not_called()
        cleanup()

    def test_navigate_calls_worker_execute(self):
        tools, cleanup = create_browser_tools()
        navigate = tools[0]
        mock_worker = MagicMock()
        mock_worker.execute.return_value = "Title: X\nURL: https://example.com\n\nbody"
        with patch.object(browser, "_get_worker", return_value=mock_worker):
            result = navigate.invoke({"url": "https://example.com"})
        mock_worker.execute.assert_called_once_with("navigate", url="https://example.com")
        assert result == "Title: X\nURL: https://example.com\n\nbody"
        cleanup()

    def test_click_calls_worker_execute(self):
        tools, cleanup = create_browser_tools()
        click = tools[1]
        mock_worker = MagicMock()
        mock_worker.execute.return_value = "Clicked: button#ok"
        with patch.object(browser, "_get_worker", return_value=mock_worker):
            result = click.invoke({"selector": "button#ok"})
        mock_worker.execute.assert_called_once_with("click", selector="button#ok")
        assert result == "Clicked: button#ok"
        cleanup()

    def test_extract_with_selector_calls_worker_execute(self):
        tools, cleanup = create_browser_tools()
        extract = tools[2]
        mock_worker = MagicMock()
        mock_worker.execute.return_value = "Extracted text"
        with patch.object(browser, "_get_worker", return_value=mock_worker):
            result = extract.invoke({"selector": "div.content"})
        mock_worker.execute.assert_called_once_with("extract", selector="div.content")
        assert result == "Extracted text"
        cleanup()

    def test_extract_without_selector_calls_worker_execute(self):
        tools, cleanup = create_browser_tools()
        extract = tools[2]
        mock_worker = MagicMock()
        mock_worker.execute.return_value = "Full page text"
        with patch.object(browser, "_get_worker", return_value=mock_worker):
            result = extract.invoke({})
        mock_worker.execute.assert_called_once_with("extract", selector="")
        assert result == "Full page text"
        cleanup()

    def test_fill_calls_worker_execute(self):
        tools, cleanup = create_browser_tools()
        fill = tools[3]
        mock_worker = MagicMock()
        mock_worker.execute.return_value = "Filled input#name with 4 chars"
        with patch.object(browser, "_get_worker", return_value=mock_worker):
            result = fill.invoke({"selector": "input#name", "value": "test"})
        mock_worker.execute.assert_called_once_with(
            "fill", selector="input#name", value="test"
        )
        assert result == "Filled input#name with 4 chars"
        cleanup()

    def test_screenshot_calls_worker_execute(self):
        tools, cleanup = create_browser_tools()
        screenshot = tools[4]
        mock_worker = MagicMock()
        mock_worker.execute.return_value = "Screenshot saved: output/x.png"
        with patch.object(browser, "_get_worker", return_value=mock_worker):
            result = screenshot.invoke({})
        mock_worker.execute.assert_called_once_with("screenshot")
        assert result == "Screenshot saved: output/x.png"
        cleanup()


class TestCloseBrowser:
    """Module-level close_browser function."""

    def test_close_browser_no_session(self):
        # No worker exists; close_browser must be a safe no-op.
        assert browser._worker is None
        close_browser()
        assert browser._worker is None

    def test_close_browser_shuts_down_worker(self):
        mock_worker = MagicMock()
        browser._worker = mock_worker
        close_browser()
        mock_worker.shutdown.assert_called_once()
        assert browser._worker is None
