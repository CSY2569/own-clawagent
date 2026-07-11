"""Tests for clawagent.security.middleware."""

# mypy: disable-error-code="no-untyped-def"

from unittest.mock import MagicMock

from langchain_core.tools import tool

from clawagent.security.audit import AuditLogger
from clawagent.security.middleware import (
    _merge_args,
    wrap_tool_with_permission,
    wrap_tools,
)
from clawagent.security.permissions import PermissionConfig, PermissionLevel


def _make_read_tool(func=None):
    @tool
    def read_file(path: str) -> str:
        """Read a file."""
        if func:
            return func(path)
        return "ok"

    return read_file


def _make_write_tool(func=None):
    @tool
    def write_file(path: str, content: str) -> str:
        """Write a file."""
        if func:
            return func(path, content)
        return "ok"

    return write_file


class TestMergeArgs:
    def test_kwargs_only(self):
        assert _merge_args((), {"path": "x"}) == {"path": "x"}

    def test_positional_only(self):
        assert _merge_args(("a", "b"), {}) == {"arg1": "a", "arg2": "b"}

    def test_mixed(self):
        result = _merge_args(("a",), {"path": "x"})
        assert result == {"arg1": "a", "path": "x"}

    def test_empty(self):
        assert _merge_args((), {}) == {}


class TestAllowPath:
    def test_allow_calls_original(self):
        mock_func = MagicMock(return_value="result")
        tool_obj = _make_read_tool(mock_func)
        config = PermissionConfig()
        audit = AuditLogger(log_path="/tmp/test_audit_allow.jsonl")
        wrapped = wrap_tool_with_permission(tool_obj, config, audit)
        result = wrapped.invoke({"path": "README.md"})
        assert result == "result"
        mock_func.assert_called_once()

    def test_allow_with_auto_confirm(self):
        mock_func = MagicMock(return_value="done")
        tool_obj = _make_write_tool(mock_func)
        config = PermissionConfig()
        audit = AuditLogger(log_path="/tmp/test_audit_auto.jsonl")
        wrapped = wrap_tool_with_permission(tool_obj, config, audit, auto_confirm=True)
        result = wrapped.invoke({"path": "output/x.txt", "content": "hi"})
        assert result == "done"


class TestDenyPath:
    def test_deny_returns_message(self):
        mock_func = MagicMock(return_value="should not reach")
        tool_obj = _make_write_tool(mock_func)
        config = PermissionConfig()
        audit = AuditLogger(log_path="/tmp/test_audit_deny.jsonl")
        wrapped = wrap_tool_with_permission(tool_obj, config, audit)
        result = wrapped.invoke({"path": ".env", "content": "secret"})
        assert "操作被拒绝" in result
        mock_func.assert_not_called()


class TestConfirmPath:
    def test_confirm_with_confirm_fn_approved(self):
        mock_func = MagicMock(return_value="written")
        tool_obj = _make_write_tool(mock_func)
        config = PermissionConfig()
        audit = AuditLogger(log_path="/tmp/test_audit_confirm_ok.jsonl")
        confirm_fn = MagicMock(return_value=True)
        wrapped = wrap_tool_with_permission(tool_obj, config, audit, confirm_fn=confirm_fn)
        result = wrapped.invoke({"path": "src/main.py", "content": "code"})
        assert result == "written"
        confirm_fn.assert_called_once()

    def test_confirm_with_confirm_fn_denied(self):
        mock_func = MagicMock(return_value="should not reach")
        tool_obj = _make_write_tool(mock_func)
        config = PermissionConfig()
        audit = AuditLogger(log_path="/tmp/test_audit_confirm_no.jsonl")
        confirm_fn = MagicMock(return_value=False)
        wrapped = wrap_tool_with_permission(tool_obj, config, audit, confirm_fn=confirm_fn)
        result = wrapped.invoke({"path": "src/main.py", "content": "code"})
        assert "用户拒绝操作" in result
        mock_func.assert_not_called()

    def test_confirm_without_confirm_fn(self):
        mock_func = MagicMock(return_value="should not reach")
        tool_obj = _make_write_tool(mock_func)
        config = PermissionConfig()
        audit = AuditLogger(log_path="/tmp/test_audit_no_confirm.jsonl")
        wrapped = wrap_tool_with_permission(tool_obj, config, audit, confirm_fn=None)
        result = wrapped.invoke({"path": "src/main.py", "content": "code"})
        assert "需确认" in result
        mock_func.assert_not_called()

    def test_confirm_auto_confirm_skips_confirm_fn(self):
        mock_func = MagicMock(return_value="done")
        tool_obj = _make_write_tool(mock_func)
        config = PermissionConfig()
        audit = AuditLogger(log_path="/tmp/test_audit_auto_confirm.jsonl")
        confirm_fn = MagicMock(return_value=False)
        wrapped = wrap_tool_with_permission(
            tool_obj, config, audit, auto_confirm=True, confirm_fn=confirm_fn
        )
        result = wrapped.invoke({"path": "src/main.py", "content": "code"})
        assert result == "done"
        confirm_fn.assert_not_called()


class TestCustomRules:
    def test_custom_deny_rule(self):
        mock_func = MagicMock(return_value="should not reach")
        tool_obj = _make_read_tool(mock_func)
        config = PermissionConfig()
        config.add_rule("read_file", "secret/*", PermissionLevel.DENY, "no secrets")
        audit = AuditLogger(log_path="/tmp/test_audit_custom.jsonl")
        wrapped = wrap_tool_with_permission(tool_obj, config, audit)
        result = wrapped.invoke({"path": "secret/key.pem"})
        assert "操作被拒绝" in result
        mock_func.assert_not_called()


class TestWrapTools:
    def test_batch_wrap(self):
        func1 = MagicMock(return_value="r1")
        func2 = MagicMock(return_value="r2")
        tool1 = _make_read_tool(func1)
        tool2 = _make_write_tool(func2)
        config = PermissionConfig()
        audit = AuditLogger(log_path="/tmp/test_audit_batch.jsonl")
        wrapped = wrap_tools([tool1, tool2], config, audit, auto_confirm=True)
        assert len(wrapped) == 2
        assert wrapped[0].invoke({"path": "x"}) == "r1"
        assert wrapped[1].invoke({"path": "output/y", "content": "c"}) == "r2"
