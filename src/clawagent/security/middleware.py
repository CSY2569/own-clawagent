"""Permission middleware - wraps LangChain tools with access control."""

from __future__ import annotations

import copy
import functools
from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from clawagent.security.audit import AuditLogger
from clawagent.security.permissions import PermissionConfig, PermissionLevel


def wrap_tool_with_permission(
    tool: BaseTool,
    perm_config: PermissionConfig,
    audit: AuditLogger,
    thread_id: str = "",
    auto_confirm: bool = False,
    confirm_fn: Callable[[str, dict[str, Any]], bool] | None = None,
) -> BaseTool:
    """Wrap a tool so every invocation goes through permission check + audit."""
    original_func: Any = getattr(tool, "func", None)
    if original_func is None:
        return copy.copy(tool)
    tool_name = tool.name

    @functools.wraps(original_func)
    def checked_func(*args: Any, **kwargs: Any) -> Any:
        merged_args = _merge_args(args, kwargs)
        rule = perm_config.match(tool_name, merged_args)

        if rule.level == PermissionLevel.DENY:
            audit.log(thread_id, tool_name, merged_args, rule.level.value, "denied")
            return f"操作被拒绝: {rule.description}"

        if rule.level == PermissionLevel.CONFIRM and not auto_confirm:
            if confirm_fn is None:
                audit.log(thread_id, tool_name, merged_args, rule.level.value, "denied")
                return f"此操作需确认但当前模式不支持交互: {rule.description}"
            if not confirm_fn(tool_name, merged_args):
                audit.log(thread_id, tool_name, merged_args, rule.level.value, "denied")
                return "用户拒绝操作"

        audit.log(thread_id, tool_name, merged_args, rule.level.value, "approved")
        return original_func(*args, **kwargs)

    if isinstance(tool, StructuredTool):
        return StructuredTool(
            name=tool.name,
            description=tool.description,
            func=checked_func,
            args_schema=tool.args_schema,
        )
    new_tool = copy.copy(tool)
    new_tool.func = checked_func
    return new_tool


def wrap_tools(
    tools: list[BaseTool],
    perm_config: PermissionConfig,
    audit: AuditLogger,
    thread_id: str = "",
    auto_confirm: bool = False,
    confirm_fn: Callable[[str, dict[str, Any]], bool] | None = None,
) -> list[BaseTool]:
    """Wrap a list of tools with permission checking."""
    return [
        wrap_tool_with_permission(t, perm_config, audit, thread_id, auto_confirm, confirm_fn)
        for t in tools
    ]


def _merge_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Merge positional and keyword args into a dict for pattern matching."""
    merged: dict[str, Any] = dict(kwargs)
    for i, val in enumerate(args):
        merged[f"arg{i + 1}"] = val
    return merged
