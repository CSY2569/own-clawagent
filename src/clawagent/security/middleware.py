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
    tool_name = tool.name

    def _check(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | None:
        """Run permission check; returns denial message or None if allowed."""
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
        return None

    # Path 1: StructuredTool with func attribute (most common, via @tool decorator)
    original_func: Any = getattr(tool, "func", None)
    if original_func is not None:
        @functools.wraps(original_func)
        def checked_func(*args: Any, **kwargs: Any) -> Any:
            denial = _check(args, kwargs)
            if denial is not None:
                return denial
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

    # Path 2: BaseTool subclass without func - wrap _run method
    original_run: Any = getattr(tool, "_run", None)
    if original_run is None:
        raise ValueError(
            f"Tool {tool_name!r} has neither 'func' nor '_run'; cannot wrap with permission"
        )

    @functools.wraps(original_run)
    def checked_run(*args: Any, **kwargs: Any) -> Any:
        denial = _check(args, kwargs)
        if denial is not None:
            return denial
        return original_run(*args, **kwargs)

    new_tool = copy.copy(tool)
    new_tool._run = checked_run  # type: ignore[method-assign]
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
