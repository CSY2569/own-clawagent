"""Permission levels, rules, and matching engine for tool-level access control."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PermissionLevel(Enum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"


@dataclass(frozen=True)
class PermissionRule:
    """A single permission rule: tool + pattern -> level."""

    tool_name: str
    pattern: str
    level: PermissionLevel
    description: str = ""


def _extract_pattern(tool_name: str, args: dict[str, Any]) -> str:
    """Extract the pattern-matching key from tool arguments."""
    if tool_name == "write_file":
        return str(args.get("path", args.get("arg1", "")))
    if tool_name == "read_file":
        return str(args.get("path", args.get("arg1", "")))
    if tool_name == "run_command":
        cmd = str(args.get("command", args.get("arg1", "")))
        return cmd.split()[0] if cmd else ""
    return "*"


_DEFAULT_RULES: list[PermissionRule] = [
    PermissionRule("write_file", "*.env*", PermissionLevel.DENY, "禁止写环境配置文件"),
    PermissionRule("write_file", "src/*.py", PermissionLevel.CONFIRM, "改源码需确认"),
    PermissionRule("write_file", "output/*", PermissionLevel.ALLOW, "输出目录自动允许"),
    PermissionRule("write_file", "*", PermissionLevel.CONFIRM, "其他写操作需确认"),
    PermissionRule("run_command", "rm*", PermissionLevel.DENY, "删除命令禁止"),
    PermissionRule("run_command", "rmdir*", PermissionLevel.DENY, "删除命令禁止"),
    PermissionRule("run_command", "git status*", PermissionLevel.ALLOW, "只读 git 操作"),
    PermissionRule("run_command", "git diff*", PermissionLevel.ALLOW, "只读 git 操作"),
    PermissionRule("run_command", "git log*", PermissionLevel.ALLOW, "只读 git 操作"),
    PermissionRule("run_command", "git show*", PermissionLevel.ALLOW, "只读 git 操作"),
    PermissionRule("run_command", "git push*", PermissionLevel.CONFIRM, "写 git 操作需确认"),
    PermissionRule("run_command", "git commit*", PermissionLevel.CONFIRM, "写 git 操作需确认"),
    PermissionRule("run_command", "git reset*", PermissionLevel.CONFIRM, "写 git 操作需确认"),
    PermissionRule("run_command", "git checkout*", PermissionLevel.CONFIRM, "写 git 操作需确认"),
    PermissionRule("run_command", "git merge*", PermissionLevel.CONFIRM, "写 git 操作需确认"),
    PermissionRule("run_command", "git rebase*", PermissionLevel.CONFIRM, "写 git 操作需确认"),
    PermissionRule("run_command", "python*", PermissionLevel.CONFIRM, "Python 可执行任意代码"),
    PermissionRule("run_command", "python3*", PermissionLevel.CONFIRM, "Python 可执行任意代码"),
    PermissionRule("run_command", "docker*", PermissionLevel.CONFIRM, "Docker 需确认"),
    PermissionRule("run_command", "npm*", PermissionLevel.CONFIRM, "npm 需确认"),
    PermissionRule("run_command", "*", PermissionLevel.ALLOW, "白名单内其他命令自动允许"),
    PermissionRule("read_file", "*", PermissionLevel.ALLOW, "读文件自动允许"),
    PermissionRule("web_search", "*", PermissionLevel.ALLOW, "联网搜索自动允许"),
    PermissionRule("search_documents", "*", PermissionLevel.ALLOW, "RAG 检索自动允许"),
    PermissionRule("delegate_task", "*", PermissionLevel.ALLOW, "Worker 委托自动允许"),
    PermissionRule("list_sessions", "*", PermissionLevel.ALLOW, "会话列表自动允许"),
    PermissionRule("recall_session", "*", PermissionLevel.ALLOW, "会话回顾自动允许"),
    PermissionRule("summarize_session", "*", PermissionLevel.ALLOW, "会话摘要自动允许"),
]

_WORKER_RESTRICTIONS: dict[str, list[PermissionRule]] = {
    "researcher": [
        PermissionRule("write_file", "*", PermissionLevel.DENY, "Researcher 不可写文件"),
        PermissionRule("run_command", "*", PermissionLevel.DENY, "Researcher 不可执行命令"),
    ],
    "critic": [
        PermissionRule("write_file", "*", PermissionLevel.DENY, "Critic 不可写文件"),
        PermissionRule("run_command", "*", PermissionLevel.DENY, "Critic 不可执行命令"),
    ],
    "writer": [
        PermissionRule("run_command", "*", PermissionLevel.DENY, "Writer 不可执行命令"),
    ],
}


@dataclass
class PermissionConfig:
    """Permission configuration with rules and default level."""

    rules: list[PermissionRule] = field(default_factory=lambda: list(_DEFAULT_RULES))
    default_level: PermissionLevel = PermissionLevel.ALLOW

    def match(self, tool_name: str, args: dict[str, Any]) -> PermissionRule:
        """Find the first matching rule for a tool invocation."""
        pattern = _extract_pattern(tool_name, args)
        for rule in self.rules:
            if rule.tool_name != tool_name and rule.tool_name != "*":
                continue
            if rule.pattern == "*" or fnmatch.fnmatch(pattern, rule.pattern):
                return rule
        return PermissionRule("*", "*", self.default_level, "default")

    def add_rule(self, tool_name: str, pattern: str, level: PermissionLevel, description: str = "") -> None:
        """Add a rule to the front of the list (highest priority)."""
        self.rules.insert(0, PermissionRule(tool_name, pattern, level, description))

    def reset(self) -> None:
        """Reset to default rules."""
        self.rules = list(_DEFAULT_RULES)

    @classmethod
    def for_worker(cls, role: str) -> PermissionConfig:
        """Create a permission config for a worker, with role-specific restrictions."""
        config = cls()
        for rule in _WORKER_RESTRICTIONS.get(role, []):
            config.rules.insert(0, rule)
        return config
