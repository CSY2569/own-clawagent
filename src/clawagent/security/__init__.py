"""Security module - tool-level permission control and audit logging."""

from clawagent.security.audit import AuditLogger
from clawagent.security.middleware import wrap_tool_with_permission, wrap_tools
from clawagent.security.permissions import (
    PermissionConfig,
    PermissionLevel,
    PermissionRule,
)

__all__ = [
    "AuditLogger",
    "PermissionConfig",
    "PermissionLevel",
    "PermissionRule",
    "wrap_tool_with_permission",
    "wrap_tools",
]
