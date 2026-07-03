"""Multi-agent worker system — independent sub-agents for task delegation."""

# Trigger worker class registration via @register_worker decorators
import clawagent.worker.coder
import clawagent.worker.critic
import clawagent.worker.researcher
import clawagent.worker.writer  # noqa: F401
from clawagent.worker.config import BUILTIN_WORKER_ROLES
from clawagent.worker.factory import WorkerFactory

__all__ = ["BUILTIN_WORKER_ROLES", "WorkerFactory"]
