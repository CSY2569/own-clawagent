"""Multi-agent worker system — independent sub-agents for task delegation."""

from clawagent.worker.config import BUILTIN_WORKER_ROLES
from clawagent.worker.factory import WorkerFactory

__all__ = ["BUILTIN_WORKER_ROLES", "WorkerFactory"]
