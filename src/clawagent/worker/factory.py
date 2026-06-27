"""WorkerFactory — creates Worker instances by role on demand."""

from __future__ import annotations

from typing import TYPE_CHECKING

from clawagent.worker.config import load_worker_configs
from clawagent.worker.registry import get_worker_class

if TYPE_CHECKING:
    from clawagent.worker.base import BaseWorker


class WorkerFactory:
    """Factory that creates worker instances on demand.

    Responsibilities:
    1. Load all worker configs at startup (once)
    2. Look up the worker class by role
    3. Instantiate the worker with its config
    """

    def __init__(self) -> None:
        self._configs = load_worker_configs()

    @property
    def available_roles(self) -> list[str]:
        """Return list of configured worker role names."""
        return sorted(self._configs.keys())

    def create(self, role: str) -> BaseWorker:
        """Create a worker instance by role name.

        Args:
            role: Role identifier (e.g. "coder", "researcher")

        Returns:
            Configured but not-yet-spawned worker instance

        Raises:
            ValueError: if the role is not configured or registered
        """
        config = self._configs.get(role)
        if config is None:
            available = self.available_roles
            raise ValueError(
                f"Unconfigured worker role: {role}. Configured: {available}"
            )

        cls = get_worker_class(role)
        return cls(config)
