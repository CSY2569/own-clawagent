"""Worker role registry — maps role names to their worker class."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from clawagent.worker.base import BaseWorker

WORKER_REGISTRY: dict[str, type[BaseWorker]] = {}


def register_worker(role: str) -> Callable[[type[BaseWorker]], type[BaseWorker]]:
    """Decorator: register a worker class in WORKER_REGISTRY.

    Usage:
        @register_worker("coder")
        class CoderWorker(BaseWorker):
            ...
    """

    def decorator(cls: type[BaseWorker]) -> type[BaseWorker]:
        WORKER_REGISTRY[role] = cls
        return cls

    return decorator


def get_worker_class(role: str) -> type[BaseWorker]:
    """Look up a worker class by role name.

    Raises ValueError if the role is not registered.
    """
    cls = WORKER_REGISTRY.get(role)
    if cls is None:
        available = list(WORKER_REGISTRY.keys())
        raise ValueError(
            f"Unknown worker role: {role}. Available: {available}"
        )
    return cls
