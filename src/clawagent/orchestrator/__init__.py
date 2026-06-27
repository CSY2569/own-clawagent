"""Orchestrator layer — task delegation and (future) planning."""

from clawagent.orchestrator.delegator import configure_worker_factory, delegate_task

__all__ = ["configure_worker_factory", "delegate_task"]
