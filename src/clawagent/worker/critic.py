"""CriticWorker — code review and solution assessment specialist.

Uses the qwen model via SiliconFlow chat API by default.
"""

from typing import Any

from clawagent.worker.base import BaseWorker
from clawagent.worker.registry import register_worker


@register_worker("critic")
class CriticWorker(BaseWorker):
    """Code review, solution assessment, issue discovery. Read-only."""

    def _get_tools(self) -> list[Any]:
        from clawagent.tools import read_file, search_documents

        return [read_file, search_documents]
