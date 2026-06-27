"""WriterWorker — documentation and content creation specialist."""

from typing import Any

from clawagent.worker.base import BaseWorker
from clawagent.worker.registry import register_worker


@register_worker("writer")
class WriterWorker(BaseWorker):
    """Write documentation, reports, and generated content."""

    def _get_tools(self) -> list[Any]:
        from clawagent.tools import get_current_time, read_file, write_file

        return [read_file, write_file, get_current_time]
