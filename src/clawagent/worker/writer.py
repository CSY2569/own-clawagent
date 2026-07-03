"""WriterWorker — documentation and content creation specialist."""

from typing import ClassVar

from clawagent.worker.base import BaseWorker
from clawagent.worker.registry import register_worker


@register_worker("writer")
class WriterWorker(BaseWorker):
    """Write documentation, reports, and generated content."""

    _TOOLS: ClassVar[list[str]] = ["read_file", "write_file"]
