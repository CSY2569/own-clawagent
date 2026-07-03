"""CoderWorker — code writing and debugging specialist."""

from typing import ClassVar

from clawagent.worker.base import BaseWorker
from clawagent.worker.registry import register_worker


@register_worker("coder")
class CoderWorker(BaseWorker):
    """File I/O, command execution, debugging. Uses deepseek-v4-flash by default."""

    _TOOLS: ClassVar[list[str]] = ["read_file", "write_file", "run_command"]
