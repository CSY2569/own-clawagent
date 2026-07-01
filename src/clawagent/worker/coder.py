"""CoderWorker — code writing and debugging specialist."""

from typing import Any

from clawagent.worker.base import BaseWorker
from clawagent.worker.registry import register_worker


@register_worker("coder")
class CoderWorker(BaseWorker):
    """File I/O, command execution, debugging. Uses deepseek-v4-flash by default."""

    def _get_tools(self) -> list[Any]:
        from clawagent.tools import read_file, run_command, write_file

        return [read_file, write_file, run_command]
