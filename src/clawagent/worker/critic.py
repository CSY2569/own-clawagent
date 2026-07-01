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

    def _customize_prompt(self, prompt: str, task: str) -> str:
        """Append review output format requirements."""
        base = super()._customize_prompt(prompt, task)
        return (
            base
            + "\n\n请按以下格式输出审查结果：\n"
            "## 问题列表\n"
            "| 严重程度 | 位置 | 问题描述 | 建议修复 |\n"
            "|----------|------|----------|----------|\n"
            "## 总体评价\n"
            "[摘要性评语]"
        )
