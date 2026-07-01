"""ResearcherWorker — information retrieval and research specialist.

Uses the qwen model via SiliconFlow chat API by default.
Configure WORKER_RESEARCHER_API_KEY / WORKER_COMMON_API_KEY for provider access.
"""

from typing import Any

from clawagent.worker.base import BaseWorker
from clawagent.worker.registry import register_worker


@register_worker("researcher")
class ResearcherWorker(BaseWorker):
    """Search local knowledge base, collect and summarize information.

    Uses search_documents for RAG retrieval.
    Reserved: web_search tool for internet access.
    """

    def _get_tools(self) -> list[Any]:
        from clawagent.tools import search_documents

        return [search_documents]
