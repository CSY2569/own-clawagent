"""ResearcherWorker — information retrieval and research specialist.

Uses the qwen model via SiliconFlow chat API by default.
Configure WORKER_RESEARCHER_API_KEY / WORKER_COMMON_API_KEY for provider access.
"""

from typing import ClassVar

from clawagent.worker.base import BaseWorker
from clawagent.worker.registry import register_worker


@register_worker("researcher")
class ResearcherWorker(BaseWorker):
    """Search local knowledge base AND the internet, collect and summarize information.

    Uses search_documents for RAG retrieval + web_search for Bing search
    with automatic full-text extraction of top results.
    """

    _TOOLS: ClassVar[list[str]] = [
        "search_documents", "web_search",
        "browser_navigate_standalone", "browser_extract_standalone",
    ]
