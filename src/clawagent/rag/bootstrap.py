"""RAG system bootstrap — embedding, vector store, BM25, hybrid search setup."""

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from clawagent.config import PROJECT_ROOT


@dataclass
class RAGContext:
    """Initialized RAG components."""
    bm25_ready_signal: list[bool]


def bootstrap_rag(
    settings: Any,
    configure_hybrid_search_fn: Callable[..., None],
) -> RAGContext | None:
    """Initialize RAG system if SILICONFLOW_API_KEY is configured.

    Returns RAGContext with a bm25_ready_signal list that gets appended
    when BM25 index building completes. Returns None if RAG is not configured.
    """
    if not settings.siliconflow_api_key:
        return None

    from clawagent.rag import BM25Retriever, HybridSearcher, RAGStore, SiliconFlowEmbedding

    embedding = SiliconFlowEmbedding(
        api_key=settings.siliconflow_api_key,
        model=settings.siliconflow_model,
        dimensions=settings.siliconflow_dimensions,
        base_url=settings.siliconflow_base_url,
    )
    rag_store = RAGStore(db_path=str(PROJECT_ROOT / "chroma_db"), embedding=embedding)

    all_docs = rag_store.get_all_documents()
    corpus = [d["text"] for d in all_docs]
    cache_dir = str(PROJECT_ROOT / "chroma_db")

    bm25 = BM25Retriever()

    def _knn_retrieve(query: str, k: int) -> list[dict[str, str]]:
        return rag_store.retrieve(query, top_k=k)

    hybrid = HybridSearcher(
        knn_retriever=_knn_retrieve,
        bm25_retriever=bm25,
        all_docs=all_docs,
    )
    configure_hybrid_search_fn(hybrid)

    bm25_ready_signal: list[bool]

    if bm25.try_load_cache(cache_dir, corpus):
        bm25_ready_signal = [True]
    else:
        bm25_ready_signal = []

        def _build_bm25() -> None:
            bm25.build_async(corpus, cache_dir=cache_dir)
            bm25_ready_signal.append(True)

        threading.Thread(target=_build_bm25, daemon=True).start()

    return RAGContext(bm25_ready_signal=bm25_ready_signal)
