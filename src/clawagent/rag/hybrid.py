"""Hybrid search: KNN vector retrieval + BM25 lexical retrieval with RRF fusion."""

import hashlib
from collections.abc import Callable

from clawagent.rag.bm25 import BM25Retriever


class HybridSearcher:
    """KNN + BM25 hybrid retriever with Reciprocal Rank Fusion.

    Runs both retrievers in parallel, then merges results via RRF.
    Text deduplication uses SHA256 of the full text (robust against collision and prefix-mismatch bugs).

    Args:
        knn_retriever: Callable(query, top_k) -> list[dict] backed by RAGStore.
        bm25_retriever: BM25Retriever instance with pre-built index.
        all_docs: Full document metadata list (for mapping BM25 corpus index
            back to document metadata like source/chapter).
    """

    def __init__(
        self,
        knn_retriever: Callable[..., list[dict[str, str]]],
        bm25_retriever: BM25Retriever,
        all_docs: list[dict[str, str]] | None = None,
    ) -> None:
        self._knn = knn_retriever
        self._bm25 = bm25_retriever
        self._all_docs = all_docs or []

    def search(self, query: str, top_k: int = 5) -> list[dict[str, str]]:
        """Hybrid search with RRF fusion.

        Args:
            query: Search query string.
            top_k: Number of final results (clamped to 1-10).

        Returns:
            List of hit dicts with keys: id, text, score, source, chapter.
        """
        if top_k < 1:
            top_k = 1
        if top_k > 10:
            top_k = 10
        fetch_k = top_k * 2

        try:
            knn_hits = self._knn(query, fetch_k)
        except Exception:
            knn_hits = []

        if not self._bm25.ready:
            return knn_hits[:top_k]

        bm25_hits = self._bm25.retrieve(query, top_k=fetch_k)
        return self._rrf_fusion(knn_hits, bm25_hits, top_k)

    def _rrf_fusion(
        self,
        knn_hits: list[dict[str, str]],
        bm25_hits: list[tuple[int, float]],
        top_k: int,
        k: int = 60,
    ) -> list[dict[str, str]]:
        """Reciprocal Rank Fusion.

        score(d) = sum over retrievers r: 1 / (k + rank_r(d))

        Args:
            knn_hits: Results from KNN retriever.
            bm25_hits: (corpus_index, score) from BM25 retriever.
            top_k: Number of results to return after fusion.
            k: RRF constant to dampen rank dominance (default 60).

        Returns:
            Fused and re-ranked hit list.
        """
        rrf: dict[str, tuple[float, dict[str, str]]] = {}

        for rank, hit in enumerate(knn_hits, 1):
            key = self._hash(hit.get("text", ""))
            score = 1.0 / (k + rank)
            if key in rrf:
                old_score, old_hit = rrf[key]
                rrf[key] = (old_score + score, old_hit)
            else:
                rrf[key] = (score, hit)

        for rank, (corpus_idx, bm_score) in enumerate(bm25_hits, 1):
            if corpus_idx >= len(self._all_docs):
                continue
            doc = self._all_docs[corpus_idx]
            key = self._hash(doc.get("text", ""))
            score = 1.0 / (k + rank)
            if key in rrf:
                old_score, old_hit = rrf[key]
                rrf[key] = (old_score + score, old_hit)
            else:
                doc_hit: dict[str, str] = {
                    "id": doc.get("id", ""),
                    "text": doc.get("text", ""),
                    "score": f"{bm_score:.4f}",
                }
                if "source" in doc:
                    doc_hit["source"] = doc["source"]
                if "chapter" in doc:
                    doc_hit["chapter"] = doc["chapter"]
                rrf[key] = (score, doc_hit)

        sorted_hits = sorted(rrf.values(), key=lambda x: x[0], reverse=True)
        return [
            {**hit, "score": f"{rrf_score:.4f}"}
            for rrf_score, hit in sorted_hits[:top_k]
        ]

    @staticmethod
    def _hash(text: str) -> str:
        """Hash full text with SHA256 for dedup key."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
