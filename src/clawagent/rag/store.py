"""Chroma-backed vector store for document retrieval."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.types import EmbeddingFunction

from clawagent.rag.embedding import SiliconFlowEmbedding


class _ChromaEmbeddingAdapter(EmbeddingFunction[Any]):
    """Adapt SiliconFlowEmbedding to Chroma's EmbeddingFunction interface."""

    def __init__(self, embedding: SiliconFlowEmbedding) -> None:
        self._embedding = embedding

    def __call__(self, input: list[str]) -> list[list[float]]:  # type: ignore[override]
        return self._embedding.embed_documents(input)


class RAGStore:
    """Chroma vector store with SiliconFlow cloud embeddings.

    Args:
        db_path: Path to the Chroma persistence directory.
        embedding: SiliconFlowEmbedding instance.
        collection_name: Chroma collection name.
    """

    def __init__(
        self,
        db_path: str | Path,
        embedding: SiliconFlowEmbedding,
        collection_name: str = "clawagent_docs",
    ) -> None:
        db_path = Path(db_path)
        db_path.mkdir(parents=True, exist_ok=True)
        self._embedding = embedding
        self._client = chromadb.PersistentClient(path=str(db_path))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=_ChromaEmbeddingAdapter(embedding),
        )

    def add_documents(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        add_batch_size: int = 2000,
    ) -> None:
        """Add documents to the vector store in batches.

        Large document sets are split into batches to avoid hitting Chroma's
        per-add limit and to preserve partial progress on failure.

        Args:
            texts: List of document chunks.
            metadatas: Optional metadata for each chunk.
            ids: Optional unique IDs (auto-generated if omitted).
            add_batch_size: Number of documents per Chroma add() call.
        """
        if not texts:
            return
        total = len(texts)
        if ids is None:
            total_count = self.count()
            ids = [f"doc_{total_count + j}" for j in range(total)]
        for batch_start in range(0, total, add_batch_size):
            batch_end = min(batch_start + add_batch_size, total)
            batch_texts = texts[batch_start:batch_end]
            batch_ids = ids[batch_start:batch_end]
            kwargs: dict[str, Any] = {"documents": batch_texts, "ids": batch_ids}
            if metadatas is not None:
                kwargs["metadatas"] = metadatas[batch_start:batch_end]
            self._collection.add(**kwargs)
            batch_num = batch_start // add_batch_size + 1
            total_batches = (total - 1) // add_batch_size + 1
            print(f"  Stored batch {batch_num}/{total_batches} ({len(batch_texts)} docs)")

    def retrieve(
        self, query: str, top_k: int = 3
    ) -> list[dict[str, str]]:
        """Search for documents relevant to the query.

        Returns:
            List of dicts with keys: id, text, score.
        """
        query_vec: Sequence[float] = self._embedding.embed_query(query)
        results = self._collection.query(
            query_embeddings=[query_vec], n_results=top_k
        )
        hits: list[dict[str, str]] = []
        ids_list: list[str] = (results.get("ids") or [[]])[0]
        docs_list: list[str] = (results.get("documents") or [[]])[0]
        distances: Sequence[float] = (results.get("distances") or [[]])[0]
        metas_list: Sequence[Mapping[str, Any]] = (results.get("metadatas") or [[]])[0]
        for i in range(len(ids_list)):
            score = 1.0 / (1.0 + distances[i]) if distances and i < len(distances) else 0.0
            hit: dict[str, str] = {
                "id": ids_list[i],
                "text": docs_list[i] if docs_list else "",
                "score": f"{score:.4f}",
            }
            if metas_list and i < len(metas_list):
                meta = metas_list[i]
                if meta.get("chapter"):
                    hit["chapter"] = str(meta["chapter"])
                if meta.get("source"):
                    hit["source"] = str(meta["source"])
            hits.append(hit)
        return hits

    def get_all_documents(self) -> list[dict[str, str]]:
        """Return all documents with metadata for external indexing.

        Returns:
            List of dicts with keys: id, text, source, chapter.
        """
        results = self._collection.get()
        docs: list[dict[str, str]] = []
        ids_list: list[str] = results.get("ids") or []
        docs_list: list[str] = results.get("documents") or []
        metas_list: list[Any] = results.get("metadatas") or []
        for i in range(len(ids_list)):
            doc: dict[str, str] = {
                "id": ids_list[i],
                "text": docs_list[i] if i < len(docs_list) else "",
            }
            if i < len(metas_list):
                meta = metas_list[i]
                if meta:
                    if meta.get("source"):
                        doc["source"] = str(meta["source"])
                    if meta.get("chapter"):
                        doc["chapter"] = str(meta["chapter"])
            docs.append(doc)
        return docs

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self._collection.count()
