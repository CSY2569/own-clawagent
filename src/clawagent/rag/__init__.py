"""RAG module — embedding, chunking, vector store, and document ingestion."""

from clawagent.rag.bm25 import BM25Retriever
from clawagent.rag.chunker import chunk_text
from clawagent.rag.embedding import SiliconFlowEmbedding
from clawagent.rag.hybrid import HybridSearcher
from clawagent.rag.store import RAGStore

__all__ = [
    "BM25Retriever",
    "HybridSearcher",
    "RAGStore",
    "SiliconFlowEmbedding",
    "chunk_text",
]
