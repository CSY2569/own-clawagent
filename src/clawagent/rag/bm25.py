"""BM25 lexical retriever with jieba tokenization."""

from math import log

import jieba  # type: ignore[import-untyped]


class BM25Retriever:
    """BM25 lexical retriever for Chinese text.

    Uses jieba for tokenization and the standard BM25 ranking function.
    Index is built in-memory from the corpus provided at construction time.

    BM25 formula:
        score(D, Q) = sum over terms t in Q:
            IDF(t) * tf(t,D) * (k1+1) / (tf(t,D) + k1*(1-b+b*|D|/avgdl))

    Args:
        corpus: Full list of document texts.
        k1: Term frequency saturation parameter (default 1.5).
        b: Length normalization parameter (default 0.75).
    """

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self._corpus = corpus
        self._k1 = k1
        self._b = b
        self._N = 0
        self._avgdl = 0.0
        self._df: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._doc_len: list[int] = []
        self._tf: list[dict[str, int]] = []
        self._build_index()

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text with jieba, dropping whitespace and pure-punctuation tokens."""
        tokens = jieba.lcut(text)
        return [
            t for t in tokens
            if not t.isspace() and not all(c in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~，。、；：？！""''（）【】《》" for c in t)
        ]

    def _build_index(self) -> None:
        """Build document frequency and term frequency indexes from corpus."""
        self._N = len(self._corpus)
        if self._N == 0:
            return
        for doc_text in self._corpus:
            tokens = self._tokenize(doc_text)
            self._doc_len.append(len(tokens))
            tf: dict[str, int] = {}
            seen: set[str] = set()
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
                if t not in seen:
                    self._df[t] = self._df.get(t, 0) + 1
                    seen.add(t)
            self._tf.append(tf)
        self._avgdl = sum(self._doc_len) / self._N
        self._idf = {
            term: log((self._N - df + 0.5) / (df + 0.5) + 1)
            for term, df in self._df.items()
        }

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Search corpus and return top_k (corpus_index, bm25_score) tuples.

        Args:
            query: Search query string.
            top_k: Maximum number of results to return.

        Returns:
            List of (corpus_index, bm25_score) sorted by score descending.
        """
        if self._N == 0 or not query.strip():
            return []
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        scores: list[tuple[int, float]] = []
        for idx in range(self._N):
            score = 0.0
            doc_tf = self._tf[idx]
            doc_len = self._doc_len[idx]
            for t in query_tokens:
                tf = doc_tf.get(t, 0)
                if tf == 0:
                    continue
                idf = self._idf.get(t, 0.0)
                numerator = tf * (self._k1 + 1)
                denominator = tf + self._k1 * (
                    1 - self._b + self._b * doc_len / self._avgdl
                )
                score += idf * numerator / denominator
            if score > 0:
                scores.append((idx, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
