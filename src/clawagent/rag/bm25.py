"""BM25 lexical retriever with jieba tokenization."""

import contextlib
import hashlib
import hmac
import json
import logging
from math import log
from pathlib import Path
from typing import Any

import jieba  # type: ignore[import-untyped]

jieba.setLogLevel(logging.WARNING)

_CACHE_FILE = "bm25_index.json"
_LEGACY_CACHE_FILE = "bm25_index.pkl"


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
        cache_secret: Optional HMAC secret to sign cache files. When set,
            tampered cache files are rejected. When empty, signing disabled.
    """

    def __init__(
        self,
        corpus: list[str] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
        cache_secret: str = "",
    ) -> None:
        self._corpus = corpus or []
        self._k1 = k1
        self._b = b
        self._cache_secret = cache_secret
        self._ready = False
        self._N = 0
        self._avgdl = 0.0
        self._df: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._doc_len: list[int] = []
        self._tf: list[dict[str, int]] = []
        if self._corpus:
            self._build_index()
            self._ready = True

    @property
    def ready(self) -> bool:
        """Whether the BM25 index has been built."""
        return self._ready

    def build_async(self, corpus: list[str], cache_dir: str = "") -> None:
        """Build BM25 index from corpus. Called from a background thread.

        If cache_dir is provided, the index is persisted to disk after building.
        """
        self._corpus = corpus
        self._build_index()
        self._ready = True
        if cache_dir:
            self.save_cache(cache_dir)

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
        self._doc_len.clear()
        self._df.clear()
        self._tf.clear()
        self._idf.clear()
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

    def _corpus_hash(self, corpus: list[str]) -> str:
        """SHA256 hash of corpus content for cache validation."""
        h = hashlib.sha256()
        for doc in corpus:
            h.update(doc.encode("utf-8"))
        return h.hexdigest()[:16]

    def _sign(self, payload: str) -> str:
        return hmac.new(
            self._cache_secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def try_load_cache(self, cache_dir: str, corpus: list[str]) -> bool:
        """Load BM25 index from disk cache. Returns True on success.

        Reads JSON cache (signed when a secret is configured). Legacy .pkl
        files are ignored — index will be rebuilt.
        """
        cache_path = Path(cache_dir) / _CACHE_FILE
        if not cache_path.exists():
            return self._try_load_legacy_pickle(cache_dir, corpus)
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            corpus_hash = data.get("hash", "")
            if corpus_hash != self._corpus_hash(corpus):
                return False

            if self._cache_secret:
                signature = data.get("signature", "")
                if not signature:
                    return False
                payload = json.dumps(
                    {k: v for k, v in data.items() if k != "signature"},
                    sort_keys=True,
                )
                if not hmac.compare_digest(self._sign(payload), signature):
                    return False

            self._N = data["N"]
            self._avgdl = data["avgdl"]
            self._df = data["df"]
            self._idf = data["idf"]
            self._doc_len = data["doc_len"]
            self._tf = data["tf"]
            self._corpus = corpus
            self._ready = True
            return True
        except (json.JSONDecodeError, KeyError, TypeError):
            return False

    def _try_load_legacy_pickle(self, cache_dir: str, corpus: list[str]) -> bool:
        legacy_path = Path(cache_dir) / _LEGACY_CACHE_FILE
        if legacy_path.exists():
            with contextlib.suppress(OSError):
                legacy_path.unlink()
        return False

    def save_cache(self, cache_dir: str) -> None:
        """Persist current BM25 index to disk as signed JSON."""
        cache_path = Path(cache_dir) / _CACHE_FILE
        data: dict[str, Any] = {
            "hash": self._corpus_hash(self._corpus),
            "N": self._N,
            "avgdl": self._avgdl,
            "df": self._df,
            "idf": self._idf,
            "doc_len": self._doc_len,
            "tf": self._tf,
        }
        if self._cache_secret:
            payload = json.dumps(data, sort_keys=True)
            data["signature"] = self._sign(payload)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

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
