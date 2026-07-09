"""Tests for RAG core components: chunker, BM25, bootstrap."""

# mypy: disable-error-code="no-untyped-def"

from unittest.mock import MagicMock, patch

import pytest

from clawagent.rag.bm25 import BM25Retriever
from clawagent.rag.bootstrap import RAGContext, bootstrap_rag
from clawagent.rag.chunker import chunk_text


class TestChunker:
    def test_empty_text(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_small_text_fits_in_one_chunk(self):
        chunks = chunk_text("hello world", chunk_size=100)
        assert len(chunks) == 1
        assert chunks[0] == ("hello world", 0)

    def test_chinese_text_chunking(self):
        text = "这是一段中文测试文本。" * 50  # ~500 chars
        chunks = chunk_text(text, chunk_size=200, overlap=40)
        assert len(chunks) >= 2
        for chunk_text_val, start in chunks:
            assert len(chunk_text_val) <= 200
            assert start >= 0

    def test_overlap_between_chunks(self):
        text = "0123456789" * 10  # 100 chars
        chunks = chunk_text(text, chunk_size=30, overlap=10)
        assert len(chunks) >= 3
        # Adjacent chunks should overlap
        if len(chunks) >= 2:
            second_start = chunks[1][1]
            first_end = chunks[0][1] + 30
            assert second_start < first_end  # overlap exists

    def test_chunk_progress(self):
        """Chunks should advance through the text monotonically."""
        text = "x" * 500
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        starts = [c[1] for c in chunks]
        assert starts == sorted(starts)
        assert starts[-1] + 100 >= 500  # last chunk reaches near end

    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_size"):
            chunk_text("hello", chunk_size=0)

    def test_overlap_not_less_than_chunk_size(self):
        with pytest.raises(ValueError, match="overlap"):
            chunk_text("hello", chunk_size=10, overlap=10)

    def test_overlap_greater_than_chunk_size(self):
        with pytest.raises(ValueError, match="overlap"):
            chunk_text("hello", chunk_size=10, overlap=15)


class TestBM25:
    def test_empty_corpus(self):
        bm = BM25Retriever([])
        assert not bm.ready  # empty corpus → not ready
        assert bm.retrieve("test") == []

    def test_corpus_from_init(self):
        bm = BM25Retriever(["hello world", "foo bar"])
        assert bm.ready

    def test_build_async(self):
        bm = BM25Retriever()
        assert not bm.ready
        bm.build_async(["doc one", "doc two", "doc three"])
        assert bm.ready

    def test_chinese_tokenization(self):
        bm = BM25Retriever(["你好世界", "世界你好", "另一个文档"])
        assert bm.ready
        results = bm.retrieve("你好")
        assert len(results) > 0
        # Should match docs containing 你好
        indices = [r[0] for r in results]
        assert 0 in indices or 1 in indices  # first two docs

    def test_query_with_no_results(self):
        bm = BM25Retriever(["apple banana", "cherry date"])
        results = bm.retrieve("xyzxyz")
        assert results == []

    def test_empty_query(self):
        bm = BM25Retriever(["test"])
        assert bm.retrieve("") == []
        assert bm.retrieve("   ") == []

    def test_single_doc_always_matches(self):
        bm = BM25Retriever(["the quick brown fox"])
        results = bm.retrieve("quick")
        assert len(results) == 1
        assert results[0][0] == 0  # corpus index

    def test_top_k_limit(self):
        bm = BM25Retriever([f"doc{i}" for i in range(20)])
        results = bm.retrieve("doc", top_k=5)
        assert len(results) <= 5

    def test_build_async_on_fresh_instance(self):
        bm = BM25Retriever()
        assert not bm.ready
        bm.build_async(["苹果香蕉", "苹果橘子"])
        assert bm.ready
        results = bm.retrieve("苹果")
        assert len(results) == 2
        indices = {r[0] for r in results}
        assert indices == {0, 1}

    def test_punctuation_filtered(self):
        """Chinese punctuation should be filtered from tokens."""
        bm = BM25Retriever(["你好，世界！", "你好。世界"])
        results = bm.retrieve("你好世界")
        assert len(results) == 2  # both docs contain 你好 and 世界


    def test_try_load_cache_miss(self, tmp_path):
        """Cache file doesn't exist → returns False."""
        bm = BM25Retriever()
        result = bm.try_load_cache(str(tmp_path), ["doc one", "doc two"])
        assert result is False
        assert not bm.ready

    def test_save_and_load_roundtrip(self, tmp_path):
        """Build index → save → new instance loads → results match."""
        corpus = ["苹果香蕉橘子", "香蕉西瓜", "苹果葡萄"]
        cache_dir = str(tmp_path)

        bm1 = BM25Retriever(corpus)
        bm1.save_cache(cache_dir)

        bm2 = BM25Retriever()
        assert bm2.try_load_cache(cache_dir, corpus)
        assert bm2.ready

        results1 = bm1.retrieve("苹果")
        results2 = bm2.retrieve("苹果")
        assert results1 == results2

    def test_cache_hash_mismatch(self, tmp_path):
        """Corpus changed → cache invalidated."""
        corpus1 = ["苹果香蕉", "西瓜葡萄"]
        cache_dir = str(tmp_path)

        bm1 = BM25Retriever(corpus1)
        bm1.save_cache(cache_dir)

        bm2 = BM25Retriever()
        assert not bm2.try_load_cache(cache_dir, ["完全不同的文档"])
        assert not bm2.ready

    def test_cache_handles_corrupt_file(self, tmp_path):
        """Corrupt JSON file → returns False, no crash."""
        cache_path = tmp_path / "bm25_index.json"
        cache_path.write_text("this is not valid json")

        bm = BM25Retriever()
        result = bm.try_load_cache(str(tmp_path), ["test"])
        assert result is False

    def test_cache_legacy_pickle_ignored(self, tmp_path):
        """Legacy .pkl cache is ignored (format migrated to signed JSON)."""
        cache_path = tmp_path / "bm25_index.pkl"
        cache_path.write_bytes(b"\x80\x04")  # truncated pickle header

        bm = BM25Retriever()
        result = bm.try_load_cache(str(tmp_path), ["test"])
        assert result is False
        assert not bm.ready

    def test_cache_signed_roundtrip(self, tmp_path):
        """With a secret, cache is signed and reloads correctly."""
        corpus = ["苹果香蕉橘子", "香蕉西瓜", "苹果葡萄"]
        cache_dir = str(tmp_path)

        bm1 = BM25Retriever(corpus, cache_secret="my-secret")
        bm1.save_cache(cache_dir)

        bm2 = BM25Retriever(cache_secret="my-secret")
        assert bm2.try_load_cache(cache_dir, corpus)
        assert bm2.ready
        assert bm1.retrieve("苹果") == bm2.retrieve("苹果")

    def test_cache_tampered_rejected(self, tmp_path):
        """Tampered signed cache → signature mismatch → returns False."""
        corpus = ["苹果香蕉橘子", "香蕉西瓜", "苹果葡萄"]
        cache_dir = str(tmp_path)

        bm1 = BM25Retriever(corpus, cache_secret="my-secret")
        bm1.save_cache(cache_dir)

        cache_path = tmp_path / "bm25_index.json"
        content = cache_path.read_text(encoding="utf-8")
        tampered = content.replace("苹果", "橘子")  # alter data, signature now invalid
        cache_path.write_text(tampered, encoding="utf-8")

        bm2 = BM25Retriever(cache_secret="my-secret")
        assert not bm2.try_load_cache(cache_dir, corpus)
        assert not bm2.ready

    def test_cache_wrong_secret_rejected(self, tmp_path):
        """Cache saved with secret A, loaded with secret B → rejected."""
        corpus = ["苹果香蕉橘子"]
        cache_dir = str(tmp_path)

        bm1 = BM25Retriever(corpus, cache_secret="secret-a")
        bm1.save_cache(cache_dir)

        bm2 = BM25Retriever(cache_secret="secret-b")
        assert not bm2.try_load_cache(cache_dir, corpus)


class TestBootstrapRAG:
    def test_returns_none_when_no_api_key(self):
        settings = MagicMock()
        settings.siliconflow_api_key = ""
        result = bootstrap_rag(settings, lambda x: None)
        assert result is None

    def test_returns_context_when_configured(self):
        settings = MagicMock()
        settings.siliconflow_api_key = "sk-test"
        settings.siliconflow_model = "test-model"
        settings.siliconflow_dimensions = 512
        settings.siliconflow_base_url = "http://test"

        mock_store = MagicMock()
        mock_store.get_all_documents.return_value = [
            {"text": "测试文档内容", "id": "1", "source": "test.md", "chapter": "1"}
        ]

        # bootstrap_rag imports these from clawagent.rag at runtime
        with (
            patch("clawagent.rag.SiliconFlowEmbedding") as mock_emb,
            patch("clawagent.rag.RAGStore", return_value=mock_store) as mock_rag,
            patch("clawagent.rag.BM25Retriever") as mock_bm25,
            patch("clawagent.rag.HybridSearcher") as mock_hybrid,
        ):
            mock_bm25_instance = MagicMock()
            mock_bm25.return_value = mock_bm25_instance

            configure_fn = MagicMock()
            result = bootstrap_rag(settings, configure_fn)

            assert result is not None
            assert isinstance(result, RAGContext)
            assert isinstance(result.bm25_ready_signal, list)
            # Thread may or may not have completed (mocked build_async is instant)

            mock_emb.assert_called_once()
            mock_rag.assert_called_once()
            mock_bm25.assert_called_once()
            mock_hybrid.assert_called_once()
            configure_fn.assert_called_once()
