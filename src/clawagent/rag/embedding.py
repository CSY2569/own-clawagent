"""SiliconFlow cloud embedding via REST API."""

import time
import urllib.request
from json import JSONDecodeError


class SiliconFlowEmbedding:
    """Embedding client backed by SiliconFlow's embedding API.

    Args:
        api_key: SiliconFlow API key.
        model: Embedding model name.
        dimensions: Output vector dimensions.
        base_url: API base URL.
        batch_size: Maximum number of texts per API call.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "Qwen/Qwen3-VL-Embedding-8B",
        dimensions: int = 768,
        base_url: str = "https://api.siliconflow.cn/v1/embeddings",
        batch_size: int = 100,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._dimensions = dimensions
        self._url = base_url
        self._batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Batch-embed a list of document texts."""
        return self._call(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        results = self._call([text])
        return results[0]

    def _call(self, texts: list[str], max_retries: int = 3) -> list[list[float]]:
        """Call the SiliconFlow embedding API with retry on failure.

        Large text lists are automatically split into batches.
        """
        import json

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            payload = json.dumps({
                "model": self._model,
                "input": batch,
                "dimensions": self._dimensions,
            }).encode("utf-8")

            req = urllib.request.Request(
                self._url,
                data=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )

            last_error: Exception | None = None
            for attempt in range(max_retries):
                try:
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        body = json.loads(resp.read().decode("utf-8"))
                    batch_embeddings = [item["embedding"] for item in body["data"]]
                    all_embeddings.extend(batch_embeddings)
                    print(
                        f"  Embedded batch {i // self._batch_size + 1}/"
                        f"{(len(texts) - 1) // self._batch_size + 1} "
                        f"({len(batch)} texts)"
                    )
                    break
                except (OSError, JSONDecodeError, KeyError) as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
            else:
                raise RuntimeError(
                    f"SiliconFlow embedding API failed after {max_retries} attempts "
                    f"on batch {i // self._batch_size + 1}"
                ) from last_error

        return all_embeddings
