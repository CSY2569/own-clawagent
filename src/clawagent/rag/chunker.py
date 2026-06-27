"""Fixed-window text chunking with overlap."""


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[str]:
    """Split text into overlapping fixed-size chunks.

    Args:
        text: Input text to split.
        chunk_size: Maximum characters per chunk.
        overlap: Number of characters to overlap between adjacent chunks.

    Returns:
        List of text chunks.
    """
    if not text.strip():
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = max(end - overlap, start + 1)
        if start >= len(text):
            break
    return chunks
