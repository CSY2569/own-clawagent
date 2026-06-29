"""Fixed-window text chunking with overlap."""


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[tuple[str, int]]:
    """Split text into overlapping fixed-size chunks.

    Args:
        text: Input text to split.
        chunk_size: Maximum characters per chunk.
        overlap: Number of characters to overlap between adjacent chunks.

    Returns:
        List of (chunk_text, start_position) tuples.

    Raises:
        ValueError: If overlap >= chunk_size or chunk_size <= 0.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        )
    if not text.strip():
        return []
    if len(text) <= chunk_size:
        return [(text, 0)]

    chunks: list[tuple[str, int]] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append((chunk, start))
        start = max(end - overlap, start + 1)
        if start >= len(text):
            break
    return chunks
