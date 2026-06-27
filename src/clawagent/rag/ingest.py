"""Ingest documents from a directory into the RAG vector store.

Usage:
    uv run python -m clawagent.rag.ingest docs/ --chunk-size 512 --overlap 64
"""

import argparse
import re
import sys
from pathlib import Path

_CHAPTER_RE = re.compile(
    r"(第[零一二三四五六七八九十百千万\d]+[章节]"
    r"|序章|尾声|终章|番外[一二三四五六七八九十]?"
    r"|楔子|后记|附录"
    r"|Chapter\s+\d+"
    r"|Part\s+\d+"
    r"|Volume\s+\d+)",
)


def _find_chapter_spans(text: str) -> list[tuple[int, str]]:
    """Find chapter markers in text and return (char_position, chapter_name) pairs."""
    spans: list[tuple[int, str]] = []
    for m in _CHAPTER_RE.finditer(text):
        spans.append((m.start(), m.group()))
    return spans


def _chapter_for_pos(spans: list[tuple[int, str]], pos: int) -> str:
    """Return the chapter name for a given character position."""
    chapter = ""
    for span_pos, name in spans:
        if span_pos <= pos:
            chapter = name
        else:
            break
    return chapter


def _collect_files(root: Path) -> list[tuple[Path, str]]:
    """Recursively collect readable files from a directory.

    Returns list of (path, content) tuples.
    """
    files: list[tuple[Path, str]] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix in (".md", ".txt", ".rst", ".py"):
            try:
                content = p.read_text(encoding="utf-8")
                if content.strip():
                    files.append((p, content))
            except UnicodeDecodeError:
                print(f"  [skip] {p} — not UTF-8", file=sys.stderr)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into RAG store")
    parser.add_argument("docs_dir", help="Directory containing documents to ingest")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--db-path", default="./chroma_db")
    args = parser.parse_args()

    docs_root = Path(args.docs_dir)
    if not docs_root.is_dir():
        print(f"Error: {args.docs_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    from clawagent.config import Settings
    from clawagent.rag import RAGStore, SiliconFlowEmbedding, chunk_text

    settings = Settings.from_env()
    if not settings.siliconflow_api_key:
        print("Error: SILICONFLOW_API_KEY not set in .env", file=sys.stderr)
        sys.exit(1)

    embedding = SiliconFlowEmbedding(
        api_key=settings.siliconflow_api_key,
        model=settings.siliconflow_model,
        dimensions=settings.siliconflow_dimensions,
    )
    store = RAGStore(db_path=args.db_path, embedding=embedding)

    files = _collect_files(docs_root)
    if not files:
        print(f"No readable files found in {args.docs_dir}")
        sys.exit(0)

    print(f"Found {len(files)} files. Chunking (size={args.chunk_size}, overlap={args.overlap})...")

    texts: list[str] = []
    metadatas: list[dict[str, str]] = []
    for filepath, content in files:
        chapter_spans = _find_chapter_spans(content)
        if chapter_spans:
            chapter_names = ", ".join({n for _, n in chapter_spans[:5]})
            print(f"  Found {len(chapter_spans)} chapter markers "
                  f"(e.g. {chapter_names})")
        chunks = chunk_text(content, chunk_size=args.chunk_size, overlap=args.overlap)
        step = args.chunk_size - args.overlap
        for i, chunk in enumerate(chunks):
            chapter = _chapter_for_pos(chapter_spans, i * step)
            meta: dict[str, str] = {
                "source": str(filepath.relative_to(docs_root)),
                "chunk_index": str(i),
            }
            if chapter:
                meta["chapter"] = chapter
            texts.append(chunk)
            metadatas.append(meta)

    print(f"Produced {len(texts)} chunks. Embedding and storing...")
    store.add_documents(texts, metadatas=metadatas)
    print(f"Done. {store.count()} documents in vector store at {args.db_path}/")


if __name__ == "__main__":
    main()
