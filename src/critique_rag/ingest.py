"""Ingest the sample corpus into ChromaDB.

Reads every ``*.md`` under ``data/sample_docs/``, splits each into short
paragraph-packed chunks, embeds them locally, and upserts them. Idempotent:
chunk ids are stable (``<filename>::<index>``), so re-running re-embeds in place.

Run: ``uv run python -m critique_rag.ingest``
"""

import re

from . import vector_store
from .config import SAMPLE_DOCS_DIR
from .embeddings import embed


def chunk_text(text: str, max_chars: int = 600) -> list[str]:
    """Split on blank lines, then pack paragraphs into <= max_chars windows."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > max_chars:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


def load_docs() -> list[tuple[str, str]]:
    return [
        (path.stem, path.read_text(encoding="utf-8"))
        for path in sorted(SAMPLE_DOCS_DIR.glob("*.md"))
    ]


def main() -> None:
    ids, documents, metadatas = [], [], []
    for stem, text in load_docs():
        for i, chunk in enumerate(chunk_text(text)):
            ids.append(f"{stem}::{i}")
            documents.append(chunk)
            metadatas.append({"source": stem, "chunk": i})

    if not documents:
        print(f"No .md documents found in {SAMPLE_DOCS_DIR}")
        return

    vector_store.add_chunks(ids, documents, embed(documents), metadatas)
    n_docs = len({m["source"] for m in metadatas})
    print(f"Ingested {len(documents)} chunks from {n_docs} docs.")
    print(f"Collection now holds {vector_store.count()} chunks.")


if __name__ == "__main__":
    main()
