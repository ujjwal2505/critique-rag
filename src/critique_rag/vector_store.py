"""Thin wrapper over a local persistent ChromaDB collection.

We compute embeddings ourselves (see `embeddings.py`) and pass them in
explicitly, so the retrieval path mirrors the `retrieve` node's description:
embed the query, then pull the nearest chunks.
"""

import chromadb

from .config import CHROMA_DIR, COLLECTION_NAME


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # Cosine space pairs with the normalized embeddings from embeddings.py.
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(ids, documents, embeddings, metadatas) -> None:
    get_collection().upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def query(query_embedding, k: int) -> list[dict]:
    """Return the top-k chunks as flat dicts: {id, text, distance, source}."""
    res = get_collection().query(query_embeddings=[query_embedding], n_results=k)
    ids = res["ids"][0]
    docs = res["documents"][0]
    dists = res["distances"][0]
    metas = res["metadatas"][0]
    chunks = []
    for i in range(len(ids)):
        meta = metas[i] or {}
        chunks.append(
            {
                "id": ids[i],
                "text": docs[i],
                "distance": dists[i],
                "source": meta.get("source"),
            }
        )
    return chunks


def count() -> int:
    return get_collection().count()
