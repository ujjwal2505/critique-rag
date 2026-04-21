"""Local sentence-transformers embeddings (no external API)."""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from .config import EMBED_MODEL


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    # Loaded once per process; first call downloads the model (~80MB).
    return SentenceTransformer(EMBED_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts into normalized vectors."""
    vectors = _model().encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vectors]


def embed_one(text: str) -> list[float]:
    return embed([text])[0]
