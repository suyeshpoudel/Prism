import hashlib

import numpy as np


class HashingEmbedder:
    """Deterministic, offline embedding stand-in for tests.

    Signed bag-of-words hashing: texts sharing tokens get higher cosine
    similarity, which is enough to test retrieval plumbing without the
    heavyweight model. Uses SHA-1 (stable across processes, unlike hash()).
    """

    def __init__(self, dimension: int = 64) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return f"hashing-{self._dimension}"

    def _embed_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self._dimension, dtype=np.float32)
        for token in text.lower().split():
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self._dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        return vector

    def embed_texts(self, texts):
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        return np.stack([self._embed_one(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed_one(text)


class FailingEmbedder:
    dimension = 8
    model_name = "failing"

    def embed_texts(self, texts):
        from app.embedding.provider import EmbeddingError

        raise EmbeddingError("embedder is broken")

    def embed_query(self, text: str):
        from app.embedding.provider import EmbeddingError

        raise EmbeddingError("embedder is broken")
