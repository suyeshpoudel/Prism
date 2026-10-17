import threading
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingError(Exception):
    """Raised when the embedding model fails to load or encode."""


@runtime_checkable
class Embedder(Protocol):
    """Replaceable embedding backend. Vectors may be unnormalized; the
    vector index is responsible for L2 normalization."""

    @property
    def dimension(self) -> int: ...

    @property
    def model_name(self) -> str: ...

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch of documents. Returns float32 array of shape (n, dim)."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query. Returns float32 array of shape (dim,)."""
        ...


class LazySentenceTransformerEmbedder:
    """Local, free embedding backend via Sentence Transformers.

    The model is loaded lazily on first use and shared across instances that
    load the same model name, so app startup stays fast and tests that inject
    a fake embedder never pay the model cost.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self._model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:
                    raise EmbeddingError(
                        "sentence-transformers is not installed. "
                        "Install it with: pip install -r requirements-ml.txt"
                    ) from exc
                try:
                    self._model = SentenceTransformer(self._model_name)
                except Exception as exc:
                    raise EmbeddingError(
                        f"Failed to load embedding model '{self._model_name}'."
                    ) from exc
        return self._model

    @property
    def dimension(self) -> int:
        return int(self._load().get_sentence_embedding_dimension())

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        try:
            vectors = self._load().encode(
                list(texts),
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=False,
            )
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError("Failed to embed documents.") from exc
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]


_SHARED_EMBEDDERS: dict[str, LazySentenceTransformerEmbedder] = {}
_SHARED_EMBEDDERS_LOCK = threading.Lock()


def get_shared_embedder(model_name: str = DEFAULT_EMBEDDING_MODEL) -> LazySentenceTransformerEmbedder:
    """Return one lazily-loading embedder per model name (process-wide)."""
    with _SHARED_EMBEDDERS_LOCK:
        if model_name not in _SHARED_EMBEDDERS:
            _SHARED_EMBEDDERS[model_name] = LazySentenceTransformerEmbedder(model_name)
        return _SHARED_EMBEDDERS[model_name]
