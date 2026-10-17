from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from app.schemas.documents import Chunk


@dataclass(frozen=True)
class IndexEntry:
    """One indexed unit: a chunk plus the metadata a caller needs to
    attribute results (collection isolation, display)."""

    chunk: Chunk
    filename: str
    collection_id: str


@dataclass(frozen=True)
class ScoredResult:
    entry: IndexEntry
    score: float


def l2_normalize(vectors: np.ndarray, axis: int = -1) -> np.ndarray:
    """L2-normalize rows (or a single vector); zero vectors stay zero."""
    norms = np.linalg.norm(vectors, axis=axis, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (vectors / norms).astype(np.float32)


@runtime_checkable
class VectorIndex(Protocol):
    """Replaceable vector index interface (NumPy baseline; FAISS later)."""

    def add(self, entries: Sequence[IndexEntry], vectors: np.ndarray) -> None: ...

    def search(
        self, query_vector: np.ndarray, *, top_k: int, collection_id: str | None = None
    ) -> list[ScoredResult]: ...

    def size(self) -> int: ...


class NumpyVectorIndex:
    """Transparent in-memory index: L2-normalized matrix, cosine via dot product.

    ``collection_id`` filtering happens inside search so chunks from unrelated
    collections can never surface in each other's results.
    """

    def __init__(self) -> None:
        self._entries: list[IndexEntry] = []
        self._matrix: np.ndarray | None = None
        self._dimension: int | None = None

    @property
    def dimension(self) -> int | None:
        return self._dimension

    def add(self, entries: Sequence[IndexEntry], vectors: np.ndarray) -> None:
        if len(entries) == 0:
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(entries):
            raise ValueError("vectors must have shape (len(entries), dimension)")
        if self._dimension is None:
            self._dimension = int(vectors.shape[1])
        elif vectors.shape[1] != self._dimension:
            raise ValueError(
                f"vector dimension {vectors.shape[1]} does not match index dimension {self._dimension}"
            )

        normalized = l2_normalize(vectors, axis=1)
        if self._matrix is None:
            self._matrix = normalized
        else:
            self._matrix = np.vstack([self._matrix, normalized])
        self._entries.extend(entries)

    def search(
        self, query_vector: np.ndarray, *, top_k: int, collection_id: str | None = None
    ) -> list[ScoredResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if self._matrix is None or not self._entries:
            return []

        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != self._dimension:
            raise ValueError(
                f"query dimension {query.shape[0]} does not match index dimension {self._dimension}"
            )
        query = l2_normalize(query)

        scores = self._matrix @ query
        if collection_id is not None:
            keep = np.array(
                [entry.collection_id == collection_id for entry in self._entries],
                dtype=np.bool_,
            )
            scores = np.where(keep, scores, np.float32(-np.inf))

        count = min(top_k, scores.shape[0])
        order = np.argsort(-scores, kind="stable")[:count]
        return [
            ScoredResult(entry=self._entries[index], score=float(scores[index]))
            for index in order
            if scores[index] > -np.inf
        ]

    def size(self) -> int:
        return len(self._entries)
