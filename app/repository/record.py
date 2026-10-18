from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol

import numpy as np

from app.schemas.documents import Chunk, PageExtraction


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class DocumentRecord:
    id: str
    filename: str
    content_type: str | None
    size_bytes: int
    page_count: int
    extracted_char_count: int
    pages: list[PageExtraction]
    chunks: list[Chunk]
    warnings: list[str]
    collection_id: str = "default"
    status: str = "processed"
    owner_id: str = "default"
    embedding_model: str = ""
    created_at: datetime = field(default_factory=_utcnow)


class DocumentStore(Protocol):
    """Persistence boundary for documents (PostgreSQL-backed in Milestone 4)."""

    def add(self, record: DocumentRecord, embeddings: np.ndarray) -> None: ...

    def get(self, document_id: str, *, owner_id: str) -> DocumentRecord | None: ...

    def list(self, *, owner_id: str) -> list[DocumentRecord]: ...

    def delete(self, document_id: str, *, owner_id: str) -> bool: ...

    def load_all_with_embeddings(self) -> list[tuple[DocumentRecord, np.ndarray]]: ...
