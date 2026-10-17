from dataclasses import dataclass
from threading import Lock
from typing import Protocol

from app.schemas.documents import Chunk, PageExtraction


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
    status: str = "processed"


class DocumentStore(Protocol):
    """Persistence boundary for documents. PostgreSQL-backed in Milestone 4."""

    def add(self, record: DocumentRecord) -> None: ...

    def get(self, document_id: str) -> DocumentRecord | None: ...

    def list(self) -> list[DocumentRecord]: ...


class InMemoryDocumentStore:
    def __init__(self) -> None:
        self._documents: dict[str, DocumentRecord] = {}
        self._lock = Lock()

    def add(self, record: DocumentRecord) -> None:
        with self._lock:
            self._documents[record.id] = record

    def get(self, document_id: str) -> DocumentRecord | None:
        with self._lock:
            return self._documents.get(document_id)

    def list(self) -> list[DocumentRecord]:
        with self._lock:
            return list(self._documents.values())
