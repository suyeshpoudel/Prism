from __future__ import annotations

import json

import numpy as np
from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, selectinload, sessionmaker

from app.repository.models import ChunkRow, DocumentRow, PageRow
from app.repository.record import DocumentRecord
from app.schemas.documents import Chunk, PageExtraction

_EMBEDDING_DTYPE = np.dtype("<f4")


def _embeddings_to_blob(matrix: np.ndarray) -> list[bytes]:
    array = np.asarray(matrix, dtype=_EMBEDDING_DTYPE)
    return [np.ascontiguousarray(row).tobytes() for row in array]


def _blob_to_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=_EMBEDDING_DTYPE).copy()


class SqlDocumentStore:
    """SQLAlchemy-backed document store (PostgreSQL in dev, SQLite in tests)."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    @property
    def engine(self) -> Engine:
        bind = self._session_factory.kw.get("bind")
        if bind is None:
            raise RuntimeError("session factory is not bound to an engine")
        return bind

    def add(self, record: DocumentRecord, embeddings: np.ndarray) -> None:
        document = DocumentRow(
            id=record.id,
            filename=record.filename,
            content_type=record.content_type,
            size_bytes=record.size_bytes,
            page_count=record.page_count,
            extracted_char_count=record.extracted_char_count,
            warnings_json=json.dumps(record.warnings),
            collection_id=record.collection_id,
            owner_id=record.owner_id,
            status=record.status,
            embedding_model=record.embedding_model,
            created_at=record.created_at,
        )
        pages = [
            PageRow(
                document_id=record.id,
                page_number=page.page_number,
                text=page.text,
                char_count=page.char_count,
                is_empty=page.is_empty,
            )
            for page in record.pages
        ]
        blobs = _embeddings_to_blob(embeddings)
        if len(blobs) != len(record.chunks):
            raise ValueError(
                f"embeddings count {len(blobs)} does not match chunks count {len(record.chunks)}"
            )
        chunks = [
            ChunkRow(
                id=chunk.id,
                document_id=record.id,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                char_count=chunk.char_count,
                start_offset=chunk.start_offset,
                end_offset=chunk.end_offset,
                embedding=blob,
            )
            for chunk, blob in zip(record.chunks, blobs, strict=True)
        ]
        with self._session_factory() as session:
            session.add(document)
            session.add_all(pages)
            session.add_all(chunks)
            session.commit()

    def get(self, document_id: str, *, owner_id: str) -> DocumentRecord | None:
        with self._session_factory() as session:
            row = session.execute(
                select(DocumentRow)
                .where(DocumentRow.id == document_id, DocumentRow.owner_id == owner_id)
                .options(selectinload(DocumentRow.pages), selectinload(DocumentRow.chunks))
            ).scalar_one_or_none()
            if row is None:
                return None
            return self._row_to_record(row)

    def list(self, *, owner_id: str) -> list[DocumentRecord]:
        with self._session_factory() as session:
            rows = (
                session.execute(
                    select(DocumentRow)
                    .where(DocumentRow.owner_id == owner_id)
                    .options(selectinload(DocumentRow.pages), selectinload(DocumentRow.chunks))
                    .order_by(DocumentRow.created_at)
                )
                .scalars()
                .all()
            )
            return [self._row_to_record(row) for row in rows]

    def delete(self, document_id: str, *, owner_id: str) -> bool:
        with self._session_factory() as session:
            row = session.execute(
                select(DocumentRow).where(
                    DocumentRow.id == document_id, DocumentRow.owner_id == owner_id
                )
            ).scalar_one_or_none()
            if row is None:
                return False
            session.delete(row)
            session.commit()
            return True

    def load_all_with_embeddings(self) -> list[tuple[DocumentRecord, np.ndarray]]:
        with self._session_factory() as session:
            rows = (
                session.execute(
                    select(DocumentRow)
                    .options(selectinload(DocumentRow.pages), selectinload(DocumentRow.chunks))
                    .order_by(DocumentRow.created_at)
                )
                .scalars()
                .all()
            )
            loaded: list[tuple[DocumentRecord, np.ndarray]] = []
            for row in rows:
                if not row.chunks:
                    continue
                matrix = np.vstack([_blob_to_embedding(chunk.embedding) for chunk in row.chunks])
                loaded.append((self._row_to_record(row), matrix))
            return loaded

    @staticmethod
    def _row_to_record(row: DocumentRow) -> DocumentRecord:
        return DocumentRecord(
            id=row.id,
            filename=row.filename,
            content_type=row.content_type,
            size_bytes=row.size_bytes,
            page_count=row.page_count,
            extracted_char_count=row.extracted_char_count,
            pages=[
                PageExtraction(
                    page_number=page.page_number,
                    text=page.text,
                    char_count=page.char_count,
                    is_empty=page.is_empty,
                )
                for page in row.pages
            ],
            chunks=[
                Chunk(
                    id=chunk.id,
                    document_id=chunk.document_id,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    char_count=chunk.char_count,
                    start_offset=chunk.start_offset,
                    end_offset=chunk.end_offset,
                )
                for chunk in row.chunks
            ],
            warnings=json.loads(row.warnings_json),
            collection_id=row.collection_id,
            status=row.status,
            owner_id=row.owner_id,
            embedding_model=row.embedding_model,
            created_at=row.created_at,
        )
