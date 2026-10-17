import uuid

import numpy as np
from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from app.config import Settings
from app.core.errors import DocumentNotFoundError, EmbeddingFailedError, FileTooLargeError, InvalidRequestError
from app.embedding import Embedder
from app.embedding.provider import EmbeddingError
from app.processing.chunking import ChunkingConfig, chunk_pages
from app.processing.extraction import extract_pages
from app.processing.validation import (
    ensure_pdf_magic,
    ensure_size_within_limit,
    validate_media_type,
)
from app.repository import DocumentRecord, DocumentStore
from app.retrieval import IndexEntry, NumpyVectorIndex, VectorIndex
from app.schemas.documents import DocumentResponse, DocumentSummary, ErrorResponse

router = APIRouter(prefix="/documents", tags=["documents"])

READ_CHUNK_BYTES = 64 * 1024
MAX_COLLECTION_ID_LENGTH = 128

ERROR_RESPONSES = {
    400: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
    413: {"model": ErrorResponse},
    415: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
    500: {"model": ErrorResponse},
}


def get_document_store(request: Request) -> DocumentStore:
    return request.app.state.document_store


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_embedder(request: Request) -> Embedder:
    return request.app.state.embedder


def get_vector_index(request: Request) -> NumpyVectorIndex:
    return request.app.state.vector_index


def chunking_config_from_settings(settings: Settings) -> ChunkingConfig:
    return ChunkingConfig(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)


def normalize_collection_id(collection_id: str | None) -> str:
    cleaned = (collection_id or "default").strip()
    if not cleaned:
        raise InvalidRequestError("collection_id must not be empty.")
    if len(cleaned) > MAX_COLLECTION_ID_LENGTH:
        raise InvalidRequestError(
            f"collection_id must be at most {MAX_COLLECTION_ID_LENGTH} characters."
        )
    return cleaned


def index_chunks(
    *,
    index: VectorIndex,
    embedder: Embedder,
    chunks: list,
    filename: str,
    collection_id: str,
    batch_size: int,
) -> None:
    if not chunks:
        return
    entries = [
        IndexEntry(chunk=chunk, filename=filename, collection_id=collection_id)
        for chunk in chunks
    ]
    try:
        vectors = []
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors.append(embedder.embed_texts([chunk.text for chunk in batch]))
        matrix = np.vstack(vectors).astype(np.float32)
        index.add(entries, matrix)
    except EmbeddingError as exc:
        raise EmbeddingFailedError(
            "The embedding model failed while processing this document."
        ) from exc


async def read_upload_limited(file: UploadFile, max_bytes: int) -> bytes:
    """Stream the upload in chunks, aborting as soon as the limit is exceeded."""
    buffer = bytearray()
    while True:
        chunk = await file.read(READ_CHUNK_BYTES)
        if not chunk:
            break
        buffer.extend(chunk)
        if len(buffer) > max_bytes:
            raise FileTooLargeError(f"File exceeds the upload limit of {max_bytes} bytes.")
    return bytes(buffer)


def record_to_response(record: DocumentRecord) -> DocumentResponse:
    return DocumentResponse(
        id=record.id,
        filename=record.filename,
        content_type=record.content_type,
        size_bytes=record.size_bytes,
        page_count=record.page_count,
        extracted_char_count=record.extracted_char_count,
        chunk_count=len(record.chunks),
        pages=record.pages,
        chunks=record.chunks,
        warnings=record.warnings,
        status=record.status,
    )


@router.post("", response_model=DocumentResponse, status_code=201, responses=ERROR_RESPONSES)
async def upload_document(
    file: UploadFile = File(...),
    collection_id: str | None = Form(default=None),
    store: DocumentStore = Depends(get_document_store),
    settings: Settings = Depends(get_app_settings),
    embedder: Embedder = Depends(get_embedder),
    index: NumpyVectorIndex = Depends(get_vector_index),
) -> DocumentResponse:
    filename = validate_media_type(file.filename, file.content_type)
    collection = normalize_collection_id(collection_id)
    data = await read_upload_limited(file, settings.max_upload_bytes)
    ensure_size_within_limit(data, settings.max_upload_bytes)
    ensure_pdf_magic(data)

    extracted = extract_pages(data)
    record_id = uuid.uuid4().hex
    chunks = chunk_pages(
        document_id=record_id,
        pages=extracted.pages,
        config=chunking_config_from_settings(settings),
    )
    index_chunks(
        index=index,
        embedder=embedder,
        chunks=chunks,
        filename=filename,
        collection_id=collection,
        batch_size=settings.embedding_batch_size,
    )
    record = DocumentRecord(
        id=record_id,
        filename=filename,
        content_type=file.content_type,
        size_bytes=len(data),
        page_count=extracted.page_count,
        extracted_char_count=extracted.extracted_char_count,
        pages=list(extracted.pages),
        chunks=chunks,
        warnings=list(extracted.warnings),
        collection_id=collection,
    )
    store.add(record)
    return record_to_response(record)


@router.get("", response_model=list[DocumentSummary], responses=ERROR_RESPONSES)
def list_documents(store: DocumentStore = Depends(get_document_store)) -> list[DocumentSummary]:
    return [
        DocumentSummary(
            id=record.id,
            filename=record.filename,
            size_bytes=record.size_bytes,
            page_count=record.page_count,
            chunk_count=len(record.chunks),
            status=record.status,
        )
        for record in store.list()
    ]


@router.get("/{document_id}", response_model=DocumentResponse, responses=ERROR_RESPONSES)
def get_document(
    document_id: str, store: DocumentStore = Depends(get_document_store)
) -> DocumentResponse:
    record = store.get(document_id)
    if record is None:
        raise DocumentNotFoundError(f"Document '{document_id}' was not found.")
    return record_to_response(record)
