import uuid

from fastapi import APIRouter, Depends, File, Request, UploadFile

from app.config import Settings
from app.core.errors import DocumentNotFoundError, FileTooLargeError
from app.processing.chunking import ChunkingConfig, chunk_pages
from app.processing.extraction import extract_pages
from app.processing.validation import (
    ensure_pdf_magic,
    ensure_size_within_limit,
    validate_media_type,
)
from app.repository import DocumentRecord, DocumentStore
from app.schemas.documents import DocumentResponse, DocumentSummary, ErrorResponse

router = APIRouter(prefix="/documents", tags=["documents"])

READ_CHUNK_BYTES = 64 * 1024

ERROR_RESPONSES = {
    400: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
    413: {"model": ErrorResponse},
    415: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
}


def get_document_store(request: Request) -> DocumentStore:
    return request.app.state.document_store


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def chunking_config_from_settings(settings: Settings) -> ChunkingConfig:
    return ChunkingConfig(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)


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
    store: DocumentStore = Depends(get_document_store),
    settings: Settings = Depends(get_app_settings),
) -> DocumentResponse:
    filename = validate_media_type(file.filename, file.content_type)
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
