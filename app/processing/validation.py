from pathlib import Path

from app.core.errors import (
    EmptyFileError,
    FileTooLargeError,
    InvalidPDFError,
    InvalidRequestError,
    UnsupportedMediaTypeError,
)

PDF_EXTENSIONS = {".pdf"}
PDF_CONTENT_TYPES = {"", "application/pdf", "application/x-pdf", "application/octet-stream"}
PDF_MAGIC = b"%PDF-"
MAGIC_SCAN_WINDOW = 1024


def validate_media_type(filename: str | None, content_type: str | None) -> str:
    """Validate declared filename and content type. Returns the cleaned filename.

    Neither the filename nor the MIME type is trusted as proof the content is a
    PDF; callers must also check the file magic (``ensure_pdf_magic``).
    """
    cleaned = (filename or "").strip()
    if not cleaned:
        raise InvalidRequestError("A filename is required for the upload.")
    if Path(cleaned).suffix.lower() not in PDF_EXTENSIONS:
        raise UnsupportedMediaTypeError("Only PDF files are supported.")
    media_type = (content_type or "").split(";")[0].strip().lower()
    if media_type not in PDF_CONTENT_TYPES:
        raise UnsupportedMediaTypeError(
            f"Unsupported content type '{content_type}'. Expected application/pdf."
        )
    return cleaned


def ensure_size_within_limit(data: bytes, max_bytes: int) -> None:
    if len(data) == 0:
        raise EmptyFileError("The uploaded file is empty.")
    if len(data) > max_bytes:
        raise FileTooLargeError(f"File exceeds the upload limit of {max_bytes} bytes.")


def ensure_pdf_magic(data: bytes) -> None:
    if PDF_MAGIC not in data[:MAGIC_SCAN_WINDOW]:
        raise InvalidPDFError("File content does not contain a valid PDF header.")
