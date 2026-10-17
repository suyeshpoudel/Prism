from dataclasses import dataclass

import pymupdf

from app.core.errors import EncryptedPDFError, UnreadablePDFError
from app.schemas.documents import PageExtraction

WARNING_NO_TEXT = "no_text_extracted"
WARNING_FAILED_PAGES = "extraction_failed_pages"


@dataclass(frozen=True)
class ExtractedDocument:
    page_count: int
    pages: list[PageExtraction]
    warnings: list[str]
    extracted_char_count: int


def _normalize_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def extract_pages(data: bytes) -> ExtractedDocument:
    """Extract text page by page from PDF bytes, preserving page numbers.

    Raises EncryptedPDFError for password-protected documents and
    UnreadablePDFError for corrupt or page-less documents.
    """
    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise UnreadablePDFError(
            "The PDF could not be opened. The file may be corrupted."
        ) from exc

    try:
        if document.needs_pass:
            raise EncryptedPDFError(
                "The PDF is password-protected and cannot be processed."
            )
        if document.page_count == 0:
            raise UnreadablePDFError("The PDF contains no pages.")

        pages: list[PageExtraction] = []
        failed_pages: list[int] = []
        for index in range(document.page_count):
            page_number = index + 1
            try:
                raw = document[index].get_text("text") or ""
            except Exception:
                failed_pages.append(page_number)
                pages.append(
                    PageExtraction(page_number=page_number, text="", char_count=0, is_empty=True)
                )
                continue
            text = _normalize_line_endings(raw)
            pages.append(
                PageExtraction(
                    page_number=page_number,
                    text=text,
                    char_count=len(text),
                    is_empty=not text.strip(),
                )
            )
    finally:
        document.close()

    warnings: list[str] = []
    if failed_pages:
        warnings.append(
            f"{WARNING_FAILED_PAGES}:{','.join(str(number) for number in failed_pages)}"
        )
    if all(page.is_empty for page in pages):
        warnings.append(WARNING_NO_TEXT)

    extracted_char_count = sum(page.char_count for page in pages)
    return ExtractedDocument(
        page_count=len(pages),
        pages=pages,
        warnings=warnings,
        extracted_char_count=extracted_char_count,
    )
