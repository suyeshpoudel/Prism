import pytest

from app.core.errors import EncryptedPDFError, UnreadablePDFError
from app.processing.extraction import WARNING_NO_TEXT, extract_pages
from tests.pdf_helpers import make_pdf


def test_extract_pages_preserves_page_numbers_and_text():
    data = make_pdf(["Page one content.", "Page two content."])

    extracted = extract_pages(data)

    assert extracted.page_count == 2
    assert [page.page_number for page in extracted.pages] == [1, 2]
    assert "Page one content." in extracted.pages[0].text
    assert "Page two content." in extracted.pages[1].text
    assert extracted.extracted_char_count == sum(page.char_count for page in extracted.pages)
    assert extracted.warnings == []


def test_extract_marks_blank_pages():
    data = make_pdf(["Some text.", ""])

    extracted = extract_pages(data)

    assert extracted.pages[0].is_empty is False
    assert extracted.pages[1].is_empty is True
    assert extracted.pages[1].text == ""
    assert WARNING_NO_TEXT not in extracted.warnings


def test_extract_warns_when_no_text_extracted():
    data = make_pdf(["", ""])

    extracted = extract_pages(data)

    assert extracted.warnings == [WARNING_NO_TEXT]
    assert extracted.extracted_char_count == 0


def test_extract_rejects_encrypted_pdf():
    data = make_pdf(["Secret."], encrypt=True)

    with pytest.raises(EncryptedPDFError):
        extract_pages(data)


def test_extract_rejects_corrupt_pdf():
    with pytest.raises(UnreadablePDFError):
        extract_pages(b"%PDF-1.4\nthis is not a real pdf document")
