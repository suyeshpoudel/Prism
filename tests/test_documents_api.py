import pytest

from app.core.errors import (
    EmptyFileError,
    FileTooLargeError,
    InvalidPDFError,
    InvalidRequestError,
    UnsupportedMediaTypeError,
)
from app.main import create_app
from app.processing.validation import (
    ensure_pdf_magic,
    ensure_size_within_limit,
    validate_media_type,
)
from tests.pdf_helpers import make_pdf


def test_upload_valid_pdf_extracts_pages(client, pdf_bytes):
    response = client.post(
        "/api/v1/documents",
        files={"file": ("research.pdf", pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "research.pdf"
    assert body["content_type"] == "application/pdf"
    assert body["size_bytes"] == len(pdf_bytes)
    assert body["page_count"] == 2
    assert body["status"] == "processed"
    assert body["warnings"] == []
    assert [page["page_number"] for page in body["pages"]] == [1, 2]
    assert "retrieval" in body["pages"][0]["text"]
    assert body["pages"][0]["is_empty"] is False
    assert body["id"]


def test_upload_accepts_octet_stream_content_type(client, pdf_bytes):
    response = client.post(
        "/api/v1/documents",
        files={"file": ("notes.pdf", pdf_bytes, "application/octet-stream")},
    )

    assert response.status_code == 201


def test_upload_rejects_non_pdf_extension(client, pdf_bytes):
    response = client.post(
        "/api/v1/documents",
        files={"file": ("notes.txt", pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media_type"


def test_upload_rejects_pdf_name_with_wrong_content_type(client, pdf_bytes):
    response = client.post(
        "/api/v1/documents",
        files={"file": ("notes.pdf", pdf_bytes, "text/plain")},
    )

    assert response.status_code == 415


def test_upload_rejects_missing_filename(client, pdf_bytes):
    # A blank filename is rejected by Prism validation. (A fully absent filename
    # is rejected earlier by framework-level multipart validation with a 422.)
    response = client.post(
        "/api/v1/documents",
        files={"file": ("   ", pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


def test_upload_rejects_content_without_pdf_magic(client):
    response = client.post(
        "/api/v1/documents",
        files={"file": ("fake.pdf", b"plain text pretending to be a pdf", "application/pdf")},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_pdf"


def test_upload_rejects_empty_file(client):
    response = client.post(
        "/api/v1/documents",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "empty_file"


def test_upload_rejects_oversized_file(tight_client, pdf_bytes):
    response = tight_client.post(
        "/api/v1/documents",
        files={"file": ("big.pdf", pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "file_too_large"


def test_upload_rejects_encrypted_pdf(client):
    data = make_pdf(["Secret."], encrypt=True)

    response = client.post(
        "/api/v1/documents",
        files={"file": ("locked.pdf", data, "application/pdf")},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "encrypted_pdf"


def test_upload_rejects_corrupt_pdf(client):
    response = client.post(
        "/api/v1/documents",
        files={"file": ("broken.pdf", b"%PDF-1.4\nnot a real pdf", "application/pdf")},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unreadable_pdf"


def test_upload_blank_pages_reported(client):
    data = make_pdf(["Text on page one.", ""])

    response = client.post(
        "/api/v1/documents",
        files={"file": ("blank.pdf", data, "application/pdf")},
    )

    body = response.json()
    assert body["pages"][1]["is_empty"] is True
    assert "no_text_extracted" not in body["warnings"]


def test_upload_no_text_pdf_warns(client):
    data = make_pdf(["", ""])

    response = client.post(
        "/api/v1/documents",
        files={"file": ("scan.pdf", data, "application/pdf")},
    )

    assert response.json()["warnings"] == ["no_text_extracted"]


def test_list_and_get_document(client, pdf_bytes):
    upload = client.post(
        "/api/v1/documents",
        files={"file": ("research.pdf", pdf_bytes, "application/pdf")},
    )
    document_id = upload.json()["id"]

    listing = client.get("/api/v1/documents")
    assert listing.status_code == 200
    summaries = listing.json()
    assert len(summaries) == 1
    assert summaries[0]["id"] == document_id
    assert summaries[0]["filename"] == "research.pdf"

    fetched = client.get(f"/api/v1/documents/{document_id}")
    assert fetched.status_code == 200
    assert fetched.json()["id"] == document_id
    assert len(fetched.json()["pages"]) == 2


def test_get_unknown_document_returns_404(client):
    response = client.get("/api/v1/documents/does-not-exist")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "document_not_found"


def test_upload_returns_chunks_with_source_metadata(client, pdf_bytes):
    response = client.post(
        "/api/v1/documents",
        files={"file": ("research.pdf", pdf_bytes, "application/pdf")},
    )

    body = response.json()
    assert body["chunk_count"] == len(body["chunks"]) >= 1
    for chunk in body["chunks"]:
        assert chunk["document_id"] == body["id"]
        assert chunk["id"].startswith(f"{body['id']}:")
        page = body["pages"][chunk["page_number"] - 1]
        assert page["text"][chunk["start_offset"] : chunk["end_offset"]] == chunk["text"]


def test_upload_respects_chunk_settings_from_settings(pdf_bytes, db_url):
    from fastapi.testclient import TestClient

    from app.config import Settings
    from tests.conftest import TEST_API_KEY, TEST_OWNER
    from tests.embedding_stub import HashingEmbedder

    settings = Settings(
        max_upload_bytes=1_000_000,
        chunk_size=20,
        chunk_overlap=5,
        database_url=db_url,
        api_keys={TEST_API_KEY: TEST_OWNER},
    )
    with TestClient(create_app(settings, embedder=HashingEmbedder())) as app_client:
        app_client.headers.update({"X-API-Key": TEST_API_KEY})
        response = app_client.post(
            "/api/v1/documents",
            files={"file": ("research.pdf", pdf_bytes, "application/pdf")},
        )

    body = response.json()
    assert body["chunk_count"] > 1
    assert all(chunk["char_count"] <= 20 for chunk in body["chunks"])


def test_list_includes_chunk_count(client, pdf_bytes):
    client.post(
        "/api/v1/documents",
        files={"file": ("research.pdf", pdf_bytes, "application/pdf")},
    )

    summaries = client.get("/api/v1/documents").json()
    assert summaries[0]["chunk_count"] >= 1


def test_validate_media_type_helpers():
    assert validate_media_type("a.pdf", "application/pdf") == "a.pdf"
    assert validate_media_type("A.PDF", None) == "A.PDF"
    with pytest.raises(InvalidRequestError):
        validate_media_type(None, "application/pdf")
    with pytest.raises(UnsupportedMediaTypeError):
        validate_media_type("a.docx", "application/pdf")


def test_ensure_size_within_limit_helpers():
    ensure_size_within_limit(b"x", 10)
    with pytest.raises(EmptyFileError):
        ensure_size_within_limit(b"", 10)
    with pytest.raises(FileTooLargeError):
        ensure_size_within_limit(b"xx", 1)


def test_ensure_pdf_magic_helpers():
    ensure_pdf_magic(b"%PDF-1.7\nrest")
    with pytest.raises(InvalidPDFError):
        ensure_pdf_magic(b"nope")
