# Prism

> Find the signal in your documents.

Prism is a document intelligence system that answers questions about uploaded
PDF documents using evidence retrieved from the original sources. This repository
currently contains the **FastAPI backend**; Milestone 1 is complete.

## Status

| Milestone | Description | Status |
|-----------|-------------|--------|
| 1 | FastAPI app, health endpoint, PDF upload + page-level extraction, validation, tests | **Done** |
| 2 | Chunking with page metadata + edge-case tests | Next |
| 3 | Local embeddings, NumPy cosine retrieval, retrieval benchmark | Planned |
| 4 | PostgreSQL persistence + ownership/access control | Planned |
| 5 | Grounded answer generation + citations | Planned |
| 6 | Next.js UI, evaluation lab, Docker | Planned |

## Requirements

- Python 3.14+ (developed and tested on 3.14.5)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optional; defaults work for local dev
```

ML/retrieval dependencies (Milestone 3+) live separately in
`requirements-ml.txt` — they are large and not needed yet:

```bash
pip install -r requirements-ml.txt
```

## Run

```bash
uvicorn app.main:app --reload
```

- API docs: http://127.0.0.1:8000/docs
- Health: `GET /health`

## Test

```bash
pytest
```

## Configuration

Environment variables use the `PRISM_` prefix (or a `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `PRISM_APP_NAME` | `Prism` | Service name |
| `PRISM_ENVIRONMENT` | `development` | Environment label |
| `PRISM_API_V1_PREFIX` | `/api/v1` | API prefix |
| `PRISM_MAX_UPLOAD_BYTES` | `10485760` (10 MB) | Upload size limit |

## API

### `GET /health`

```json
{"status": "ok", "service": "Prism", "version": "0.1.0", "environment": "development"}
```

### `POST /api/v1/documents`

Multipart upload (`file` field). Returns `201` with document identity, page-level
extracted text, and extraction warnings.

```bash
curl -F "file=@sample.pdf;type=application/pdf" http://127.0.0.1:8000/api/v1/documents
```

Response (abridged):

```json
{
  "id": "379ebe97961e4c3d883236d12b0006ee",
  "filename": "sample.pdf",
  "content_type": "application/pdf",
  "size_bytes": 1212,
  "page_count": 2,
  "extracted_char_count": 56,
  "pages": [
    {"page_number": 1, "text": "…", "char_count": 35, "is_empty": false}
  ],
  "warnings": [],
  "status": "processed"
}
```

Warnings:

- `no_text_extracted` — every page produced no text (typical of scanned PDFs;
  OCR is a later phase).
- `extraction_failed_pages:2,5` — text extraction failed on specific pages;
  those pages are returned empty.

### `GET /api/v1/documents`

Lists uploaded documents (id, filename, size, page count, status).

### `GET /api/v1/documents/{id}`

Returns the full document record, or `404` if unknown.

### Error model

Prism errors use a consistent envelope:

```json
{"error": {"code": "invalid_pdf", "message": "File content does not contain a valid PDF header."}}
```

| Status | Code | Trigger |
|--------|------|---------|
| 400 | `invalid_request` | Missing/blank filename |
| 400 | `empty_file` | Zero-byte upload |
| 400 | `invalid_pdf` | No `%PDF-` header in the first 1024 bytes |
| 404 | `document_not_found` | Unknown document id |
| 413 | `file_too_large` | Exceeds `PRISM_MAX_UPLOAD_BYTES` (checked while streaming) |
| 415 | `unsupported_media_type` | Non-`.pdf` filename or non-PDF content type |
| 422 | `unreadable_pdf` | Corrupt file or document with no pages |
| 422 | `encrypted_pdf` | Password-protected PDF |

Framework-level validation failures (e.g. a missing `file` field) return
FastAPI's standard `422` detail format.

## Project structure

```
app/
  main.py                  # app factory (create_app), module-level app
  config.py                # pydantic-settings configuration
  api/routes/              # HTTP layer only (thin)
    health.py
    documents.py
  core/errors.py           # domain errors + exception handlers
  processing/              # pipeline, independent of HTTP
    validation.py          # filename/type/size/magic-byte checks
    extraction.py          # PyMuPDF page-level extraction
  repository/              # DocumentStore protocol + in-memory impl
  schemas/                 # Pydantic response models
tests/                     # pytest suite (24 tests)
```

## Security notes (Milestone 1)

- Filenames and MIME types are not trusted; content is verified via the
  `%PDF-` magic in the first 1024 bytes (per the PDF spec).
- Uploads are streamed and rejected as soon as they exceed the size limit.
- Encrypted and corrupt files fail with explicit, non-leaking error codes.
- Extracted PDF text is treated as untrusted data (prompt-injection defenses
  become critical at Milestone 5 and are documented there).

## Known limitations

- Document storage is **in-memory only**; data is lost on restart
  (PostgreSQL arrives in Milestone 4).
- No authentication/authorization yet (Milestone 4).
- Scanned/image-only PDFs extract zero text; OCR is a later phase.
- Pathological PDFs that hang the parser are not time-limited yet.
- No maximum page-count guard yet.

## Development rules

- Tests run with `pytest` from the repository root; keep them green.
- The processing pipeline must stay independent of FastAPI routes.
- No secrets in source control; configuration is environment-based.
