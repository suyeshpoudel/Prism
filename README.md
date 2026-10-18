# Prism

> Find the signal in your documents.

Prism is a document intelligence system that answers questions about uploaded
PDF documents using evidence retrieved from the original sources. This repository
currently contains the **FastAPI backend**; Milestones 1–4 are complete.

## Status

| Milestone | Description | Status |
|-----------|-------------|--------|
| 1 | FastAPI app, health endpoint, PDF upload + page-level extraction, validation, tests | **Done** |
| 2 | Chunking with page metadata + edge-case tests | **Done** |
| 3 | Local embeddings, NumPy cosine retrieval, retrieval benchmark | **Done** |
| 4 | PostgreSQL persistence + ownership/access control | **Done** |
| 5 | Grounded answer generation + citations | Next |
| 6 | Next.js UI, evaluation lab, Docker | Planned |

## Requirements

- Python 3.14+ (developed and tested on 3.14.5)
- PostgreSQL 14+ (or SQLite for a zero-dependency tryout — see Configuration)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-ml.txt   # embeddings/search (Sentence Transformers, PyTorch)
cp .env.example .env   # set PRISM_API_KEYS and PRISM_DATABASE_URL
```

### Database

Start a local PostgreSQL (Docker example):

```bash
docker run --name prism-postgres -e POSTGRES_USER=prism -e POSTGRES_PASSWORD=prism \
  -e POSTGRES_DB=prism -p 5432:5432 -d postgres:16
```

Apply migrations:

```bash
alembic upgrade head
```

`PRISM_AUTO_CREATE_SCHEMA=true` (the default) also creates tables on startup
for local development; the Alembic migration is the authoritative schema.

For a zero-dependency tryout without PostgreSQL, set
`PRISM_DATABASE_URL=sqlite+pysqlite:///./prism.db` and skip the migration step.

The first search or upload downloads the local embedding model
(`all-MiniLM-L6-v2`, ~90 MB) via Sentence Transformers; after that it is
cached and works offline. No API keys or paid services are involved.

## Run

```bash
uvicorn app.main:app --reload
```

- API docs: http://127.0.0.1:8000/docs
- Health: `GET /health`

On startup Prism rebuilds the in-memory vector index from the database, so
uploaded documents remain searchable across restarts without re-embedding.

## Test

```bash
pytest                        # full suite (includes real-model integration tests)
pytest --ignore=tests/test_real_embeddings.py   # fast subset, fake embedder only
```

Tests run against a temporary SQLite database per test — no PostgreSQL needed
for the test suite.

## Benchmark

```bash
python scripts/run_retrieval_benchmark.py
python scripts/run_retrieval_benchmark.py --chunk-size 500 --chunk-overlap 50 --output report.json
```

The corpus (`benchmarks/retrieval/corpus.json`, 3 documents × 3 pages) and
questions (`benchmarks/retrieval/cases.json`, 6 cases with expected
`(filename, page_number)` evidence) are version-controlled. Metrics are
deterministic — Recall@k, Precision@k, MRR — never LLM judgments. Latest run
with defaults (`all-MiniLM-L6-v2`, chunk 1000/150):

| Metric | Value |
|--------|-------|
| MRR | 0.917 |
| Recall@1 | 0.833 |
| Recall@3 | 1.0 |
| Precision@1 | 0.833 |
| Mean query embedding | ~5.8 ms |
| Mean retrieval | ~0.05 ms |

## Configuration

Environment variables use the `PRISM_` prefix (or a `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `PRISM_APP_NAME` | `Prism` | Service name |
| `PRISM_ENVIRONMENT` | `development` | Environment label |
| `PRISM_API_V1_PREFIX` | `/api/v1` | API prefix |
| `PRISM_MAX_UPLOAD_BYTES` | `10485760` (10 MB) | Upload size limit |
| `PRISM_CHUNK_SIZE` | `1000` | Target characters per chunk |
| `PRISM_CHUNK_OVERLAP` | `150` | Characters shared between consecutive chunks (must be < chunk size) |
| `PRISM_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local Sentence Transformers model |
| `PRISM_EMBEDDING_BATCH_SIZE` | `32` | Texts embedded per batch |
| `PRISM_DEFAULT_TOP_K` | `5` | Results returned when the request omits `top_k` |
| `PRISM_DATABASE_URL` | `postgresql+psycopg://prism:prism@localhost:5432/prism` | SQLAlchemy database URL |
| `PRISM_API_KEYS` | `{}` | JSON object mapping API keys to owner ids, e.g. `{"dev-key":"dev-owner"}` |
| `PRISM_AUTO_CREATE_SCHEMA` | `true` | Create tables on startup (dev convenience; use `alembic upgrade head` for the migration workflow) |

## Authentication & ownership

Every document and search endpoint requires an `X-API-Key` header whose value
is a key in `PRISM_API_KEYS`. The key maps to an owner id; all documents are
scoped to that owner:

- `GET /api/v1/documents` lists only the caller's documents.
- `GET`/`DELETE /api/v1/documents/{id}` return `404` for documents owned by
  someone else (existence is not leaked).
- `POST /api/v1/search` only surfaces the caller's chunks (and the requested
  `collection_id` within them).
- Missing or unknown keys return `401 unauthorized`.

This is a pragmatic key→owner mapping for local/multi-user development, not a
full user-account system; production auth (tokens, users, rotation) is a later
phase.

## API

### `GET /health`

No authentication required.

```json
{"status": "ok", "service": "Prism", "version": "0.3.0", "environment": "development"}
```

### `POST /api/v1/documents`

Requires `X-API-Key`. Multipart upload (`file` field, optional `collection_id`
form field, default `"default"`). Text is extracted, chunked, embedded,
persisted to the database, and indexed synchronously; returns `201` with
document identity, page-level extracted text, chunks, and warnings.

```bash
curl -H "X-API-Key: dev-key" \
  -F "file=@sample.pdf;type=application/pdf" -F "collection_id=research" \
  http://127.0.0.1:8000/api/v1/documents
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
  "chunk_count": 2,
  "pages": [
    {"page_number": 1, "text": "…", "char_count": 35, "is_empty": false}
  ],
  "chunks": [
    {
      "id": "379ebe97…:0",
      "document_id": "379ebe97…",
      "page_number": 1,
      "chunk_index": 0,
      "text": "…",
      "char_count": 35,
      "start_offset": 0,
      "end_offset": 35
    }
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

Requires `X-API-Key`. Lists the caller's uploaded documents (id, filename,
size, page count, status).

### `GET /api/v1/documents/{id}`

Requires `X-API-Key`. Returns the full document record, or `404` if unknown or
owned by another caller.

### `DELETE /api/v1/documents/{id}`

Requires `X-API-Key`. Deletes the document (pages, chunks, and embeddings are
removed with it) and rebuilds the in-memory index from the database. Returns
`204` on success, `404` if unknown or owned by another caller.

### `POST /api/v1/search`

Requires `X-API-Key`. Semantic top-k search over the caller's indexed chunks.
Results carry cosine scores plus full source metadata (document, filename,
page number, chunk offsets).

```bash
curl -X POST http://127.0.0.1:8000/api/v1/search \
  -H "Content-Type: application/json" -H "X-API-Key: dev-key" \
  -d '{"query": "How does the Calvin cycle fix carbon?", "top_k": 3, "collection_id": "research"}'
```

Response (abridged):

```json
{
  "query": "How does the Calvin cycle fix carbon?",
  "top_k": 3,
  "collection_id": "research",
  "results": [
    {
      "score": 0.768,
      "document_id": "…",
      "filename": "plants.pdf",
      "chunk": {"page_number": 2, "chunk_index": 0, "text": "…", "start_offset": 0, "end_offset": 612}
    }
  ]
}
```

- `top_k` defaults to `PRISM_DEFAULT_TOP_K` (max 50).
- `collection_id` scopes the search within the caller's documents; chunks from
  other collections or other owners can never appear (tested end-to-end).

## Retrieval architecture

- **Embedder** (`app/embedding/`) — replaceable `Embedder` protocol; the
  default `LazySentenceTransformerEmbedder` loads a local MiniLM model lazily
  and shares it process-wide. Tests inject a deterministic hashing embedder.
- **Vector index** (`app/retrieval/index.py`) — `VectorIndex` protocol with a
  transparent NumPy baseline: L2-normalized float32 matrix, cosine similarity
  via dot product, stable score ordering, in-index `owner_id`/`collection_id`
  filtering. A FAISS (or other) backend can replace it without touching routes.
- **Persistence** (`app/repository/`) — `DocumentStore` protocol with a
  SQLAlchemy implementation (`SqlDocumentStore`). Documents, pages, chunks,
  and float32 embedding vectors are stored transactionally. On startup the
  vector index is rebuilt from the database; documents indexed with a
  different embedding model than the current configuration are skipped with a
  warning.
- **Search service** — `POST /api/v1/search` embeds the query and returns
  scored chunks with source metadata.

## Chunking

Text is chunked **per page** during upload, so every chunk maps to exactly one
page and citations can never be ambiguous.

- Sliding window of `chunk_size` characters with `chunk_overlap` characters of
  overlap between consecutive chunks.
- Cut points prefer paragraph breaks (`\n\n`), then line breaks, then sentence
  endings (`. ? !` followed by whitespace), then spaces — falling back to a
  hard cut only for unbreakable runs (e.g. long URLs).
- Cuts never fall below half the requested window, keeping chunks substantive.
- Chunks are exact slices of the page text: `page.text[start_offset:end_offset]
  == chunk.text` always holds. Nothing is trimmed, so offsets stay truthful.
- Blank pages produce no chunks, but later pages keep their true page numbers.
- Chunk ids are deterministic: `{document_id}:{chunk_index}`, with
  `chunk_index` sequential across the document in page order.

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
| 401 | `unauthorized` | Missing or unknown `X-API-Key` |
| 404 | `document_not_found` | Unknown document id, or owned by another caller |
| 413 | `file_too_large` | Exceeds `PRISM_MAX_UPLOAD_BYTES` (checked while streaming) |
| 415 | `unsupported_media_type` | Non-`.pdf` filename or non-PDF content type |
| 422 | `unreadable_pdf` | Corrupt file or document with no pages |
| 422 | `encrypted_pdf` | Password-protected PDF |
| 500 | `embedding_failed` | Embedding model failed to load or encode |

Framework-level validation failures (e.g. a missing `file` field) return
FastAPI's standard `422` detail format.

## Project structure

```
app/
  main.py                  # app factory (create_app), lifespan, module-level app
  config.py                # pydantic-settings configuration
  api/routes/              # HTTP layer only (thin)
    health.py
    documents.py
    search.py
  core/
    errors.py              # domain errors + exception handlers
    auth.py                # X-API-Key → owner resolution
  processing/              # pipeline, independent of HTTP
    validation.py          # filename/type/size/magic-byte checks
    extraction.py          # PyMuPDF page-level extraction
    chunking.py            # boundary-aware chunking with source offsets
  embedding/               # Embedder protocol + lazy SentenceTransformer backend
  retrieval/               # VectorIndex protocol + NumPy cosine index + rebuild
  evaluation/              # retrieval metrics + benchmark runner
  repository/              # DocumentStore protocol + SQLAlchemy impl + models
  schemas/                 # Pydantic response models
alembic/                   # migrations (alembic upgrade head)
benchmarks/retrieval/      # versioned benchmark corpus + cases (JSON)
scripts/                   # run_retrieval_benchmark.py CLI
tests/                     # pytest suite (100 tests)
```

## Security notes

- Filenames and MIME types are not trusted; content is verified via the
  `%PDF-` magic in the first 1024 bytes (per the PDF spec).
- Uploads are streamed and rejected as soon as they exceed the size limit.
- Encrypted and corrupt files fail with explicit, non-leaking error codes.
- Every document and search endpoint enforces API-key authentication and
  per-owner scoping; non-owned documents return `404`, not `403`.
- Extracted PDF text is treated as untrusted data (prompt-injection defenses
  become critical at Milestone 5 and are documented there).

## Known limitations

- The API-key → owner mapping is a development-grade stand-in for real user
  accounts; key rotation, expiry, and per-user management are not included.
- Embedding vectors are stored in the database and reloaded into the NumPy
  index at startup; deleting a document rebuilds the whole in-memory index
  (fine at MVP scale, not yet optimized for large corpora).
- If `PRISM_EMBEDDING_MODEL` changes after documents are indexed, documents
  from the old model are skipped at rebuild (their vectors do not match the
  new query space); re-upload or a future re-index tool is required.
- The upload route performs the database write synchronously inside the async
  handler (consistent with its already-blocking embedding work); async
  processing is a later concern.
- Chunks never span page boundaries; a sentence split across a page break is
  chunked separately on each page (keeps citations exact).
- Scanned/image-only PDFs extract zero text; OCR is a later phase.
- Pathological PDFs that hang the parser are not time-limited yet.
- No maximum page-count guard yet.

## Development rules

- Tests run with `pytest` from the repository root; keep them green.
- The processing pipeline must stay independent of FastAPI routes.
- Schema changes go through Alembic (`alembic revision --autogenerate`).
- No secrets in source control; configuration is environment-based.
