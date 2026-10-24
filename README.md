# Prism

> Find the signal in your documents.

Prism is a document intelligence system that answers questions about uploaded
PDF documents using evidence retrieved from the original sources. It is a
complete vertical slice: ingestion, page-level extraction, boundary-aware
chunking, local embeddings, a transparent NumPy vector index, PostgreSQL
persistence with per-owner isolation, grounded answer generation with citation
validation, a retrieval/answer evaluation lab, and a Next.js interface.

All six MVP milestones are complete. The evaluation lab reports deterministic
retrieval and citation metrics separately from an optional, explicitly
subjective LLM judge.

## Status

| Milestone | Description | Status |
|-----------|-------------|--------|
| 1 | FastAPI app, health endpoint, PDF upload + page-level extraction, validation, tests | **Done** |
| 2 | Chunking with page metadata + edge-case tests | **Done** |
| 3 | Local embeddings, NumPy cosine retrieval, retrieval benchmark | **Done** |
| 4 | PostgreSQL persistence + ownership/access control | **Done** |
| 5 | Grounded answer generation + citations | **Done** |
| 6 | Next.js UI, evaluation lab, Docker, documentation | **Done** |

## What it does

1. **Ingests** a PDF after filename, MIME, size, and `%PDF-` magic-byte checks.
2. **Extracts** text page by page with PyMuPDF and reports empty/failed pages.
3. **Chunks** each page with a sliding window that prefers paragraphs, lines,
   sentences, and spaces over hard cuts. Every chunk knows its document, page,
   and exact character offsets.
4. **Embeds** chunks locally with Sentence Transformers (`all-MiniLM-L6-v2` by
   default) and stores normalized vectors in PostgreSQL.
5. **Retrieves** the top-k chunks with cosine similarity inside a NumPy index,
   filtered by owner and collection so unrelated documents never mix.
6. **Answers** from retrieved passages only, instructing the model to refuse
   when evidence is insufficient and treating document text as untrusted data.
7. **Validates** every citation marker against the passages that were actually
   retrieved; fabricated markers are reported and never shown as sources.
8. **Evaluates** retrieval (Recall@k, Precision@k, MRR) and answers (citation
   validity/precision/recall, structural grounding proxies, optional LLM judge,
   latency, tokens, and optional cost) on a versioned benchmark.
9. **Exposes** the library, ask interface, source viewer, and evaluation
   dashboard in a responsive, accessible Next.js UI.

## Architecture

```
PDF upload → validation → page extraction → chunking → local embeddings
    → PostgreSQL + in-memory NumPy index

question → query embedding → cosine top-k → optional LLM answer
    → citation validation → evidence-backed UI

benchmark corpus/cases → retrieval metrics + answer metrics
    → JSON run reports (reproducible, owner-scoped)
```

The processing, embedding, retrieval, answer, and evaluation modules do not
depend on FastAPI or the frontend. HTTP routes are thin adapters.

## Requirements

- Python 3.14+ (developed and tested on 3.14.5)
- Node.js 22+ and npm
- PostgreSQL 14+ (or SQLite for a zero-dependency tryout)
- Docker and Docker Compose (optional)

## Quick start with Docker

```bash
cp .env.example .env
# Edit PRISM_API_KEYS before sharing the instance. Optionally set an LLM.
docker compose up --build
```

- UI: http://localhost:3000
- API docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

The backend container runs `alembic upgrade head` on startup. The embedding
model and evaluation reports are kept in named volumes, so restarts do not
re-download the model or lose saved runs. The first upload or search downloads
the local embedding model (~90 MB).

`PRISM_PUBLIC_API_URL` controls the browser-visible API URL baked into the
frontend build (`http://localhost:8000` by default). `PRISM_CORS_ORIGINS` is a
JSON array of allowed browser origins.

## Local development

### Backend

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-ml.txt   # embeddings/search (Sentence Transformers, PyTorch)
cp .env.example .env                 # set PRISM_API_KEYS and PRISM_DATABASE_URL
```

Start PostgreSQL (Docker example):

```bash
docker run --name prism-postgres -e POSTGRES_USER=prism \
  -e POSTGRES_PASSWORD=prism -e POSTGRES_DB=prism \
  -p 5432:5432 -d postgres:16
```

Apply migrations and run:

```bash
alembic upgrade head
uvicorn app.main:app --reload
```

For a zero-dependency tryout without PostgreSQL, set
`PRISM_DATABASE_URL=sqlite+pysqlite:///./prism.db` and skip the migration step
(`PRISM_AUTO_CREATE_SCHEMA=true` creates tables on startup).

On startup Prism rebuilds the in-memory vector index from the database, so
uploaded documents remain searchable across restarts without re-embedding.
Documents indexed with a different embedding model than the current
configuration are skipped with a warning.

### Frontend

```bash
cd frontend
npm install
cp .env.example .env.local   # NEXT_PUBLIC_PRISM_API_URL
npm run dev
```

Open http://localhost:3000 and use **Set API key** in the header. The key is
sent as `X-API-Key` and stored only in that browser's `localStorage`.

## Test and build

```bash
# Backend: 152 tests, including real-model integration tests
pytest

# Fast subset with the deterministic hashing embedder
pytest --ignore=tests/test_real_embeddings.py

# Frontend type check and production build
cd frontend
npm run typecheck
npm run build
```

Tests run against a temporary SQLite database per test, so PostgreSQL is not
needed for the suite. Evaluation API tests use a temporary results directory.

## Benchmark

```bash
python scripts/run_retrieval_benchmark.py
python scripts/run_retrieval_benchmark.py --chunk-size 500 --chunk-overlap 50 --output report.json
```

The corpus (`benchmarks/retrieval/corpus.json`, 3 documents × 3 pages) and
questions (`benchmarks/retrieval/cases.json`, 6 cases with expected
`(filename, page_number)` evidence) are version-controlled. Metrics are
deterministic — Recall@k, Precision@k, MRR — never LLM judgments.

Latest local run with defaults (`all-MiniLM-L6-v2`, chunk 1000/150, 9 chunks):

| Metric | Value |
|--------|-------|
| MRR | 0.917 |
| Recall@1 | 0.833 |
| Recall@3 | 1.0 |
| Recall@5 | 1.0 |
| Precision@1 | 0.833 |
| Precision@3 | 0.333 |
| Mean query embedding | ~5.8 ms |
| Mean retrieval | ~0.06 ms |

Timings are wall-clock values and vary between machines and runs.

## Configuration

Environment variables use the `PRISM_` prefix (or a `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `PRISM_APP_NAME` | `Prism` | Service name |
| `PRISM_ENVIRONMENT` | `development` | Environment label |
| `PRISM_API_V1_PREFIX` | `/api/v1` | API prefix |
| `PRISM_CORS_ORIGINS` | `["http://localhost:3000"]` | JSON array of allowed browser origins |
| `PRISM_MAX_UPLOAD_BYTES` | `10485760` (10 MB) | Upload size limit |
| `PRISM_CHUNK_SIZE` | `1000` | Target characters per chunk |
| `PRISM_CHUNK_OVERLAP` | `150` | Characters shared between consecutive chunks (must be < chunk size) |
| `PRISM_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local Sentence Transformers model |
| `PRISM_EMBEDDING_BATCH_SIZE` | `32` | Texts embedded per batch |
| `PRISM_DEFAULT_TOP_K` | `5` | Results returned when the request omits `top_k` |
| `PRISM_DATABASE_URL` | `postgresql+psycopg://prism:prism@localhost:5432/prism` | SQLAlchemy database URL |
| `PRISM_API_KEYS` | `{}` | JSON object mapping API keys to owner ids, e.g. `{"dev-key":"dev-owner"}` |
| `PRISM_AUTO_CREATE_SCHEMA` | `true` | Create tables on startup (dev convenience; Alembic is authoritative) |
| `PRISM_LLM_BASE_URL` | _(empty)_ | OpenAI-compatible endpoint base URL (e.g. `https://api.openai.com/v1`, `http://localhost:11434/v1` for Ollama). Empty disables answer generation |
| `PRISM_LLM_API_KEY` | _(empty)_ | Bearer key for the LLM endpoint (optional for local servers) |
| `PRISM_LLM_MODEL` | `gpt-4o-mini` | Model name sent to the endpoint |
| `PRISM_LLM_MAX_TOKENS` | `512` | Max completion tokens |
| `PRISM_LLM_TEMPERATURE` | `0.0` | Sampling temperature (0 for determinism) |
| `PRISM_LLM_TIMEOUT_SECONDS` | `60.0` | HTTP timeout for LLM calls |
| `PRISM_LLM_INPUT_COST_PER_1M` | `0` | Optional input price (USD per 1M tokens) for evaluation cost estimates |
| `PRISM_LLM_OUTPUT_COST_PER_1M` | `0` | Optional output price (USD per 1M tokens) for evaluation cost estimates |
| `PRISM_EVALUATION_RESULTS_DIR` | `benchmarks/results` | Directory for saved evaluation run JSON |
| `PRISM_EVALUATION_MAX_CONFIGURATIONS` | `8` | Maximum retrieval configurations per matrix run |

Compose-only variables: `POSTGRES_PASSWORD` (default `prism`) and
`PRISM_PUBLIC_API_URL` (default `http://localhost:8000`). The frontend's
`NEXT_PUBLIC_PRISM_API_URL` is a build-time public setting, not a secret.

## Authentication & ownership

Every document, search, answer, and evaluation endpoint requires an `X-API-Key`
header whose value is a key in `PRISM_API_KEYS`. The key maps to an owner id;
all documents and saved evaluation runs are scoped to that owner:

- `GET /api/v1/documents` lists only the caller's documents.
- `GET`/`DELETE /api/v1/documents/{id}` return `404` for documents owned by
  someone else (existence is not leaked).
- `POST /api/v1/search` and `POST /api/v1/answer` only surface the caller's
  chunks (and the requested `collection_id` within them).
- Evaluation runs are listed and fetched only for their owner.
- Missing or unknown keys return `401 unauthorized`.

This is a pragmatic key→owner mapping for local/multi-user development, not a
full account system; production auth (tokens, users, rotation) is a later phase.
The Next.js UI keeps the key in browser storage and never hardcodes one.

## API

### `GET /health`

No authentication required.

```json
{"status": "ok", "service": "Prism", "version": "0.6.0", "environment": "development"}
```

### `POST /api/v1/documents`

Requires `X-API-Key`. Multipart upload (`file` field, optional `collection_id`
form field, default `"default"`). Text is extracted, chunked, embedded,
persisted, and indexed synchronously; returns `201` with document identity,
page-level text, chunks, and warnings.

```bash
curl -H "X-API-Key: dev-key" \
  -F "file=@sample.pdf;type=application/pdf" -F "collection_id=research" \
  http://127.0.0.1:8000/api/v1/documents
```

Warnings:

- `no_text_extracted` — every page produced no text (typical of scanned PDFs;
  OCR is a later phase).
- `extraction_failed_pages:2,5` — extraction failed on specific pages; those
  pages are returned empty.

### `GET /api/v1/documents`, `GET /api/v1/documents/{id}`

List the caller's documents or fetch one full record (`404` if unknown or not
owned).

### `DELETE /api/v1/documents/{id}`

Deletes the document (pages, chunks, embeddings) and rebuilds the in-memory
index. `204` on success, `404` if unknown or not owned.

### `POST /api/v1/search`

Requires `X-API-Key`. Returns top-k chunks with cosine scores and full source
metadata.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/search \
  -H "Content-Type: application/json" -H "X-API-Key: dev-key" \
  -d '{"query": "How does the Calvin cycle fix carbon?", "top_k": 3, "collection_id": "research"}'
```

`top_k` defaults to `PRISM_DEFAULT_TOP_K` (max 50). Chunks from other
collections or owners never appear.

### `POST /api/v1/answer`

Requires `X-API-Key` and a configured LLM endpoint (`PRISM_LLM_BASE_URL`);
returns `503 answer_generation_unavailable` when no LLM is configured.

Retrieves top-k passages, builds a grounding prompt, and generates an answer
through the OpenAI-compatible provider. Citation markers like `[1]` are
validated against the actual retrieved passages. `citations` contains only
markers that map to a real retrieved chunk; markers that do not map appear in
`invalid_citation_markers`.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/answer \
  -H "Content-Type: application/json" -H "X-API-Key: dev-key" \
  -d '{"question": "How does the Calvin cycle fix carbon?", "top_k": 3}'
```

Response (abridged):

```json
{
  "question": "How does the Calvin cycle fix carbon?",
  "answer": "The Calvin cycle fixes carbon dioxide using RuBisCO [1].",
  "insufficient_evidence": false,
  "citations": [
    {
      "marker": 1,
      "chunk_id": "…:0",
      "document_id": "…",
      "filename": "plants.pdf",
      "page_number": 2,
      "text": "…",
      "score": 0.768
    }
  ],
  "invalid_citation_markers": [],
  "evidence": [
    {"score": 0.768, "document_id": "…", "filename": "plants.pdf", "chunk": {"page_number": 2, "text": "…"}}
  ],
  "top_k": 3,
  "collection_id": "default",
  "model": "gpt-4o-mini",
  "input_tokens": 812,
  "output_tokens": 42
}
```

Insufficient-evidence behavior (tested):

- No passages retrieved → deterministic `"Insufficient evidence."` answer
  **without calling the LLM** (`insufficient_evidence: true`).
- The model is instructed to reply exactly `"Insufficient evidence."` when the
  passages do not answer the question; such replies are flagged with empty
  citations.
- Empty model replies are treated as insufficient.

### Evaluation lab

All evaluation endpoints require `X-API-Key`.

#### `GET /api/v1/evaluation/benchmark`

Returns the versioned corpus metadata (filenames, page counts) and benchmark
questions with expected `(filename, page_number)` evidence.

#### `POST /api/v1/evaluation/retrieval`

Runs one or more configurations over the benchmark and persists a reproducible
report by default.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/evaluation/retrieval \
  -H "Content-Type: application/json" -H "X-API-Key: dev-key" \
  -d '{
    "configurations": [
      {"chunk_size": 500, "chunk_overlap": 50},
      {"chunk_size": 1000, "chunk_overlap": 150}
    ],
    "k_values": [1, 3, 5]
  }'
```

Each result contains the configuration, aggregate Recall@k/Precision@k/MRR,
per-question metrics, top results, and timings. `comparison` is a flat table for
the dashboard. `persist: false` returns the report without writing it.

#### `POST /api/v1/evaluation/answers`

Generates an answer for every benchmark question and scores it. Requires a
configured LLM (`503 answer_generation_unavailable` otherwise).

```bash
curl -X POST http://127.0.0.1:8000/api/v1/evaluation/answers \
  -H "Content-Type: application/json" -H "X-API-Key: dev-key" \
  -d '{"chunk_size": 1000, "chunk_overlap": 150, "top_k": 5, "include_llm_judge": false}'
```

Reports remain available when individual generations fail: failures are counted
in `aggregate.generation_errors` and recorded per case. Token usage is reported;
`estimated_cost_usd` is `null` unless pricing is configured.

#### `GET /api/v1/evaluation/runs`, `GET /api/v1/evaluation/runs/{id}`

List or fetch saved runs for the caller. Results are JSON files under
`PRISM_EVALUATION_RESULTS_DIR` with a generated `id` and `created_at` timestamp.
Unknown, malformed, or non-owned ids return `404 evaluation_run_not_found`.

### Error model

Prism errors use a consistent envelope:

```json
{"error": {"code": "invalid_pdf", "message": "File content does not contain a valid PDF header."}}
```

| Status | Code | Trigger |
|--------|------|---------|
| 400 | `invalid_request` | Missing/blank filename or question, empty collection id, too many configurations |
| 400 | `empty_file` | Zero-byte upload |
| 400 | `invalid_pdf` | No `%PDF-` header in the first 1024 bytes |
| 401 | `unauthorized` | Missing or unknown `X-API-Key` |
| 404 | `document_not_found` | Unknown document id, or owned by another caller |
| 404 | `evaluation_run_not_found` | Unknown, malformed, or non-owned evaluation run id |
| 413 | `file_too_large` | Exceeds `PRISM_MAX_UPLOAD_BYTES` (checked while streaming) |
| 415 | `unsupported_media_type` | Non-`.pdf` filename or non-PDF content type |
| 422 | `unreadable_pdf` | Corrupt file or document with no pages |
| 422 | `encrypted_pdf` | Password-protected PDF |
| 500 | `embedding_failed` | Embedding model failed to load or encode |
| 502 | `answer_generation_failed` | LLM endpoint failed or returned a malformed response |
| 503 | `answer_generation_unavailable` | No LLM configured (`PRISM_LLM_BASE_URL` empty) |

Framework-level validation failures (e.g. a missing `file` field, invalid
chunk/overlap combinations) return FastAPI's standard `422` detail format.

## Evaluation methodology

The lab deliberately separates deterministic measurements from subjective
model judgments. The returned `method_notes` spell this out in every report.

**Retrieval (deterministic).** A retrieved chunk is relevant iff its
`(filename, page_number)` appears in the case's expected set. Recall@k is the
share of expected keys found in the top-k; Precision@k is the share of top-k
results that are relevant; MRR is the reciprocal rank of the first relevant
result. Corpus, questions, and expected evidence are version-controlled JSON.

**Citations (deterministic).** Citation validity is the share of `[n]` markers
that map to a retrieved passage. Citation precision is the share of uniquely
cited evidence keys that are expected; citation recall is the share of expected
keys that were cited. Invalid markers are counted and never presented as
sources.

**Grounding proxies (deterministic, not semantic proof).** For each substantive
answer sentence, Prism checks whether it contains a valid citation and whether
its content tokens overlap the cited passage above a documented threshold
(`0.5`). `lexical_grounding_rate` is the share of cited sentences that pass;
`uncited_sentence_rate` is the share of sentences with no valid citation. These
are structural heuristics that catch obvious uncited claims and disconnected
citations. They do **not** prove entailment or factual faithfulness.

**LLM judge (optional, subjective).** When `include_llm_judge` is true, the
configured model is asked for a JSON judgement of whether the answer is
supported by the evidence. This is labelled separately, may be self-judging
when the same model generated the answer, and can fail parsing; failures are
reported instead of being coerced into a score. It is not ground truth.

**Reproducibility.** Corpus and cases are versioned; embeddings and scoring are
deterministic; reports record model names, chunk settings, k values, timings,
usage, and timestamps. Timings vary by machine and run. Stored reports are JSON
and owner-scoped. Use the same dependency versions and corpus revision to
compare runs.

## Retrieval architecture

- **Embedder** (`app/embedding/`) — replaceable `Embedder` protocol; the default
  `LazySentenceTransformerEmbedder` loads a local MiniLM model lazily and shares
  it process-wide. Tests inject a deterministic hashing embedder.
- **Vector index** (`app/retrieval/index.py`) — `VectorIndex` protocol with a
  transparent NumPy baseline: L2-normalized float32 matrix, cosine similarity
  via dot product, stable score ordering, in-index `owner_id`/`collection_id`
  filtering. A FAISS backend can replace it without touching routes.
- **Persistence** (`app/repository/`) — `DocumentStore` protocol with a
  SQLAlchemy implementation. Documents, pages, chunks, and float32 embedding
  vectors are stored transactionally. On startup the index is rebuilt from the
  database.
- **Evaluation** (`app/evaluation/`) — metrics, retrieval runner, answer
  runner, and the file-backed run store.

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

## Frontend

`frontend/` is a Next.js App Router application using TypeScript and Tailwind
CSS.

- **Library** (`/`) — upload with processing status, warnings, and document
  deletion.
- **Document view** (`/documents/[id]`) — page tabs, extracted text, chunk
  offsets, and extraction warnings.
- **Ask** (`/ask`) — grounded answers or retrieval-only mode, citation cards,
  invalid-marker warnings, and a retrieved-passage viewer with scores and chunk
  offsets.
- **Evaluation lab** (`/evaluation`) — benchmark metadata, retrieval
  configuration matrix, comparison table, answer/citation metrics, methodology
  notes, per-case details, and saved runs.

Loading, empty, error, and success states are explicit. Controls have labels and
focus styles; the layout is responsive.

## Project structure

```
app/
  main.py                  # app factory (create_app), lifespan, CORS, routers
  config.py                # pydantic-settings configuration
  api/routes/              # HTTP layer only (thin)
    health.py
    documents.py
    search.py
    answer.py
    evaluation.py
  core/
    errors.py              # domain errors + exception handlers
    auth.py                # X-API-Key → owner resolution
  processing/              # pipeline, independent of HTTP
    validation.py          # filename/type/size/magic-byte checks
    extraction.py          # PyMuPDF page-level extraction
    chunking.py            # boundary-aware chunking with source offsets
  embedding/               # Embedder protocol + lazy SentenceTransformer backend
  llm/                     # LLMProvider protocol + OpenAI-compatible client
  answer/                  # prompts, citation validation (no HTTP)
  retrieval/               # VectorIndex protocol + NumPy cosine index + rebuild
  evaluation/
    metrics.py             # Recall@k, Precision@k, MRR
    benchmark.py           # retrieval benchmark runner
    answer_eval.py         # citation/grounding metrics + optional LLM judge
    store.py               # JSON run persistence, owner-scoped ids
  repository/              # DocumentStore protocol + SQLAlchemy impl + models
  schemas/                 # Pydantic request/response models
alembic/                   # migrations (alembic upgrade head)
benchmarks/retrieval/      # versioned benchmark corpus + cases (JSON)
scripts/                   # run_retrieval_benchmark.py CLI
frontend/                  # Next.js App Router UI
tests/                     # pytest suite (152 tests)
Dockerfile                 # backend image (CPU torch)
docker-compose.yml         # db + backend + frontend
```

## Security notes

- Filenames and MIME types are not trusted; content is verified via the
  `%PDF-` magic in the first 1024 bytes (per the PDF spec).
- Uploads are streamed and rejected as soon as they exceed the size limit.
- Encrypted and corrupt files fail with explicit, non-leaking error codes.
- Every document, search, answer, and evaluation endpoint enforces API-key
  authentication and per-owner scoping; non-owned resources return `404`.
- CORS allows only configured browser origins and only the headers/methods the
  UI needs.
- Extracted PDF text is treated as untrusted data in answer and judge prompts:
  passages are delimited in `<evidence>` tags and the system prompt instructs
  the model never to follow instructions inside them. This raises the cost of
  document-based prompt injection but is not a formal guarantee — treat LLM
  output as untrusted as well.
- The UI stores the API key in `localStorage`, which is readable by any script
  on the page. For production, serve the UI and API over HTTPS, avoid untrusted
  third-party scripts, and plan a real session-based auth system.
- Evaluation run ids are validated against a safe charset before filesystem
  access, preventing path traversal.

## Known limitations

- Citation validation confirms that `[n]` markers map to actually-retrieved
  passages; deterministic grounding metrics are lexical proxies, not semantic
  entailment. Only the optional LLM judge attempts semantic faithfulness, and it
  is subjective.
- Insufficient-evidence detection relies on the model following the system
  prompt (plus a deterministic no-retrieval path); it is not a proof of
  groundedness.
- LLM token usage is surfaced in answer responses and evaluation reports. Cost
  is estimated only when input/output token prices are configured.
- The API-key → owner mapping is a development-grade stand-in for real user
  accounts; key rotation, expiry, and per-user management are not included.
- Embedding vectors are stored in the database and reloaded into the NumPy
  index at startup; deleting a document rebuilds the whole in-memory index
  (fine at MVP scale, not yet optimized for large corpora).
- If `PRISM_EMBEDDING_MODEL` changes after documents are indexed, documents
  from the old model are skipped at rebuild; re-upload or a future re-index tool
  is required.
- The upload route performs the database write synchronously inside the async
  handler (consistent with its already-blocking embedding work); async
  processing is a later concern.
- Chunks never span page boundaries; a sentence split across a page break is
  chunked separately on each page (keeps citations exact).
- Scanned/image-only PDFs extract zero text; OCR is a later phase.
- Pathological PDFs that hang the parser are not time-limited yet, and there is
  no maximum page-count guard.
- Evaluation model comparison loads each requested Sentence Transformers model
  into the backend process; this is intended for a local lab, not concurrent
  multi-tenant use.
- The Docker images are provided for local deployment; they have not been
  hardened for public internet exposure.

## Development rules

- Tests run with `pytest` from the repository root; keep them green.
- The processing pipeline must stay independent of FastAPI routes.
- Schema changes go through Alembic (`alembic revision --autogenerate`).
- No secrets in source control; configuration is environment-based.
- Do not present subjective LLM judgments as deterministic ground truth.
