import numpy as np
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.retrieval import NumpyVectorIndex, rebuild_index
from tests.conftest import TEST_API_KEY, TEST_OWNER
from tests.embedding_stub import HashingEmbedder
from tests.pdf_helpers import make_pdf

PLANT_PAGES = [
    "Chlorophyll in the thylakoid membrane absorbs light and drives the electron transport chain, producing ATP and NADPH while splitting water and releasing oxygen.",
    "The Calvin cycle fixes carbon dioxide in the stroma using RuBisCO, building sugar molecules from carbon using the ATP and NADPH made by the light reactions.",
]


def build_app(db_url: str):
    settings = Settings(
        max_upload_bytes=1_000_000,
        database_url=db_url,
        api_keys={TEST_API_KEY: TEST_OWNER},
    )
    app = create_app(settings, embedder=HashingEmbedder())
    client = TestClient(app)
    client.headers.update({"X-API-Key": TEST_API_KEY})
    return app, client


def upload(client: TestClient, filename: str, pages: list[str]) -> str:
    response = client.post(
        "/api/v1/documents",
        files={"file": (filename, make_pdf(pages), "application/pdf")},
    )
    assert response.status_code == 201
    return response.json()["id"]


def test_documents_survive_restart(db_url):
    _, client = build_app(db_url)
    with client:
        doc_id = upload(client, "plants.pdf", PLANT_PAGES)
        listing_before = client.get("/api/v1/documents").json()

    # Simulate a restart: new app instance, same database.
    _, restarted = build_app(db_url)
    with restarted:
        listing_after = restarted.get("/api/v1/documents").json()
        fetched = restarted.get(f"/api/v1/documents/{doc_id}")

    assert listing_after == listing_before
    assert fetched.status_code == 200
    assert fetched.json()["id"] == doc_id
    assert len(fetched.json()["pages"]) == 2


def test_search_works_after_restart_without_reembedding(db_url):
    _, client = build_app(db_url)
    with client:
        upload(client, "plants.pdf", PLANT_PAGES)
        before = client.post(
            "/api/v1/search", json={"query": "Calvin cycle carbon dioxide", "top_k": 3}
        ).json()["results"]
        assert before

    _, restarted = build_app(db_url)
    with restarted:
        after = restarted.post(
            "/api/v1/search", json={"query": "Calvin cycle carbon dioxide", "top_k": 3}
        ).json()["results"]

    assert after
    assert after[0]["filename"] == "plants.pdf"
    assert [r["chunk"]["id"] for r in after] == [r["chunk"]["id"] for r in before]
    assert after[0]["score"] == before[0]["score"]


def test_blank_document_survives_restart_without_breaking_search(db_url):
    _, client = build_app(db_url)
    with client:
        upload(client, "blank.pdf", ["", ""])

    _, restarted = build_app(db_url)
    with restarted:
        response = restarted.post("/api/v1/search", json={"query": "anything"})

    assert response.status_code == 200
    assert response.json()["results"] == []


def test_rebuild_index_skips_model_mismatch():
    from app.repository.record import DocumentRecord
    from app.schemas.documents import Chunk, PageExtraction

    def record(doc_id: str, model: str) -> DocumentRecord:
        return DocumentRecord(
            id=doc_id,
            filename=f"{doc_id}.pdf",
            content_type="application/pdf",
            size_bytes=1,
            page_count=1,
            extracted_char_count=4,
            pages=[PageExtraction(page_number=1, text="text", char_count=4, is_empty=False)],
            chunks=[
                Chunk(
                    id=f"{doc_id}:0",
                    document_id=doc_id,
                    page_number=1,
                    chunk_index=0,
                    text="text",
                    char_count=4,
                    start_offset=0,
                    end_offset=4,
                )
            ],
            warnings=[],
            owner_id="o",
            embedding_model=model,
        )

    index = NumpyVectorIndex()
    count = rebuild_index(
        index,
        [(record("matching", "model-v1"), np.ones((1, 3), dtype=np.float32)),
         (record("stale", "model-v0"), np.ones((1, 3), dtype=np.float32))],
        expected_model="model-v1",
    )

    assert count == 1
    assert index.size() == 1
