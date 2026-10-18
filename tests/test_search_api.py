from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.conftest import TEST_API_KEY, TEST_OWNER
from tests.embedding_stub import FailingEmbedder, HashingEmbedder
from tests.pdf_helpers import make_pdf

PLANT_PAGES = [
    "Chlorophyll in the thylakoid membrane absorbs light and drives the electron transport chain, producing ATP and NADPH while splitting water and releasing oxygen.",
    "The Calvin cycle fixes carbon dioxide in the stroma using RuBisCO, building sugar molecules from carbon using the ATP and NADPH made by the light reactions.",
]
PASTA_PAGES = [
    "Salt the pasta water generously to about ten grams per liter so the pasta seasons evenly while it absorbs water during boiling.",
]


def upload(client: TestClient, filename: str, pages: list[str], collection_id: str | None = None):
    data = make_pdf(pages)
    files = {"file": (filename, data, "application/pdf")}
    form = {"collection_id": collection_id} if collection_id else None
    return client.post("/api/v1/documents", files=files, data=form)


def test_search_returns_ranked_results_with_source_metadata(client):
    upload(client, "plants.pdf", PLANT_PAGES)
    upload(client, "pasta.pdf", PASTA_PAGES)

    response = client.post(
        "/api/v1/search",
        json={"query": "Calvin cycle fixes carbon dioxide into sugar", "top_k": 3},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "Calvin cycle fixes carbon dioxide into sugar"
    assert body["top_k"] == 3
    assert len(body["results"]) == 3
    scores = [result["score"] for result in body["results"]]
    assert scores == sorted(scores, reverse=True)
    top = body["results"][0]
    assert top["filename"] == "plants.pdf"
    assert top["chunk"]["page_number"] == 2
    assert top["document_id"] == top["chunk"]["document_id"]
    assert top["chunk"]["text"]


def test_search_collection_isolation(client):
    upload(client, "alpha.pdf", ["Alpha collection talks about glaciers and ice sheets."], "alpha")
    upload(client, "beta.pdf", ["Beta collection talks about volcanoes and magma."], "beta")

    response = client.post(
        "/api/v1/search",
        json={"query": "glaciers ice", "top_k": 5, "collection_id": "alpha"},
    )

    assert response.status_code == 200
    filenames = {result["filename"] for result in response.json()["results"]}
    assert filenames == {"alpha.pdf"}


def test_search_unknown_collection_returns_empty(client):
    upload(client, "alpha.pdf", ["Alpha collection content about glaciers."], "alpha")

    response = client.post(
        "/api/v1/search",
        json={"query": "glaciers", "collection_id": "nobody-here"},
    )

    assert response.status_code == 200
    assert response.json()["results"] == []


def test_search_on_empty_index_returns_empty(client):
    response = client.post("/api/v1/search", json={"query": "anything"})

    assert response.status_code == 200
    assert response.json()["results"] == []


def test_search_rejects_blank_query(client):
    response = client.post("/api/v1/search", json={"query": "   "})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


def test_search_uses_default_top_k(client):
    upload(client, "plants.pdf", PLANT_PAGES)

    response = client.post("/api/v1/search", json={"query": "carbon dioxide sugar"})

    assert response.json()["top_k"] == Settings().default_top_k


def test_upload_rejects_blank_collection_id(client):
    data = make_pdf(["text"])
    response = client.post(
        "/api/v1/documents",
        files={"file": ("a.pdf", data, "application/pdf")},
        data={"collection_id": "   "},
    )

    assert response.status_code == 400


def test_upload_embedding_failure_returns_500(db_url):
    settings = Settings(
        max_upload_bytes=1_000_000,
        database_url=db_url,
        api_keys={TEST_API_KEY: TEST_OWNER},
    )
    with TestClient(create_app(settings, embedder=FailingEmbedder())) as app_client:
        app_client.headers.update({"X-API-Key": TEST_API_KEY})
        response = app_client.post(
            "/api/v1/documents",
            files={"file": ("a.pdf", make_pdf(["text"]), "application/pdf")},
        )

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "embedding_failed"


def test_search_embedding_failure_returns_500(db_url):
    settings = Settings(
        database_url=db_url,
        api_keys={TEST_API_KEY: TEST_OWNER},
    )
    with TestClient(create_app(settings, embedder=FailingEmbedder())) as app_client:
        app_client.headers.update({"X-API-Key": TEST_API_KEY})
        response = app_client.post("/api/v1/search", json={"query": "anything"})

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "embedding_failed"


def test_blank_document_upload_succeeds_without_indexing(client):
    response = upload(client, "blank.pdf", ["", ""])

    assert response.status_code == 201
    assert response.json()["chunk_count"] == 0
