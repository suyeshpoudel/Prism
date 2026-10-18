from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.conftest import (
    SECOND_OWNER,
    SECOND_OWNER_KEY,
    TEST_API_KEY,
    TEST_OWNER,
)
from tests.embedding_stub import HashingEmbedder
from tests.pdf_helpers import make_pdf


def build_client(db_url: str, api_key: str) -> TestClient:
    settings = Settings(
        max_upload_bytes=1_000_000,
        database_url=db_url,
        api_keys={TEST_API_KEY: TEST_OWNER, SECOND_OWNER_KEY: SECOND_OWNER},
    )
    client = TestClient(create_app(settings, embedder=HashingEmbedder()))
    client.headers.update({"X-API-Key": api_key})
    return client


def upload(client: TestClient, filename: str, text: str) -> str:
    response = client.post(
        "/api/v1/documents",
        files={"file": (filename, make_pdf([text]), "application/pdf")},
    )
    assert response.status_code == 201
    return response.json()["id"]


def test_list_is_scoped_to_owner(db_url):
    with build_client(db_url, TEST_API_KEY) as alice, build_client(db_url, SECOND_OWNER_KEY) as bob:
        alice_doc = upload(alice, "alice.pdf", "Alice secret research about glaciers.")
        upload(bob, "bob.pdf", "Bob secret research about volcanoes.")

        alice_listing = alice.get("/api/v1/documents").json()
        bob_listing = bob.get("/api/v1/documents").json()

    assert [d["id"] for d in alice_listing] == [alice_doc]
    assert all(d["filename"] != "bob.pdf" for d in alice_listing)
    assert all(d["filename"] != "alice.pdf" for d in bob_listing)


def test_get_other_owners_document_returns_404(db_url):
    with build_client(db_url, TEST_API_KEY) as alice, build_client(db_url, SECOND_OWNER_KEY) as bob:
        alice_doc = upload(alice, "alice.pdf", "Alice private notes.")

        response = bob.get(f"/api/v1/documents/{alice_doc}")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "document_not_found"


def test_delete_other_owners_document_returns_404_and_preserves_document(db_url):
    with build_client(db_url, TEST_API_KEY) as alice, build_client(db_url, SECOND_OWNER_KEY) as bob:
        alice_doc = upload(alice, "alice.pdf", "Alice private notes.")

        response = bob.delete(f"/api/v1/documents/{alice_doc}")
        assert response.status_code == 404

        # Owner can still fetch it afterwards.
        assert alice.get(f"/api/v1/documents/{alice_doc}").status_code == 200


def test_search_does_not_cross_owners(db_url):
    with build_client(db_url, TEST_API_KEY) as alice, build_client(db_url, SECOND_OWNER_KEY) as bob:
        upload(alice, "alice.pdf", "Glacier ice sheet research by Alice.")
        upload(bob, "bob.pdf", "Glacier ice sheet research by Bob.")

        alice_results = alice.post(
            "/api/v1/search", json={"query": "glacier ice", "top_k": 10}
        ).json()["results"]
        bob_results = bob.post(
            "/api/v1/search", json={"query": "glacier ice", "top_k": 10}
        ).json()["results"]

    assert {r["filename"] for r in alice_results} == {"alice.pdf"}
    assert {r["filename"] for r in bob_results} == {"bob.pdf"}


def test_owner_delete_reindexes_so_document_disappears_from_search(db_url):
    with build_client(db_url, TEST_API_KEY) as alice:
        doc_id = upload(alice, "alice.pdf", "Glacier ice sheet research by Alice.")

        before = alice.post("/api/v1/search", json={"query": "glacier ice"}).json()["results"]
        assert before

        response = alice.delete(f"/api/v1/documents/{doc_id}")
        assert response.status_code == 204

        after = alice.post("/api/v1/search", json={"query": "glacier ice"}).json()["results"]
        assert after == []
        assert alice.get(f"/api/v1/documents/{doc_id}").status_code == 404
