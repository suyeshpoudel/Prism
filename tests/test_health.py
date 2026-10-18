from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app as module_app, create_app
from tests.conftest import TEST_API_KEY, TEST_OWNER
from tests.embedding_stub import HashingEmbedder


def test_health_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "Prism"
    assert body["version"]
    assert body["environment"]


def test_factory_built_app_serves_health(db_url):
    settings = Settings(
        database_url=db_url,
        api_keys={TEST_API_KEY: TEST_OWNER},
    )
    with TestClient(create_app(settings, embedder=HashingEmbedder())) as client:
        assert client.get("/health").status_code == 200


def test_module_level_app_wires_routers():
    paths = set(module_app.openapi()["paths"])
    assert "/health" in paths
    assert "/api/v1/documents" in paths
    assert "/api/v1/search" in paths
