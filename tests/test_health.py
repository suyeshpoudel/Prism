from fastapi.testclient import TestClient

from app.main import app as module_app


def test_health_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "Prism"
    assert body["version"]
    assert body["environment"]


def test_module_level_app_serves_health():
    with TestClient(module_app) as client:
        assert client.get("/health").status_code == 200
