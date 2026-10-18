import pytest

from app.config import Settings
from app.core.auth import resolve_owner
from app.core.errors import UnauthorizedError
from tests.conftest import SECOND_OWNER, SECOND_OWNER_KEY, TEST_API_KEY, TEST_OWNER


def test_missing_key_header_returns_401(client):
    response = client.get("/api/v1/documents", headers={"X-API-Key": ""})

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


def test_forged_header_without_value_returns_401(client):
    response = client.get("/api/v1/documents", headers={"X-API-Key": "not-a-real-key"})

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


def test_search_requires_api_key(client):
    response = client.post("/api/v1/search", json={"query": "x"}, headers={"X-API-Key": ""})

    assert response.status_code == 401


def test_upload_requires_api_key(client, pdf_bytes):
    response = client.post(
        "/api/v1/documents",
        files={"file": ("a.pdf", pdf_bytes, "application/pdf")},
        headers={"X-API-Key": ""},
    )

    assert response.status_code == 401


def test_delete_requires_api_key(client, pdf_bytes):
    response = client.delete("/api/v1/documents/some-id", headers={"X-API-Key": ""})

    assert response.status_code == 401


def test_valid_key_maps_to_owner(client):
    response = client.get(
        "/api/v1/documents",
        headers={"X-API-Key": SECOND_OWNER_KEY},
    )

    assert response.status_code == 200


def test_resolve_owner_unit():
    settings = Settings(api_keys={TEST_API_KEY: TEST_OWNER, SECOND_OWNER_KEY: SECOND_OWNER})
    assert resolve_owner(TEST_API_KEY, settings) == TEST_OWNER
    assert resolve_owner(SECOND_OWNER_KEY, settings) == SECOND_OWNER


def test_resolve_owner_missing_key_unit():
    settings = Settings(api_keys={TEST_API_KEY: TEST_OWNER})
    with pytest.raises(UnauthorizedError):
        resolve_owner(None, settings)
    with pytest.raises(UnauthorizedError):
        resolve_owner("", settings)


def test_resolve_owner_unknown_key_unit():
    settings = Settings(api_keys={TEST_API_KEY: TEST_OWNER})
    with pytest.raises(UnauthorizedError):
        resolve_owner("nope", settings)
