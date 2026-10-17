import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.embedding_stub import HashingEmbedder
from tests.pdf_helpers import make_pdf


@pytest.fixture
def settings() -> Settings:
    return Settings(max_upload_bytes=1_000_000)


@pytest.fixture
def client(settings: Settings) -> TestClient:
    return TestClient(create_app(settings, embedder=HashingEmbedder()))


@pytest.fixture
def tight_client() -> TestClient:
    return TestClient(create_app(Settings(max_upload_bytes=100), embedder=HashingEmbedder()))


@pytest.fixture
def pdf_bytes() -> bytes:
    return make_pdf(["First page text about retrieval.", "Second page text about chunking."])
