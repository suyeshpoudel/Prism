import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.embedding_stub import HashingEmbedder
from tests.llm_stub import StubLLMProvider
from tests.pdf_helpers import make_pdf

TEST_API_KEY = "test-key"
TEST_OWNER = "test-owner"
SECOND_OWNER_KEY = "second-key"
SECOND_OWNER = "second-owner"


@pytest.fixture
def db_url(tmp_path) -> str:
    return f"sqlite+pysqlite:///{tmp_path / 'test.db'}"


@pytest.fixture
def settings(db_url: str, tmp_path) -> Settings:
    return Settings(
        max_upload_bytes=1_000_000,
        database_url=db_url,
        api_keys={TEST_API_KEY: TEST_OWNER, SECOND_OWNER_KEY: SECOND_OWNER},
        evaluation_results_dir=tmp_path / "evaluation-results",
    )


@pytest.fixture
def client(settings: Settings) -> TestClient:
    app = create_app(settings, embedder=HashingEmbedder(), llm=StubLLMProvider())
    with TestClient(app) as test_client:
        test_client.headers.update({"X-API-Key": TEST_API_KEY})
        yield test_client


@pytest.fixture
def tight_client(db_url: str) -> TestClient:
    settings = Settings(
        max_upload_bytes=100,
        database_url=db_url,
        api_keys={TEST_API_KEY: TEST_OWNER},
    )
    app = create_app(settings, embedder=HashingEmbedder())
    with TestClient(app) as test_client:
        test_client.headers.update({"X-API-Key": TEST_API_KEY})
        yield test_client


@pytest.fixture
def pdf_bytes() -> bytes:
    return make_pdf(["First page text about retrieval.", "Second page text about chunking."])
