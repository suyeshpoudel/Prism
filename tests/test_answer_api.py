from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.conftest import TEST_API_KEY, TEST_OWNER
from tests.embedding_stub import FailingEmbedder, HashingEmbedder
from tests.llm_stub import FailingLLMProvider, StubLLMProvider
from tests.pdf_helpers import make_pdf

PLANT_PAGES = [
    "Chlorophyll in the thylakoid membrane absorbs light and drives the electron transport chain, producing ATP and NADPH while splitting water and releasing oxygen.",
    "The Calvin cycle fixes carbon dioxide in the stroma using RuBisCO, building sugar molecules from carbon using the ATP and NADPH made by the light reactions.",
]


def build_answer_client(db_url: str, llm, embedder=None) -> TestClient:
    settings = Settings(
        max_upload_bytes=1_000_000,
        database_url=db_url,
        api_keys={TEST_API_KEY: TEST_OWNER},
    )
    app = create_app(settings, embedder=embedder or HashingEmbedder(), llm=llm)
    client = TestClient(app)
    client.headers.update({"X-API-Key": TEST_API_KEY})
    return client


def upload(client: TestClient, filename: str, pages: list[str], collection_id: str | None = None) -> str:
    form = {"collection_id": collection_id} if collection_id else None
    response = client.post(
        "/api/v1/documents",
        files={"file": (filename, make_pdf(pages), "application/pdf")},
        data=form,
    )
    assert response.status_code == 201
    return response.json()["id"]


def test_answer_requires_api_key(db_url):
    with build_answer_client(db_url, StubLLMProvider()) as client:
        response = client.post(
            "/api/v1/answer", json={"question": "anything"}, headers={"X-API-Key": ""}
        )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


def test_answer_returns_grounded_citations(db_url):
    llm = StubLLMProvider(
        reply="The Calvin cycle fixes carbon dioxide using RuBisCO [1].",
        input_tokens=321,
        output_tokens=12,
    )
    with build_answer_client(db_url, llm) as client:
        upload(client, "plants.pdf", PLANT_PAGES)
        response = client.post(
            "/api/v1/answer",
            json={"question": "How does the Calvin cycle fix carbon?", "top_k": 3},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "How does the Calvin cycle fix carbon?"
    assert body["insufficient_evidence"] is False
    assert body["model"] == "stub-model"
    assert body["input_tokens"] == 321
    assert body["output_tokens"] == 12
    assert body["invalid_citation_markers"] == []

    # Evidence is the retrieved set with scores; citation [1] must map to the
    # first retrieved passage exactly (validated against real sources).
    assert body["evidence"]
    first = body["evidence"][0]
    assert len(body["citations"]) == 1
    citation = body["citations"][0]
    assert citation["marker"] == 1
    assert citation["chunk_id"] == first["chunk"]["id"]
    assert citation["document_id"] == first["document_id"]
    assert citation["filename"] == first["filename"]
    assert citation["page_number"] == first["chunk"]["page_number"]
    assert citation["text"] == first["chunk"]["text"]
    assert citation["score"] == first["score"]


def test_answer_flags_invalid_citation_markers(db_url):
    llm = StubLLMProvider(reply="Supported [1]. Fabricated [9].")
    with build_answer_client(db_url, llm) as client:
        upload(client, "plants.pdf", PLANT_PAGES)
        response = client.post("/api/v1/answer", json={"question": "carbon fixation?"})

    body = response.json()
    assert [c["marker"] for c in body["citations"]] == [1]
    assert body["invalid_citation_markers"] == [9]


def test_answer_insufficient_evidence_when_nothing_retrieved(db_url):
    # FailingLLMProvider raises if called — proof the LLM is skipped.
    with build_answer_client(db_url, FailingLLMProvider()) as client:
        response = client.post("/api/v1/answer", json={"question": "quantum foam?"})

    assert response.status_code == 200
    body = response.json()
    assert body["insufficient_evidence"] is True
    assert body["answer"].startswith("Insufficient evidence")
    assert body["citations"] == []
    assert body["evidence"] == []


def test_answer_insufficient_evidence_from_llm_reply(db_url):
    llm = StubLLMProvider(reply="Insufficient evidence.")
    with build_answer_client(db_url, llm) as client:
        upload(client, "plants.pdf", PLANT_PAGES)
        response = client.post("/api/v1/answer", json={"question": "unrelated topic?"})

    body = response.json()
    assert body["insufficient_evidence"] is True
    assert body["citations"] == []


def test_answer_503_when_llm_not_configured(db_url):
    settings = Settings(
        max_upload_bytes=1_000_000,
        database_url=db_url,
        api_keys={TEST_API_KEY: TEST_OWNER},
        llm_base_url="",
    )
    app = create_app(settings, embedder=HashingEmbedder())
    with TestClient(app) as client:
        client.headers.update({"X-API-Key": TEST_API_KEY})
        response = client.post("/api/v1/answer", json={"question": "anything"})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "answer_generation_unavailable"


def test_answer_llm_failure_returns_502(db_url):
    with build_answer_client(db_url, FailingLLMProvider()) as client:
        upload(client, "plants.pdf", PLANT_PAGES)
        response = client.post("/api/v1/answer", json={"question": "carbon fixation?"})

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "answer_generation_failed"


def test_answer_embedding_failure_returns_500(db_url):
    with build_answer_client(db_url, StubLLMProvider(), embedder=FailingEmbedder()) as client:
        response = client.post("/api/v1/answer", json={"question": "anything"})

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "embedding_failed"


def test_answer_rejects_blank_question(db_url):
    with build_answer_client(db_url, StubLLMProvider()) as client:
        response = client.post("/api/v1/answer", json={"question": "   "})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


def test_answer_respects_collection_scoping(db_url):
    llm = StubLLMProvider(reply="Glaciers are ice [1].")
    with build_answer_client(db_url, llm) as client:
        upload(client, "alpha.pdf", ["Alpha collection talks about glaciers and ice."], "alpha")
        upload(client, "beta.pdf", ["Beta collection talks about volcanoes and magma."], "beta")

        response = client.post(
            "/api/v1/answer",
            json={"question": "glaciers?", "collection_id": "beta"},
        )

    body = response.json()
    assert body["evidence"]
    assert all(e["filename"] == "beta.pdf" for e in body["evidence"])
    # [1] maps to beta's passage, never alpha's.
    assert body["citations"][0]["filename"] == "beta.pdf"


def test_answer_does_not_cross_owners(db_url):
    from tests.conftest import SECOND_OWNER_KEY

    llm = StubLLMProvider(reply="Something [1].")
    with build_answer_client(db_url, llm) as client:
        upload(client, "mine.pdf", ["My private glacier research notes."])

    other_settings = Settings(
        max_upload_bytes=1_000_000,
        database_url=db_url,
        api_keys={TEST_API_KEY: TEST_OWNER, SECOND_OWNER_KEY: "second-owner"},
    )
    app = create_app(other_settings, embedder=HashingEmbedder(), llm=StubLLMProvider(reply="Insufficient evidence."))
    with TestClient(app) as other_client:
        other_client.headers.update({"X-API-Key": SECOND_OWNER_KEY})
        response = other_client.post("/api/v1/answer", json={"question": "glaciers?"})

    body = response.json()
    assert body["evidence"] == []
    assert body["insufficient_evidence"] is True


def test_answer_prompt_contains_evidence_and_injection_guard(db_url):
    llm = StubLLMProvider(reply="Answer [1].")
    with build_answer_client(db_url, llm) as client:
        upload(client, "plants.pdf", PLANT_PAGES)
        client.post("/api/v1/answer", json={"question": "carbon?"})

    messages = llm.calls[0]["messages"]
    system = messages[0]["content"]
    user = messages[1]["content"]
    assert "untrusted" in system.lower()
    assert "<evidence>" in user
    assert "Chlorophyll" in user or "Calvin cycle" in user
