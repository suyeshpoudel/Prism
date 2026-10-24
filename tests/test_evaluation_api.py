from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import SECOND_OWNER_KEY, TEST_API_KEY
from tests.embedding_stub import HashingEmbedder
from tests.llm_stub import StubLLMProvider


def test_benchmark_info_requires_api_key(client):
    response = client.get("/api/v1/evaluation/benchmark", headers={"X-API-Key": ""})
    assert response.status_code == 401


def test_benchmark_info_lists_corpus_and_cases(client):
    response = client.get("/api/v1/evaluation/benchmark")
    assert response.status_code == 200
    body = response.json()
    assert len(body["documents"]) == 3
    assert len(body["cases"]) == 6
    assert all(document["page_count"] >= 1 for document in body["documents"])
    assert all(case["relevant"] for case in body["cases"])


def test_retrieval_evaluation_persists_and_is_retrievable(client, settings):
    response = client.post(
        "/api/v1/evaluation/retrieval",
        json={"k_values": [1, 3], "configurations": [{"chunk_size": 800, "chunk_overlap": 100}]},
    )
    assert response.status_code == 201
    record = response.json()
    assert record["type"] == "retrieval"
    assert record["id"].startswith("eval-")
    assert len(record["results"]) == 1
    assert record["results"][0]["config"]["chunk_size"] == 800
    assert record["comparison"][0]["recall@1"] is not None
    assert (settings.evaluation_results_dir / f"{record['id']}.json").is_file()

    listed = client.get("/api/v1/evaluation/runs")
    assert listed.status_code == 200
    summaries = listed.json()
    assert [summary["id"] for summary in summaries] == [record["id"]]

    fetched = client.get(f"/api/v1/evaluation/runs/{record['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["id"] == record["id"]


def test_retrieval_evaluation_compares_configurations(client):
    response = client.post(
        "/api/v1/evaluation/retrieval",
        json={
            "configurations": [
                {"chunk_size": 500, "chunk_overlap": 50},
                {"chunk_size": 1000, "chunk_overlap": 150},
            ]
        },
    )
    assert response.status_code == 201
    record = response.json()
    assert len(record["results"]) == 2
    assert len(record["comparison"]) == 2
    assert {row["chunk_size"] for row in record["comparison"]} == {500, 1000}
    assert all("mrr" in row for row in record["comparison"])


def test_retrieval_evaluation_can_skip_persistence(client):
    response = client.post("/api/v1/evaluation/retrieval", json={"persist": False})
    assert response.status_code == 201
    assert client.get("/api/v1/evaluation/runs").json() == []


def test_retrieval_evaluation_rejects_invalid_configuration(client):
    response = client.post(
        "/api/v1/evaluation/retrieval",
        json={"configurations": [{"chunk_size": 500, "chunk_overlap": 500}]},
    )
    assert response.status_code == 422


def test_retrieval_evaluation_enforces_configured_limit(settings):
    limited = settings.model_copy(update={"evaluation_max_configurations": 1})
    app = create_app(limited, embedder=HashingEmbedder(), llm=StubLLMProvider())
    with TestClient(app) as client:
        client.headers.update({"X-API-Key": TEST_API_KEY})
        response = client.post(
            "/api/v1/evaluation/retrieval",
            json={
                "configurations": [
                    {"chunk_size": 500, "chunk_overlap": 50},
                    {"chunk_size": 800, "chunk_overlap": 100},
                ]
            },
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


def test_answer_evaluation_reports_grounded_answer_metrics(settings):
    app = create_app(
        settings,
        embedder=HashingEmbedder(),
        llm=StubLLMProvider(reply="Calvin cycle fixes carbon [1].", input_tokens=10, output_tokens=20),
    )
    with TestClient(app) as client:
        client.headers.update({"X-API-Key": TEST_API_KEY})
        response = client.post(
            "/api/v1/evaluation/answers",
            json={"top_k": 3, "chunk_size": 1000, "chunk_overlap": 150},
        )

    assert response.status_code == 201
    record = response.json()
    assert record["type"] == "answer"
    assert record["config"]["top_k"] == 3
    assert record["aggregate"]["cases"] == 6
    assert record["aggregate"]["generation_errors"] == 0
    assert record["aggregate"]["answers_with_citations_rate"] == 1.0
    assert record["usage"]["input_tokens"] == 60
    assert record["usage"]["estimated_cost_usd"] is None
    assert len(record["cases"]) == 6


def test_answer_evaluation_503_without_llm(settings):
    app = create_app(settings, embedder=HashingEmbedder())
    with TestClient(app) as client:
        client.headers.update({"X-API-Key": TEST_API_KEY})
        response = client.post("/api/v1/evaluation/answers", json={})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "answer_generation_unavailable"


def test_evaluation_runs_are_owner_scoped(client):
    created = client.post("/api/v1/evaluation/retrieval", json={"persist": True})
    assert created.status_code == 201
    run_id = created.json()["id"]

    other = client.get(
        f"/api/v1/evaluation/runs/{run_id}", headers={"X-API-Key": SECOND_OWNER_KEY}
    )
    assert other.status_code == 404
    assert other.json()["error"]["code"] == "evaluation_run_not_found"

    listed = client.get("/api/v1/evaluation/runs", headers={"X-API-Key": SECOND_OWNER_KEY})
    assert listed.status_code == 200
    assert listed.json() == []


def test_unknown_or_unsafe_run_id_returns_404(client):
    for run_id in ["does-not-exist", "..%2Fescape", "bad*id"]:
        response = client.get(f"/api/v1/evaluation/runs/{run_id}")
        assert response.status_code == 404
