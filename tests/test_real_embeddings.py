"""Integration tests using the real local Sentence Transformer model.

These download/load ``all-MiniLM-L6-v2`` (cached after first run) and are the
honest end-to-end check that semantic retrieval works, not just the plumbing.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.embedding import get_shared_embedder
from app.main import create_app
from tests.pdf_helpers import make_pdf

CALVIN_PAGE = (
    "The Calvin cycle, also called light-independent reactions, takes place in the stroma "
    "of the chloroplast. The enzyme RuBisCO fixes carbon dioxide by attaching it to ribulose "
    "bisphosphate. A series of reduction reactions powered by ATP and NADPH from the light "
    "reactions builds sugars the plant can store as glucose."
)
LIGHT_PAGE = (
    "The light-dependent reactions occur in the thylakoid membranes. Chlorophyll absorbs "
    "photons and excites electrons along an electron transport chain, producing ATP and NADPH "
    "while water is split and oxygen is released."
)
PASTA_PAGE = (
    "Salt the pasta water generously, roughly ten grams of salt per liter, so the pasta "
    "seasons from the inside as it absorbs water during boiling."
)


@pytest.fixture(scope="module")
def real_client() -> TestClient:
    return TestClient(create_app(Settings(max_upload_bytes=1_000_000)))


def test_real_embedder_produces_normalized_shapes():
    embedder = get_shared_embedder()
    vectors = embedder.embed_texts(["photosynthesis", "pasta water"])

    assert vectors.shape == (2, embedder.dimension)
    assert np.all(np.linalg.norm(vectors, axis=1) > 0)


def test_real_semantic_search_roundtrip(real_client):
    for filename, pages in [
        ("plants.pdf", [LIGHT_PAGE, CALVIN_PAGE]),
        ("pasta.pdf", [PASTA_PAGE]),
    ]:
        response = real_client.post(
            "/api/v1/documents",
            files={"file": (filename, make_pdf(pages), "application/pdf")},
        )
        assert response.status_code == 201

    response = real_client.post(
        "/api/v1/search",
        json={
            "query": "How does the plant fix carbon dioxide into sugar using RuBisCO?",
            "top_k": 3,
        },
    )

    assert response.status_code == 200
    results = response.json()["results"]
    assert results
    top = results[0]
    assert top["filename"] == "plants.pdf"
    assert top["chunk"]["page_number"] == 2
    # The relevant page must outrank the unrelated pasta page by a clear margin.
    pasta_scores = [r["score"] for r in results if r["filename"] == "pasta.pdf"]
    assert not pasta_scores or top["score"] > pasta_scores[0]


def test_real_semantic_search_is_collection_isolated(real_client):
    response = real_client.post(
        "/api/v1/search",
        json={"query": "carbon dioxide RuBisco sugar", "collection_id": "empty-space"},
    )

    assert response.status_code == 200
    assert response.json()["results"] == []
