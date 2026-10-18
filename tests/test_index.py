import numpy as np
import pytest

from app.retrieval.index import IndexEntry, NumpyVectorIndex, ScoredResult, l2_normalize
from app.schemas.documents import Chunk


def make_chunk(chunk_id: str = "d:0", page_number: int = 1) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id="d",
        page_number=page_number,
        chunk_index=0,
        text="text",
        char_count=4,
        start_offset=0,
        end_offset=4,
    )


def make_entry(
    chunk_id: str,
    collection_id: str = "default",
    page_number: int = 1,
    owner_id: str = "owner",
) -> IndexEntry:
    return IndexEntry(
        chunk=make_chunk(chunk_id, page_number),
        filename="f.pdf",
        collection_id=collection_id,
        owner_id=owner_id,
    )


def test_l2_normalize_unit_rows():
    matrix = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)

    normalized = l2_normalize(matrix, axis=1)

    assert np.isclose(np.linalg.norm(normalized[0]), 1.0)
    assert np.allclose(normalized[1], 0.0)


def test_cosine_similarity_of_known_vectors():
    index = NumpyVectorIndex()
    index.add(
        [make_entry("a"), make_entry("b")],
        np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32),
    )

    results = index.search(np.array([1.0, 0.0], dtype=np.float32), top_k=2)

    assert results[0].entry.chunk.id == "a"
    assert results[0].score == pytest.approx(1.0, abs=1e-6)
    assert results[1].entry.chunk.id == "b"
    assert results[1].score == pytest.approx(1.0 / np.sqrt(2.0), abs=1e-6)


def test_search_orders_by_score_descending():
    index = NumpyVectorIndex()
    index.add(
        [make_entry("far"), make_entry("mid"), make_entry("near")],
        np.array([[1.0, 0.0], [0.9, 0.1], [0.99, 0.01]], dtype=np.float32),
    )

    results = index.search(np.array([1.0, 0.0], dtype=np.float32), top_k=3)

    # "far" is exactly aligned; after normalization "near" ([0.99, 0.01])
    # beats "mid" ([0.9, 0.1]).
    assert [r.entry.chunk.id for r in results] == ["far", "near", "mid"]
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_top_k_limits_results_and_clamps_to_size():
    index = NumpyVectorIndex()
    index.add(
        [make_entry("a"), make_entry("b")],
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )

    assert len(index.search(np.array([1.0, 0.0], dtype=np.float32), top_k=1)) == 1
    assert len(index.search(np.array([1.0, 0.0], dtype=np.float32), top_k=99)) == 2


def test_search_on_empty_index_returns_empty():
    index = NumpyVectorIndex()
    assert index.search(np.array([1.0, 0.0], dtype=np.float32), top_k=5) == []
    assert index.size() == 0


def test_collection_isolation_filters_other_collections():
    index = NumpyVectorIndex()
    # Identical vectors: without filtering, either could rank first.
    index.add(
        [
            make_entry("alpha-1", collection_id="alpha"),
            make_entry("beta-1", collection_id="beta"),
            make_entry("alpha-2", collection_id="alpha"),
        ],
        np.array([[1.0, 0.0], [1.0, 0.0], [0.9, 0.1]], dtype=np.float32),
    )

    results = index.search(
        np.array([1.0, 0.0], dtype=np.float32), top_k=10, collection_id="alpha"
    )

    assert {r.entry.collection_id for r in results} == {"alpha"}
    assert [r.entry.chunk.id for r in results] == ["alpha-1", "alpha-2"]


def test_owner_isolation_filters_other_owners():
    index = NumpyVectorIndex()
    index.add(
        [
            make_entry("alice-1", owner_id="alice"),
            make_entry("bob-1", owner_id="bob"),
            make_entry("alice-2", owner_id="alice"),
        ],
        np.array([[1.0, 0.0], [1.0, 0.0], [0.9, 0.1]], dtype=np.float32),
    )

    results = index.search(np.array([1.0, 0.0], dtype=np.float32), top_k=10, owner_id="alice")

    assert {r.entry.owner_id for r in results} == {"alice"}
    assert [r.entry.chunk.id for r in results] == ["alice-1", "alice-2"]


def test_owner_and_collection_filters_combine():
    index = NumpyVectorIndex()
    index.add(
        [
            make_entry("keep", collection_id="alpha", owner_id="alice"),
            make_entry("same-owner-other-collection", collection_id="beta", owner_id="alice"),
            make_entry("other-owner-same-collection", collection_id="alpha", owner_id="bob"),
        ],
        np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
    )

    results = index.search(
        np.array([1.0, 0.0], dtype=np.float32),
        top_k=10,
        collection_id="alpha",
        owner_id="alice",
    )

    assert [r.entry.chunk.id for r in results] == ["keep"]


def test_search_without_filter_returns_all_collections():
    index = NumpyVectorIndex()
    index.add(
        [make_entry("a", "alpha"), make_entry("b", "beta")],
        np.array([[1.0, 0.0], [0.99, 0.01]], dtype=np.float32),
    )

    results = index.search(np.array([1.0, 0.0], dtype=np.float32), top_k=10)

    assert len(results) == 2


def test_invalid_top_k_and_dimension_errors():
    index = NumpyVectorIndex()
    with pytest.raises(ValueError):
        index.search(np.array([1.0], dtype=np.float32), top_k=0)

    # Empty index short-circuits before dimension checks.
    assert index.search(np.array([1.0, 2.0, 3.0], dtype=np.float32), top_k=1) == []

    with pytest.raises(ValueError):
        index.add([make_entry("a")], np.array([1.0, 2.0], dtype=np.float32))

    index.add([make_entry("a")], np.array([[1.0, 2.0]], dtype=np.float32))
    with pytest.raises(ValueError):
        index.add([make_entry("b")], np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
    with pytest.raises(ValueError):
        index.search(np.array([1.0, 2.0, 3.0], dtype=np.float32), top_k=1)


def test_zero_vectors_score_zero():
    index = NumpyVectorIndex()
    index.add(
        [make_entry("zero"), make_entry("one")],
        np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
    )

    results = index.search(np.array([1.0, 0.0], dtype=np.float32), top_k=2)

    assert results[0].entry.chunk.id == "one"
    assert results[1].score == pytest.approx(0.0, abs=1e-6)


def test_scored_result_type():
    index = NumpyVectorIndex()
    index.add([make_entry("a")], np.array([[1.0, 0.0]], dtype=np.float32))
    results = index.search(np.array([1.0, 0.0], dtype=np.float32), top_k=1)
    assert isinstance(results[0], ScoredResult)
