import numpy as np
import pytest

from app.repository.database import build_engine, build_session_factory, create_schema
from app.repository.record import DocumentRecord
from app.repository.sql_store import SqlDocumentStore
from app.schemas.documents import Chunk, PageExtraction


@pytest.fixture
def store(tmp_path) -> SqlDocumentStore:
    engine = build_engine(f"sqlite+pysqlite:///{tmp_path / 'store.db'}")
    create_schema(engine)
    return SqlDocumentStore(build_session_factory(engine))


def make_record(
    doc_id: str = "doc1",
    owner_id: str = "owner-a",
    collection_id: str = "default",
    chunk_count: int = 2,
    embedding_model: str = "model-v1",
) -> DocumentRecord:
    pages = [
        PageExtraction(page_number=1, text="Hello world about retrieval.", char_count=28, is_empty=False),
    ]
    chunks = [
        Chunk(
            id=f"{doc_id}:{i}",
            document_id=doc_id,
            page_number=1,
            chunk_index=i,
            text=f"chunk {i} text",
            char_count=len(f"chunk {i} text"),
            start_offset=i * 5,
            end_offset=i * 5 + 10,
        )
        for i in range(chunk_count)
    ]
    return DocumentRecord(
        id=doc_id,
        filename=f"{doc_id}.pdf",
        content_type="application/pdf",
        size_bytes=123,
        page_count=1,
        extracted_char_count=28,
        pages=pages,
        chunks=chunks,
        warnings=["no_text_extracted"],
        collection_id=collection_id,
        owner_id=owner_id,
        embedding_model=embedding_model,
    )


def embeddings_for(record: DocumentRecord, dimension: int = 4) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(record.id)) % (2**32))
    return rng.random((len(record.chunks), dimension), dtype=np.float32)


def test_add_and_get_round_trip(store: SqlDocumentStore):
    record = make_record()
    embeddings = embeddings_for(record)

    store.add(record, embeddings)
    fetched = store.get(record.id, owner_id="owner-a")

    assert fetched is not None
    assert fetched.id == record.id
    assert fetched.filename == "doc1.pdf"
    assert fetched.owner_id == "owner-a"
    assert fetched.embedding_model == "model-v1"
    assert fetched.warnings == ["no_text_extracted"]
    assert len(fetched.pages) == 1
    assert fetched.pages[0].text == "Hello world about retrieval."
    assert len(fetched.chunks) == 2
    assert fetched.chunks[1].text == "chunk 1 text"


def test_get_scoped_by_owner(store: SqlDocumentStore):
    record = make_record(owner_id="owner-a")
    store.add(record, embeddings_for(record))

    assert store.get(record.id, owner_id="owner-a") is not None
    assert store.get(record.id, owner_id="owner-b") is None


def test_list_scoped_by_owner(store: SqlDocumentStore):
    store.add(make_record("doc-a", owner_id="owner-a"), embeddings_for(make_record("doc-a")))
    store.add(make_record("doc-b", owner_id="owner-b"), embeddings_for(make_record("doc-b")))

    listing = store.list(owner_id="owner-a")

    assert [r.id for r in listing] == ["doc-a"]


def test_delete_removes_document_and_returns_false_for_missing(store: SqlDocumentStore):
    record = make_record()
    store.add(record, embeddings_for(record))

    assert store.delete(record.id, owner_id="owner-a") is True
    assert store.get(record.id, owner_id="owner-a") is None
    assert store.delete(record.id, owner_id="owner-a") is False


def test_delete_scoped_by_owner(store: SqlDocumentStore):
    record = make_record(owner_id="owner-a")
    store.add(record, embeddings_for(record))

    assert store.delete(record.id, owner_id="owner-b") is False
    assert store.get(record.id, owner_id="owner-a") is not None


def test_load_all_with_embeddings_round_trip(store: SqlDocumentStore):
    record = make_record(chunk_count=3)
    embeddings = embeddings_for(record, dimension=8)
    store.add(record, embeddings)

    loaded = store.load_all_with_embeddings()

    assert len(loaded) == 1
    loaded_record, matrix = loaded[0]
    assert loaded_record.id == record.id
    assert matrix.shape == (3, 8)
    assert np.allclose(matrix, embeddings, atol=1e-6)


def test_load_all_with_embeddings_multiple_documents(store: SqlDocumentStore):
    first = make_record("doc1", owner_id="a")
    second = make_record("doc2", owner_id="b")
    store.add(first, embeddings_for(first))
    store.add(second, embeddings_for(second))

    loaded = store.load_all_with_embeddings()

    assert {r.id for r, _ in loaded} == {"doc1", "doc2"}


def test_embedding_count_mismatch_rejected(store: SqlDocumentStore):
    record = make_record(chunk_count=2)

    with pytest.raises(ValueError, match="embeddings count"):
        store.add(record, np.zeros((1, 4), dtype=np.float32))
