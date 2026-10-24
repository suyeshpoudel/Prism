import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.embedding import Embedder
from app.evaluation.metrics import RelevanceKey, precision_at_k, recall_at_k, reciprocal_rank
from app.processing.chunking import ChunkingConfig, chunk_pages
from app.retrieval import IndexEntry, NumpyVectorIndex
from app.schemas.documents import PageExtraction

DEFAULT_K_VALUES = (1, 3, 5)
BENCHMARK_COLLECTION = "benchmark"
BENCHMARK_OWNER = "benchmark"


@dataclass(frozen=True)
class BenchmarkDocument:
    filename: str
    pages: list[str]


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    question: str
    relevant: set[RelevanceKey]
    k_values: tuple[int, ...]


def load_benchmark(
    corpus_path: Path, cases_path: Path
) -> tuple[list[BenchmarkDocument], list[BenchmarkCase]]:
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    documents = [
        BenchmarkDocument(filename=entry["filename"], pages=list(entry["pages"]))
        for entry in corpus["documents"]
    ]

    case_data = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = []
    for entry in case_data["cases"]:
        relevant = {
            (item["filename"], int(item["page_number"])) for item in entry["relevant"]
        }
        k_values = tuple(entry.get("k_values", DEFAULT_K_VALUES))
        cases.append(
            BenchmarkCase(
                id=entry["id"],
                question=entry["question"],
                relevant=relevant,
                k_values=k_values,
            )
        )
    return documents, cases


def build_corpus_index(
    *,
    embedder: Embedder,
    documents: list[BenchmarkDocument],
    config: ChunkingConfig,
    batch_size: int,
) -> tuple[NumpyVectorIndex, int, float]:
    index = NumpyVectorIndex()
    entries: list[IndexEntry] = []
    vectors: list[np.ndarray] = []

    for document in documents:
        pages = [
            PageExtraction(
                page_number=number,
                text=text,
                char_count=len(text),
                is_empty=not text.strip(),
            )
            for number, text in enumerate(document.pages, start=1)
        ]
        chunks = chunk_pages(
            document_id=f"corpus:{document.filename}", pages=pages, config=config
        )
        for chunk in chunks:
            entries.append(
                IndexEntry(
                    chunk=chunk,
                    filename=document.filename,
                    collection_id=BENCHMARK_COLLECTION,
                    owner_id=BENCHMARK_OWNER,
                )
            )

    embedding_start = time.perf_counter()
    for start in range(0, len(entries), batch_size):
        batch = entries[start : start + batch_size]
        vectors.append(embedder.embed_texts([entry.chunk.text for entry in batch]))
    matrix = np.vstack(vectors).astype(np.float32) if vectors else np.zeros((0, 1), dtype=np.float32)
    embedding_ms = (time.perf_counter() - embedding_start) * 1000.0

    if entries:
        index.add(entries, matrix)
    return index, len(entries), embedding_ms


_build_index = build_corpus_index


def run_benchmark(
    *,
    embedder: Embedder,
    documents: list[BenchmarkDocument],
    cases: list[BenchmarkCase],
    chunking_config: ChunkingConfig | None = None,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    batch_size: int = 32,
) -> dict:
    """Run the retrieval benchmark and return a JSON-serializable report.

    Timings: corpus embedding once; per-query embedding and retrieval
    separately (milliseconds). Quality: Recall@k, Precision@k, MRR against
    expected ``(filename, page_number)`` keys — deterministic, no LLM judging.
    """
    config = chunking_config or ChunkingConfig()
    index, corpus_chunks, corpus_embedding_ms = _build_index(
        embedder=embedder, documents=documents, config=config, batch_size=batch_size
    )

    max_k = max((*k_values, *(k for case in cases for k in case.k_values)), default=1)
    per_case = []
    totals: dict[str, list[float]] = {}
    embedding_times: list[float] = []
    retrieval_times: list[float] = []

    for case in cases:
        embed_start = time.perf_counter()
        query_vector = embedder.embed_query(case.question)
        embedding_ms = (time.perf_counter() - embed_start) * 1000.0

        retrieval_start = time.perf_counter()
        results = index.search(
            query_vector, top_k=max_k, collection_id=BENCHMARK_COLLECTION
        )
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0

        embedding_times.append(embedding_ms)
        retrieval_times.append(retrieval_ms)

        retrieved_keys = [(result.entry.filename, result.entry.chunk.page_number) for result in results]
        case_metrics = {}
        for k in case.k_values:
            case_metrics[f"recall@{k}"] = recall_at_k(retrieved_keys, case.relevant, k)
            case_metrics[f"precision@{k}"] = precision_at_k(retrieved_keys, case.relevant, k)
        case_metrics["mrr"] = reciprocal_rank(retrieved_keys, case.relevant)
        for name, value in case_metrics.items():
            totals.setdefault(name, []).append(value)

        per_case.append(
            {
                "id": case.id,
                "question": case.question,
                "relevant": [
                    {"filename": filename, "page_number": page}
                    for filename, page in sorted(case.relevant)
                ],
                "metrics": case_metrics,
                "timings_ms": {
                    "embedding": round(embedding_ms, 3),
                    "retrieval": round(retrieval_ms, 3),
                    "total": round(embedding_ms + retrieval_ms, 3),
                },
                "top_results": [
                    {
                        "rank": rank,
                        "filename": result.entry.filename,
                        "page_number": result.entry.chunk.page_number,
                        "chunk_id": result.entry.chunk.id,
                        "score": round(result.score, 6),
                    }
                    for rank, result in enumerate(results[:max_k], start=1)
                ],
            }
        )

    aggregate = {
        name: round(sum(values) / len(values), 6) for name, values in sorted(totals.items())
    }
    aggregate["embedding_ms_mean"] = round(sum(embedding_times) / len(embedding_times), 3) if embedding_times else 0.0
    aggregate["retrieval_ms_mean"] = round(sum(retrieval_times) / len(retrieval_times), 3) if retrieval_times else 0.0
    aggregate["query_ms_mean"] = round(
        aggregate["embedding_ms_mean"] + aggregate["retrieval_ms_mean"], 3
    )

    return {
        "config": {
            "embedding_model": embedder.model_name,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "corpus_documents": len(documents),
            "corpus_chunks": corpus_chunks,
            "cases": len(cases),
            "k_values": list(k_values),
        },
        "timings_ms": {
            "corpus_embedding": round(corpus_embedding_ms, 3),
        },
        "aggregate": aggregate,
        "cases": per_case,
    }
