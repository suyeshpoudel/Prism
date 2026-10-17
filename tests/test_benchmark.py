from pathlib import Path

from app.evaluation.benchmark import load_benchmark, run_benchmark
from app.processing.chunking import ChunkingConfig
from tests.embedding_stub import HashingEmbedder

BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "retrieval"


def test_load_benchmark_reads_versioned_files():
    documents, cases = load_benchmark(BENCHMARK_DIR / "corpus.json", BENCHMARK_DIR / "cases.json")

    assert len(documents) == 3
    assert all(document.pages for document in documents)
    assert len(cases) >= 5
    for case in cases:
        assert case.question
        assert case.relevant
        assert all(filename.endswith(".pdf") and page >= 1 for filename, page in case.relevant)


def test_run_benchmark_produces_valid_report():
    documents, cases = load_benchmark(BENCHMARK_DIR / "corpus.json", BENCHMARK_DIR / "cases.json")

    report = run_benchmark(
        embedder=HashingEmbedder(),
        documents=documents,
        cases=cases,
        chunking_config=ChunkingConfig(chunk_size=1000, chunk_overlap=150),
        k_values=(1, 3, 5),
    )

    assert report["config"]["corpus_documents"] == 3
    assert report["config"]["corpus_chunks"] > 0
    assert report["config"]["cases"] == len(cases)

    aggregate = report["aggregate"]
    for name, value in aggregate.items():
        if name.startswith(("recall", "precision", "mrr")):
            assert 0.0 <= value <= 1.0, name
    # The hashing embedder shares keywords with these cases, so it should
    # rank at least one relevant page first on average.
    assert aggregate["mrr"] > 0.0

    assert len(report["cases"]) == len(cases)
    for case_report in report["cases"]:
        assert case_report["timings_ms"]["total"] >= 0.0
        assert case_report["top_results"]
        assert case_report["top_results"][0]["rank"] == 1


def test_run_benchmark_is_deterministic_apart_from_timings():
    documents, cases = load_benchmark(BENCHMARK_DIR / "corpus.json", BENCHMARK_DIR / "cases.json")

    def strip_timings(report):
        report = dict(report)
        report.pop("timings_ms")
        report["aggregate"] = {
            k: v for k, v in report["aggregate"].items() if not k.endswith("_ms_mean") and k != "query_ms_mean"
        }
        for case in report["cases"]:
            case.pop("timings_ms")
        return report

    first = run_benchmark(embedder=HashingEmbedder(), documents=documents, cases=cases)
    second = run_benchmark(embedder=HashingEmbedder(), documents=documents, cases=cases)

    assert strip_timings(first) == strip_timings(second)
