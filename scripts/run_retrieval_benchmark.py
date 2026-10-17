#!/usr/bin/env python
"""Run the versioned retrieval benchmark and print a JSON report.

Usage (from the repository root):

    python scripts/run_retrieval_benchmark.py
    python scripts/run_retrieval_benchmark.py --chunk-size 500 --chunk-overlap 50
    python scripts/run_retrieval_benchmark.py --output report.json
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.embedding import get_shared_embedder  # noqa: E402
from app.evaluation.benchmark import load_benchmark, run_benchmark  # noqa: E402
from app.processing.chunking import ChunkingConfig  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "retrieval" / "corpus.json",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "retrieval" / "cases.json",
    )
    parser.add_argument("--model", default=None, help="Embedding model name (uses app default if omitted)")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--top", type=int, nargs="*", default=[1, 3, 5], metavar="K")
    parser.add_argument("--output", type=Path, default=None, help="Write the JSON report to this path")
    args = parser.parse_args()

    documents, cases = load_benchmark(args.corpus, args.cases)
    from app.config import get_settings

    settings = get_settings()
    model_name = args.model or settings.embedding_model
    embedder = get_shared_embedder(model_name)
    report = run_benchmark(
        embedder=embedder,
        documents=documents,
        cases=cases,
        chunking_config=ChunkingConfig(
            chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
        ),
        k_values=tuple(args.top),
        batch_size=settings.embedding_batch_size,
    )

    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
