from pathlib import Path

from app.evaluation.answer_eval import (
    content_tokens,
    judge_answer,
    lexical_support,
    run_answer_evaluation,
    score_answer,
    split_sentences,
)
from app.evaluation.benchmark import load_benchmark
from app.llm.provider import LLMCompletion
from app.retrieval.index import IndexEntry, ScoredResult
from app.schemas.answers import Citation
from app.schemas.documents import Chunk
from tests.embedding_stub import HashingEmbedder
from tests.llm_stub import StubLLMProvider

BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "retrieval"

PASSAGE = (
    "The Calvin cycle fixes carbon dioxide in the stroma using RuBisCO, "
    "building sugar molecules from carbon."
)


def make_chunk(text: str = PASSAGE, page: int = 3) -> Chunk:
    return Chunk(
        id=f"doc:0",
        document_id="doc",
        page_number=page,
        chunk_index=0,
        text=text,
        char_count=len(text),
        start_offset=0,
        end_offset=len(text),
    )


def make_evidence(text: str = PASSAGE, page: int = 3) -> list[ScoredResult]:
    return [
        ScoredResult(
            entry=IndexEntry(
                chunk=make_chunk(text, page),
                filename="plant-photosynthesis.pdf",
                collection_id="benchmark",
                owner_id="benchmark",
            ),
            score=0.9,
        )
    ]


def make_citation(marker: int = 1, page: int = 3) -> Citation:
    return Citation(
        marker=marker,
        chunk_id="doc:0",
        document_id="doc",
        filename="plant-photosynthesis.pdf",
        page_number=page,
        text=PASSAGE,
        score=0.9,
    )


def test_split_sentences_and_content_tokens():
    assert split_sentences("One thing. Two things! Three?") == [
        "One thing.",
        "Two things!",
        "Three?",
    ]
    tokens = content_tokens("The Calvin cycle uses RuBisCO.")
    assert "calvin" in tokens
    assert "the" not in tokens


def test_lexical_support_scores_overlap():
    assert lexical_support("The Calvin cycle fixes carbon dioxide using RuBisCO", [PASSAGE])
    assert not lexical_support("Glaciers carve valleys through granite bedrock", [PASSAGE])
    assert lexical_support("the and of", [PASSAGE]) is None


def test_score_answer_computes_citation_and_grounding_metrics():
    metrics = score_answer(
        answer=(
            "The Calvin cycle fixes carbon dioxide using RuBisCO [1]. "
            "Bananas are rich in potassium and magnesium [9]."
        ),
        citations=[make_citation(1)],
        invalid_markers=[9],
        evidence=make_evidence(),
        relevant={("plant-photosynthesis.pdf", 3)},
        insufficient_evidence=False,
    )

    assert metrics["citation_validity"] == 0.5
    assert metrics["citation_precision"] == 1.0
    assert metrics["citation_recall"] == 1.0
    assert metrics["cited_sentences"] == 1
    assert metrics["unsupported_cited_sentences"] == 0
    assert metrics["sentences"] == 2
    assert metrics["uncited_sentence_rate"] == 0.5
    assert metrics["lexical_grounding_rate"] == 1.0


def test_score_answer_flags_lexically_unsupported_citation():
    metrics = score_answer(
        answer="Glaciers carve deep valleys through granite bedrock [1].",
        citations=[make_citation(1)],
        invalid_markers=[],
        evidence=make_evidence(),
        relevant={("plant-photosynthesis.pdf", 3)},
        insufficient_evidence=False,
    )

    assert metrics["lexical_grounding_rate"] == 0.0
    assert metrics["unsupported_cited_sentences"] == 1


def test_score_answer_handles_uncited_and_insufficient_answers():
    uncited = score_answer(
        answer="Bananas are rich in potassium and magnesium.",
        citations=[],
        invalid_markers=[],
        evidence=make_evidence(),
        relevant={("plant-photosynthesis.pdf", 3)},
        insufficient_evidence=False,
    )
    assert uncited["citation_validity"] is None
    assert uncited["citation_precision"] is None
    assert uncited["uncited_sentence_rate"] == 1.0

    insufficient = score_answer(
        answer="Insufficient evidence.",
        citations=[],
        invalid_markers=[],
        evidence=[],
        relevant={("plant-photosynthesis.pdf", 3)},
        insufficient_evidence=True,
    )
    assert insufficient["sentences"] == 0
    assert insufficient["uncited_sentence_rate"] is None
    assert insufficient["lexical_grounding_rate"] is None


class JudgeStub:
    def __init__(self, judge_reply: str) -> None:
        self._judge_reply = judge_reply

    @property
    def model_name(self) -> str:
        return "judge-stub"

    def complete(self, messages, *, max_tokens: int, temperature: float) -> LLMCompletion:
        if "evaluation judge" in messages[0]["content"]:
            return LLMCompletion(text=self._judge_reply, model="judge-stub")
        return LLMCompletion(text="The Calvin cycle fixes carbon [1].", model="answer-stub")


def test_judge_answer_parses_json_and_rejects_malformed_replies():
    judgement = judge_answer(
        llm=JudgeStub('{"faithful": false, "unsupported_claims": ["extra"], "reason": "no"}'),
        question="How does the Calvin cycle work?",
        answer="The Calvin cycle fixes carbon [1].",
        evidence=make_evidence(),
    )
    assert judgement["faithful"] is False
    assert judgement["unsupported_claims"] == ["extra"]

    malformed = judge_answer(
        llm=JudgeStub("not json"),
        question="q",
        answer="a",
        evidence=make_evidence(),
    )
    assert "error" in malformed


def test_run_answer_evaluation_reports_metrics_usage_and_cost():
    documents, cases = load_benchmark(
        BENCHMARK_DIR / "corpus.json", BENCHMARK_DIR / "cases.json"
    )
    llm = StubLLMProvider(reply="The Calvin cycle fixes carbon dioxide [1].", input_tokens=100, output_tokens=200)
    report = run_answer_evaluation(
        embedder=HashingEmbedder(),
        documents=documents,
        cases=cases,
        llm=llm,
        top_k=5,
        input_cost_per_1m=1.0,
        output_cost_per_1m=2.0,
    )

    assert report["type"] == "answer"
    assert report["config"]["cases"] == len(cases)
    assert report["aggregate"]["generation_errors"] == 0
    assert report["aggregate"]["answers_with_citations_rate"] == 1.0
    assert report["usage"]["input_tokens"] == 100 * len(cases)
    assert report["usage"]["output_tokens"] == 200 * len(cases)
    assert report["usage"]["estimated_cost_usd"] == round(
        100 * len(cases) / 1_000_000 * 1.0 + 200 * len(cases) / 1_000_000 * 2.0, 6
    )
    for case in report["cases"]:
        assert case["answer"]
        assert case["metrics"]["citation_validity"] == 1.0
        assert case["timings_ms"]["total"] >= 0.0


def test_run_answer_evaluation_with_judge_and_failure_is_reported():
    documents, cases = load_benchmark(
        BENCHMARK_DIR / "corpus.json", BENCHMARK_DIR / "cases.json"
    )
    report = run_answer_evaluation(
        embedder=HashingEmbedder(),
        documents=documents,
        cases=cases[:2],
        llm=JudgeStub('{"faithful": true, "unsupported_claims": [], "reason": "ok"}'),
        include_llm_judge=True,
    )

    assert report["aggregate"]["llm_judge_cases"] == 2
    assert report["aggregate"]["llm_judge_faithful_rate"] == 1.0
    assert all(case["llm_judge"]["faithful"] is True for case in report["cases"])


def test_run_answer_evaluation_is_deterministic_apart_from_timings():
    documents, cases = load_benchmark(
        BENCHMARK_DIR / "corpus.json", BENCHMARK_DIR / "cases.json"
    )

    def strip(report):
        report = dict(report)
        report.pop("timings_ms")
        report["aggregate"] = {
            key: value
            for key, value in report["aggregate"].items()
            if not key.endswith("_ms_mean")
        }
        report["cases"] = [
            {
                key: value
                for key, value in case.items()
                if key != "timings_ms"
            }
            for case in report["cases"]
        ]
        return report

    first = run_answer_evaluation(
        embedder=HashingEmbedder(), documents=documents, cases=cases,
        llm=StubLLMProvider(reply="Calvin cycle fixes carbon [1]."),
    )
    second = run_answer_evaluation(
        embedder=HashingEmbedder(), documents=documents, cases=cases,
        llm=StubLLMProvider(reply="Calvin cycle fixes carbon [1]."),
    )

    assert strip(first) == strip(second)
