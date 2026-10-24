from pathlib import Path

from fastapi import APIRouter, Depends, Request

from app.api.routes.documents import get_app_settings, get_embedder
from app.config import Settings
from app.core.auth import get_current_owner
from app.core.errors import (
    AnswerUnavailableError,
    EmbeddingFailedError,
    EvaluationRunNotFoundError,
    InvalidRequestError,
)
from app.embedding import Embedder, get_shared_embedder
from app.embedding.provider import EmbeddingError
from app.evaluation.answer_eval import run_answer_evaluation
from app.evaluation.benchmark import load_benchmark, run_benchmark
from app.evaluation.store import EvaluationRunStore
from app.llm.provider import LLMProvider
from app.processing.chunking import ChunkingConfig
from app.schemas.evaluation import (
    AnswerEvaluationRequest,
    BenchmarkInfo,
    RetrievalConfiguration,
    RetrievalEvaluationRequest,
    EvaluationRunSummary,
)

router = APIRouter(prefix="/evaluation", tags=["evaluation"])

BENCHMARK_DIR = Path(__file__).resolve().parents[3] / "benchmarks" / "retrieval"
CORPUS_PATH = BENCHMARK_DIR / "corpus.json"
CASES_PATH = BENCHMARK_DIR / "cases.json"

ERROR_RESPONSES = {
    400: {"description": "Invalid evaluation request"},
    401: {"description": "Missing or invalid API key"},
    404: {"description": "Evaluation run not found"},
    500: {"description": "Embedding failed"},
    503: {"description": "Answer generation unavailable"},
}


def get_evaluation_store(request: Request) -> EvaluationRunStore:
    return request.app.state.evaluation_store


def get_llm(request: Request) -> LLMProvider | None:
    return request.app.state.llm


def _load():
    return load_benchmark(CORPUS_PATH, CASES_PATH)


def _embedder_for(
    model_name: str | None,
    *,
    default_embedder: Embedder,
    settings: Settings,
) -> Embedder:
    if model_name is None or model_name == settings.embedding_model:
        return default_embedder
    return get_shared_embedder(model_name)


def _validate_configuration_limit(count: int, settings: Settings) -> None:
    if count > settings.evaluation_max_configurations:
        raise InvalidRequestError(
            "Too many configurations; use at most "
            f"{settings.evaluation_max_configurations} per run."
        )


@router.get("/benchmark", response_model=BenchmarkInfo, responses=ERROR_RESPONSES)
def benchmark_info(owner_id: str = Depends(get_current_owner)) -> BenchmarkInfo:
    documents, cases = _load()
    return BenchmarkInfo(
        documents=[
            {
                "filename": document.filename,
                "page_count": len(document.pages),
                "char_count": sum(len(page) for page in document.pages),
            }
            for document in documents
        ],
        cases=[
            {
                "id": case.id,
                "question": case.question,
                "relevant": [
                    {"filename": filename, "page_number": page}
                    for filename, page in sorted(case.relevant)
                ],
                "k_values": list(case.k_values),
            }
            for case in cases
        ],
    )


@router.post("/retrieval", status_code=201, responses=ERROR_RESPONSES)
def run_retrieval_evaluation(
    request: RetrievalEvaluationRequest,
    owner_id: str = Depends(get_current_owner),
    embedder: Embedder = Depends(get_embedder),
    settings: Settings = Depends(get_app_settings),
    store: EvaluationRunStore = Depends(get_evaluation_store),
) -> dict:
    configurations = request.configurations or [
        RetrievalConfiguration(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    ]
    _validate_configuration_limit(len(configurations), settings)

    documents, cases = _load()
    results = []
    comparison = []
    for configuration in configurations:
        active_embedder = _embedder_for(
            configuration.embedding_model,
            default_embedder=embedder,
            settings=settings,
        )
        try:
            report = run_benchmark(
                embedder=active_embedder,
                documents=documents,
                cases=cases,
                chunking_config=ChunkingConfig(
                    chunk_size=configuration.chunk_size,
                    chunk_overlap=configuration.chunk_overlap,
                ),
                k_values=tuple(request.k_values),
                batch_size=settings.embedding_batch_size,
            )
        except EmbeddingError as exc:
            raise EmbeddingFailedError(
                "The embedding model failed while running the evaluation."
            ) from exc

        report["type"] = "retrieval"
        results.append(report)
        comparison.append(
            {
                "embedding_model": report["config"]["embedding_model"],
                "chunk_size": configuration.chunk_size,
                "chunk_overlap": configuration.chunk_overlap,
                **report["aggregate"],
            }
        )

    record = store.build_record(
        {
            "type": "retrieval",
            "config": {
                "k_values": list(request.k_values),
                "configurations": len(results),
            },
            "results": results,
            "comparison": comparison,
        },
        owner_id=owner_id,
    )
    if request.persist:
        store.save(record)
    return record


@router.post("/answers", status_code=201, responses=ERROR_RESPONSES)
def run_answer_evaluation_route(
    request: AnswerEvaluationRequest,
    owner_id: str = Depends(get_current_owner),
    embedder: Embedder = Depends(get_embedder),
    settings: Settings = Depends(get_app_settings),
    store: EvaluationRunStore = Depends(get_evaluation_store),
    llm: LLMProvider | None = Depends(get_llm),
) -> dict:
    if llm is None:
        raise AnswerUnavailableError(
            "Answer evaluation requires a configured LLM. Set PRISM_LLM_BASE_URL."
        )

    active_embedder = _embedder_for(
        request.embedding_model,
        default_embedder=embedder,
        settings=settings,
    )
    documents, cases = _load()
    try:
        report = run_answer_evaluation(
            embedder=active_embedder,
            documents=documents,
            cases=cases,
            llm=llm,
            chunking_config=ChunkingConfig(
                chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap,
            ),
            top_k=request.top_k,
            batch_size=settings.embedding_batch_size,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            include_llm_judge=request.include_llm_judge,
            input_cost_per_1m=settings.llm_input_cost_per_1m,
            output_cost_per_1m=settings.llm_output_cost_per_1m,
        )
    except EmbeddingError as exc:
        raise EmbeddingFailedError(
            "The embedding model failed while running the evaluation."
        ) from exc

    record = store.build_record(report, owner_id=owner_id)
    if request.persist:
        store.save(record)
    return record


@router.get("/runs", response_model=list[EvaluationRunSummary], responses=ERROR_RESPONSES)
def list_evaluation_runs(
    owner_id: str = Depends(get_current_owner),
    store: EvaluationRunStore = Depends(get_evaluation_store),
) -> list[EvaluationRunSummary]:
    return [EvaluationRunSummary(**summary) for summary in store.list(owner_id=owner_id)]


@router.get("/runs/{run_id}", responses=ERROR_RESPONSES)
def get_evaluation_run(
    run_id: str,
    owner_id: str = Depends(get_current_owner),
    store: EvaluationRunStore = Depends(get_evaluation_store),
) -> dict:
    record = store.get(run_id, owner_id=owner_id)
    if record is None:
        raise EvaluationRunNotFoundError(f"Evaluation run '{run_id}' was not found.")
    return record
