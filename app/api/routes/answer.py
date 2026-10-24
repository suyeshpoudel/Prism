from fastapi import APIRouter, Depends, Request

from app.answer.service import generate_answer
from app.api.routes.documents import get_app_settings, get_embedder, get_vector_index
from app.config import Settings
from app.core.auth import get_current_owner
from app.core.errors import (
    AnswerGenerationFailedError,
    AnswerUnavailableError,
    EmbeddingFailedError,
    InvalidRequestError,
)
from app.embedding import Embedder
from app.embedding.provider import EmbeddingError
from app.llm.provider import LLMError, LLMProvider
from app.retrieval import NumpyVectorIndex
from app.schemas.answers import AnswerRequest, AnswerResponse
from app.schemas.documents import ErrorResponse, SearchResult

router = APIRouter(tags=["answer"])

ANSWER_ERROR_RESPONSES = {
    400: {"model": ErrorResponse},
    401: {"model": ErrorResponse},
    502: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
}


def get_llm(request: Request) -> LLMProvider | None:
    return request.app.state.llm


@router.post(
    "/answer",
    response_model=AnswerResponse,
    responses=ANSWER_ERROR_RESPONSES,
)
def answer(
    request: AnswerRequest,
    owner_id: str = Depends(get_current_owner),
    embedder: Embedder = Depends(get_embedder),
    index: NumpyVectorIndex = Depends(get_vector_index),
    settings: Settings = Depends(get_app_settings),
    llm: LLMProvider | None = Depends(get_llm),
) -> AnswerResponse:
    if llm is None:
        raise AnswerUnavailableError(
            "Answer generation is not configured. Set PRISM_LLM_BASE_URL to an "
            "OpenAI-compatible endpoint."
        )

    question = request.question.strip()
    if not question:
        raise InvalidRequestError("Question must not be empty.")
    top_k = request.top_k or settings.default_top_k

    try:
        query_vector = embedder.embed_query(question)
    except EmbeddingError as exc:
        raise EmbeddingFailedError("The embedding model failed while processing this question.") from exc

    evidence = index.search(
        query_vector,
        top_k=top_k,
        collection_id=request.collection_id,
        owner_id=owner_id,
    )

    try:
        outcome = generate_answer(
            llm=llm,
            question=question,
            evidence=evidence,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
        )
    except LLMError as exc:
        raise AnswerGenerationFailedError(
            "The language model failed while generating an answer."
        ) from exc

    return AnswerResponse(
        question=question,
        answer=outcome.answer,
        insufficient_evidence=outcome.insufficient_evidence,
        citations=outcome.citations,
        invalid_citation_markers=outcome.invalid_markers,
        evidence=[
            SearchResult(
                score=result.score,
                chunk=result.entry.chunk,
                document_id=result.entry.chunk.document_id,
                filename=result.entry.filename,
            )
            for result in outcome.evidence
        ],
        top_k=top_k,
        collection_id=request.collection_id,
        model=outcome.model,
        input_tokens=outcome.input_tokens,
        output_tokens=outcome.output_tokens,
    )
