from fastapi import APIRouter, Depends

from app.api.routes.documents import get_app_settings, get_embedder, get_vector_index
from app.config import Settings
from app.core.auth import get_current_owner
from app.core.errors import EmbeddingFailedError, InvalidRequestError
from app.embedding import Embedder
from app.embedding.provider import EmbeddingError
from app.retrieval import NumpyVectorIndex
from app.schemas.documents import SearchRequest, SearchResponse, SearchResult

router = APIRouter(tags=["search"])


@router.post(
    "/search",
    response_model=SearchResponse,
    responses={
        400: {"description": "Invalid query"},
        401: {"description": "Missing or invalid API key"},
    },
)
def search(
    request: SearchRequest,
    owner_id: str = Depends(get_current_owner),
    embedder: Embedder = Depends(get_embedder),
    index: NumpyVectorIndex = Depends(get_vector_index),
    settings: Settings = Depends(get_app_settings),
) -> SearchResponse:
    query = request.query.strip()
    if not query:
        raise InvalidRequestError("Query must not be empty.")
    top_k = request.top_k or settings.default_top_k

    try:
        query_vector = embedder.embed_query(query)
    except EmbeddingError as exc:
        raise EmbeddingFailedError("The embedding model failed while processing this query.") from exc

    results = index.search(
        query_vector,
        top_k=top_k,
        collection_id=request.collection_id,
        owner_id=owner_id,
    )
    return SearchResponse(
        query=query,
        top_k=top_k,
        collection_id=request.collection_id,
        results=[
            SearchResult(
                score=result.score,
                chunk=result.entry.chunk,
                document_id=result.entry.chunk.document_id,
                filename=result.entry.filename,
            )
            for result in results
        ],
    )
