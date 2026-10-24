from pydantic import BaseModel, Field

from app.schemas.documents import SearchResult


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    collection_id: str = Field(default="default", min_length=1, max_length=128)


class Citation(BaseModel):
    marker: int = Field(ge=1)
    chunk_id: str
    document_id: str
    filename: str
    page_number: int = Field(ge=1)
    text: str
    score: float


class AnswerResponse(BaseModel):
    question: str
    answer: str
    insufficient_evidence: bool
    citations: list[Citation]
    invalid_citation_markers: list[int]
    evidence: list[SearchResult]
    top_k: int
    collection_id: str
    model: str | None
    input_tokens: int | None = None
    output_tokens: int | None = None
