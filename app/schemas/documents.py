from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str
    version: str
    environment: str


class PageExtraction(BaseModel):
    page_number: int = Field(ge=1)
    text: str
    char_count: int = Field(ge=0)
    is_empty: bool


class Chunk(BaseModel):
    id: str
    document_id: str
    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=0)
    text: str
    char_count: int = Field(ge=0)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)


class DocumentResponse(BaseModel):
    id: str
    filename: str
    content_type: str | None
    size_bytes: int
    page_count: int
    extracted_char_count: int
    chunk_count: int
    pages: list[PageExtraction]
    chunks: list[Chunk]
    warnings: list[str]
    status: str


class DocumentSummary(BaseModel):
    id: str
    filename: str
    size_bytes: int
    page_count: int
    chunk_count: int
    status: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    collection_id: str = Field(default="default", min_length=1, max_length=128)


class SearchResult(BaseModel):
    score: float
    chunk: Chunk
    document_id: str
    filename: str


class SearchResponse(BaseModel):
    query: str
    top_k: int
    collection_id: str
    results: list[SearchResult]
