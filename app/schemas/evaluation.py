from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class BenchmarkCaseInfo(BaseModel):
    id: str
    question: str
    relevant: list[dict[str, Any]]
    k_values: list[int]


class BenchmarkDocumentInfo(BaseModel):
    filename: str
    page_count: int
    char_count: int


class BenchmarkInfo(BaseModel):
    documents: list[BenchmarkDocumentInfo]
    cases: list[BenchmarkCaseInfo]


class RetrievalConfiguration(BaseModel):
    chunk_size: int = Field(default=1000, ge=100, le=5000)
    chunk_overlap: int = Field(default=150, ge=0, le=4999)
    embedding_model: str | None = Field(default=None, min_length=1, max_length=200)

    @model_validator(mode="after")
    def _overlap_smaller_than_size(self) -> "RetrievalConfiguration":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class RetrievalEvaluationRequest(BaseModel):
    configurations: list[RetrievalConfiguration] = Field(default_factory=list, max_length=8)
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5], min_length=1, max_length=10)
    persist: bool = True

    @field_validator("k_values")
    @classmethod
    def _validate_k_values(cls, values: list[int]) -> list[int]:
        if any(value < 1 or value > 50 for value in values):
            raise ValueError("k values must be between 1 and 50")
        return sorted(set(values))


class AnswerEvaluationRequest(BaseModel):
    chunk_size: int = Field(default=1000, ge=100, le=5000)
    chunk_overlap: int = Field(default=150, ge=0, le=4999)
    top_k: int = Field(default=5, ge=1, le=50)
    embedding_model: str | None = Field(default=None, min_length=1, max_length=200)
    include_llm_judge: bool = False
    persist: bool = True

    @model_validator(mode="after")
    def _overlap_smaller_than_size(self) -> "AnswerEvaluationRequest":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class EvaluationRunSummary(BaseModel):
    id: str
    created_at: str | None = None
    type: str | None = None
    config: dict[str, Any] | None = None
    aggregate: dict[str, Any] | None = None
    configurations: int | None = None
    comparison: list[dict[str, Any]] | None = None
