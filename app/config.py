from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.embedding.provider import DEFAULT_EMBEDDING_MODEL


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PRISM_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "Prism"
    environment: str = "development"
    api_v1_prefix: str = "/api/v1"
    cors_origins: list[str] = ["http://localhost:3000"]
    max_upload_bytes: int = 10 * 1024 * 1024
    chunk_size: int = 1000
    chunk_overlap: int = 150
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_batch_size: int = 32
    default_top_k: int = 5
    database_url: str = "postgresql+psycopg://prism:prism@localhost:5432/prism"
    api_keys: dict[str, str] = {}
    auto_create_schema: bool = True
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_max_tokens: int = 512
    llm_temperature: float = 0.0
    llm_timeout_seconds: float = 60.0

    evaluation_results_dir: Path = Path("benchmarks/results")
    evaluation_max_configurations: int = 8
    llm_input_cost_per_1m: float = 0.0
    llm_output_cost_per_1m: float = 0.0

    @model_validator(mode="after")
    def _validate_chunking(self) -> "Settings":
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
