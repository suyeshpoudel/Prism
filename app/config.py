from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PRISM_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "Prism"
    environment: str = "development"
    api_v1_prefix: str = "/api/v1"
    max_upload_bytes: int = 10 * 1024 * 1024
    chunk_size: int = 1000
    chunk_overlap: int = 150

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
