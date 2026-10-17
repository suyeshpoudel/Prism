from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PRISM_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "Prism"
    environment: str = "development"
    api_v1_prefix: str = "/api/v1"
    max_upload_bytes: int = 10 * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
