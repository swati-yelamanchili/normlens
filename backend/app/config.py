import os
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "NormLens"
    app_version: str = "1.0.0"
    debug: bool = False

    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql://normlens:normlens@localhost:5432/normlens",
    )
    chroma_host: str = os.getenv("CHROMA_HOST", "localhost")
    chroma_port: int = int(os.getenv("CHROMA_PORT", "8001"))

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    embedding_backend: Literal["auto", "sentence-transformers", "hashing"] = os.getenv(
        "EMBEDDING_BACKEND", "auto"
    )
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "384"))
    similarity_top_k: int = int(os.getenv("SIMILARITY_TOP_K", "20"))

    upload_dir: str = os.getenv("UPLOAD_DIR", "/tmp/normlens/uploads")
    extraction_language: str = os.getenv("EXTRACTION_LANGUAGE", "en")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = os.getenv(
        "LOG_LEVEL", "INFO"
    )

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug(cls, value):
        if isinstance(value, str) and value.lower() in {"release", "prod", "production"}:
            return False
        return value

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
