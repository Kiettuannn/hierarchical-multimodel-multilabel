from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    APP_NAME: str = "HMM-Net Moderation API"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = False

    # Inference mode: "mock" (default for local PC without heavy GPU) or "pipeline"
    INFERENCE_MODE: str = "mock"

    # Temporary directory for video uploads
    UPLOAD_DIR: Path = Path("/tmp/hmm_uploads")
    MAX_UPLOAD_SIZE_MB: int = 100

    # CORS settings
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
