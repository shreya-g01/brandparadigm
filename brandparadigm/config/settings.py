"""Application settings, loaded from environment variables / .env.

All configurable runtime paths and service parameters are centralized here
so every module (API, dashboard, scripts) depends on a single source of
truth instead of reading `os.environ` directly.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from brandparadigm.config.paths import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    log_level: str = Field(default="INFO")

    raw_data_dir: str = Field(default=str(RAW_DATA_DIR))
    processed_data_dir: str = Field(default=str(PROCESSED_DATA_DIR))

    model_sentiment_path: str = Field(default=str(MODELS_DIR / "sentiment"))
    model_topic_model_path: str = Field(default=str(MODELS_DIR / "bertopic_model"))
    model_topic_classifier_path: str = Field(default=str(MODELS_DIR / "topic_classifier"))

    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    dashboard_api_base_url: str = Field(default="http://localhost:8000")


@lru_cache
def get_settings() -> Settings:
    """Return a cached, process-wide Settings instance."""
    return Settings()
