"""
PROPCAST – Application Configuration
Reads settings from environment variables / .env file using pydantic-settings.
"""
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Central settings object.  All values can be overridden by env vars."""

    # --- Database ---
    database_url: str = "sqlite:///./propcast.db"

    # --- Security ---
    secret_key: str = "change-me-in-production"

    # --- External APIs ---
    odds_api_key: str = ""

    # --- Runtime ---
    environment: str = "development"
    log_level: str = "INFO"

    # --- Model Season & Recency Weights (Configurable & Dynamic) ---
    season_weights_w1_3_hist: float = 0.60
    season_weights_w1_3_curr: float = 0.40
    season_weights_w4_6_hist: float = 0.45
    season_weights_w4_6_curr: float = 0.55
    season_weights_w7_10_hist: float = 0.30
    season_weights_w7_10_curr: float = 0.70
    season_weights_w11_plus_hist: float = 0.18
    season_weights_w11_plus_curr: float = 0.82

    # --- Recency Decay Factor ---
    recency_decay_alpha: float = 0.92

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # Allow extra env vars without raising validation errors
        extra = "ignore"


settings = Settings()
