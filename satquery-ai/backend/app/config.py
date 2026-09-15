"""SatQuery AI — Configuration"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings


BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    # Database
    database_url: str = f"sqlite+aiosqlite:///{BACKEND_DIR / 'satquery.db'}"

    # Auth
    secret_key: str = "dev-secret-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440

    # LLM
    gemini_api_key: str = ""

    # Storage
    upload_dir: str = str(BACKEND_DIR / "uploads")
    model_cache_dir: str = str(BACKEND_DIR / "model_cache")
    report_dir: str = str(BACKEND_DIR / "reports")
    demo_dir: str = str(BACKEND_DIR / "data" / "demo")

    # Server
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    frontend_url: str = "http://localhost:3000"

    # Limits
    max_upload_size_mb: int = 500
    max_image_dimension: int = 10000

    class Config:
        env_file = [str(BACKEND_DIR / ".env"), str(PROJECT_ROOT / ".env"), ".env"]
        extra = "ignore"


settings = Settings()

# Ensure directories exist
for d in [settings.upload_dir, settings.model_cache_dir, settings.report_dir, settings.demo_dir]:
    Path(d).mkdir(parents=True, exist_ok=True)
