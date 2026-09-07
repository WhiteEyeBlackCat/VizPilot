from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VIZPILOT_")

    data_dir: Path = BACKEND_DIR / "data"
    max_upload_mb: int = 200
    preview_rows: int = 50
    # Profiling runs on a seeded random sample above this row count.
    profile_sample_threshold: int = 100_000
    # Optional local LLM layer (D4): any OpenAI-compatible server.
    llm_enabled: bool = False
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = ""
    llm_api_key: str = ""  # local servers usually need none
    llm_timeout_seconds: float = 30
    llm_include_sample_rows: bool = True

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024
