import ipaddress
from pathlib import Path
from urllib.parse import urlparse

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VIZPILOT_")

    data_dir: Path = BACKEND_DIR / "data"
    max_upload_mb: int = 200
    preview_rows: int = 50
    # DataFrames kept in memory (LRU); evicted ones are re-read from parquet.
    max_cached_frames: int = 4
    # Profiling runs on a seeded random sample above this row count.
    profile_sample_threshold: int = 100_000
    # Optional local LLM layer (D4): any OpenAI-compatible server.
    llm_enabled: bool = False
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = ""
    llm_api_key: str = ""  # local servers usually need none
    llm_timeout_seconds: float = 60  # the hypothesis prompt is longer than the stage-5 one (stage 17.3)
    llm_include_sample_rows: bool = True
    # Data sharing with a non-loopback endpoint (C1). Column names and summary
    # statistics always go to the LLM; sample rows and example category values
    # are actual cell contents, so they are withheld from any endpoint that is
    # not on this machine unless this is set explicitly.
    llm_allow_remote_data: bool = False

    @property
    def llm_endpoint_is_local(self) -> bool:
        return is_loopback_url(self.llm_base_url)

    @property
    def llm_may_send_cell_values(self) -> bool:
        """Sample rows / example category values may leave the process."""
        return self.llm_endpoint_is_local or self.llm_allow_remote_data

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


def is_loopback_url(url: str) -> bool:
    """True only for endpoints on this machine (localhost / loopback IPs).

    A LAN address is not local: the data would leave the machine.
    """
    host = (urlparse(url).hostname or "").lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
