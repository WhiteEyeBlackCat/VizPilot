from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.data.make_fixtures import generate_all

FIXTURE_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="session", autouse=True)
def fixture_files() -> Path:
    generate_all(FIXTURE_DIR)
    return FIXTURE_DIR


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data")


@pytest.fixture()
def client(settings: Settings) -> TestClient:
    return TestClient(create_app(settings))


@pytest.fixture()
def small_limit_client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(Settings(data_dir=tmp_path / "data-small", max_upload_mb=1)))
