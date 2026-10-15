import pytest

from app.config import Settings, is_loopback_url
from app.main import create_app


@pytest.mark.parametrize(
    "url, local",
    [
        ("http://localhost:11434/v1", True),
        ("http://127.0.0.1:8080/v1", True),
        ("http://127.5.0.1/v1", True),
        ("http://[::1]:11434/v1", True),
        ("http://LOCALHOST:11434/v1", True),
        ("http://192.168.1.20:11434/v1", False),  # LAN: data leaves the machine
        ("http://10.0.0.5/v1", False),
        ("https://api.example.com/v1", False),
        ("http://localhost.example.com/v1", False),  # look-alike host
        ("", False),
    ],
)
def test_loopback_detection(url, local):
    assert is_loopback_url(url) is local


def _provider(tmp_path, **kw):
    settings = Settings(data_dir=tmp_path, llm_enabled=True, llm_model="m", **kw)
    return create_app(settings).state.recommendations._provider


def test_local_endpoint_keeps_sample_rows(tmp_path):
    assert _provider(tmp_path)._include_sample_rows is True


def test_remote_endpoint_withholds_sample_rows_by_default(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        provider = _provider(tmp_path, llm_base_url="https://api.example.com/v1")
    assert provider._include_sample_rows is False
    assert "not on this machine" in caplog.text


def test_remote_endpoint_explicit_opt_in(tmp_path):
    provider = _provider(
        tmp_path, llm_base_url="https://api.example.com/v1", llm_allow_remote_data=True
    )
    assert provider._include_sample_rows is True


def test_user_opt_out_still_wins_locally(tmp_path):
    assert _provider(tmp_path, llm_include_sample_rows=False)._include_sample_rows is False
