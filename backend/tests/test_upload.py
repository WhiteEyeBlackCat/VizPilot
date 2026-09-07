from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

from .conftest import FIXTURE_DIR

MISSING_ID = "0" * 32


def _upload(client: TestClient, name: str, filename: str | None = None) -> dict:
    resp = client.post(
        "/api/datasets",
        files={"file": (filename or name, (FIXTURE_DIR / name).read_bytes())},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_health(client: TestClient) -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_upload_csv_metadata(client: TestClient) -> None:
    meta = _upload(client, "basic.csv")
    assert len(meta["dataset_id"]) == 32
    assert meta["filename"] == "basic.csv"
    assert meta["n_rows"] == 3
    assert meta["n_cols"] == 3
    assert [c["name"] for c in meta["columns"]] == ["ts", "city", "value"]
    dtypes = {c["name"]: c["dtype"] for c in meta["columns"]}
    assert dtypes["ts"].startswith("Datetime")
    assert dtypes["value"] == "Float64"


def test_upload_xlsx(client: TestClient) -> None:
    meta = _upload(client, "basic.xlsx")
    assert meta["n_rows"] == 2
    assert [c["name"] for c in meta["columns"]] == ["city", "pop"]


def test_upload_parquet(client: TestClient) -> None:
    meta = _upload(client, "basic.parquet")
    assert meta["n_rows"] == 2
    assert meta["n_cols"] == 2


def test_upload_uppercase_extension_accepted(client: TestClient) -> None:
    meta = _upload(client, "basic.csv", filename="BASIC.CSV")
    assert meta["n_rows"] == 3


def test_get_dataset_metadata(client: TestClient) -> None:
    meta = _upload(client, "basic.csv")
    resp = client.get(f"/api/datasets/{meta['dataset_id']}")
    assert resp.status_code == 200
    assert resp.json() == meta


def test_list_datasets(client: TestClient) -> None:
    first = _upload(client, "basic.csv")["dataset_id"]
    second = _upload(client, "basic.parquet")["dataset_id"]
    resp = client.get("/api/datasets")
    assert resp.status_code == 200
    listed = [m["dataset_id"] for m in resp.json()]
    assert set(listed) == {first, second}
    # Sorted by uploaded_at descending: most recent upload first.
    uploaded = [m["uploaded_at"] for m in resp.json()]
    assert uploaded == sorted(uploaded, reverse=True)


def test_preview_content_and_datetime_iso(client: TestClient) -> None:
    meta = _upload(client, "basic.csv")
    resp = client.get(f"/api/datasets/{meta['dataset_id']}/preview", params={"rows": 2})
    assert resp.status_code == 200
    records = resp.json()
    assert len(records) == 2
    assert records[0] == {"ts": "2024-01-01T08:00:00", "city": "Taipei", "value": 12.5}
    assert records[1]["value"] is None  # null preserved


def test_preview_default_rows_capped_by_dataset_size(client: TestClient) -> None:
    meta = _upload(client, "basic.csv")
    resp = client.get(f"/api/datasets/{meta['dataset_id']}/preview")
    assert len(resp.json()) == 3  # default 50 > n_rows -> all rows


def test_preview_rows_larger_than_dataset(client: TestClient) -> None:
    meta = _upload(client, "basic.csv")
    resp = client.get(f"/api/datasets/{meta['dataset_id']}/preview", params={"rows": 1000})
    assert len(resp.json()) == 3


def test_preview_rows_out_of_bounds_rejected(client: TestClient) -> None:
    meta = _upload(client, "basic.csv")
    for rows in (0, -1, 1001):
        resp = client.get(f"/api/datasets/{meta['dataset_id']}/preview", params={"rows": rows})
        assert resp.status_code == 422


def test_preview_nan_and_inf_become_null(client: TestClient) -> None:
    meta = _upload(client, "nan_inf.csv")
    records = client.get(f"/api/datasets/{meta['dataset_id']}/preview").json()
    assert [r["v"] for r in records] == [1.5, None, None, None]


def test_upload_unsupported_extension(client: TestClient) -> None:
    resp = client.post("/api/datasets", files={"file": ("notes.txt", b"plain text")})
    assert resp.status_code == 400
    assert "Unsupported" in resp.json()["detail"]


def test_upload_corrupt_parquet(client: TestClient) -> None:
    resp = client.post(
        "/api/datasets",
        files={"file": ("corrupt.parquet", (FIXTURE_DIR / "corrupt.parquet").read_bytes())},
    )
    assert resp.status_code == 400


def test_upload_empty_csv(client: TestClient) -> None:
    resp = client.post("/api/datasets", files={"file": ("empty.csv", b"")})
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()


def test_upload_header_only_csv(client: TestClient) -> None:
    resp = client.post(
        "/api/datasets",
        files={"file": ("header_only.csv", (FIXTURE_DIR / "header_only.csv").read_bytes())},
    )
    assert resp.status_code == 400
    assert "no data rows" in resp.json()["detail"]


def test_upload_latin1_csv(client: TestClient) -> None:
    meta = _upload(client, "latin1.csv")
    assert meta["n_rows"] == 2


def test_upload_over_size_limit_rejected(small_limit_client: TestClient) -> None:
    # 2 MB body against a 1 MB limit -> 413 (body is spooled by the framework,
    # but is never loaded into memory by the handler)
    payload = b"a,b\n" + b"1,2\n" * 500_000
    resp = small_limit_client.post("/api/datasets", files={"file": ("big.csv", payload)})
    assert resp.status_code == 413
    assert "limit" in resp.json()["detail"]


def test_invalid_dataset_id_format_is_404(client: TestClient) -> None:
    for bad_id in ("not-an-id", "Z" * 32, "0" * 31, "0" * 33, MISSING_ID.upper()):
        assert client.get(f"/api/datasets/{bad_id}").status_code == 404
        assert client.get(f"/api/datasets/{bad_id}/preview").status_code == 404


def test_unknown_dataset_id_is_404(client: TestClient) -> None:
    assert client.get(f"/api/datasets/{MISSING_ID}").status_code == 404
    assert client.get(f"/api/datasets/{MISSING_ID}/preview").status_code == 404


def test_lazy_reload_through_new_app(client: TestClient, settings: Settings) -> None:
    meta = _upload(client, "basic.csv")

    fresh = TestClient(create_app(settings))  # new store instance, same DATA_DIR
    assert fresh.get(f"/api/datasets/{meta['dataset_id']}").json() == meta
    records = fresh.get(f"/api/datasets/{meta['dataset_id']}/preview").json()
    assert len(records) == 3
    assert records[0]["ts"] == "2024-01-01T08:00:00"
