import json
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from fastapi.testclient import TestClient

import app.profiling.profiler as profiler_module
from app.profiling.models import PROFILE_VERSION, DatasetProfile
from app.profiling.profiler import ProfileService, profile_dataset
from app.profiling.types import apply_semantic_casts

BIG = 10**6


def _profile(df: pl.DataFrame, threshold: int = BIG) -> DatasetProfile:
    return profile_dataset(df, "0" * 32, threshold)


def _col(profile: DatasetProfile, name: str):
    return next(c for c in profile.columns if c.name == name)


# --- statistics -------------------------------------------------------------


def test_numeric_stats_on_known_values() -> None:
    c = _col(_profile(pl.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0]})), "v")
    assert c.semantic_type == "numeric"
    assert (c.min, c.max, c.mean, c.median) == (1.0, 5.0, 3.0, 3.0)
    # polars default quantile interpolation is "nearest"
    assert (c.q25, c.q75) == (2.0, 4.0)
    assert c.std == pytest.approx(1.5811388300841898)
    assert c.skewness == pytest.approx(0.0)


def test_missing_and_unique_counts() -> None:
    c = _col(_profile(pl.DataFrame({"v": [1.0, None, 3.0, 3.0]})), "v")
    assert c.missing_count == 1
    assert c.missing_ratio == pytest.approx(0.25)
    assert c.unique_count == 2  # nulls excluded


def test_nan_and_inf_excluded_from_stats() -> None:
    c = _col(_profile(pl.DataFrame({"v": [1.0, float("nan"), float("inf"), 3.0]})), "v")
    assert (c.min, c.max, c.mean) == (1.0, 3.0, 2.0)


def test_categorical_top_values() -> None:
    c = _col(_profile(pl.DataFrame({"c": ["a", "a", "a", "b", "b", "c", None]})), "c")
    assert c.semantic_type == "categorical"
    assert c.n_categories == 3
    assert [(t.value, t.count) for t in c.top_values] == [("a", 3), ("b", 2), ("c", 1)]


def test_boolean_stats() -> None:
    c = _col(_profile(pl.DataFrame({"b": [True, True, False]})), "b")
    assert c.semantic_type == "boolean"
    assert [(t.value, t.count) for t in c.top_values] == [(True, 2), (False, 1)]


def test_datetime_stats_and_daily_frequency() -> None:
    days = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(10)]
    c = _col(_profile(pl.DataFrame({"ts": days})), "ts")
    assert c.semantic_type == "datetime"
    assert c.min == "2024-01-01T00:00:00"
    assert c.max == "2024-01-10T00:00:00"
    assert c.inferred_frequency == "daily"


def test_long_format_repeated_dates_still_daily() -> None:
    # 3 groups x 10 days: each date appears 3 times; gaps must be computed
    # on distinct timestamps or the zeros drown the true daily cadence
    days = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(10)]
    df = pl.DataFrame({"ts": days * 3, "city": ["a"] * 10 + ["b"] * 10 + ["c"] * 10})
    assert _col(_profile(df), "ts").inferred_frequency == "daily"


def test_datetime_irregular_frequency() -> None:
    ts = [datetime(2024, 1, 1), datetime(2024, 1, 4), datetime(2024, 3, 1), datetime(2024, 3, 2)]
    assert _col(_profile(pl.DataFrame({"ts": ts})), "ts").inferred_frequency == "irregular"


def test_text_stats() -> None:
    values = [f"quite a long unique descriptive sentence {i}" for i in range(30)]
    c = _col(_profile(pl.DataFrame({"description": values})), "description")
    assert c.semantic_type == "text"
    assert c.max_length == max(len(v) for v in values)
    assert c.avg_length == pytest.approx(sum(len(v) for v in values) / 30)


def test_all_null_column_unknown_with_empty_stats() -> None:
    c = _col(_profile(pl.DataFrame({"v": [1.0, 2.0], "empty": pl.Series([None, None], dtype=pl.String)})), "empty")
    assert c.semantic_type == "unknown"
    assert c.missing_count == 2
    assert c.unique_count == 0
    assert c.mean is None and c.top_values is None and c.min is None


def test_sample_rows_are_first_five_raw_rows() -> None:
    profile = _profile(pl.DataFrame({"v": [float(i) for i in range(10)]}))
    assert [r["v"] for r in profile.sample_rows] == [0.0, 1.0, 2.0, 3.0, 4.0]


# --- correlations -----------------------------------------------------------


def test_correlations_known_values() -> None:
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    profile = _profile(pl.DataFrame({"x": x, "y": [2 * v for v in x], "z": [-v for v in x]}))
    corr = profile.correlations
    assert corr is not None and corr.columns == ["x", "y", "z"] and not corr.truncated
    i, j, k = 0, 1, 2
    assert corr.matrix[i][i] == 1.0
    assert corr.matrix[i][j] == pytest.approx(1.0)
    assert corr.matrix[i][k] == pytest.approx(-1.0)
    assert corr.matrix[j][i] == corr.matrix[i][j]


def test_correlations_pairwise_with_nulls() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, None], "y": [2.0, 4.0, 6.0, None, 10.0]})
    assert _profile(df).correlations.matrix[0][1] == pytest.approx(1.0)


def test_correlations_exclude_nan_rows() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, float("nan"), 8.0]})
    assert _profile(df).correlations.matrix[0][1] == pytest.approx(1.0)


def test_correlations_include_string_derived_numeric() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "amount": ["1,000", "2,000", "3,000"]})
    corr = _profile(df).correlations
    assert corr is not None and set(corr.columns) == {"x", "amount"}
    assert corr.matrix[0][1] == pytest.approx(1.0)


def test_correlations_none_for_single_numeric() -> None:
    assert _profile(pl.DataFrame({"x": [1.0, 2.0], "c": ["a", "b"]})).correlations is None


def test_correlations_truncated_at_30_columns() -> None:
    data = {f"c{i:02d}": [float(i), float(i) + 1, float(i) * 2, 0.0] for i in range(35)}
    corr = _profile(pl.DataFrame(data)).correlations
    assert corr is not None and corr.truncated
    assert len(corr.columns) == 30 and len(corr.matrix) == 30


# --- casts round-trip and sampling ------------------------------------------


def _mixed_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": ["31/12/2024", "01/06/2024", "15/03/2024", None],
            "amount": ["1,234", "$5,678", "42", "9,000"],
            "city": ["Taipei", "Tainan", "Taipei", "Hualien"],
        }
    )


def test_apply_semantic_casts_round_trip_matches_profile() -> None:
    df = _mixed_df()
    profile = _profile(df)
    assert _col(profile, "date").semantic_type == "datetime"
    assert _col(profile, "amount").semantic_type == "numeric"

    casted = apply_semantic_casts(df, profile)  # replay from stored params only
    assert casted.schema["date"] == pl.Datetime("us")
    assert casted.schema["amount"] == pl.Float64
    assert casted["amount"].to_list() == [1234.0, 5678.0, 42.0, 9000.0]

    # Stage 4 contract: profiling the replayed frame yields identical stats
    reprofile = _profile(casted)
    for name in ("date", "amount"):
        a, b = _col(profile, name), _col(reprofile, name)
        assert (a.min, a.max, a.mean, a.median, a.std) == (b.min, b.max, b.mean, b.median, b.std)
        assert a.unique_count == b.unique_count


def test_sampling_path_is_deterministic() -> None:
    df = pl.DataFrame({"v": [float(i) for i in range(200)], "c": ["a", "b"] * 100})
    p1 = profile_dataset(df, "0" * 32, sample_threshold=50)
    p2 = profile_dataset(df, "0" * 32, sample_threshold=50)
    assert p1.sampled and p1.n_rows == 200  # n_rows stays full-count
    assert p1.model_dump() == p2.model_dump()  # seeded sample -> reproducible


def test_profile_version_embedded() -> None:
    assert _profile(pl.DataFrame({"v": [1.0, 2.0]})).profile_version == PROFILE_VERSION


# --- ProfileService cache ---------------------------------------------------


@pytest.fixture()
def counting_profile(monkeypatch: pytest.MonkeyPatch):
    calls = {"n": 0}
    real = profiler_module.profile_dataset

    def counted(df: pl.DataFrame, dataset_id: str, sample_threshold: int) -> DatasetProfile:
        calls["n"] += 1
        return real(df, dataset_id, sample_threshold)

    monkeypatch.setattr(profiler_module, "profile_dataset", counted)
    return calls


def test_profile_cached_in_memory_and_on_disk(tmp_path: Path, counting_profile: dict) -> None:
    df = pl.DataFrame({"v": [1.0, 2.0, 3.0]})
    service = ProfileService(tmp_path, BIG)
    first = service.get("a" * 32, df)
    assert service.get("a" * 32, df) == first
    assert counting_profile["n"] == 1
    assert (tmp_path / f"{'a' * 32}.profile.json").is_file()

    fresh = ProfileService(tmp_path, BIG)  # process restart: reads the file
    assert fresh.get("a" * 32, df) == first
    assert counting_profile["n"] == 1


def test_profile_cache_version_mismatch_recomputes(tmp_path: Path, counting_profile: dict) -> None:
    df = pl.DataFrame({"v": [1.0, 2.0, 3.0]})
    ProfileService(tmp_path, BIG).get("a" * 32, df)
    path = tmp_path / f"{'a' * 32}.profile.json"
    stale = json.loads(path.read_text())
    stale["profile_version"] = PROFILE_VERSION - 1
    path.write_text(json.dumps(stale))

    profile = ProfileService(tmp_path, BIG).get("a" * 32, df)
    assert counting_profile["n"] == 2
    assert profile.profile_version == PROFILE_VERSION
    assert json.loads(path.read_text())["profile_version"] == PROFILE_VERSION  # overwritten


def test_profile_cache_corrupt_file_recomputes(tmp_path: Path, counting_profile: dict) -> None:
    path = tmp_path / f"{'a' * 32}.profile.json"
    path.write_text("{not json")
    ProfileService(tmp_path, BIG).get("a" * 32, pl.DataFrame({"v": [1.0, 2.0]}))
    assert counting_profile["n"] == 1
    assert json.loads(path.read_text())["profile_version"] == PROFILE_VERSION


# --- API --------------------------------------------------------------------

MIXED_CSV = (
    "date,city,amount,user_id,score\n"
    "31/12/2024,Taipei,\"1,234\",u001,4\n"
    "01/06/2024,Tainan,\"5,678\",u002,5\n"
    "15/03/2024,Taipei,\"2,000\",u003,3\n"
    "20/08/2024,Hualien,\"9,999\",u004,4\n"
)


def _upload_mixed(client: TestClient) -> str:
    resp = client.post("/api/datasets", files={"file": ("mixed.csv", MIXED_CSV.encode())})
    assert resp.status_code == 200, resp.text
    return resp.json()["dataset_id"]


def test_profile_endpoint_semantic_types(client: TestClient) -> None:
    dataset_id = _upload_mixed(client)
    resp = client.get(f"/api/datasets/{dataset_id}/profile")
    assert resp.status_code == 200
    profile = DatasetProfile.model_validate(resp.json())  # schema-valid
    assert profile.dataset_id == dataset_id and profile.n_rows == 4
    types = {c.name: c.semantic_type for c in profile.columns}
    assert types == {
        "date": "datetime",
        "city": "categorical",
        "amount": "numeric",
        "user_id": "id",
        "score": "categorical",
    }
    assert _col(profile, "amount").max == 9999.0
    # the loader's try_parse_dates already yields a native Date column, so the
    # string-cast path is asserted on amount (thousands-separated numbers)
    assert _col(profile, "amount").cast_params.target == "numeric"
    assert _col(profile, "amount").cast_params.thousands is True
    assert len(profile.sample_rows) == 4


def test_profile_endpoint_404(client: TestClient) -> None:
    assert client.get(f"/api/datasets/{'0' * 32}/profile").status_code == 404
    assert client.get("/api/datasets/not-an-id/profile").status_code == 404


def test_profile_json_has_no_nan_or_infinity(client: TestClient) -> None:
    csv = "v\n1.5\nNaN\ninf\n-inf\n2.5\n"
    resp = client.post("/api/datasets", files={"file": ("nan.csv", csv.encode())})
    dataset_id = resp.json()["dataset_id"]
    raw = client.get(f"/api/datasets/{dataset_id}/profile").text
    assert "NaN" not in raw and "Infinity" not in raw
    json.loads(raw)  # strict-parseable


def test_profile_cache_file_and_dataset_listing(client: TestClient, settings) -> None:
    dataset_id = _upload_mixed(client)
    assert client.get(f"/api/datasets/{dataset_id}/profile").status_code == 200
    assert (settings.data_dir / f"{dataset_id}.profile.json").is_file()
    # profile cache files must not leak into the dataset listing
    listed = client.get("/api/datasets").json()
    assert [m["dataset_id"] for m in listed] == [dataset_id]
