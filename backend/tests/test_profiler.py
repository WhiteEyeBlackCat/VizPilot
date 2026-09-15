import json
import threading
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


# --- ProfileService concurrency (stage 9b) -----------------------------------


def _run_threads(targets) -> list[str]:
    errors: list[str] = []

    def wrap(fn):
        def run() -> None:
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - collected for the assertion
                errors.append(repr(exc))

        return run

    threads = [threading.Thread(target=wrap(fn)) for fn in targets]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


def test_profile_service_concurrent_get_computes_once(tmp_path: Path, counting_profile: dict) -> None:
    # the post-upload burst: profile + recs(llm=false) + recs(llm=true) all
    # miss the cache at the same time; the original code wrote one shared
    # .tmp and the second os.replace raised FileNotFoundError
    df = pl.DataFrame({"v": [1.0, 2.0, 3.0, 4.0], "g": ["a", "b", "a", "b"]})
    for rnd in range(20):
        service = ProfileService(tmp_path / str(rnd), BIG)
        (tmp_path / str(rnd)).mkdir()
        results: list[DatasetProfile] = []
        errors = _run_threads([lambda: results.append(service.get("a" * 32, df))] * 3)
        assert errors == [], f"round {rnd}: {errors}"
        assert len(results) == 3 and all(r is results[0] for r in results)
        assert counting_profile["n"] == rnd + 1  # exactly one computation per round
        assert (tmp_path / str(rnd) / f"{'a' * 32}.profile.json").is_file()


def test_profile_service_leaves_no_tmp_files(tmp_path: Path) -> None:
    df = pl.DataFrame({"v": [1.0, 2.0, 3.0]})
    service = ProfileService(tmp_path, BIG)
    assert _run_threads([lambda: service.get("b" * 32, df)] * 3) == []
    assert list(tmp_path.glob("*.tmp")) == []
    assert [p.name for p in tmp_path.iterdir()] == [f"{'b' * 32}.profile.json"]


def test_profile_service_write_failure_cleans_tmp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(src, dst):  # noqa: ANN001
        raise OSError("disk full")

    monkeypatch.setattr(profiler_module.os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        ProfileService(tmp_path, BIG).get("c" * 32, pl.DataFrame({"v": [1.0]}))
    assert list(tmp_path.iterdir()) == []


def test_profile_service_different_datasets_do_not_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # dataset A's computation is held on a barrier until dataset B has
    # finished: with a single global lock B would deadlock behind A
    real = profiler_module.profile_dataset
    b_done = threading.Event()

    def gated(df: pl.DataFrame, dataset_id: str, sample_threshold: int) -> DatasetProfile:
        if dataset_id == "a" * 32:
            assert b_done.wait(timeout=5), "dataset B was blocked behind dataset A"
        return real(df, dataset_id, sample_threshold)

    monkeypatch.setattr(profiler_module, "profile_dataset", gated)
    service = ProfileService(tmp_path, BIG)
    df = pl.DataFrame({"v": [1.0, 2.0]})

    def get_b() -> None:
        service.get("b" * 32, df)
        b_done.set()

    errors = _run_threads([lambda: service.get("a" * 32, df), get_b])
    assert errors == []
    assert set(service._cache) == {"a" * 32, "b" * 32}


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


# --- stage 9: single-execution pairwise correlations --------------------------


def _reference_pairwise(df: pl.DataFrame, cols: list[str]) -> list[list[float | None]]:
    """The pre-stage-9 per-pair algorithm (select/drop_nulls/filter/corr),
    kept here as the semantic oracle for the batched implementation."""
    n = len(cols)
    matrix: list[list[float | None]] = [[None] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1.0
        for j in range(i + 1, n):
            pair = df.select(cols[i], cols[j]).drop_nulls()
            for name in (cols[i], cols[j]):
                if pair.schema[name].is_float():
                    pair = pair.filter(pl.col(name).is_finite())
            value = pair.select(pl.corr(cols[i], cols[j])).item() if pair.height >= 2 else None
            if value is not None:
                value = float(value)
                value = value if value == value and abs(value) != float("inf") else None
            matrix[i][j] = matrix[j][i] = value
    return matrix


def _dirty_numeric_df() -> pl.DataFrame:
    import random

    rng = random.Random(9)
    n = 300

    def noisy(scale):
        out = []
        for i in range(n):
            r = rng.random()
            if r < 0.05:
                out.append(None)
            elif r < 0.08:
                out.append(float("nan"))
            elif r < 0.10:
                out.append(float("inf"))
            else:
                out.append(scale * i + rng.gauss(0, 20))
        return out

    return pl.DataFrame(
        {
            "a": noisy(1.0),
            "b": noisy(-2.0),
            "c": [rng.gauss(0, 1) for _ in range(n)],
            "i": [i if i % 7 else None for i in range(n)],  # Int64 with nulls
            "k": [i * 3 for i in range(n)],  # Int64, no nulls
            "const": [5.0] * n,
            "sparse": [float(i) if i < 2 else None for i in range(n)],  # only 2 valid rows
        }
    )


def _assert_matrix_close(actual, expected) -> None:
    for row_a, row_e in zip(actual, expected):
        for a, e in zip(row_a, row_e):
            if e is None:
                assert a is None
            else:
                assert a == pytest.approx(e, abs=1e-12)


def test_pairwise_pearson_matches_single_pair_reference() -> None:
    from app.profiling.pairwise import pairwise_pearson

    df = _dirty_numeric_df()
    cols = df.columns
    matrix, counts = pairwise_pearson(df, cols)
    _assert_matrix_close(matrix, _reference_pairwise(df, cols))
    # symmetric, unit diagonal
    for i in range(len(cols)):
        assert matrix[i][i] == 1.0
        for j in range(len(cols)):
            assert matrix[i][j] == matrix[j][i] and counts[i][j] == counts[j][i]


def test_pairwise_pearson_counts_are_pairwise_complete() -> None:
    from app.profiling.pairwise import pairwise_pearson

    df = pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, None, float("nan"), 6.0],
            "y": [1.0, None, 3.0, 4.0, 5.0, float("inf")],
            "z": [1, 2, 3, 4, 5, 6],
        }
    )
    matrix, counts = pairwise_pearson(df, ["x", "y", "z"])
    assert counts[0][0] == 4 and counts[1][1] == 4 and counts[2][2] == 6  # valid rows per column
    assert counts[0][1] == 2  # rows 0 and 2 only
    assert counts[0][2] == 4 and counts[1][2] == 4
    assert matrix[0][1] == pytest.approx(1.0)  # (1,1),(3,3)
    assert matrix[0][2] == pytest.approx(1.0)


def test_pairwise_pearson_sparse_and_constant_pairs_are_none() -> None:
    from app.profiling.pairwise import pairwise_pearson

    df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "one": [1.0, None, None], "c": [2.0, 2.0, 2.0]})
    matrix, counts = pairwise_pearson(df, ["a", "one", "c"])
    assert counts[0][1] == 1 and matrix[0][1] is None  # < 2 rows
    assert counts[0][2] == 3 and matrix[0][2] is None  # constant -> NaN -> None


def test_profile_correlations_carry_pair_counts() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, None], "y": [2.0, 4.0, 6.0, None, 10.0]})
    corr = _profile(df).correlations
    assert corr.pair_counts == [[4, 3], [3, 4]]
    assert corr.matrix[0][1] == pytest.approx(1.0)
    profile = _profile(pl.DataFrame({"x": [1.0, 2.0], "c": ["a", "b"]}))
    assert profile.profiled_rows == 2 and profile.correlations is None


def test_correlations_use_a_bounded_number_of_executions(monkeypatch: pytest.MonkeyPatch) -> None:
    # stage 9 #1 regression: 40 numeric columns used to cost 435 per-pair
    # dataframe executions; the matrix must now come from one select
    calls = {"select": 0, "collect": 0}
    real_select = pl.DataFrame.select
    real_collect = pl.LazyFrame.collect

    def counting_select(self, *args, **kwargs):
        calls["select"] += 1
        return real_select(self, *args, **kwargs)

    def counting_collect(self, *args, **kwargs):
        calls["collect"] += 1
        return real_collect(self, *args, **kwargs)

    monkeypatch.setattr(pl.DataFrame, "select", counting_select)
    monkeypatch.setattr(pl.LazyFrame, "collect", counting_collect)
    data = {f"c{i:02d}": [float((i + 1) * j % 17) for j in range(50)] for i in range(40)}
    corr = profiler_module._correlations(pl.DataFrame(data), list(data))
    assert corr is not None and corr.truncated and len(corr.columns) == 30
    assert calls["select"] + calls["collect"] <= 2


def test_sampled_profile_counts_are_sample_level() -> None:
    df = pl.DataFrame({"v": [float(i) for i in range(200)], "w": [float(i * i) for i in range(200)]})
    profile = profile_dataset(df, "0" * 32, sample_threshold=50)
    assert profile.sampled and profile.n_rows == 200 and profile.profiled_rows == 50
    assert profile.correlations.pair_counts[0][1] == 50


# --- stage 9 (stage 2): per-column quality counts ---------------------------


def test_every_column_has_quality_with_consistent_counts() -> None:
    df = pl.DataFrame(
        {
            "num": [1.0, None, float("nan"), float("inf"), 5.0, 6.0],
            "ints": [1, 2, None, 4, 5, 6],
            "cat": ["a", "b", None, "a", "b", "a"],
            "flag": [True, False, True, None, True, False],
            "empty": pl.Series([None] * 6, dtype=pl.String),
            "text": [f"a long free-form sentence with number {i} in it" for i in range(6)],
        }
    )
    profile = _profile(df)
    assert profile.profiled_rows == 6
    quality = {c.name: c.quality for c in profile.columns}
    assert all(q is not None for q in quality.values())
    for q in quality.values():
        assert q.profiled_rows == 6
        assert q.valid_count == 6 - q.missing_count - q.missing_token_count - q.invalid_count
        assert q.valid_ratio == pytest.approx(q.valid_count / 6)
        assert q.missing_token_count == 0  # attributed in stage 3
        assert q.q1 is None and q.robust_range is None and q.suspected_sentinels == []

    assert (quality["num"].missing_count, quality["num"].invalid_count, quality["num"].valid_count) == (1, 2, 3)
    assert (quality["ints"].missing_count, quality["ints"].invalid_count) == (1, 0)
    assert (quality["cat"].missing_count, quality["cat"].valid_count) == (1, 5)
    assert (quality["flag"].missing_count, quality["flag"].valid_count) == (1, 5)
    assert (quality["empty"].missing_count, quality["empty"].valid_count, quality["empty"].valid_ratio) == (6, 0, 0.0)
    assert quality["text"].valid_count == 6


def test_quality_is_sample_level_when_sampled() -> None:
    df = pl.DataFrame({"v": [float(i) if i % 4 else None for i in range(400)]})
    profile = profile_dataset(df, "0" * 32, sample_threshold=100)
    q = _col(profile, "v").quality
    assert profile.sampled and q.profiled_rows == 100
    assert q.missing_count + q.valid_count == 100
    assert _col(profile, "v").missing_count == 100  # full-table raw nulls


def test_quality_and_nominal_survive_json_roundtrip() -> None:
    df = pl.DataFrame({"v": [1.0, None, 3.0], "c": ["x", "y", "x"]})
    profile = _profile(df)
    assert all(not c.nominal for c in profile.columns)
    reloaded = DatasetProfile.model_validate_json(profile.model_dump_json())
    assert reloaded == profile
    assert reloaded.profile_version == PROFILE_VERSION


def test_quality_counts_datetime_parse_failures_as_invalid() -> None:
    # 10 parseable of 11 non-null (> 90% probe ratio) -> datetime; the one
    # unparseable value becomes a null the raw column did not have
    values = [f"2024-01-{i + 1:02d}" for i in range(10)] + ["bogus", None]
    profile = _profile(pl.DataFrame({"when": values}))
    col = _col(profile, "when")
    assert col.semantic_type == "datetime"
    assert (col.quality.missing_count, col.quality.invalid_count, col.quality.valid_count) == (1, 1, 10)
    assert col.missing_count == 1  # raw nulls only


def test_quality_counts_numeric_cast_failures_as_invalid() -> None:
    # 58 of 59 non-token strings parse -> numeric; "n/a" is a known missing
    # token (stage 3), "five" fails the numeric gate: both become nulls the
    # raw column did not have, attributed to their own counters
    values = [f"{i:,}" for i in range(1000, 1058)] + ["n/a", "five", None]
    profile = _profile(pl.DataFrame({"amount": values}))
    col = _col(profile, "amount")
    assert col.semantic_type == "numeric" and col.cast_params.thousands
    q = col.quality
    assert (q.missing_count, q.missing_token_count, q.invalid_count, q.valid_count) == (1, 1, 1, 58)
    assert col.quality.valid_ratio == pytest.approx(58 / 61)
    assert col.missing_count == 1  # raw full-table nulls keep their meaning


# --- stage 9 (stage 3): dirty numeric through the profile and render API ------

# 25 values per cycle: 2 missing tokens, 1 invalid, 22 numeric (22/23 non-token
# values parse, above the 0.95 gate share) -> numeric, 4 cycles = 100 rows
DIRTY_PRICE_VALUES = [
    "$1,234.50", "$899", " 99 ", "N/A", "1234.5", "$2,000", "42", "unknown", "7.25", "$15", "16",
    "17.5", "$18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "12kg",
] * 4
DIRTY_PRICE_CSV = (
    "id,price\n"
    + "\n".join(f'{i},"{v}"' for i, v in enumerate(DIRTY_PRICE_VALUES))
    + "\n"
)


def test_profile_attributes_tokens_and_invalids_for_price() -> None:
    values = DIRTY_PRICE_VALUES
    profile = _profile(pl.DataFrame({"price": values}))
    col = _col(profile, "price")
    assert col.semantic_type == "numeric"
    assert col.cast_params.currency and col.cast_params.thousands and not col.cast_params.percent
    q = col.quality
    assert (q.missing_count, q.missing_token_count, q.invalid_count, q.valid_count) == (0, 8, 4, 88)
    assert q.valid_ratio == pytest.approx(0.88)
    assert col.missing_count == 0  # raw nulls only
    assert col.max == 2000.0 and col.min == 7.25
    # replay on the full frame reproduces the probe's nulls
    replayed = apply_semantic_casts(pl.DataFrame({"price": values}), profile)["price"]
    assert replayed.null_count() == q.missing_token_count + q.invalid_count


def test_render_histogram_over_dirty_price_via_api(client: TestClient) -> None:
    dataset_id = client.post(
        "/api/datasets", files={"file": ("dirty.csv", DIRTY_PRICE_CSV.encode())}
    ).json()["dataset_id"]
    profile = client.get(f"/api/datasets/{dataset_id}/profile").json()
    price = next(c for c in profile["columns"] if c["name"] == "price")
    assert price["semantic_type"] == "numeric"
    assert price["quality"]["missing_token_count"] == 8 and price["quality"]["invalid_count"] == 4
    resp = client.post(
        "/api/charts/render",
        json={"dataset_id": dataset_id, "spec": {"title": "t", "type": "histogram", "x": "price"}},
    )
    assert resp.status_code == 200, resp.text
    bins = resp.json()["chart_data"]["bins"]
    assert sum(bins["counts"]) == 88  # tokens and invalid values are not binned
    assert bins["edges"][0] == pytest.approx(7.25) and bins["edges"][-1] == pytest.approx(2000.0)


# --- stage 9 (stage 4): nominal / id in the profile --------------------------


def test_profile_marks_nominal_codes_and_keeps_id_metadata() -> None:
    import random

    rng = random.Random(103)
    n = 500
    df = pl.DataFrame(
        {
            "user_id": [rng.randint(1, 60) for _ in range(n)],
            "code": [rng.randint(1, 49) for _ in range(n)],
            "zip_code": [rng.randint(10_000, 99_999) for _ in range(n)],
            "amount": [rng.uniform(1, 100) for _ in range(n)],
        }
    )
    profile = _profile(df)
    code = _col(profile, "code")
    assert code.semantic_type == "categorical" and code.nominal
    assert code.n_categories == 49 and code.top_values  # categorical stats present
    zip_code = _col(profile, "zip_code")
    assert zip_code.semantic_type == "id" and not zip_code.nominal
    assert zip_code.unique_count > 400 and zip_code.missing_ratio == 0.0  # metadata retained
    assert zip_code.quality is not None and zip_code.quality.valid_count == n
    assert _col(profile, "user_id").semantic_type == "id"
    assert profile.correlations is None  # amount is the only numeric column left
    assert all(c.name not in ("code", "zip_code", "user_id") for c in profile.columns if c.semantic_type == "numeric")
    reloaded = DatasetProfile.model_validate_json(profile.model_dump_json())
    assert reloaded == profile
