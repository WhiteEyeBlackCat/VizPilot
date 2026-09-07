from pathlib import Path

import polars as pl
import pytest

from app.datasets.loader import LoaderError, load_dataframe
from app.datasets.store import DatasetNotFoundError, DatasetStore

from .conftest import FIXTURE_DIR


def _load(name: str) -> pl.DataFrame:
    return load_dataframe((FIXTURE_DIR / name).read_bytes(), name)


def test_csv_basic_parses_dates_and_nulls() -> None:
    df = _load("basic.csv")
    assert df.shape == (3, 3)
    assert df.schema["ts"] == pl.Datetime("us")
    assert df.schema["value"] == pl.Float64
    assert df["value"].to_list()[1] is None


def test_csv_latin1_falls_back_to_utf8_lossy() -> None:
    df = _load("latin1.csv")
    assert df.shape == (2, 2)
    assert df.columns == ["name", "val"]


def test_csv_empty_file_rejected() -> None:
    with pytest.raises(LoaderError, match="empty"):
        _load("empty.csv")


def test_csv_header_only_rejected_distinctly() -> None:
    with pytest.raises(LoaderError, match="no data rows"):
        _load("header_only.csv")


def test_unsupported_extension_rejected() -> None:
    with pytest.raises(LoaderError, match="Unsupported"):
        _load("notes.txt")


def test_extension_check_is_case_insensitive() -> None:
    data = (FIXTURE_DIR / "basic.csv").read_bytes()
    assert load_dataframe(data, "BASIC.CSV").shape == (3, 3)


def test_xlsx_reads_first_sheet() -> None:
    df = _load("basic.xlsx")
    assert df.columns == ["city", "pop"]
    assert df.height == 2


def test_parquet_roundtrip() -> None:
    df = _load("basic.parquet")
    assert df.schema["ts"] == pl.Datetime("us")
    assert df["value"].to_list() == [1.0, None]


def test_corrupt_parquet_rejected() -> None:
    with pytest.raises(LoaderError, match="Failed to parse"):
        _load("corrupt.parquet")


def test_store_lazy_reload_across_instances(tmp_path: Path) -> None:
    store = DatasetStore(tmp_path)
    meta = store.save(_load("basic.csv"), "basic.csv")

    reloaded = DatasetStore(tmp_path)  # simulates a process restart
    assert reloaded.get_meta(meta["dataset_id"]) == meta
    assert reloaded.get_df(meta["dataset_id"]).shape == (3, 3)
    assert [m["dataset_id"] for m in reloaded.list_meta()] == [meta["dataset_id"]]


def test_store_missing_id_raises(tmp_path: Path) -> None:
    store = DatasetStore(tmp_path)
    with pytest.raises(DatasetNotFoundError):
        store.get_meta("0" * 32)
    with pytest.raises(DatasetNotFoundError):
        store.get_df("0" * 32)


def test_duplicate_and_blank_column_names_full_roundtrip(tmp_path: Path) -> None:
    # critique #8: read -> parquet persist -> reload in a fresh store
    df = _load("dup_blank_cols.csv")
    assert df.columns == ["a", "a_duplicated_0", ""]

    store = DatasetStore(tmp_path)
    meta = store.save(df, "dup_blank_cols.csv")
    assert [c["name"] for c in meta["columns"]] == ["a", "a_duplicated_0", ""]

    reloaded = DatasetStore(tmp_path)
    df2 = reloaded.get_df(meta["dataset_id"])
    assert df2.columns == df.columns
    assert df2.shape == (2, 3)
