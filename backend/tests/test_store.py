import polars as pl
import pytest

from app.datasets.store import DatasetNotFoundError, DatasetStore


def _frame(seed: int) -> pl.DataFrame:
    return pl.DataFrame({"a": [seed, seed + 1, seed + 2], "b": ["x", "y", "z"]})


def test_frame_cache_is_bounded_lru(tmp_path):
    store = DatasetStore(tmp_path, max_cached_frames=2)
    ids = [store.save(_frame(i), f"f{i}.csv")["dataset_id"] for i in range(3)]

    # the oldest upload was evicted from memory, the two newest stay resident
    assert list(store._df_cache) == ids[1:]

    # an evicted dataset is transparently re-read from parquet, identical content
    assert store.get_df(ids[0]).equals(_frame(0))
    assert list(store._df_cache) == [ids[2], ids[0]]

    # an access refreshes recency: ids[2] is now the most recent, ids[0] goes next
    store.get_df(ids[2])
    store.get_df(ids[1])
    assert list(store._df_cache) == [ids[2], ids[1]]
    assert len(store._df_cache) == 2


def test_metadata_survives_frame_eviction(tmp_path):
    store = DatasetStore(tmp_path, max_cached_frames=1)
    first = store.save(_frame(0), "a.csv")["dataset_id"]
    store.save(_frame(10), "b.csv")
    assert store.get_meta(first)["filename"] == "a.csv"
    assert store.get_df(first)["a"].to_list() == [0, 1, 2]


def test_missing_dataset_still_raises(tmp_path):
    store = DatasetStore(tmp_path, max_cached_frames=1)
    with pytest.raises(DatasetNotFoundError):
        store.get_df("nope")


def test_non_positive_limit_keeps_one_frame(tmp_path):
    store = DatasetStore(tmp_path, max_cached_frames=0)
    ds = store.save(_frame(0), "a.csv")["dataset_id"]
    assert list(store._df_cache) == [ds]
