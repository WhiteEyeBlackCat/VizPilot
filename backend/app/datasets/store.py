import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

Metadata = dict[str, Any]


class DatasetNotFoundError(KeyError):
    pass


class DatasetStore:
    """Persists each dataset as {data_dir}/{id}.parquet + {id}.json metadata.

    The in-memory caches are deliberately unbounded: this is a single-user
    local tool with short sessions (stage1 critique #6). A restarted process
    lazily reloads datasets from disk on first access.
    """

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        data_dir.mkdir(parents=True, exist_ok=True)
        self._meta_cache: dict[str, Metadata] = {}
        self._df_cache: dict[str, pl.DataFrame] = {}

    def save(self, df: pl.DataFrame, filename: str) -> Metadata:
        dataset_id = uuid.uuid4().hex
        meta: Metadata = {
            "dataset_id": dataset_id,
            "filename": filename,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "n_rows": df.height,
            "n_cols": df.width,
            "columns": [{"name": name, "dtype": str(dtype)} for name, dtype in df.schema.items()],
        }
        df.write_parquet(self._parquet_path(dataset_id))
        # Atomic write so a crash mid-write cannot leave a truncated metadata file.
        json_path = self._json_path(dataset_id)
        tmp_path = json_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(meta, ensure_ascii=False))
        os.replace(tmp_path, json_path)
        self._meta_cache[dataset_id] = meta
        self._df_cache[dataset_id] = df
        return meta

    def get_meta(self, dataset_id: str) -> Metadata:
        if dataset_id not in self._meta_cache:
            path = self._json_path(dataset_id)
            if not path.is_file():
                raise DatasetNotFoundError(dataset_id)
            self._meta_cache[dataset_id] = json.loads(path.read_text())
        return self._meta_cache[dataset_id]

    def get_df(self, dataset_id: str) -> pl.DataFrame:
        if dataset_id not in self._df_cache:
            path = self._parquet_path(dataset_id)
            if not path.is_file():
                raise DatasetNotFoundError(dataset_id)
            self._df_cache[dataset_id] = pl.read_parquet(path)
        return self._df_cache[dataset_id]

    def list_meta(self) -> list[Metadata]:
        metas = [json.loads(path.read_text()) for path in self._data_dir.glob("*.json")]
        return sorted(metas, key=lambda m: m.get("uploaded_at", ""), reverse=True)

    def _parquet_path(self, dataset_id: str) -> Path:
        return self._data_dir / f"{dataset_id}.parquet"

    def _json_path(self, dataset_id: str) -> Path:
        return self._data_dir / f"{dataset_id}.json"
