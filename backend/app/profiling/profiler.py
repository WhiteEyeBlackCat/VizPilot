import math
import os
from datetime import timedelta
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import ValidationError

from ..serialization import df_to_records
from .evidence import compute_evidence
from .models import (
    PROFILE_VERSION,
    CastParams,
    ColumnProfile,
    Correlations,
    DatasetProfile,
    Frequency,
    TopValue,
)
from .types import SAMPLE_SEED, apply_casts, infer_semantic_type

MAX_CORRELATION_COLUMNS = 30
SAMPLE_ROWS_N = 5


def profile_dataset(df: pl.DataFrame, dataset_id: str, sample_threshold: int) -> DatasetProfile:
    n_rows = df.height
    sampled = n_rows > sample_threshold
    sample = df.sample(sample_threshold, seed=SAMPLE_SEED) if sampled else df

    inferred = {name: infer_semantic_type(sample[name]) for name in df.columns}
    casts = {name: params for name, (_, params) in inferred.items() if params is not None}
    casted = apply_casts(sample, casts)

    columns = [
        _column_profile(name, df[name], casted[name], sem, casts.get(name), n_rows)
        for name, (sem, _) in inferred.items()
    ]
    numeric_cols = [c.name for c in columns if c.semantic_type == "numeric"]
    correlations = _correlations(casted, numeric_cols)
    return DatasetProfile(
        profile_version=PROFILE_VERSION,
        dataset_id=dataset_id,
        n_rows=n_rows,
        n_cols=df.width,
        sampled=sampled,
        columns=columns,
        correlations=correlations,
        evidence=compute_evidence(casted, columns, correlations),
        sample_rows=df_to_records(df.head(SAMPLE_ROWS_N)),
    )


def _column_profile(
    name: str,
    full: pl.Series,
    casted: pl.Series,
    semantic_type: str,
    cast_params: CastParams | None,
    n_rows: int,
) -> ColumnProfile:
    missing_count = full.null_count()
    non_null = casted.drop_nulls()
    profile = ColumnProfile(
        name=name,
        original_dtype=str(full.dtype),
        semantic_type=semantic_type,  # type: ignore[arg-type]
        missing_count=missing_count,
        missing_ratio=missing_count / n_rows if n_rows else 0.0,
        unique_count=non_null.n_unique(),
        cast_params=cast_params,
    )
    if semantic_type == "numeric":
        values = non_null.filter(non_null.is_finite()) if non_null.dtype.is_float() else non_null
        profile.min = _finite(values.min())
        profile.max = _finite(values.max())
        profile.mean = _finite(values.mean())
        profile.median = _finite(values.median())
        profile.std = _finite(values.std())
        profile.q25 = _finite(values.quantile(0.25))  # polars default interpolation: nearest
        profile.q75 = _finite(values.quantile(0.75))
        profile.skewness = _finite(values.skew())
    elif semantic_type in ("categorical", "boolean"):
        counts = non_null.value_counts(sort=True).head(10)
        profile.top_values = [TopValue(value=v, count=c) for v, c in counts.rows()]
        profile.n_categories = non_null.n_unique()
    elif semantic_type == "datetime":
        profile.min = non_null.min().isoformat() if len(non_null) else None
        profile.max = non_null.max().isoformat() if len(non_null) else None
        profile.inferred_frequency = _infer_frequency(non_null)
    elif semantic_type == "text":
        lengths = non_null.str.len_chars()
        profile.avg_length = _finite(lengths.mean())
        profile.max_length = int(lengths.max()) if len(non_null) else None
    # id / unknown: shared fields only
    return profile


def _infer_frequency(non_null: pl.Series) -> Frequency:
    # unique() first: long-format data repeats each timestamp per group
    distinct = non_null.unique()
    if len(distinct) < 3:
        return "unknown"
    median_gap = distinct.sort().diff().drop_nulls().median()
    if median_gap is None:
        return "unknown"
    days = median_gap / timedelta(days=1)
    if 0.9 <= days <= 1.1:
        return "daily"
    if 6.5 <= days <= 7.5:
        return "weekly"
    if 27 <= days <= 32:
        return "monthly"
    return "irregular"


def _correlations(casted: pl.DataFrame, numeric_cols: list[str]) -> Correlations | None:
    if len(numeric_cols) < 2:
        return None
    truncated = len(numeric_cols) > MAX_CORRELATION_COLUMNS
    cols = numeric_cols[:MAX_CORRELATION_COLUMNS]
    n = len(cols)
    matrix: list[list[float | None]] = [[None] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1.0
        for j in range(i + 1, n):
            # per-pair pl.corr on pairwise non-null rows (df.corr needs numpy);
            # NaN/inf rows are dropped too, matching the numeric-stats treatment
            pair = casted.select(cols[i], cols[j]).drop_nulls()
            for name in (cols[i], cols[j]):
                if pair.schema[name].is_float():
                    pair = pair.filter(pl.col(name).is_finite())
            value = pair.select(pl.corr(cols[i], cols[j])).item() if pair.height >= 2 else None
            matrix[i][j] = matrix[j][i] = _finite(value)
    return Correlations(columns=cols, matrix=matrix, truncated=truncated)


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


class ProfileService:
    """Profiles cached in memory and at {data_dir}/{id}.profile.json.

    Cached files embed profile_version: a mismatch or a payload that fails
    Pydantic validation triggers recomputation and overwrite (critique #2).
    """

    def __init__(self, data_dir: Path, sample_threshold: int) -> None:
        self._data_dir = data_dir
        self._sample_threshold = sample_threshold
        self._cache: dict[str, DatasetProfile] = {}

    def get(self, dataset_id: str, df: pl.DataFrame) -> DatasetProfile:
        if dataset_id in self._cache:
            return self._cache[dataset_id]

        path = self._path(dataset_id)
        if path.is_file():
            try:
                profile = DatasetProfile.model_validate_json(path.read_text())
                if profile.profile_version == PROFILE_VERSION:
                    self._cache[dataset_id] = profile
                    return profile
            except ValidationError:
                pass  # stale/corrupt cache -> recompute below

        profile = profile_dataset(df, dataset_id, self._sample_threshold)
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(profile.model_dump_json())
        os.replace(tmp_path, path)
        self._cache[dataset_id] = profile
        return profile

    def _path(self, dataset_id: str) -> Path:
        return self._data_dir / f"{dataset_id}.profile.json"
