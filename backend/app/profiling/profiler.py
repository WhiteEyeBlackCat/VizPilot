import math
import os
import threading
from datetime import timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import polars as pl
from pydantic import ValidationError

from ..serialization import df_to_records
from .evidence import compute_evidence
from .models import (
    PROFILE_VERSION,
    CastParams,
    ColumnProfile,
    ColumnQuality,
    Correlations,
    DatasetProfile,
    Frequency,
    TopValue,
)
from .pairwise import pairwise_pearson
from .quality import robust_quality
from .types import SAMPLE_SEED, apply_casts, infer_column, missing_token_count

MAX_CORRELATION_COLUMNS = 30
SAMPLE_ROWS_N = 5


def profile_dataset(df: pl.DataFrame, dataset_id: str, sample_threshold: int) -> DatasetProfile:
    n_rows = df.height
    sampled = n_rows > sample_threshold
    sample = df.sample(sample_threshold, seed=SAMPLE_SEED) if sampled else df

    inferred = {name: infer_column(sample[name]) for name in df.columns}
    casts = {name: params for name, (_, params, _) in inferred.items() if params is not None}
    casted = apply_casts(sample, casts)

    columns = [
        _column_profile(name, df[name], sample[name], casted[name], sem, casts.get(name), n_rows, nominal)
        for name, (sem, _, nominal) in inferred.items()
    ]
    numeric_cols = [c.name for c in columns if c.semantic_type == "numeric"]
    correlations = _correlations(casted, numeric_cols)
    return DatasetProfile(
        profile_version=PROFILE_VERSION,
        dataset_id=dataset_id,
        n_rows=n_rows,
        n_cols=df.width,
        sampled=sampled,
        profiled_rows=casted.height,
        columns=columns,
        correlations=correlations,
        evidence=compute_evidence(casted, columns, correlations),
        sample_rows=df_to_records(df.head(SAMPLE_ROWS_N)),
    )


def _column_profile(
    name: str,
    full: pl.Series,
    raw_sample: pl.Series,
    casted: pl.Series,
    semantic_type: str,
    cast_params: CastParams | None,
    n_rows: int,
    nominal: bool = False,
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
        nominal=nominal,
        quality=_column_quality(raw_sample, casted, cast_params, semantic_type == "numeric"),
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


def _column_quality(
    raw_sample: pl.Series, casted: pl.Series, cast_params: CastParams | None, numeric: bool = False
) -> ColumnQuality:
    """Sample-level validity counts for every column (stage 9 #2/#3/#5). Cast
    failures show up as nulls the raw sample did not have; on the numeric cast
    path the known missing tokens among them are reported separately; NaN/inf
    are counted on the casted column. Numeric columns also get the robust
    block (stage 9 #6) computed on their finite values."""
    profiled_rows = len(casted)
    missing = raw_sample.null_count()
    cast_failures = casted.null_count() - missing  # >= 0: casts only ever add nulls
    tokens = 0
    if cast_params is not None and cast_params.target == "numeric" and raw_sample.dtype == pl.String:
        tokens = missing_token_count(raw_sample)
    non_null = casted.drop_nulls()
    non_finite = int((~non_null.is_finite()).sum()) if non_null.dtype.is_float() else 0
    invalid = cast_failures - tokens + non_finite
    valid = profiled_rows - missing - tokens - invalid
    robust = {}
    if numeric:
        finite = non_null.filter(non_null.is_finite()) if non_null.dtype.is_float() else non_null
        robust = robust_quality(finite)
    return ColumnQuality(
        profiled_rows=profiled_rows,
        missing_count=missing,
        missing_token_count=tokens,
        invalid_count=invalid,
        valid_count=valid,
        valid_ratio=valid / profiled_rows if profiled_rows else 0.0,
        **robust,
    )


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
    matrix, counts = pairwise_pearson(casted, cols)
    return Correlations(columns=cols, matrix=matrix, truncated=truncated, pair_counts=counts)


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


class ProfileService:
    """Profiles cached in memory and at {data_dir}/{id}.profile.json.

    Cached files embed profile_version: a mismatch or a payload that fails
    Pydantic validation triggers recomputation and overwrite (critique #2).

    Concurrency (stage 9b): the frontend fires profile + two recommendation
    requests right after upload, so the first miss for a dataset arrives on
    three threadpool workers at once. A per-dataset lock makes exactly one of
    them compute and write; the others wait and take the in-memory result.
    The temp file carries a unique suffix so even independent writers (other
    processes) never move each other's file out from under os.replace.
    """

    def __init__(self, data_dir: Path, sample_threshold: int) -> None:
        self._data_dir = data_dir
        self._sample_threshold = sample_threshold
        self._cache: dict[str, DatasetProfile] = {}
        self._lock = threading.Lock()
        self._key_locks: dict[str, threading.Lock] = {}

    def get(self, dataset_id: str, df: pl.DataFrame) -> DatasetProfile:
        cached = self._cache.get(dataset_id)  # hot path: no lock
        if cached is not None:
            return cached
        with self._lock:
            key_lock = self._key_locks.setdefault(dataset_id, threading.Lock())
        with key_lock:
            cached = self._cache.get(dataset_id)  # a concurrent miss already filled it
            if cached is not None:
                return cached
            profile = self._load(dataset_id)
            if profile is None:
                profile = profile_dataset(df, dataset_id, self._sample_threshold)
                self._write(dataset_id, profile)
            self._cache[dataset_id] = profile
            return profile

    def _load(self, dataset_id: str) -> DatasetProfile | None:
        path = self._path(dataset_id)
        if not path.is_file():
            return None
        try:
            profile = DatasetProfile.model_validate_json(path.read_text())
        except ValidationError:
            return None  # stale/corrupt cache -> recompute
        return profile if profile.profile_version == PROFILE_VERSION else None

    def _write(self, dataset_id: str, profile: DatasetProfile) -> None:
        path = self._path(dataset_id)
        tmp_path = self._data_dir / f"{dataset_id}.profile.{uuid4().hex}.tmp"
        try:
            tmp_path.write_text(profile.model_dump_json())
            os.replace(tmp_path, path)  # atomic publish
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    def _path(self, dataset_id: str) -> Path:
        return self._data_dir / f"{dataset_id}.profile.json"
