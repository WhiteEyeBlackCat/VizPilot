from typing import Any, Literal

from pydantic import BaseModel

# Bump whenever the profile schema or inference rules change: cached
# {id}.profile.json files with a different version are recomputed.
PROFILE_VERSION = 1

SemanticType = Literal["numeric", "categorical", "datetime", "boolean", "text", "id", "unknown"]
Frequency = Literal["daily", "weekly", "monthly", "irregular", "unknown"]


class CastParams(BaseModel):
    """Recorded at inference time; apply_semantic_casts replays these verbatim
    (never re-infers) so Stage 4 reproduces exactly the profiled columns."""

    target: Literal["datetime", "numeric"]
    # "infer" (polars format inference) or an explicit strptime format
    datetime_format: str | None = None
    thousands: bool = False
    currency: bool = False
    # '%' is stripped but the value is kept as-is (45% -> 45.0), per stage2 brief
    percent: bool = False


class TopValue(BaseModel):
    value: Any
    count: int


class ColumnProfile(BaseModel):
    name: str
    original_dtype: str
    semantic_type: SemanticType
    missing_count: int
    missing_ratio: float
    # nulls excluded; unique_ratio uses the non-null count as denominator
    unique_count: int
    cast_params: CastParams | None = None
    # numeric stats (min/max double as ISO strings for datetime columns)
    min: float | str | None = None
    max: float | str | None = None
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    q25: float | None = None
    q75: float | None = None
    skewness: float | None = None
    # categorical / boolean
    top_values: list[TopValue] | None = None
    n_categories: int | None = None
    # datetime
    inferred_frequency: Frequency | None = None
    # text
    avg_length: float | None = None
    max_length: int | None = None


class Correlations(BaseModel):
    columns: list[str]
    matrix: list[list[float | None]]
    truncated: bool = False


class DatasetProfile(BaseModel):
    profile_version: int
    dataset_id: str
    n_rows: int
    n_cols: int
    sampled: bool
    columns: list[ColumnProfile]
    correlations: Correlations | None = None
    sample_rows: list[dict[str, Any]]
