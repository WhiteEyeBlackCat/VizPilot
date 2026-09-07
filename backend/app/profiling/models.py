from typing import Any, Literal

from pydantic import BaseModel

# Bump whenever the profile schema or inference rules change: cached
# {id}.profile.json files with a different version are recomputed.
PROFILE_VERSION = 2  # v2: evidence table (stage 7)

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


TimeBucket = Literal["day", "month", "year"]


class CatNumEffect(BaseModel):
    """Adjusted (df-corrected) eta-squared of `num` explained by `cat` groups,
    clipped at 0 — raw eta-squared inflates on high-cardinality small samples."""

    cat: str
    num: str
    eta_squared: float
    n_groups: int


class TimeEffect(BaseModel):
    """Adjusted eta-squared of `num` across time buckets of `datetime_col`.
    `bucket` records the granularity actually used (span-adaptive, coarsened
    when the finer bucket leaves no within-group degrees of freedom)."""

    datetime_col: str
    num: str
    eta_squared: float
    bucket: TimeBucket


class InteractionEffect(BaseModel):
    """Two-factor interaction share of variance (additive-prediction residual;
    not df-corrected, so small cells bias it upward — cells below the minimum
    count are dropped to bound that). cat1 may be a `col@bucket` time marker."""

    cat1: str
    cat2: str
    num: str
    strength: float


class SlopeHet(BaseModel):
    """Per-group Pearson correlations of a numeric pair; spread = max - min
    over groups with enough rows. Keys are stringified group values."""

    x: str
    y: str
    group: str
    corrs: dict[str, float | None]
    spread: float
    n_min: int  # rows in the smallest contributing group (for dynamic thresholds)


class Evidence(BaseModel):
    cat_num: list[CatNumEffect] = []
    time_effects: list[TimeEffect] = []
    # same columns/shape as DatasetProfile.correlations, Spearman method
    num_num_spearman: Correlations | None = None
    interactions: list[InteractionEffect] = []
    slope_heterogeneity: list[SlopeHet] = []


class DatasetProfile(BaseModel):
    profile_version: int
    dataset_id: str
    n_rows: int
    n_cols: int
    sampled: bool
    columns: list[ColumnProfile]
    correlations: Correlations | None = None
    evidence: Evidence = Evidence()
    sample_rows: list[dict[str, Any]]
