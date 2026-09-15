from typing import Any, Literal

from pydantic import BaseModel

# Bump whenever the profile schema or inference rules change: cached
# {id}.profile.json files with a different version are recomputed.
PROFILE_VERSION = 9  # v9: near-duplicate groups (stage 14); v8: derived columns (stage 13)

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


class SentinelCandidate(BaseModel):
    """A distinct value flagged as a suspected sentinel (stage 9 #6A). The
    label is deliberately "suspected": the signals are statistical, never
    schema knowledge. signals is a subset of {extreme, repeated, pattern, scale}."""

    value: float
    count: int
    signals: list[str]


class RobustRange(BaseModel):
    """Display-range hint: Tukey far-out fences (q1 - 3 IQR, q3 + 3 IQR)
    intersected with the data range. Raw min/max stay on ColumnProfile."""

    lo: float
    hi: float


class ColumnQuality(BaseModel):
    """Per-column data-quality counts and robust statistics (stage 9 #2/#5/#6).

    Every count is SAMPLE-level (computed on the profiled rows, == the full
    table unless DatasetProfile.sampled); the confidence layer scales by
    n_rows / profiled_rows. valid_count = profiled_rows - missing_count -
    missing_token_count - invalid_count. The robust block is None / empty for
    non-numeric columns (and, until stage 5 lands, for numeric ones too)."""

    profiled_rows: int
    missing_count: int  # raw nulls in the profiled sample
    missing_token_count: int = 0  # non-null raw values recognised as missing tokens (numeric cast path)
    # non-null raw values lost otherwise: numeric-gate/cast failures, NaN/+-inf,
    # datetime parse failures
    invalid_count: int = 0
    valid_count: int
    valid_ratio: float
    # robust location / scale (linear-interpolated quantiles, like the box render)
    q1: float | None = None
    q3: float | None = None
    iqr: float | None = None
    mad: float | None = None  # median |x - median|
    robust_z_max: float | None = None  # max |x - median| / (1.4826 * mad)
    extreme_value_count: int = 0  # beyond the far-out fences, sentinel rows excluded
    extreme_value_ratio: float = 0.0  # / valid_count
    suspected_sentinels: list[SentinelCandidate] = []
    sentinel_row_count: int = 0
    robust_range: RobustRange | None = None


class ColumnProfile(BaseModel):
    name: str
    original_dtype: str
    semantic_type: SemanticType
    missing_count: int
    missing_ratio: float
    # nulls excluded; unique_ratio uses the non-null count as denominator
    unique_count: int
    cast_params: CastParams | None = None
    # numeric-backed categorical whose values are labels (codes), never a
    # numeric y / mean / correlation input (stage 4 sets it)
    nominal: bool = False
    quality: ColumnQuality | None = None
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
    # pairwise-complete row counts behind each cell (diagonal = valid rows of
    # the column); sample-level like every other count in the profile
    pair_counts: list[list[int]] | None = None


TimeBucket = Literal["day", "month", "year"]


class CatNumEffect(BaseModel):
    """Adjusted (df-corrected) eta-squared of `num` explained by `cat` groups,
    clipped at 0 — raw eta-squared inflates on high-cardinality small samples."""

    cat: str
    num: str
    eta_squared: float
    n_groups: int
    n_total: int = 0  # pairwise-complete rows (cat non-null, num finite) in the profiled sample
    n_min: int = 0  # rows in the smallest non-empty group
    # per-group pairwise-complete rows, str(value) keys like SlopeHet.corrs and
    # the render group labels; only groups with >= 1 valid row (sum == n_total)
    group_counts: dict[str, int] = {}


class TimeEffect(BaseModel):
    """Adjusted eta-squared of `num` across time buckets of `datetime_col`.
    `bucket` records the granularity actually used (span-adaptive, coarsened
    when the finer bucket leaves no within-group degrees of freedom)."""

    datetime_col: str
    num: str
    eta_squared: float
    bucket: TimeBucket
    n_total: int = 0  # pairwise-complete rows in the profiled sample
    n_min: int = 0  # rows in the smallest non-empty time bucket


class InteractionEffect(BaseModel):
    """Two-factor interaction share of variance (additive-prediction residual;
    not df-corrected, so small cells bias it upward — cells below the minimum
    count are dropped to bound that). cat1 may be a `col@bucket` time marker."""

    cat1: str
    cat2: str
    num: str
    strength: float
    n_total: int = 0  # rows in the kept cells


class SlopeHet(BaseModel):
    """Per-group Pearson correlations of a numeric pair; spread = max - min
    over groups with enough rows. Keys are stringified group values."""

    x: str
    y: str
    group: str
    corrs: dict[str, float | None]
    spread: float
    n_min: int  # rows in the smallest contributing group (for dynamic thresholds)


DerivedKind = Literal["product", "product_discount", "sum", "difference", "ratio", "near_copy"]


class DerivedColumn(BaseModel):
    """A column that is (almost) a deterministic function of other columns
    (stage 13). A chart of `target` against one of its `components` draws the
    definition, not a finding, so the rule engine demotes such charts and the
    LLM is told never to present them as insights.

    kind != near_copy: `target ≈ formula(components)` row-wise within a
    rounding tolerance on >= 99% of the rows where every column is finite.
    kind == near_copy: a near-duplicate (stage 14 NearDuplicateGroup member)
    of its single component, the group's representative. Disclosed; the
    duplicate itself is suppressed from candidate generation."""

    target: str
    components: list[str]
    formula: str  # human-readable, e.g. "unit_price × quantity × (1 − discount)"
    kind: DerivedKind
    match_ratio: float  # matched / valid rows (kind near_copy: |rho|)
    n: int  # valid rows the check ran on (profiled-sample level, capped)


class NearDuplicateGroup(BaseModel):
    """Columns that rank (almost) identically (stage 14): a strong rank
    correlation alone (|rho| >= 0.995) or a slightly weaker one corroborated
    by related names (|rho| >= 0.98, e.g. temp / atemp). Only the
    representative takes part in chart candidates (heatmap excepted); the
    duplicates are disclosed."""

    representative: str
    duplicates: list[str]
    rho: dict[str, float]  # duplicate -> |Spearman| with the representative
    n: int  # shared rows behind the weakest duplicate/representative pair


class Evidence(BaseModel):
    cat_num: list[CatNumEffect] = []
    time_effects: list[TimeEffect] = []
    # same columns/shape as DatasetProfile.correlations, Spearman method
    num_num_spearman: Correlations | None = None
    interactions: list[InteractionEffect] = []
    slope_heterogeneity: list[SlopeHet] = []
    derived_columns: list[DerivedColumn] = []  # stage 13, additive
    near_duplicate_groups: list[NearDuplicateGroup] = []  # stage 14, additive


class DatasetProfile(BaseModel):
    profile_version: int
    dataset_id: str
    n_rows: int
    n_cols: int
    sampled: bool
    # rows every statistic / correlation / evidence entry was computed on
    # (== n_rows unless sampled); 0 only for pre-v3 payloads, which the
    # version check recomputes anyway
    profiled_rows: int = 0
    columns: list[ColumnProfile]
    correlations: Correlations | None = None
    evidence: Evidence = Evidence()
    sample_rows: list[dict[str, Any]]
