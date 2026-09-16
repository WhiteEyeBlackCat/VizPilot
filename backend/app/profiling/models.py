from typing import Any, Literal

from pydantic import BaseModel

# Bump whenever the profile schema or inference rules change: cached
# {id}.profile.json files with a different version are recomputed.
PROFILE_VERSION = 14  # v14: ColumnProfile.categories (full label set of small categoricals for role detection, stage 19.2); v13: dispersion compared groups / F-test p-value / definitional flag, bucket derived kind (stage 19.2); v12: dispersion evidence (stage 19.1); v11: layer-2 shapes/strength (stage 17.1b); v10: evidence layer 2 (stage 17.1)

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


CATEGORY_LABELS_MAX = 13  # stage 19.2: full label set kept for role detection up to this many categories


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
    # stage 19.2: every distinct label (as str, sorted by value) when the
    # column has at most CATEGORY_LABELS_MAX categories — the input of
    # semantic-role detection (charts/roles.py: month / weekday names, label
    # scales), which top_values (10 most frequent) cannot serve for 11-13
    # categories such as the twelve month names. None otherwise.
    categories: list[str] | None = None
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


DerivedKind = Literal["product", "product_discount", "sum", "difference", "ratio", "near_copy", "bucket"]


class DerivedColumn(BaseModel):
    """A column that is (almost) a deterministic function of other columns
    (stage 13). A chart of `target` against one of its `components` draws the
    definition, not a finding, so the rule engine demotes such charts and the
    LLM is told never to present them as insights.

    kind != near_copy: `target ≈ formula(components)` row-wise within a
    rounding tolerance on >= 99% of the rows where every column is finite.
    kind == near_copy: a near-duplicate (stage 14 NearDuplicateGroup member)
    of its single component, the group's representative. Disclosed; the
    duplicate itself is suppressed from candidate generation.
    kind == bucket (stage 19.2): a categorical `target` whose groups are
    ordered, disjoint, contiguous intervals of its numeric single component
    (`price_band = bucket(price)`): the per-group ranges tile the column and
    every shared boundary is dense on at least one side (see
    evidence._is_bucket). formula is `bucket(<component>)`, match_ratio 1.0."""

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


DispersionStrength = Literal["strong", "weak"]


class DispersionEffect(BaseModel):
    """Do the *spreads* of `num` differ across the groups of `cat` (stage
    19.1)? Independent of the mean difference (CatNumEffect): two groups with
    the same mean and a different spread have eta-squared ~0 and a strong
    entry here. bf_eta_squared is the adjusted eta-squared of |num − group
    median| over the compared groups — the Brown–Forsythe statistic as an
    effect size, 0 when the spreads are equal; bf_f / bf_p_value are the
    Brown–Forsythe F statistic and its F-distribution p-value (stage 19.2);
    std_ratio / iqr_ratio / mad_ratio are the widest / narrowest per-group
    spread (None when a group's spread is 0); spread_ratio is the robust one
    the rule engine reads (IQR, MAD when an IQR is 0, std last).

    Sample sizes (stage 19.2): the comparison runs over the groups listed in
    compared_groups; a group too small to support it is EXCLUDED and listed
    in excluded_groups (label -> rows) instead of cancelling the comparison
    for the well-supported groups. When at least two groups hold
    DISPERSION_STRONG_MIN_N rows the comparison is over exactly those groups
    and may be `strong` (bf_p_value <= DISPERSION_ALPHA, effect gates met);
    when that lists nothing (or fewer than two groups are supported) the
    comparison falls back to every group with DISPERSION_WEAK_MIN_N rows and
    is at most `weak`. n_groups / n_total / n_min describe the compared
    groups only. A small group never contributes to strong evidence. A
    group of `cat` without any value of `num` is excluded with 0 rows.

    definitional (stage 19.2): the pair merely restates a definition —
    `num` is derived from `cat` (or vice versa: an identity or near copy in
    derived_columns, also through a chain — `num` a near copy of a column
    derived from `cat`, kind near_copy with the chain in `definition`) or
    `cat` is a banding of `num` (kind bucket). The
    evidence is kept and flagged; the rules never promote a flagged entry to
    a headline distribution chart and the LLM is told it is not a
    discovery."""

    cat: str
    num: str
    n_total: int  # rows in the compared groups
    n_groups: int  # compared groups
    n_min: int  # rows in the smallest compared group
    bf_eta_squared: float
    std_ratio: float | None
    iqr_ratio: float | None
    mad_ratio: float | None
    spread_ratio: float
    widest_group: str  # str(group value), like CatNumEffect.group_counts keys
    narrowest_group: str
    strength: DispersionStrength
    # stage 19.2, additive
    bf_f: float | None = None
    bf_p_value: float | None = None
    compared_groups: list[str] = []  # labels that took part, sorted
    excluded_groups: dict[str, int] = {}  # label -> valid rows of num (0: no value at all), not compared
    definitional: bool = False
    definitional_kind: DerivedKind | None = None
    definition: str | None = None  # e.g. "sales = unit_price × quantity", "price_band = bucket(price)"


# --- evidence layer 2 (stage 17.1) ------------------------------------------
#
# Bounded, deterministic summaries that let the LLM reason about distribution
# shape, group structure, conditional and non-linear relationships and changes
# over time WITHOUT seeing raw rows. Every entry carries the row count it was
# computed on (profiled-sample level, like layer 1). Nothing here feeds the
# rule engine: layer 2 is read by the LLM workflow only.

DistributionShape = Literal["symmetric", "right_skewed", "left_skewed"]


class DistributionSummary(BaseModel):
    """Coarse shape of one numeric column: quantiles, skew, a 10-bin relative
    histogram over `bin_lo..bin_hi` (the robust display range when the column
    has one, so a few extreme values cannot squash the picture; the share
    outside that window is reported, never dropped silently) and a
    multimodality signal (>= 2 separated peaks AND bimodality coefficient
    above the uniform-distribution value 0.555)."""

    column: str
    n: int  # finite non-null rows
    p05: float
    p25: float
    p50: float
    p75: float
    p95: float
    skewness: float | None
    robust_range: RobustRange | None
    bin_lo: float
    bin_hi: float
    bins: list[float]  # relative frequency per bin over [bin_lo, bin_hi]; sums to 1 - outside_ratio
    outside_ratio: float
    shape: DistributionShape
    bimodality_coefficient: float | None
    modes: int  # separated peaks found in the coarse histogram
    multimodal_signal: bool


class GroupStat(BaseModel):
    group: str  # str(group value), like CatNumEffect.group_counts keys
    n: int
    mean: float
    median: float
    std: float | None
    q25: float
    q75: float


class GroupSummary(BaseModel):
    """Per-group location and spread of `num` across the categories of `cat`
    (groups in stringified key order). max_diff_sd is the largest mean
    difference between any two groups in pooled-within-group standard
    deviations — a plain-language effect size next to the eta-squared."""

    cat: str
    num: str
    n_total: int
    eta_squared: float
    groups: list[GroupStat]
    pooled_std: float | None
    max_diff_sd: float | None
    top_group: str
    bottom_group: str


class GroupCorrelation(BaseModel):
    group: str
    n: int
    corr: float | None  # None below the per-group row floor
    slope: float | None  # OLS slope of y on x within the group


class ConditionalRelationship(BaseModel):
    """The x–y relationship inside each group of `group`: correlation and
    slope per group, so a relationship that flips or vanishes in a subgroup
    is visible (layer 1 only carries the correlation spread)."""

    x: str
    y: str
    group: str
    groups: list[GroupCorrelation]
    corr_spread: float
    n_min: int


class TimeSeriesPoint(BaseModel):
    bucket: str  # ISO start of the time bucket
    mean: float
    n: int


class GroupTimePattern(BaseModel):
    """Mean of `num` per time bucket, separately for each group of `group`
    (<= MAX_TIME_POINTS buckets per group: coarsened, then evenly thinned)."""

    datetime_col: str
    num: str
    group: str
    bucket: TimeBucket
    series: dict[str, list[TimeSeriesPoint]]  # str(group value) -> points in time order
    n_total: int


class NonlinearBin(BaseModel):
    lo: float
    hi: float
    n: int
    y_mean: float


NonlinearShape = Literal[
    "flat", "linear", "monotone_nonlinear", "multi_peak", "u_shape", "inverted_u", "other"
]
# multi_peak (stage 17.1b): >= 2 separated interior peaks in the bin means
# (a commuting double peak), decided before the quadratic fit so it is not
# mislabelled as an inverted U


class NonlinearSignal(BaseModel):
    """Binned relationship of y on x: x split into equal-frequency bins, the
    mean of y per bin, the adjusted eta-squared of y over the bins (variance
    the bin structure explains, linear or not) against the Pearson r²
    (variance a straight line explains). nonlinear_gap = binned_eta2 − r²:
    large when the dependence is real but not linear (a U shape has a small
    r² and a large binned eta-squared)."""

    x: str
    y: str
    n: int
    bins: list[NonlinearBin]
    binned_eta2: float
    r2_pearson: float | None
    nonlinear_gap: float
    monotone: bool  # bin means are monotone in x
    shape: NonlinearShape


class ChangePoint(BaseModel):
    """Single best split of the bucketed mean series of `num` over
    `datetime_col` (one step of binary segmentation on bucket means, each
    segment at least MIN_SEGMENT_BUCKETS buckets). effect_size = |after −
    before| in pooled within-segment standard deviations of the bucket means;
    flagged when it reaches CHANGE_FLAG_EFFECT."""

    datetime_col: str
    num: str
    bucket: TimeBucket
    n_buckets: int
    change_at: str  # ISO start of the first bucket after the split
    before_mean: float
    after_mean: float
    effect_size: float  # |after − before| / pooled within-segment SD of the bucket means
    diff_sd: float  # |after − before| / SD of the column (a change that matters in data units)
    n_before: int  # rows in the buckets before the split
    n_after: int
    flagged: bool  # effect_size >= CHANGE_FLAG_EFFECT and diff_sd >= CHANGE_FLAG_DIFF_SD
    # stage 17.1b grading for the LLM workflow: "strong" = flagged; "weak" =
    # the split is clear against bucket noise (effect_size >= CHANGE_FLAG_EFFECT)
    # but small in data units (diff_sd < CHANGE_FLAG_DIFF_SD); "none" otherwise
    strength: Literal["strong", "weak", "none"] = "none"


class SubgroupAnomaly(BaseModel):
    """A group whose mean of `num` sits far from the other groups' means:
    robust z = (mean − median of group means) / (1.4826 · MAD) at least
    ANOMALY_Z AND the gap at least ANOMALY_MIN_DIFF_SD pooled within-group
    SDs (so a tight cluster of means cannot flag a trivial difference), over
    groups with at least ANOMALY_MIN_ROWS rows and ANOMALY_MIN_GROUPS
    groups."""

    cat: str
    num: str
    group: str
    n: int
    mean: float
    others_median: float
    robust_z: float
    diff_sd: float  # (mean − others_median) / pooled within-group SD


class EvidenceLayer2(BaseModel):
    distributions: list[DistributionSummary] = []
    group_summaries: list[GroupSummary] = []
    conditional_relationships: list[ConditionalRelationship] = []
    group_time_patterns: list[GroupTimePattern] = []
    nonlinear: list[NonlinearSignal] = []
    change_points: list[ChangePoint] = []
    subgroup_anomalies: list[SubgroupAnomaly] = []


class Evidence(BaseModel):
    cat_num: list[CatNumEffect] = []
    time_effects: list[TimeEffect] = []
    # same columns/shape as DatasetProfile.correlations, Spearman method
    num_num_spearman: Correlations | None = None
    interactions: list[InteractionEffect] = []
    slope_heterogeneity: list[SlopeHet] = []
    derived_columns: list[DerivedColumn] = []  # stage 13, additive
    near_duplicate_groups: list[NearDuplicateGroup] = []  # stage 14, additive
    # stage 19.1, additive: spread differences across groups, decoupled from
    # the mean-difference evidence in cat_num (read by the rules and the LLM)
    dispersion: list[DispersionEffect] = []
    layer2: EvidenceLayer2 = EvidenceLayer2()  # stage 17.1, additive; LLM-only


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
