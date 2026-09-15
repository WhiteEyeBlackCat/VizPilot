"""Targeted validation probes (stage 17.2): the request / result contract.

A probe is the ONLY way the LLM workflow can ask the backend for a
statistic it does not already have. The vocabulary is closed: seven probe
types, each with a fixed set of column roles. The LLM chooses a type and
columns; it never supplies code, expressions or numbers. Every request is
validated against the profile before anything runs — a request that names a
missing column, the wrong semantic type, a derived/near-duplicate pair or a
column the rule engine itself excludes is rejected, not executed.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from ..charts.confidence import MAX_MISSING_RATIO, Confidence
from ..charts.spec import ChartSpec, _is_numeric_y
from ..profiling.evidence import MAX_CAT_CATEGORIES
from ..profiling.models import ColumnProfile, DatasetProfile

ProbeType = Literal[
    "group_difference",
    "grouped_relationship",
    "nonlinear_relationship",
    "distribution_difference",
    "time_pattern",
    "interaction",
    "slope_difference",
]
Verdict = Literal["pass", "weak", "fail"]

# role keys per probe type: (required, optional)
PROBE_ROLES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "group_difference": (("group", "target"), ()),
    "grouped_relationship": (("x", "y", "group"), ()),
    "slope_difference": (("x", "y", "group"), ()),
    "nonlinear_relationship": (("x", "y"), ()),
    "distribution_difference": (("group", "target"), ()),
    "time_pattern": (("time", "target"), ("group",)),
    "interaction": (("factor1", "factor2", "target"), ()),
}
GROUP_ROLES = frozenset({"group", "factor1", "factor2"})
NUMERIC_ROLES = frozenset({"x", "y", "target"})
TIME_ROLES = frozenset({"time"})
STRICT_NUMERIC_TYPES = frozenset({"nonlinear_relationship", "grouped_relationship", "slope_difference"})
BOX_GROUP_MAX = 50  # distribution_difference draws a box: validate_spec's box x limit


class ProbeRequest(BaseModel):
    type: ProbeType
    columns: dict[str, str]

    @field_validator("columns")
    @classmethod
    def _non_empty_names(cls, value: dict[str, str]) -> dict[str, str]:
        for role, name in value.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"role '{role}' needs a column name")
        return value

    def key(self) -> tuple:
        """Identity used for de-duplication and caching (role order free)."""
        return (self.type, tuple(sorted(self.columns.items())))


class ProbeResult(BaseModel):
    type: ProbeType
    columns: dict[str, str]
    effect_size: float
    effect_label: str
    n: int  # rows the statistic was computed on (profiled-sample level)
    n_min_group: int | None = None
    confidence: Confidence
    verdict: Verdict
    thresholds: dict[str, float]  # {"pass": ..., "weak": ...} on effect_size
    evidence: dict[str, Any]  # structured summaries only — never raw rows
    chart: ChartSpec | None = None
    notes: list[str] = []
    cached: bool = False


class ProbeRejected(BaseModel):
    type: str
    columns: dict[str, str]
    reason: str
    rejected: Literal[True] = True


ProbeOutcome = ProbeResult | ProbeRejected


# --- request validation ---------------------------------------------------------


def validate_request(req: ProbeRequest, profile: DatasetProfile) -> str | None:
    """None when the request may run; otherwise the rejection reason."""
    roles = PROBE_ROLES.get(req.type)
    if roles is None:
        return f"unknown probe type '{req.type}'"
    required, optional = roles
    allowed = set(required) | set(optional)
    missing = [r for r in required if r not in req.columns]
    if missing:
        return f"{req.type} requires columns {list(required)}; missing {missing}"
    extra = [r for r in req.columns if r not in allowed]
    if extra:
        return f"{req.type} does not accept roles {extra}"

    cols = {c.name: c for c in profile.columns}
    names = list(req.columns.values())
    if len(set(names)) != len(names):
        return "every role must name a different column"

    index = _DefinitionalIndex(profile)
    for role, name in req.columns.items():
        col = cols.get(name)
        if col is None:
            return f"column '{name}' does not exist"
        if col.semantic_type in ("id", "text", "unknown"):
            return f"column '{name}' is {col.semantic_type} and cannot be analysed"
        if col.missing_ratio > MAX_MISSING_RATIO:
            return f"column '{name}' has too many missing values ({col.missing_ratio:.0%})"
        representative = index.suppressed.get(name)
        if representative is not None:
            return f"column '{name}' is a near-duplicate of '{representative}'; use '{representative}'"
        problem = _check_role(req.type, role, col)
        if problem is not None:
            return problem

    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            derived = index.pairs.get(frozenset((a, b)))
            if derived is not None:
                return (
                    f"definitional: '{derived.target}' is computed as {derived.formula}; "
                    f"'{a}' vs '{b}' would restate the definition"
                )
    return None


def _check_role(probe_type: str, role: str, col: ColumnProfile) -> str | None:
    if role in GROUP_ROLES:
        if col.semantic_type not in ("categorical", "boolean"):
            return f"role '{role}' needs a categorical column, '{col.name}' is {col.semantic_type}"
        n_cat = col.n_categories or col.unique_count
        limit = BOX_GROUP_MAX if probe_type == "distribution_difference" else MAX_CAT_CATEGORIES
        if n_cat > limit:
            return f"'{col.name}' has {n_cat} categories (limit {limit} for {probe_type})"
        if n_cat < 2:
            return f"'{col.name}' has fewer than 2 categories"
        return None
    if role in NUMERIC_ROLES:
        if probe_type in STRICT_NUMERIC_TYPES or role in ("x", "y"):
            if col.semantic_type != "numeric":
                return f"role '{role}' needs a numeric column, '{col.name}' is {col.semantic_type}"
        elif not _is_numeric_y(col):
            return f"role '{role}' needs a numeric column, '{col.name}' is {col.semantic_type}"
        if col.std is not None and col.std == 0:
            return f"'{col.name}' is constant"
        return None
    if role in TIME_ROLES:
        if col.semantic_type != "datetime":
            return f"role '{role}' needs a datetime column, '{col.name}' is {col.semantic_type}"
        return None
    return f"unknown role '{role}'"


class _DefinitionalIndex:
    """Derived-column pairs (stage 13) and near-duplicate suppression (stage 14)."""

    def __init__(self, profile: DatasetProfile) -> None:
        evidence = profile.evidence
        self.pairs = {}
        for d in evidence.derived_columns:
            for component in d.components:
                self.pairs.setdefault(frozenset((d.target, component)), d)
        self.suppressed: dict[str, str] = {}
        for g in evidence.near_duplicate_groups:
            for dup in g.duplicates:
                self.suppressed[dup] = g.representative
