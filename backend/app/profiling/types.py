"""Semantic type inference and cast replay.

Inference decides a semantic type per column and records CastParams for
string columns that convert to datetime/numeric. apply_semantic_casts only
replays those stored params — it never re-infers (stage2 blocking fix #1).
"""

import re
from typing import Literal

import polars as pl

from .models import CastParams, DatasetProfile, SemanticType

SAMPLE_SEED = 42  # every sample anywhere in profiling uses this seed

_DATETIME_PROBE_N = 1000
_DATETIME_SUCCESS_RATIO = 0.90
# numeric iff this share of the NON-TOKEN values passes the gate (stage 9 #3:
# missing tokens are missing, not failures) ...
_NUMERIC_SUCCESS_RATIO = 0.95
# ... AND at least this share of ALL non-null values does — a column that is
# mostly missing tokens with a few numbers is not "numeric enough"
_NUMERIC_MIN_OVERALL_RATIO = 0.5
# digit strings with leading zeros ("00123") are codes, never measurements:
# above this share of non-null values the numeric probe is skipped entirely
# (stage 4's id/nominal rules take over)
_LEADING_ZERO_SHARE = 0.05
_ID_UNIQUE_RATIO = 0.95
# stage 9 #4 identifier signals (all require integer-like values):
# weak-hint columns (code / zip / no ...) with at least this unique ratio are
# identifiers; below it they are nominal codes (labels) or, past the string
# categorical limit, identifiers again
_WEAK_HINT_ID_RATIO = 0.5
# an unnamed 0/1-based gap-free all-unique integer column is a row index;
# below this many rows the pattern is too easy to hit by accident
_SEQUENTIAL_ID_MIN_ROWS = 50
# strong-hint columns that are not near-unique per row (user_id in an event
# log) need more distinct values than this to be identifiers: below it a
# region_id / category_id with a handful of codes is a plain categorical even
# on small tables, where the 5%-of-n categorical limit alone would be too low
_STRONG_HINT_MIN_UNIQUE = 20
_TEXT_MIN_AVG_LENGTH = 20
_INT_CATEGORICAL_SPAN = 100

# tried in order after polars' own format inference
_DATETIME_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
)

_ID_NAME_RE = re.compile(r"^(id|.*_id|uuid|guid|索引|index)$", re.IGNORECASE)
_ID_CAMEL_RE = re.compile(r".*Id$")  # case-sensitive so 'paid'/'valid' don't match
# weak identifier hints, suffix-anchored so number_of_orders / num_orders /
# quantity never match; camelCase variants are case-sensitive like _ID_CAMEL_RE
# "num" is deliberately absent: orders_num / category_num are counts or coded
# categories far more often than identifiers (stage 4 verification D2)
_WEAK_HINT_RE = re.compile(
    r"^(.*[_ ])?(code|zip|zipcode|zip_code|postcode|postal|no|number|phone|tel)$", re.IGNORECASE
)
_WEAK_HINT_CAMEL_RE = re.compile(r".*(Code|Zip|No|Number|Phone|Tel)$")
_DIGITS_RE = r"^[+-]?\d+$"
_CURRENCY_RE = r"NT\$|[$¥€£]"
# Known missing tokens (compared lower-cased after trimming). Applied ONLY on
# the numeric cast path (decision A.10): a categorical column keeps "unknown"
# as a category. "nan" is a token (pandas-style NA marker); "inf" is not — it
# fails the gate and counts as invalid, so no string ever becomes NaN/inf.
MISSING_TOKENS = frozenset(
    {"", "na", "n/a", "n.a.", "#n/a", "#na", "nan", "null", "none", "nil", "-", "--", "?", "missing", "unknown"}
)
# Full-match numeric gate: optional sign, optional leading currency symbol,
# well-formed thousands groups or plain digits, optional decimals, optional
# percent. Anything else (A123, 12kg, 1e5, inf, 1.234,56) is invalid — never
# coerced by stripping characters.
# Decimals: "1,234.5", "42", "5.", ".5"; exponent "1e5" / "2.5E-3" allowed;
# "1.234,56" and "12,50" (European) are rejected rather than misread.
_NUMERIC_GATE_RE = (
    r"^[+-]?(NT\$|[$¥€£])?\s*[+-]?((\d{1,3}(,\d{3})+|\d+)(\.\d*)?|\.\d+)([eE][+-]?\d+)?\s*%?$"
)
_LEADING_ZERO_RE = r"^0\d+$"


def name_looks_like_id(name: str) -> bool:
    return bool(_ID_NAME_RE.fullmatch(name) or _ID_CAMEL_RE.fullmatch(name))


def name_has_weak_id_hint(name: str) -> bool:
    return bool(_WEAK_HINT_RE.fullmatch(name) or _WEAK_HINT_CAMEL_RE.fullmatch(name))


Verdict = Literal["id", "nominal"]


def _int_categorical_limit(n: int) -> int:
    return max(5, min(20, int(0.05 * n)))


def _str_categorical_limit(n: int) -> int:
    return min(5000, max(50, int(0.05 * n)))


def _integer_like(non_null: pl.Series) -> pl.Series | None:
    """The values as Int64 when every one is an integer (int dtype,
    integer-valued float, or digit string); None otherwise."""
    if non_null.dtype.is_integer():
        return non_null
    if non_null.dtype.is_float():
        finite = non_null.filter(non_null.is_finite())
        if len(finite) and bool((finite == finite.floor()).all()):
            return finite.cast(pl.Int64)
        return None
    if non_null.dtype == pl.String:
        stripped = non_null.str.strip_chars()
        if bool(stripped.str.contains(_DIGITS_RE).all()):
            return stripped.cast(pl.Int64, strict=False)
        return None
    return None


def _identifier_verdict(name: str, non_null: pl.Series) -> Verdict | None:
    """Multi-signal identifier detection (stage 9 #4). Only integer-like
    columns qualify (a near-unique float column is a measurement); high
    uniqueness alone never does. Returns "id" (excluded from charts),
    "nominal" (categorical whose numbers are labels) or None (undecided —
    the ordinary numeric / categorical rules apply)."""
    ints = _integer_like(non_null)
    if ints is None or len(ints) == 0:
        return None
    n = len(non_null)
    unique = non_null.n_unique()
    ratio = unique / n
    strong = name_looks_like_id(name)
    weak = name_has_weak_id_hint(name) or (
        non_null.dtype == pl.String and _leading_zero_share(non_null) >= _LEADING_ZERO_SHARE
    )
    if strong:
        if ratio > _ID_UNIQUE_RATIO or unique > max(_int_categorical_limit(n), _STRONG_HINT_MIN_UNIQUE):
            # user_id in an event log is far from unique per row but still an id
            return "id"
        # region_id / category_id with a handful of codes: a coded category
        # (labels, never averaged), even on tables too small for the
        # 5%-of-n categorical limit
        return "nominal"
    if weak:
        if ratio >= _WEAK_HINT_ID_RATIO:
            return "id"  # zip_code, phone, zero-padded near-unique codes
        return "nominal" if unique <= _str_categorical_limit(n) else "id"
    # no hint: only a gap-free 0/1-based all-unique INTEGER column (a row
    # index) — integer-valued floats are a measurement grid, not an index
    if non_null.dtype.is_float() or unique < n or n < _SEQUENTIAL_ID_MIN_ROWS:
        return None
    sorted_unique = ints.unique().sort()
    if int(sorted_unique.min()) in (0, 1) and bool((sorted_unique.diff().drop_nulls() == 1).all()):
        return "id"
    return None


def infer_semantic_type(s: pl.Series) -> tuple[SemanticType, CastParams | None]:
    sem, params, _ = infer_column(s)
    return sem, params


def infer_column(s: pl.Series) -> tuple[SemanticType, CastParams | None, bool]:
    """(semantic type, cast params, nominal) — nominal marks a categorical
    whose numeric-looking values are labels (stage 9 #4)."""
    non_null = s.drop_nulls()
    if non_null.is_empty():
        return "unknown", None, False

    dtype = s.dtype
    if dtype == pl.Boolean:
        return "boolean", None, False
    if dtype == pl.Date or dtype == pl.Datetime:
        return "datetime", None, False
    if dtype.is_numeric():
        verdict = _identifier_verdict(s.name, non_null)
        if verdict is not None:
            return ("id", None, False) if verdict == "id" else ("categorical", None, True)
        return _infer_numeric_dtype(s, non_null), None, False
    if dtype == pl.String:
        return _infer_string(s, non_null)
    return "unknown", None, False  # exotic dtypes (duration, struct, ...) — skipped downstream


def _infer_numeric_dtype(s: pl.Series, non_null: pl.Series) -> SemanticType:
    unique_count = non_null.n_unique()
    if s.dtype.is_integer():
        span = non_null.max() - non_null.min()
        if unique_count <= _int_categorical_limit(len(s)) and span <= _INT_CATEGORICAL_SPAN:
            return "categorical"
    return "numeric"


def _infer_string(s: pl.Series, non_null: pl.Series) -> tuple[SemanticType, CastParams | None, bool]:
    params = _probe_datetime(non_null)
    if params is not None:
        return "datetime", params, False
    verdict = _identifier_verdict(s.name, non_null)  # digit strings: codes / zips / zero-padded ids
    if verdict is not None:
        return ("id", None, False) if verdict == "id" else ("categorical", None, True)
    unique_count = non_null.n_unique()
    unique_ratio = unique_count / len(non_null)
    params = None if _leading_zero_share(non_null) >= _LEADING_ZERO_SHARE else _probe_numeric(non_null)
    if params is not None:
        # keep string-typed numeric ids (e.g. zero-padded codes) consistent
        # with the Int64 path, which classifies them as id
        if unique_ratio > _ID_UNIQUE_RATIO and name_looks_like_id(s.name):
            return "id", None, False
        return "numeric", params, False

    # id must win before the text-by-length rule, or 36-char UUIDs become text
    if unique_ratio > _ID_UNIQUE_RATIO and name_looks_like_id(s.name):
        return "id", None, False
    avg_length = non_null.str.len_chars().mean() or 0
    if unique_ratio > _ID_UNIQUE_RATIO and avg_length > _TEXT_MIN_AVG_LENGTH:
        return "text", None, False
    if unique_count <= _str_categorical_limit(len(s)):
        return "categorical", None, False
    return "text", None, False


def _probe_datetime(non_null: pl.Series) -> CastParams | None:
    if len(non_null) > _DATETIME_PROBE_N:
        probe = non_null.sample(_DATETIME_PROBE_N, seed=SAMPLE_SEED)
    else:
        probe = non_null
    for fmt in (None, *_DATETIME_FORMATS):
        try:
            parsed = probe.str.to_datetime(format=fmt, strict=False)
        except Exception:
            # format inference raises (rather than yielding nulls) when it
            # cannot find any format, e.g. on numeric-string columns
            continue
        if parsed.drop_nulls().len() / len(probe) > _DATETIME_SUCCESS_RATIO:
            return CastParams(target="datetime", datetime_format=fmt or "infer")
    return None


def _leading_zero_share(non_null: pl.Series) -> float:
    return float(non_null.str.strip_chars().str.contains(_LEADING_ZERO_RE).mean() or 0.0)


def _apply_expr(s: pl.Series, expr: pl.Expr) -> pl.Series:
    """Evaluates a pl.col("v") expression on a Series, so the probe runs the
    very same expression apply_casts replays later."""
    return s.to_frame("v").select(expr.alias("v"))["v"]


def missing_token_mask(col: pl.Expr) -> pl.Expr:
    return col.str.strip_chars().str.to_lowercase().is_in(list(MISSING_TOKENS))


def missing_token_count(s: pl.Series) -> int:
    """Non-null values of a string column that are known missing tokens
    (what the numeric cast turns into nulls before the gate)."""
    return int(_apply_expr(s.drop_nulls(), missing_token_mask(pl.col("v"))).sum())


def _probe_numeric(non_null: pl.Series) -> CastParams | None:
    params = CastParams(
        target="numeric",
        thousands=bool(non_null.str.contains(",", literal=True).any()),
        currency=bool(non_null.str.contains(_CURRENCY_RE).any()),
        percent=bool(non_null.str.contains("%", literal=True).any()),
    )
    tokens = missing_token_count(non_null)
    non_token = len(non_null) - tokens
    if non_token == 0:
        return None  # nothing but missing tokens
    parsed = _apply_expr(non_null, _numeric_cast(pl.col("v"), params)).drop_nulls().len()
    if parsed / non_token >= _NUMERIC_SUCCESS_RATIO and parsed / len(non_null) >= _NUMERIC_MIN_OVERALL_RATIO:
        return params
    return None


def _numeric_cast(col: pl.Expr, params: CastParams) -> pl.Expr:
    """Normalisation + cast, identical at probe time and replay (stage 9 #3):
    strip -> missing token -> null -> full-match gate (non-match -> null) ->
    strip the symbols named in CastParams -> Float64. Only the named symbols
    are ever removed, and only from values that already passed the gate."""
    col = col.str.strip_chars()
    gated = pl.when(missing_token_mask(col) | ~col.str.contains(_NUMERIC_GATE_RE)).then(None).otherwise(col)
    if params.currency:
        gated = gated.str.replace_all(_CURRENCY_RE, "")
    if params.thousands:
        gated = gated.str.replace_all(",", "", literal=True)
    if params.percent:
        gated = gated.str.replace_all("%", "", literal=True)
    return gated.str.strip_chars().cast(pl.Float64, strict=False)


def _datetime_cast(col, params: CastParams):
    fmt = None if params.datetime_format == "infer" else params.datetime_format
    return col.str.to_datetime(format=fmt, strict=False)


def apply_casts(df: pl.DataFrame, casts: dict[str, CastParams]) -> pl.DataFrame:
    exprs = []
    for name, params in casts.items():
        if name not in df.columns or df.schema[name] != pl.String:
            continue  # already cast, or column gone — replay is a no-op then
        col = pl.col(name)
        expr = _datetime_cast(col, params) if params.target == "datetime" else _numeric_cast(col, params)
        exprs.append(expr.alias(name))
    return df.with_columns(exprs) if exprs else df


def apply_semantic_casts(df: pl.DataFrame, profile: DatasetProfile) -> pl.DataFrame:
    """Replays the profile's stored cast_params; the Stage 4 render contract."""
    return apply_casts(df, {c.name: c.cast_params for c in profile.columns if c.cast_params})
