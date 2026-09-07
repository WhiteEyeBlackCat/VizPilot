"""Semantic type inference and cast replay.

Inference decides a semantic type per column and records CastParams for
string columns that convert to datetime/numeric. apply_semantic_casts only
replays those stored params — it never re-infers (stage2 blocking fix #1).
"""

import re

import polars as pl

from .models import CastParams, DatasetProfile, SemanticType

SAMPLE_SEED = 42  # every sample anywhere in profiling uses this seed

_DATETIME_PROBE_N = 1000
_DATETIME_SUCCESS_RATIO = 0.90
_NUMERIC_SUCCESS_RATIO = 0.95
_ID_UNIQUE_RATIO = 0.95
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

_ID_NAME_RE = re.compile(r"^(id|.*_id|uuid|索引|index)$", re.IGNORECASE)
_ID_CAMEL_RE = re.compile(r".*Id$")  # case-sensitive so 'paid'/'valid' don't match
_CURRENCY_RE = r"NT\$|[$¥€£]"


def name_looks_like_id(name: str) -> bool:
    return bool(_ID_NAME_RE.fullmatch(name) or _ID_CAMEL_RE.fullmatch(name))


def infer_semantic_type(s: pl.Series) -> tuple[SemanticType, CastParams | None]:
    non_null = s.drop_nulls()
    if non_null.is_empty():
        return "unknown", None

    dtype = s.dtype
    if dtype == pl.Boolean:
        return "boolean", None
    if dtype == pl.Date or dtype == pl.Datetime:
        return "datetime", None
    if dtype.is_numeric():
        return _infer_numeric_dtype(s, non_null), None
    if dtype == pl.String:
        return _infer_string(s, non_null)
    return "unknown", None  # exotic dtypes (duration, struct, ...) — skipped downstream


def _infer_numeric_dtype(s: pl.Series, non_null: pl.Series) -> SemanticType:
    unique_count = non_null.n_unique()
    if name_looks_like_id(s.name) and unique_count / len(non_null) > _ID_UNIQUE_RATIO:
        return "id"
    if s.dtype.is_integer():
        limit = max(5, min(20, int(0.05 * len(s))))
        span = non_null.max() - non_null.min()
        if unique_count <= limit and span <= _INT_CATEGORICAL_SPAN:
            return "categorical"
    return "numeric"


def _infer_string(s: pl.Series, non_null: pl.Series) -> tuple[SemanticType, CastParams | None]:
    params = _probe_datetime(non_null)
    if params is not None:
        return "datetime", params
    unique_count = non_null.n_unique()
    unique_ratio = unique_count / len(non_null)
    params = _probe_numeric(non_null)
    if params is not None:
        # keep string-typed numeric ids (e.g. zero-padded codes) consistent
        # with the Int64 path, which classifies them as id
        if unique_ratio > _ID_UNIQUE_RATIO and name_looks_like_id(s.name):
            return "id", None
        return "numeric", params

    # id must win before the text-by-length rule, or 36-char UUIDs become text
    if unique_ratio > _ID_UNIQUE_RATIO and name_looks_like_id(s.name):
        return "id", None
    avg_length = non_null.str.len_chars().mean() or 0
    if unique_ratio > _ID_UNIQUE_RATIO and avg_length > _TEXT_MIN_AVG_LENGTH:
        return "text", None
    if unique_count <= min(5000, max(50, int(0.05 * len(s)))):
        return "categorical", None
    return "text", None


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


def _probe_numeric(non_null: pl.Series) -> CastParams | None:
    params = CastParams(
        target="numeric",
        thousands=bool(non_null.str.contains(",", literal=True).any()),
        currency=bool(non_null.str.contains(_CURRENCY_RE).any()),
        percent=bool(non_null.str.contains("%", literal=True).any()),
    )
    parsed = _numeric_cast(non_null, params)
    if parsed.drop_nulls().len() / len(non_null) > _NUMERIC_SUCCESS_RATIO:
        return params
    return None


def _numeric_cast(col, params: CastParams):
    """Cleaning + cast used identically at probe time (Series) and replay (Expr)."""
    col = col.str.strip_chars()
    if params.currency:
        col = col.str.replace_all(_CURRENCY_RE, "")
    if params.thousands:
        col = col.str.replace_all(",", "", literal=True)
    if params.percent:
        col = col.str.replace_all("%", "", literal=True)
    return col.str.strip_chars().cast(pl.Float64, strict=False)


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
