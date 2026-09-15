"""Single-execution pairwise Pearson (stage 9 #1), shared by the profiler
(raw values) and the evidence layer (ranks -> Spearman). Kept apart from
profiler.py because evidence.py must not import the profiler (cycle)."""

import math
from typing import Any

import polars as pl


def _valid_mask(df: pl.DataFrame, name: str) -> pl.Expr:
    """Rows a column contributes to pairwise statistics: non-null, and finite
    for floats (ints cannot hold NaN/inf) — the numeric-stats treatment."""
    if df.schema[name].is_float():
        return pl.col(name).is_not_null() & pl.col(name).is_finite()
    return pl.col(name).is_not_null()


def pairwise_pearson(
    df: pl.DataFrame, cols: list[str]
) -> tuple[list[list[float | None]], list[list[int]]]:
    """Pairwise-complete Pearson matrix plus the row count behind each cell,
    computed in ONE polars execution (stage 9 #1): every pair is an expression
    over the pair's own null/NaN/inf mask, so the semantics equal the former
    per-pair select/drop_nulls/filter/collect (which cost ~11s for 30 columns)
    without materialising a dataframe per pair. numpy is deliberately not a
    dependency, so df.corr() is not an option. A pair with < 2 rows or a
    constant column yields None (pl.corr returns NaN there)."""
    n = len(cols)
    matrix: list[list[float | None]] = [[None] * n for _ in range(n)]
    counts: list[list[int]] = [[0] * n for _ in range(n)]
    masks = {c: _valid_mask(df, c) for c in cols}
    exprs: list[pl.Expr] = [masks[c].sum().alias(f"n:{i}") for i, c in enumerate(cols)]
    for i in range(n):
        for j in range(i + 1, n):
            m = masks[cols[i]] & masks[cols[j]]
            a = pl.col(cols[i]).filter(m).cast(pl.Float64)
            b = pl.col(cols[j]).filter(m).cast(pl.Float64)
            exprs.append(pl.corr(a, b).alias(f"r:{i}:{j}"))
            exprs.append(m.sum().alias(f"n:{i}:{j}"))
    row = df.select(exprs).row(0, named=True)
    for i in range(n):
        matrix[i][i] = 1.0
        counts[i][i] = int(row[f"n:{i}"])
        for j in range(i + 1, n):
            count = int(row[f"n:{i}:{j}"])
            value = _finite(row[f"r:{i}:{j}"]) if count >= 2 else None
            matrix[i][j] = matrix[j][i] = value
            counts[i][j] = counts[j][i] = count
    return matrix, counts


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None
