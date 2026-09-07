"""Project-wide JSON boundary convention (stage1 critique #1, reused by later stages):

- float NaN / +inf / -inf are serialized as null
- datetime / date / time become ISO 8601 strings
- null values are preserved as null
"""

import math
from datetime import date, datetime, time
from typing import Any

import polars as pl


def df_to_records(df: pl.DataFrame) -> list[dict[str, Any]]:
    return [{key: _jsonify(value) for key, value in row.items()} for row in df.to_dicts()]


def _jsonify(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    return value
