import uuid
from datetime import datetime

import polars as pl
import pytest

from app.profiling.types import apply_casts, infer_semantic_type, name_looks_like_id


def _infer(name: str, values: list, dtype: pl.DataType | None = None):
    return infer_semantic_type(pl.Series(name, values, dtype=dtype))


# --- id name heuristics -----------------------------------------------------


@pytest.mark.parametrize("name", ["id", "ID", "user_id", "Order_ID", "uuid", "index", "索引", "userId", "orderId"])
def test_id_like_names(name: str) -> None:
    assert name_looks_like_id(name)


@pytest.mark.parametrize("name", ["paid", "valid", "grid", "idea", "identity", "user"])
def test_non_id_names(name: str) -> None:
    assert not name_looks_like_id(name)


# --- non-string dtypes ------------------------------------------------------


def test_boolean_dtype() -> None:
    assert _infer("flag", [True, False, None]) == ("boolean", None)


def test_native_datetime_dtype() -> None:
    assert _infer("ts", [datetime(2024, 1, 1), datetime(2024, 1, 2)]) == ("datetime", None)


def test_all_null_column_is_unknown() -> None:
    assert _infer("empty", [None, None, None], dtype=pl.String) == ("unknown", None)


def test_numeric_id_column() -> None:
    sem, params = _infer("user_id", list(range(1000, 1100)))
    assert (sem, params) == ("id", None)


def test_paid_column_is_not_id() -> None:
    sem, _ = _infer("paid", [float(v) for v in range(100)])
    assert sem == "numeric"


def test_low_cardinality_int_is_categorical() -> None:
    sem, _ = _infer("rating", [1, 2, 3, 4, 5] * 20)
    assert sem == "categorical"


def test_low_cardinality_int_with_wide_span_is_numeric() -> None:
    # only 2 unique values but span 1000 > 100 -> numeric
    sem, _ = _infer("v", [0, 1000] * 50)
    assert sem == "numeric"


def test_plain_floats_are_numeric() -> None:
    sem, _ = _infer("value", [1.5, 2.5, 3.5, 2.5])
    assert sem == "numeric"


# --- string datetime probing ------------------------------------------------


@pytest.mark.parametrize(
    "values,expected",
    [
        (["2024-01-01", "2024-02-15"], [datetime(2024, 1, 1), datetime(2024, 2, 15)]),
        (["2024/01/01", "2024/02/15"], [datetime(2024, 1, 1), datetime(2024, 2, 15)]),
        (["31/12/2024", "01/06/2024"], [datetime(2024, 12, 31), datetime(2024, 6, 1)]),
        (["12/31/2024", "06/01/2024"], [datetime(2024, 12, 31), datetime(2024, 6, 1)]),
        (["2024-01-01T08:30:00", "2024-01-02T09:00:00"], [datetime(2024, 1, 1, 8, 30), datetime(2024, 1, 2, 9)]),
        (["2024-01-01 08:30:00", "2024-01-02 09:00:00"], [datetime(2024, 1, 1, 8, 30), datetime(2024, 1, 2, 9)]),
    ],
)
def test_string_dates_infer_and_replay(values: list[str], expected: list[datetime]) -> None:
    df = pl.DataFrame({"d": values})
    sem, params = infer_semantic_type(df["d"])
    assert sem == "datetime"
    assert params is not None and params.target == "datetime"
    # the recorded format must replay to the right values without re-inference
    assert apply_casts(df, {"d": params})["d"].to_list() == expected


def test_mostly_unparseable_strings_are_not_datetime() -> None:
    sem, _ = _infer("d", ["2024-01-01", "hello", "world", "foo", "bar"] * 4)
    assert sem != "datetime"


# --- string numeric probing -------------------------------------------------


def test_thousands_separated_numbers() -> None:
    df = pl.DataFrame({"amount": ["1,234", "5,678,900", "42"]})
    sem, params = infer_semantic_type(df["amount"])
    assert sem == "numeric"
    assert params is not None and params.thousands and not params.currency and not params.percent
    assert apply_casts(df, {"amount": params})["amount"].to_list() == [1234.0, 5678900.0, 42.0]


def test_currency_numbers() -> None:
    df = pl.DataFrame({"price": ["$1,234.50", "NT$500", "€99"]})
    sem, params = infer_semantic_type(df["price"])
    assert sem == "numeric"
    assert params is not None and params.currency and params.thousands
    assert apply_casts(df, {"price": params})["price"].to_list() == [1234.5, 500.0, 99.0]


def test_percent_numbers_keep_raw_value() -> None:
    df = pl.DataFrame({"pct": ["45%", "30%", "12.5%"]})
    sem, params = infer_semantic_type(df["pct"])
    assert sem == "numeric"
    assert params is not None and params.percent
    # value is kept as-is, never divided by 100
    assert apply_casts(df, {"pct": params})["pct"].to_list() == [45.0, 30.0, 12.5]


def test_digit_strings_do_not_crash_datetime_probe() -> None:
    # str.to_datetime raises on these instead of returning nulls (critique #4)
    sem, params = _infer("code", ["12345", "67890", "11111"])
    assert sem == "numeric"
    assert params is not None and params.target == "numeric"


def test_mostly_non_numeric_strings_are_not_numeric() -> None:
    sem, _ = _infer("mixed", ["1", "two", "three", "four"] * 5)
    assert sem != "numeric"


# --- string id / text / categorical -----------------------------------------


def test_uuid_column_is_id_not_text() -> None:
    # 36-char uuids would match the text rule; id must be checked first
    values = [str(uuid.uuid4()) for _ in range(50)]
    assert _infer("uuid", values)[0] == "id"
    assert _infer("session_id", values)[0] == "id"


def test_string_numeric_id_matches_int_path() -> None:
    # zero-padded numeric codes stay String after CSV load; must classify as
    # id like their Int64 twins, not numeric
    values = [f"{10000 + i}" for i in range(50)]
    assert _infer("user_id", values)[0] == "id"
    assert _infer("amount", values)[0] == "numeric"


def test_long_unique_strings_are_text() -> None:
    values = [f"a fairly long free-form sentence number {i}" for i in range(50)]
    assert _infer("description", values)[0] == "text"


def test_repeated_short_strings_are_categorical() -> None:
    assert _infer("city", ["Taipei", "Tainan", "Hualien"] * 30)[0] == "categorical"


def test_many_unique_short_strings_are_text() -> None:
    # 60 unique values > min(5000, max(50, 5% of 60)) = 50 -> text
    values = [f"code-{i}" for i in range(60)]
    assert _infer("token", values)[0] == "text"


def test_apply_casts_skips_already_cast_and_missing_columns() -> None:
    df = pl.DataFrame({"d": ["2024-01-01", "2024-01-02"]})
    _, params = infer_semantic_type(df["d"])
    once = apply_casts(df, {"d": params, "ghost": params})
    twice = apply_casts(once, {"d": params})  # non-String column -> no-op
    assert twice.equals(once)
