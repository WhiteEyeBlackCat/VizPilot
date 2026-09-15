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
    # column named "value": a weak id hint ("code") would classify these
    # unique digit strings as an identifier (stage 9 #4); the probe crash is
    # what this test pins
    sem, params = _infer("value", ["12345", "67890", "11111"])
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


# --- stage 9 (stage 3): dirty numeric normalisation --------------------------


def _cast(values: list, name: str = "v") -> tuple[str, list]:
    df = pl.DataFrame({name: values})
    sem, params = infer_semantic_type(df[name])
    if params is None or params.target != "numeric":
        return sem, []
    return sem, apply_casts(df, {name: params})[name].to_list()


def test_price_column_with_currency_spaces_and_missing_token() -> None:
    from app.profiling.types import missing_token_count

    values = ["$1,234.50", "$899", " 99 ", "N/A", "1234.5", "$2,000", "42.0"]
    sem, casted = _cast(values, "price")
    assert sem == "numeric"
    assert casted == [1234.5, 899.0, 99.0, None, 1234.5, 2000.0, 42.0]
    assert missing_token_count(pl.Series(values)) == 1  # the only null comes from N/A


@pytest.mark.parametrize(
    "values,expected",
    [
        (["€999", "€1,000", "€5"], [999.0, 1000.0, 5.0]),
        (["42", "7", "13"], [42.0, 7.0, 13.0]),
        (["-$1,234", "$-5", "+5", "$ 12"], [-1234.0, -5.0, 5.0, 12.0]),
        (["45%", "30 %", "12.5%"], [45.0, 30.0, 12.5]),  # raw value kept (stage 2 rule)
        (["1e5", "2.5E-3", ".5", "5.", "-1.5e+2"], [100000.0, 0.0025, 0.5, 5.0, -150.0]),  # stage 3 follow-up
    ],
)
def test_gate_accepts_well_formed_numbers(values: list[str], expected: list[float]) -> None:
    sem, casted = _cast(values)
    assert sem == "numeric" and casted == expected


@pytest.mark.parametrize(
    "values",
    [
        ["A123", "B456", "C789"] * 10,
        ["12kg", "7kg", "3kg"] * 10,
        ["inf", "-inf", "nan"] * 10,  # never becomes NaN/inf
        ["1.234,56", "2.345,67"] * 10,  # European decimals are not silently misread
        ["12,50", "7,25", "3,10"] * 10,  # European decimal comma is not a thousands group
        ["1e", "e5", "1e5.5", "."] * 10,  # malformed exponent / lone dot
    ],
)
def test_gate_rejects_non_numeric_shapes(values: list[str]) -> None:
    sem, casted = _cast(values)
    assert sem != "numeric" and casted == []


def test_stray_non_numeric_values_in_numeric_column_become_invalid_nulls() -> None:
    values = [str(i) for i in range(100)] + ["A123", "12kg"]
    sem, casted = _cast(values)
    assert sem == "numeric"
    assert casted[:100] == [float(i) for i in range(100)]
    assert casted[100:] == [None, None]  # never 123 / 12


def test_leading_zero_digit_strings_are_not_numeric() -> None:
    assert _cast([f"{i:05d}" for i in range(1, 60)], "code")[0] != "numeric"
    assert _cast(["00123", "00456", "00789"] * 5)[0] != "numeric"
    # a numeric column tolerates a rare zero-padded typo below the 5% share
    values = [str(i) for i in range(100, 200)] + ["05"]
    assert _cast(values)[0] == "numeric"


def test_mostly_non_numeric_column_stays_non_numeric() -> None:
    values = [str(i) for i in range(10)] + [f"word{i}" for i in range(90)]
    assert _cast(values)[0] in ("categorical", "text")


def test_quantity_shaped_column_is_not_numeric() -> None:
    # dirty_types.quantity: ints, a few number words, some empties
    values = [str(i % 14 + 1) for i in range(89)] + ["twelve", "five", "ten", "two", "one", "three"] + [""] * 5
    assert _cast(values, "quantity")[0] != "numeric"


def test_token_only_column_is_not_numeric() -> None:
    assert _cast(["N/A"] * 20)[0] != "numeric"
    assert _cast(["N/A", "unknown", "-", "", "null"] * 4)[0] != "numeric"


def test_mostly_tokens_with_a_few_numbers_is_not_numeric() -> None:
    # 100% of the non-token values parse, but only 20% of the column does
    values = ["N/A"] * 40 + [str(i) for i in range(10)]
    assert _cast(values)[0] != "numeric"


def test_missing_tokens_are_numeric_path_only() -> None:
    # a categorical column keeps "unknown" as a category
    sem, _ = infer_semantic_type(pl.Series("gender", ["M", "F", "unknown"] * 20))
    assert sem == "categorical"


def test_replay_reproduces_probe_nulls_on_full_frame() -> None:
    import random

    rng = random.Random(71)
    values = []
    for i in range(2000):
        r = rng.random()
        if r < 0.5:
            values.append(f"${rng.uniform(10, 5000):,.2f}")
        elif r < 0.85:
            values.append(f"{rng.uniform(10, 5000):.2f}")
        elif r < 0.9:
            values.append(rng.choice(["N/A", "n/a", "unknown", ""]))
        elif r < 0.97:
            values.append(f" {rng.uniform(10, 5000):.0f} ")
        else:
            values.append(rng.choice(["A123", "12kg", "inf"]))
    df = pl.DataFrame({"price": values})
    sem, params = infer_semantic_type(df["price"].sample(500, seed=1))  # probe on a sample
    assert sem == "numeric"
    replayed = apply_casts(df, {"price": params})["price"]
    tokens = sum(v.strip().lower() in ("n/a", "unknown", "") for v in values)
    invalid = sum(v in ("A123", "12kg", "inf") for v in values)
    assert replayed.null_count() == tokens + invalid
    assert replayed.drop_nulls().is_finite().all()


# --- stage 9 (stage 4): identifier / nominal detection -----------------------

from app.profiling.types import infer_column, name_has_weak_id_hint  # noqa: E402


def _verdict(name: str, values: list, dtype=None) -> tuple[str, bool]:
    sem, _, nominal = infer_column(pl.Series(name, values, dtype=dtype))
    return sem, nominal


@pytest.mark.parametrize(
    "name", ["code", "zip_code", "ZipCode", "postal_code", "postcode", "order_no", "invoice_number", "phone", "tel", "no", "item code"]
)
def test_weak_id_hints(name: str) -> None:
    assert name_has_weak_id_hint(name)


@pytest.mark.parametrize("name", ["number_of_orders", "num_orders", "quantity", "encoded", "notes", "telemetry", "zipper_sales"])
def test_non_weak_id_hints(name: str) -> None:
    assert not name_has_weak_id_hint(name)


def test_user_id_far_from_unique_per_row_is_id() -> None:
    import random

    rng = random.Random(83)
    values = [rng.randint(1, 100_000) for _ in range(100_000)]  # 1M-event / 100k-user shape, ratio ~0.63
    assert _verdict("user_id", values) == ("id", False)


def test_zip_code_int_is_id() -> None:
    import random

    rng = random.Random(89)
    values = [rng.randint(10_000, 99_999) for _ in range(5000)]  # ratio ~0.97
    assert _verdict("zip_code", values) == ("id", False)
    assert _verdict("zip_code", [str(v) for v in values]) == ("id", False)


def test_moderate_cardinality_code_is_nominal_categorical() -> None:
    import random

    rng = random.Random(97)
    ints = [rng.randint(1, 49) for _ in range(500)]
    assert _verdict("code", ints) == ("categorical", True)
    assert _verdict("code", [f"{v:03d}" for v in ints]) == ("categorical", True)  # "001".."049"


def test_zero_padded_near_unique_digit_strings_are_id() -> None:
    values = [f"{i:05d}" for i in range(1, 400)]  # "00001".. no name hint at all
    assert _verdict("ref", values) == ("id", False)


def test_sequential_and_named_ids_stay_id() -> None:
    assert _verdict("event_id", list(range(1, 1001))) == ("id", False)
    assert _verdict("customer_id", [f"C{i:06d}" for i in range(1, 500)]) == ("id", False)  # string path unchanged
    assert _verdict("index", list(range(60))) == ("id", False)


def test_unnamed_sequential_from_origin_is_id_but_offsets_and_floats_are_not() -> None:
    assert _verdict("row", list(range(1, 101))) == ("id", False)
    assert _verdict("row", list(range(0, 100))) == ("id", False)
    assert _verdict("row", list(range(1, 40)))[0] != "id"  # below the 50-row floor
    assert _verdict("year_unique", list(range(1950, 2025)))[0] == "numeric"  # origin 1950
    assert _verdict("paid", [float(v) for v in range(100)])[0] == "numeric"  # float grid, not an index
    assert _verdict("row", [0, 1, 2] + list(range(4, 101)))[0] == "numeric"  # a gap breaks the pattern


def test_email_and_free_text_unchanged() -> None:
    values = [f"user{i}.name{i % 7}@example.org" for i in range(300)]
    assert _verdict("email", values)[0] == "text"


@pytest.mark.parametrize(
    "name,values,expected",
    [
        ("sensor_reading", [0.1 * i + 0.01 for i in range(1000)], "numeric"),
        ("year", [2000 + i % 26 for i in range(1000)], "numeric"),
        ("quantity", [i % 10 + 1 for i in range(1000)], "categorical"),
        ("rating", [i % 5 + 1 for i in range(1000)], "categorical"),
        ("age", [18 + i % 53 for i in range(1000)], "numeric"),
        ("temperature", [i * 7 % 1201 for i in range(1000)], "numeric"),  # unique-ish ints, no hint
        ("latency_ms", [i * 37 % 9973 for i in range(1000)], "numeric"),
        ("paid", [i % 100 for i in range(1000)], "numeric"),
        ("valid", [i * 13 % 997 for i in range(1000)], "numeric"),
        ("number_of_orders", [i % 57 for i in range(1000)], "numeric"),
        ("category_id", [i % 8 + 1 for i in range(1000)], "categorical"),  # nominal, asserted below
        ("timestamp", [1_700_000_000 + i * 6007 for i in range(1000)], "numeric"),  # epoch seconds
    ],
)
def test_measurements_and_codes_are_not_id(name: str, values: list, expected: str) -> None:
    sem, nominal = _verdict(name, values)
    assert sem == expected
    assert nominal is (name == "category_id")  # strong-hint few codes -> nominal; plain measurements -> not


def test_random_ints_with_high_uniqueness_and_no_hint_stay_numeric() -> None:
    import random

    rng = random.Random(101)
    values = [rng.randint(1, 50_000) for _ in range(5000)]  # ratio ~0.95
    assert _verdict("amount", values) == ("numeric", False)


def test_nominal_never_set_on_id_or_plain_columns() -> None:
    assert _verdict("id", list(range(1, 200)))[1] is False
    assert _verdict("city", ["a", "b", "c"] * 30)[1] is False


# --- stage 4 verification fixes (D2 / D3) --------------------------------------


def test_num_suffix_is_not_an_identifier_hint() -> None:
    assert not name_has_weak_id_hint("orders_num") and not name_has_weak_id_hint("num")
    assert _verdict("orders_num", [i % 30 for i in range(600)]) == ("numeric", False)
    assert _verdict("orders_num", [i % 10 for i in range(600)]) == ("categorical", False)
    assert _verdict("num", list(range(100)) * 3) == ("numeric", False)  # 100 distinct, no hint
    assert _verdict("category_num", [i % 5 + 1 for i in range(500)]) == ("categorical", False)


def test_strong_hint_with_few_codes_stays_categorical_on_small_tables() -> None:
    # few codes under a strong hint are a coded category: categorical, and
    # nominal so an LLM/manual spec cannot average the code numbers
    assert _verdict("region_id", [i % 8 + 1 for i in range(100)]) == ("categorical", True)
    assert _verdict("category_id", [i % 8 + 1 for i in range(160)]) == ("categorical", True)
    import random

    rng = random.Random(109)
    assert _verdict("user_id", [rng.randint(1, 60) for _ in range(100)])[0] == "id"  # 60/100 distinct
    assert _verdict("user_id", list(range(1, 101)))[0] == "id"  # near-unique branch
