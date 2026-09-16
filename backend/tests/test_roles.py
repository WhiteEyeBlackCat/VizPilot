"""Stage 19: semantic roles are a deterministic function of the profile."""

import polars as pl

from app.charts.roles import column_role, roles_for
from app.profiling.models import ColumnProfile, TopValue
from app.profiling.profiler import profile_dataset

BIG = 10**6


def _col(name: str, semantic: str, dtype: str = "String", **extra) -> ColumnProfile:
    base = dict(
        name=name,
        original_dtype=dtype,
        semantic_type=semantic,
        missing_count=0,
        missing_ratio=0.0,
        unique_count=extra.pop("unique_count", 3),
    )
    return ColumnProfile(**base, **extra)


def _cat(name: str, labels: list, dtype: str = "String", nominal: bool = False) -> ColumnProfile:
    return _col(
        name,
        "categorical",
        dtype,
        nominal=nominal,
        n_categories=len(labels),
        unique_count=len(labels),
        top_values=[TopValue(value=v, count=10) for v in labels[:10]],
    )


def test_datetime_numeric_boolean_roles() -> None:
    assert column_role(_col("ts", "datetime", "Datetime")).role == "datetime"
    num = column_role(_col("temp", "numeric", "Float64", min=0.1, max=0.9))
    assert num.role == "numeric" and num.ordered and not num.cyclic
    flag = column_role(_col("flag", "boolean", "Boolean", n_categories=2))
    assert flag.role == "boolean" and flag.two_state


def test_cyclic_by_name_needs_a_fitting_range() -> None:
    hour = column_role(_col("hr", "numeric", "Int64", min=0, max=23))
    assert (hour.role, hour.period, hour.cyclic, hour.ordered) == ("cyclic", 24, True, True)
    # "hour" holding 0..500 is a duration, not a clock
    assert column_role(_col("hour", "numeric", "Int64", min=0, max=500)).role == "numeric"
    month = column_role(_cat("mnth", [3, 4, 5], dtype="Int64"))
    assert month.role == "cyclic" and month.period == 12 and month.order == ["3", "4", "5"]
    # integer months beyond the ten listed top values: the name and a
    # plausible category count decide; the order is resolved at render time
    twelve = _cat("month", list(range(1, 13)), dtype="Int64")
    assert column_role(twelve).role == "cyclic" and column_role(twelve).order is None
    # a nominal code named like a cycle is still nominal
    assert column_role(_cat("season", [1, 2], dtype="Int64", nominal=True)).role != "cyclic"


def test_cyclic_by_labels() -> None:
    weekday = column_role(_cat("day", ["Wed", "Mon", "Fri"]))
    assert weekday.role == "cyclic" and weekday.period == 7 and weekday.order == ["Mon", "Wed", "Fri"]
    month = column_role(_cat("period", ["March", "January"]))
    assert month.role == "cyclic" and month.order == ["January", "March"]
    season = column_role(_cat("s", ["Fall", "Spring"]))
    assert season.role == "cyclic" and season.order == ["Spring", "Fall"]


def test_ordinal_scales_and_integer_codes() -> None:
    tier = column_role(_cat("tier", ["Gold", "Bronze", "Silver"]))
    assert tier.role == "ordinal" and tier.order == ["Bronze", "Silver", "Gold"]
    size = column_role(_cat("size", ["L", "S", "M"]))
    assert size.order == ["S", "M", "L"]
    codes = column_role(_cat("weathersit", [1, 2, 3], dtype="Int64"))
    assert codes.role == "ordinal" and codes.order == ["1", "2", "3"]
    # a nominal integer code (product id-like) is not ordered
    assert column_role(_cat("code", [10, 20], dtype="Int64", nominal=True)).role == "nominal"


def test_nominal_and_two_state() -> None:
    region = column_role(_cat("region", ["North", "South", "East"]))
    assert region.role == "nominal" and not region.ordered and not region.two_state
    ab = column_role(_cat("variant", ["A", "B"]))
    assert ab.role == "nominal" and ab.two_state
    assert column_role(_col("note", "text")).role == "nominal"


def test_roles_on_a_profiled_frame() -> None:
    df = pl.DataFrame(
        {
            "hr": [i % 24 for i in range(480)],
            "weekday": [["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][i % 7] for i in range(480)],
            "region": [["N", "S", "E", "W"][i % 4] for i in range(480)],
            "level": [["low", "medium", "high"][i % 3] for i in range(480)],
            "y": [float(i % 17) for i in range(480)],
        }
    )
    roles = roles_for(profile_dataset(df, "0" * 32, BIG).columns)
    assert roles["hr"].role == "cyclic" and roles["hr"].period == 24
    assert roles["weekday"].role == "cyclic" and roles["weekday"].order == ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    assert roles["region"].role == "nominal"
    assert roles["level"].role == "ordinal" and roles["level"].order == ["low", "medium", "high"]
    assert roles["y"].role == "numeric"
    # deterministic
    assert roles_for(profile_dataset(df, "0" * 32, BIG).columns) == roles
