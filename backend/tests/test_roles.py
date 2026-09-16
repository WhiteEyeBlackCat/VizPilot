"""Stage 19: semantic roles are a deterministic function of the profile."""

import random

import polars as pl
import pytest

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


# --- stage 19.2 (A5): month / weekday names are cyclic even beyond top_values ---

MONTHS_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTHS_FULL = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _cat_full(name: str, labels: list[str]) -> ColumnProfile:
    """A profile as the stage 19.2 profiler writes it: top_values capped at
    10, the complete label set in `categories`."""
    col = _cat(name, labels)
    col.categories = sorted(labels)
    return col


@pytest.mark.parametrize("labels", [MONTHS_ABBR, MONTHS_FULL])
def test_twelve_month_names_are_a_cycle_in_natural_order(labels: list[str]) -> None:
    role = column_role(_cat_full("period", labels))
    assert (role.role, role.cyclic, role.period) == ("cyclic", True, 12)
    assert role.order == labels  # natural month order, not the sorted / first-seen order


def test_month_names_are_recognised_whatever_the_input_order_and_case() -> None:
    shuffled = ["oct", "FEB", "Dec", "jan", "Sept.", "Mar", "Jul", "Nov", "may", "Apr", "Aug", "JUN"]
    role = column_role(_cat_full("m", shuffled))
    assert role.cyclic and role.period == 12
    assert [v.strip(".").lower()[:3] for v in role.order or []] == [m.lower() for m in MONTHS_ABBR]


def test_weekday_names_keep_working() -> None:
    role = column_role(_cat_full("day", WEEKDAYS[::-1]))
    assert (role.cyclic, role.period, role.order) == (True, 7, WEEKDAYS)


def test_twelve_unrelated_labels_are_nominal_not_cyclic() -> None:
    labels = ["ads", "email", "organic", "paid", "referral", "social", "video", "print", "radio", "tv", "event", "partner"]
    role = column_role(_cat_full("channel", labels))
    assert role.role == "nominal" and not role.cyclic and role.period is None


def test_mixed_spellings_of_the_same_month_are_not_a_cycle() -> None:
    role = column_role(_cat_full("m", ["Jan", "January", "Feb", "Mar"]))
    assert not role.cyclic


def test_profiler_keeps_the_full_label_set_only_for_small_categoricals() -> None:
    rng = random.Random(5)
    n = 600
    df = pl.DataFrame(
        {
            "month": [MONTHS_FULL[rng.randrange(12)] for _ in range(n)],
            "channel": [f"c{rng.randrange(20)}" for _ in range(n)],
            "rides": [rng.randint(0, 100) for _ in range(n)],
        }
    )
    profile = profile_dataset(df, "0" * 32, BIG)
    month = next(c for c in profile.columns if c.name == "month")
    assert month.n_categories == 12 and len(month.top_values or []) == 10  # the cap is unchanged
    assert month.categories == sorted(MONTHS_FULL)
    channel = next(c for c in profile.columns if c.name == "channel")
    assert channel.n_categories == 20 and channel.categories is None  # beyond CATEGORY_LABELS_MAX
    roles = roles_for(profile.columns)
    assert roles["month"].cyclic and roles["month"].period == 12 and roles["month"].order == MONTHS_FULL
    assert roles["channel"].role == "nominal"
