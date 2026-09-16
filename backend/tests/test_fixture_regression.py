"""Stage 19.1: the recorded RenderResult fixtures (frontend/src/charts/
__fixtures__, captured from the real backend by e2e/capture-fixtures.mjs)
whose inputs are committed under frontend/e2e/data are re-rendered here and
compared semantically: same shape, same strings and integers, floats within
a tolerance. The last float digit of an aggregation may differ between runs
(known); a changed count, category order, bin edge or density shape fails."""

import json
import math
from pathlib import Path

import pytest

from app.charts.render import render_chart
from app.charts.spec import ChartSpec, validate_spec
from app.datasets.loader import load_dataframe
from app.profiling.profiler import profile_dataset
from app.profiling.types import apply_semantic_casts

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "frontend" / "src" / "charts" / "__fixtures__"
BIG = 10**6
REL_TOL = 1e-6
ABS_TOL = 1e-9


def _entries() -> dict[str, dict]:
    index = json.loads((FIXTURES / "index.json").read_text())
    return {name: fx for name, fx in index.items() if fx["file"].startswith("e2e/") and (FIXTURES / f"{name}.json").exists()}


def _assert_close(actual, expected, path: str) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert set(actual) >= set(expected), f"{path}: missing keys {set(expected) - set(actual)}"
        for key, value in expected.items():
            _assert_close(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), f"{path}: length {len(actual)} != {len(expected)}"
        for i, (a, e) in enumerate(zip(actual, expected)):
            _assert_close(a, e, f"{path}[{i}]")
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"
    elif isinstance(expected, int) and not isinstance(actual, float):
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"
    else:
        assert isinstance(actual, (int, float)) and not isinstance(actual, bool), f"{path}: {actual!r} is not a number"
        assert math.isclose(float(actual), float(expected), rel_tol=REL_TOL, abs_tol=ABS_TOL), f"{path}: {actual!r} != {expected!r}"


_ENTRIES = _entries()


@pytest.mark.parametrize("name", sorted(_ENTRIES))
def test_recorded_fixture_matches_the_backend(name: str) -> None:
    fx = _ENTRIES[name]
    path = REPO / "frontend" / fx["file"]
    df = load_dataframe(path.read_bytes(), path.name)
    profile = profile_dataset(df, "0" * 32, BIG)
    spec = ChartSpec.model_validate(fx["spec"])
    assert validate_spec(spec, profile) == []
    result = json.loads(json.dumps(render_chart(apply_semantic_casts(df, profile), spec, profile), default=str))
    recorded = json.loads((FIXTURES / f"{name}.json").read_text())
    for key in ("chart_data", "n_points", "sampled", "display_range"):
        _assert_close(result[key], recorded[key], f"{name}.{key}")


def test_stage191_fixtures_are_recorded() -> None:
    expected = {
        "raincloud_spread", "box_same_mean", "ridgeline_ordered", "bar_ordered_natural", "line_ordered_category",
        "polar_hour", "rose_hour_sum", "rose_month_count", "line_cyclic_month", "hexbin_dense",
    }
    assert expected <= set(_ENTRIES), f"missing fixtures: {expected - set(_ENTRIES)} (npm run capture-fixtures -- --only …)"


def test_stage191_fixture_semantics() -> None:
    """The purpose of each fixture, checked on the recorded payloads."""
    load = lambda name: json.loads((FIXTURES / f"{name}.json").read_text())["chart_data"]  # noqa: E731
    rain = load("raincloud_spread")
    names = [g["name"] for g in rain["groups"]]
    assert names == ["a", "b", "c"] and len(rain["grid"]) == 64
    assert all(len(g["density"]) == 64 and max(g["density"]) == pytest.approx(1.0) for g in rain["groups"])
    # same mean, different spread: the medians agree within a unit, the IQRs do not
    medians = [g["median"] for g in rain["groups"]]
    iqrs = [g["q3"] - g["q1"] for g in rain["groups"]]
    assert max(medians) - min(medians) < 3 and iqrs[2] / iqrs[0] > 3
    ridge = load("ridgeline_ordered")
    assert [g["name"] for g in ridge["groups"]] == ["low", "medium", "high", "very high"]
    assert load("bar_ordered_natural")["categories"] == ["low", "medium", "high", "very high"]
    assert load("line_ordered_category")["series"][0]["x"] == ["low", "medium", "high", "very high"]
    polar = load("polar_hour")
    assert polar["categories"] == list(range(24))
    rose = load("rose_hour_sum")
    assert rose["categories"] == list(range(24)) and rose["series"][0]["values"][8] > rose["series"][0]["values"][3]
    assert load("rose_month_count")["categories"] == list(range(1, 13))
    assert load("line_cyclic_month")["series"][0]["x"] == list(range(1, 13))
    hexbin = load("hexbin_dense")
    assert hexbin["n"] == 2000 and sum(c[2] for c in hexbin["cells"]) == 2000
    # offset coordinates: odd rows are shifted half a cell, so the leftmost column can be -1
    assert all(-1 <= c[0] <= hexbin["columns"] and 0 <= c[1] < hexbin["rows"] for c in hexbin["cells"])
    assert hexbin["dx"] > 0 and hexbin["dy"] > 0 and hexbin["x_range"][0] < hexbin["x_range"][1]
