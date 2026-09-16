"""Stage 19.1 Task 3: order semantics (cyclic / ordered / two-state) and the
dispersion evidence reach the LLM as metadata and the Explore builder via the
profile response; an LLM-suggested rose / polar is kept when the data supports
it. The rules-only path is untouched by any of this."""

import json
import math
import random
from pathlib import Path

import polars as pl
from fastapi.testclient import TestClient

from app.charts.rules import recommend_charts
from app.llm.prompts import build_hypothesis_messages
from app.llm.provider import DisabledProvider
from app.llm.service import RecommendationService
from app.profiling.profiler import profile_dataset
from tests.test_llm_service import FakeProvider

BIG = 10**6
DATA = Path(__file__).resolve().parents[2] / "frontend" / "e2e" / "data"


def _cycles() -> pl.DataFrame:
    return pl.read_csv(DATA / "cycles.csv")


def _spread() -> pl.DataFrame:
    return pl.read_csv(DATA / "spread_groups.csv")


def _prompt(df: pl.DataFrame) -> str:
    profile = profile_dataset(df, "0" * 32, BIG)
    messages = build_hypothesis_messages(profile, recommend_charts(profile), include_sample_rows=False)
    return messages[-1]["content"]


# --- LLM #1 metadata ----------------------------------------------------------


def test_prompt_lists_cyclic_and_ordered_roles() -> None:
    text = _prompt(_cycles())
    assert "Column roles" in text
    assert "- hr: cyclic, period 24" in text
    assert "- mnth: cyclic, period 12" in text
    assert "rides" not in text.split("Column roles")[1].split("Measured evidence")[0]  # plain numerics are not listed


def test_prompt_lists_ordered_scale_and_spread_differences() -> None:
    text = _prompt(_spread())
    assert "- level: ordered category (low < medium < high < very high)" in text
    assert "Spread differences" in text
    assert "- group -> value: spread ratio" in text and "c widest, a narrowest" in text
    assert "strong (n_min=80)" in text


def test_prompt_has_no_roles_section_for_plain_columns() -> None:
    rng = random.Random(5)
    df = pl.DataFrame({"region": [["N", "S", "E"][i % 3] for i in range(90)], "sales": [rng.gauss(0, 1) for _ in range(90)]})
    text = _prompt(df)
    assert "Column roles" not in text and "Spread differences" not in text


def test_system_prompt_mentions_polar_and_rose_as_hints() -> None:
    messages = build_hypothesis_messages(profile_dataset(_cycles(), "0" * 32, BIG), [], include_sample_rows=False)
    system = messages[0]["content"]
    assert "polar_line" in system and "rose" in system and "raincloud" in system
    assert "hints" in system


# --- profile response: column_roles for the Explore builder -------------------------


def test_profile_response_carries_column_roles(client: TestClient) -> None:
    resp = client.post("/api/datasets", files={"file": ("cycles.csv", (DATA / "cycles.csv").read_bytes())})
    assert resp.status_code == 200, resp.text
    dataset_id = resp.json()["dataset_id"]
    payload = client.get(f"/api/datasets/{dataset_id}/profile").json()
    roles = {r["name"]: r for r in payload["column_roles"]}
    assert set(roles) == {"hr", "mnth", "rides", "temp"}
    assert roles["hr"]["cyclic"] and roles["hr"]["period"] == 24 and roles["hr"]["ordered"]
    assert roles["mnth"]["cyclic"] and roles["mnth"]["period"] == 12
    assert roles["rides"]["role"] == "numeric" and not roles["rides"]["cyclic"]
    # the cached profile itself is unchanged: the field is added at response time only
    assert "column_roles" not in json.loads(json.dumps(payload["columns"]))[0]


# --- an LLM rose / polar suggestion survives when the cycle supports it ---------------


def _rose_suggestion(**overrides) -> dict:
    chart = {"title": "total rides by hour", "type": "rose", "x": "hr", "y": "rides", "aggregation": "sum", "reason": "commute peaks", "priority": 1}
    chart.update(overrides)
    return chart


def test_llm_rose_over_a_cycle_is_kept_and_polar_is_an_alternative() -> None:
    df = _cycles()
    profile = profile_dataset(df, "0" * 32, BIG)
    service = RecommendationService(FakeProvider(hypotheses={"hypotheses": [], "charts": [_rose_suggestion()]}))
    result = service.get(profile, use_llm=True, df=df)
    rose = [c for c in result["charts"] if c["spec"]["type"] == "rose"]
    assert len(rose) == 1 and rose[0]["source"] == "llm"
    assert rose[0]["intent"] == "cyclic_pattern" and rose[0]["spec"]["sort"] == "natural"
    assert {a["type"] for a in rose[0]["alternatives"]} >= {"line", "polar_line"}


def test_llm_rose_with_a_mean_falls_back_to_the_preferred_cycle_chart() -> None:
    df = _cycles()
    profile = profile_dataset(df, "0" * 32, BIG)
    service = RecommendationService(FakeProvider(hypotheses={"hypotheses": [], "charts": [_rose_suggestion(aggregation="mean")]}))
    result = service.get(profile, use_llm=True, df=df)
    # a rose of a mean is rejected by validate_spec: the suggestion is dropped, never bent
    assert not [c for c in result["charts"] if c["spec"]["type"] == "rose"]


def test_llm_rose_over_a_nominal_column_is_never_kept() -> None:
    rng = random.Random(6)
    df = pl.DataFrame({"region": [["N", "S", "E", "W"][i % 4] for i in range(200)], "sales": [abs(rng.gauss(5, 1)) for _ in range(200)]})
    profile = profile_dataset(df, "0" * 32, BIG)
    service = RecommendationService(FakeProvider(hypotheses={"hypotheses": [], "charts": [_rose_suggestion(x="region", y="sales")]}))
    result = service.get(profile, use_llm=True, df=df)
    assert not [c for c in result["charts"] if c["spec"]["type"] in ("rose", "polar_line")]


# --- the rules-only path does not depend on any of this ------------------------------


def test_rules_only_output_ignores_the_llm_side_metadata() -> None:
    df = _cycles()
    profile = profile_dataset(df, "0" * 32, BIG)
    rules = RecommendationService(DisabledProvider()).get(profile, use_llm=False)
    again = RecommendationService(FakeProvider(hypotheses={"hypotheses": [], "charts": [_rose_suggestion()]})).get(profile, use_llm=False)
    assert rules == again
    assert all(c["source"] == "rules" for c in rules["charts"])
    assert not any(c["spec"]["type"] == "rose" for c in rules["charts"])  # rose stays an Explore / LLM choice
    assert math.isfinite(rules["charts"][0]["score"])


# --- stage 19.2 (A5): month names reach the Explore builder as a cycle -----------------


def test_profile_response_marks_month_names_cyclic_for_the_builder(client: TestClient) -> None:
    """The Explore builder offers rose / polar x only for `column_roles`
    entries with cyclic=True (ManualBuilder.tsx); twelve month names exceed
    the 10 top_values, so the role must come from the full label set."""
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    rows = ["month,day,rides"] + [f"{months[i % 12]},{'Mon Tue Wed Thu Fri Sat Sun'.split()[i % 7]},{(i * 37) % 100}" for i in range(360)]
    resp = client.post("/api/datasets", files={"file": ("months.csv", "\n".join(rows).encode())})
    assert resp.status_code == 200, resp.text
    payload = client.get(f"/api/datasets/{resp.json()['dataset_id']}/profile").json()
    roles = {r["name"]: r for r in payload["column_roles"]}
    assert roles["month"]["cyclic"] and roles["month"]["period"] == 12 and roles["month"]["order"] == months
    assert roles["day"]["cyclic"] and roles["day"]["period"] == 7
    assert not roles["rides"]["cyclic"]
    column = next(c for c in payload["columns"] if c["name"] == "month")
    assert len(column["top_values"]) == 10 and sorted(column["categories"]) == sorted(months)
