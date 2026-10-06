"""Stage 20b renderers: the composition (stacked_bar) and flow (sankey)
payloads. Everything the adapter draws is aggregated here — the shares, the
bar totals, the node weights — and the two disclosures a flow owes the reader
(dropped self-loops, truncated links) travel with the payload."""

import polars as pl
import pytest

from app.charts.render import SANKEY_MAX_LINKS, render_chart
from app.charts.spec import ChartSpec, validate_spec
from app.profiling.profiler import profile_dataset
from app.profiling.types import apply_semantic_casts

BIG = 10**6


def _render(df: pl.DataFrame, **kwargs):
    profile = profile_dataset(df, "0" * 32, BIG)
    casted = apply_semantic_casts(df, profile)
    spec = ChartSpec(title="t", **kwargs)
    assert validate_spec(spec, profile) == [], validate_spec(spec, profile)
    return render_chart(casted, spec, profile)


def _composition_df(n: int = 240) -> pl.DataFrame:
    """region (4) x channel (3) -> revenue; the mix differs by region, and
    (East, partner) never occurs so one stack cell is missing."""
    regions = ["North", "South", "East", "West"]
    channels = ["online", "retail", "partner"]
    rows = []
    for i in range(n):
        region = regions[i % 4]
        channel = channels[i % 3]
        if region == "East" and channel == "partner":
            channel = "online"
        rows.append({"region": region, "channel": channel, "level": ["low", "high"][i % 2], "revenue": 10.0 + (i % 7)})
    return pl.DataFrame(rows)


def _flow_df() -> pl.DataFrame:
    """from_hub -> to_hub with one self-loop per cycle and a weight column."""
    hubs = ["alpha", "bravo", "charlie"]
    rows = []
    for i in range(90):
        src = hubs[i % 3]
        dst = src if i % 5 == 0 else hubs[(i % 3 + 1) % 3]
        rows.append({"from_hub": src, "to_hub": dst, "weight": float(i % 4 + 1)})
    return pl.DataFrame(rows)


# --- composition -------------------------------------------------------------


def test_stacked_bar_sums_the_parts_of_every_bar() -> None:
    data = _render(_composition_df(), type="stacked_bar", x="region", y="revenue", group_by="channel", aggregation="sum")["chart_data"]
    assert sorted(data["categories"]) == ["East", "North", "South", "West"]
    assert [s["name"] for s in data["series"]] == ["online", "partner", "retail"]
    assert data["percent"] is False and data["truncated"] is False
    # the bar total is the sum of its parts, nulls skipped
    for i, total in enumerate(data["totals"]):
        parts = [s["values"][i] for s in data["series"] if s["values"][i] is not None]
        assert total == pytest.approx(sum(parts))
    # the combination that never occurs is a gap, never a zero
    east = data["categories"].index("East")
    partner = next(s for s in data["series"] if s["name"] == "partner")
    assert partner["values"][east] is None


def test_stacked_bar_percent_normalises_every_bar_and_keeps_the_totals() -> None:
    absolute = _render(_composition_df(), type="stacked_bar", x="region", y="revenue", group_by="channel", aggregation="sum")["chart_data"]
    share = _render(
        _composition_df(), type="stacked_bar", x="region", y="revenue", group_by="channel", aggregation="sum", percent=True
    )["chart_data"]
    assert share["percent"] is True and share["y_label"] == "share of sum(revenue) (%)"
    # the absolute magnitude behind a share is never lost
    assert sorted(share["totals"]) == sorted(absolute["totals"])
    for i in range(len(share["categories"])):
        parts = [s["values"][i] for s in share["series"] if s["values"][i] is not None]
        assert sum(parts) == pytest.approx(100.0)


def test_stacked_bar_counts_rows_without_a_measure() -> None:
    data = _render(_composition_df(), type="stacked_bar", x="region", group_by="channel", aggregation="count")["chart_data"]
    assert data["y_label"] == "count"
    assert sum(t for t in data["totals"] if t is not None) == 240


def test_stacked_bar_stacks_ordered_parts_in_their_own_order() -> None:
    # an ordered part column (low < high) stacks in that order, not alphabetically
    data = _render(_composition_df(), type="stacked_bar", x="region", y="revenue", group_by="level", aggregation="sum")["chart_data"]
    assert [s["name"] for s in data["series"]] == ["low", "high"]


def test_stacked_bar_keeps_the_natural_order_of_its_bars() -> None:
    df = _composition_df().with_columns(pl.col("level").alias("band"))
    ranked = _render(df, type="stacked_bar", x="band", y="revenue", group_by="channel", aggregation="sum")["chart_data"]
    natural = _render(df, type="stacked_bar", x="band", y="revenue", group_by="channel", aggregation="sum", sort="natural")["chart_data"]
    assert natural["categories"] == ["low", "high"]
    assert sorted(ranked["categories"]) == sorted(natural["categories"])


# --- flow --------------------------------------------------------------------


def test_sankey_counts_the_pairs_and_drops_self_loops() -> None:
    result = _render(_flow_df(), type="sankey", x="from_hub", group_by="to_hub", aggregation="count")
    data = result["chart_data"]
    assert data["self_loops"] == 18  # every 5th of 90 rows stays in its hub
    assert data["truncated"] is False and data["n_links_total"] == 3
    assert sum(link["value"] for link in data["links"]) == 90 - 18
    # no link may start and end on the same hub label
    labels = {n["name"]: n["label"] for n in data["nodes"]}
    assert all(labels[link["source"]] != labels[link["target"]] for link in data["links"])
    assert result["n_points"] == len(data["links"])


def test_sankey_node_names_separate_the_two_ends() -> None:
    data = _render(_flow_df(), type="sankey", x="from_hub", group_by="to_hub", aggregation="count")["chart_data"]
    names = [n["name"] for n in data["nodes"]]
    assert len(names) == len(set(names)), names
    sides = {n["side"] for n in data["nodes"]}
    assert sides == {"source", "target"}
    # a label appearing on both ends is two nodes, so the graph never cycles
    assert {n["label"] for n in data["nodes"] if n["side"] == "source"} & {
        n["label"] for n in data["nodes"] if n["side"] == "target"
    }
    # a node's value is the total weight of the links on its side
    for node in data["nodes"]:
        key = "source" if node["side"] == "source" else "target"
        assert node["value"] == sum(link["value"] for link in data["links"] if link[key] == node["name"])


def test_sankey_sums_a_numeric_weight() -> None:
    data = _render(_flow_df(), type="sankey", x="from_hub", y="weight", group_by="to_hub", aggregation="sum")["chart_data"]
    assert data["y_label"] == "sum(weight)"
    kept = _flow_df().filter(pl.col("from_hub") != pl.col("to_hub"))
    assert sum(link["value"] for link in data["links"]) == pytest.approx(kept["weight"].sum())


def test_sankey_keeps_only_the_heaviest_links_and_says_so() -> None:
    # 12 x 12 sources/targets = 132 off-diagonal pairs, each with a distinct weight
    rows = []
    for a in range(12):
        for b in range(12):
            for k in range(a * 12 + b + 1):
                rows.append({"from_node": f"s{a:02d}", "to_node": f"t{b:02d}", "k": k})
    data = _render(pl.DataFrame(rows), type="sankey", x="from_node", group_by="to_node", aggregation="count")["chart_data"]
    assert data["truncated"] is True
    assert data["n_links_total"] == 144 and len(data["links"]) == SANKEY_MAX_LINKS
    values = [link["value"] for link in data["links"]]
    assert values == sorted(values, reverse=True)
    # the kept links are the heaviest ones: the lightest kept beats every dropped one
    assert min(values) > 144 - SANKEY_MAX_LINKS


def test_sankey_with_nothing_but_self_loops_is_empty_and_honest() -> None:
    df = pl.DataFrame({"from_hub": ["a", "b", "c"] * 10, "to_hub": ["a", "b", "c"] * 10, "v": [1.0] * 30})
    result = _render(df, type="sankey", x="from_hub", group_by="to_hub", aggregation="count")
    assert result["chart_data"]["self_loops"] == 30
    assert result["chart_data"]["links"] == [] and result["chart_data"]["nodes"] == []
    assert result["n_points"] == 0  # the frontend shows the empty state, not a blank chart
