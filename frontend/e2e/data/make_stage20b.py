"""Deterministic, compact input for the stage-20b render fixtures
(composition / flow) and for backend/tests/test_fixture_regression.py.
Standard library + polars only; fixed seed — re-running rewrites a
byte-identical file.

  backend/.venv/bin/python frontend/e2e/data/make_stage20b.py

flows.csv  one row per shipment:
  region   4 bars of the composition (North / South / East / West)
  channel  3 parts of each bar (online / retail / partner); the mix is
           deliberately different per region, so the interaction share of
           region x channel on revenue clears COMPOSITION_INTERACTION_MIN
  revenue  non-negative magnitude (the whole each bar sums to)
  from_hub / to_hub  a declared flow pair over one stem (five hubs, with
           some rows staying in the same hub: the self-loops a sankey drops)
"""

import random
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent

REGIONS = ["North", "South", "East", "West"]
CHANNELS = ["online", "retail", "partner"]
HUBS = ["alpha", "bravo", "charlie", "delta", "echo"]
# share of each region's rows per channel: the mix shifts region by region
MIX = {
    "North": (0.70, 0.20, 0.10),
    "South": (0.15, 0.70, 0.15),
    "East": (0.30, 0.20, 0.50),
    "West": (0.34, 0.33, 0.33),
}
# mean revenue per (region, channel): not additive across the two factors
REVENUE = {
    ("North", "online"): 900, ("North", "retail"): 300, ("North", "partner"): 150,
    ("South", "online"): 200, ("South", "retail"): 950, ("South", "partner"): 180,
    ("East", "online"): 250, ("East", "retail"): 220, ("East", "partner"): 1000,
    ("West", "online"): 400, ("West", "retail"): 420, ("West", "partner"): 410,
}


def flows() -> pl.DataFrame:
    rng = random.Random(2020)
    rows = []
    for i in range(600):
        region = REGIONS[i % 4]
        channel = rng.choices(CHANNELS, weights=MIX[region])[0]
        # a flow that mostly moves one hub forward, sometimes stays put
        src = HUBS[i % 5]
        dst = src if i % 7 == 0 else HUBS[(i % 5 + 1 + (i // 5) % 4) % 5]
        rows.append(
            {
                "region": region,
                "channel": channel,
                "revenue": round(max(0.0, rng.gauss(REVENUE[(region, channel)], 60)), 2),
                "units": rng.randint(1, 40),
                "from_hub": src,
                "to_hub": dst,
            }
        )
    return pl.DataFrame(rows)


if __name__ == "__main__":
    frame = flows()
    frame.write_csv(HERE / "flows.csv")
    print(f"flows.csv: {frame.shape}")
