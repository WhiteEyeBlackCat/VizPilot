"""Deterministic, compact inputs for the stage-19.1 render fixtures
(frontend/src/charts/__fixtures__, recorded by e2e/capture-fixtures.mjs) and
for backend/tests/test_fixture_regression.py. Standard library + polars only;
fixed seeds — re-running rewrites byte-identical files.

  backend/.venv/bin/python frontend/e2e/data/make_stage191.py

spread_groups.csv  same mean / different dispersion: `value` by nominal `group`
                   (a / b / c, sd 2 / 6 / 12) and `amount` by ordered `level`
                   (low < medium < high < very high, sd 1 / 3 / 6 / 12);
                   raincloud, ridgeline, box, natural-order bar / line
cycles.csv         hour-of-day and month cycles: `rides` peaks at 8 and 18 h,
                   `temp` follows the month; polar_line, rose (sum / count),
                   line over a cycle
dense_xy.csv       2000 correlated (x, y) points: hexbin
"""

import math
import random
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent


def spread_groups() -> pl.DataFrame:
    rng = random.Random(1911)
    groups = ["a", "b", "c"]
    levels = ["low", "medium", "high", "very high"]
    sd_group = {"a": 2.0, "b": 6.0, "c": 12.0}
    sd_level = {"low": 1.0, "medium": 3.0, "high": 6.0, "very high": 12.0}
    rows = []
    for i in range(240):  # 3 and 4 are coprime: 80 rows per group, 60 per level
        g, lv = groups[i % 3], levels[i % 4]
        rows.append(
            {
                "group": g,
                "level": lv,
                "value": round(50 + rng.gauss(0, sd_group[g]), 3),
                "amount": round(20 + rng.gauss(0, sd_level[lv]), 3),
            }
        )
    return pl.DataFrame(rows)


def cycles() -> pl.DataFrame:
    rng = random.Random(1912)
    rows = []
    for i in range(480):  # 20 days of hours; 40 rows per month
        hr = i % 24
        mnth = (i // 40) % 12 + 1
        commute = math.exp(-((hr - 8) ** 2) / 3) + math.exp(-((hr - 18) ** 2) / 4)
        rides = int(round(10 + 90 * commute + rng.uniform(0, 8)))
        temp = round(15 + 8 * math.sin(2 * math.pi * (mnth - 4) / 12) + rng.gauss(0, 1.5), 2)
        rows.append({"hr": hr, "mnth": mnth, "rides": rides, "temp": temp})
    return pl.DataFrame(rows)


def dense_xy() -> pl.DataFrame:
    rng = random.Random(1913)
    xs, ys = [], []
    for _ in range(2000):
        x = rng.gauss(0, 1)
        xs.append(round(x, 4))
        ys.append(round(0.8 * x + rng.gauss(0, 0.6), 4))
    return pl.DataFrame({"x": xs, "y": ys})


if __name__ == "__main__":
    for name, frame in (("spread_groups.csv", spread_groups()), ("cycles.csv", cycles()), ("dense_xy.csv", dense_xy())):
        frame.write_csv(HERE / name)
        print(f"{name}: {frame.shape}")
