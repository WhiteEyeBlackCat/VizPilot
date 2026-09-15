# Pure-noise benchmark table (stage 17.3): every column is independent of
# every other, so the correct number of insights is ZERO. Used to check that
# the LLM workflow does not mint findings out of multiple-comparison noise
# (max KS over group pairs, correlation spread over groups, ...).
#
#| 欄位      | 型態        | 目的                                          |
#| -------- | ----------- | --------------------------------------------- |
#| ts       | datetime    | 每日一列，600 天                                 |
#| segment  | categorical | 5 群，與任何數值欄無關                            |
#| flag     | boolean     | 2 群，與任何數值欄無關                            |
#| region   | categorical | 4 群，與任何數值欄無關                            |
#| n0..n7   | numeric     | 獨立高斯雜訊                                     |
#| u0..u1   | numeric     | 獨立均勻雜訊（不同尺度）                            |

import random
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

rng = random.Random(1700)
n = 600

rows = {
    "ts": [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n)],
    "segment": [rng.choice(["a", "b", "c", "d", "e"]) for _ in range(n)],
    "flag": [rng.random() < 0.5 for _ in range(n)],
    "region": [rng.choice(["north", "south", "east", "west"]) for _ in range(n)],
}
for k in range(8):
    rows[f"n{k}"] = [rng.gauss(0, 1) for _ in range(n)]
rows["u0"] = [rng.uniform(0, 100) for _ in range(n)]
rows["u1"] = [rng.uniform(-5, 5) for _ in range(n)]

df = pl.DataFrame(rows)
print(df.shape)

output_path = Path(__file__).resolve().parent.parent / "noise.csv"
df.write_csv(output_path)
print(f"Saved to: {output_path}")
