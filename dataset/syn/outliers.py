#| 欄位            | 型態          | 目的                                   |
#| ------------- | ----------- | ------------------------------------ |
#| `id`          | integer     |                                      |
#| `group`       | categorical | 3 組，供 box plot                        |
#| `value`       | numeric     | 常態 + 2% 極端值 (±10σ)                    |
#| `revenue`     | numeric     | log-normal 右偏 + 少數超大值                 |
#| `temperature` | numeric     | 正常 20~30，混入 sentinel -999 / 9999      |
#| `duration_ms` | integer     | 大多 <1000，少數 >100000 (長尾)              |
#| `clean`       | numeric     | 完全沒有 outlier 的對照組                     |

import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)
n = 1000

id_ = np.arange(1, n + 1)
group = np.random.choice(["A", "B", "C"], size=n)
group_shift = {"A": 0, "B": 3, "C": -2}

value = np.random.normal(50, 5, size=n) + np.array([group_shift[g] for g in group])
outlier_mask = np.random.rand(n) < 0.02
value[outlier_mask] += np.random.choice([-1, 1], size=outlier_mask.sum()) * np.random.uniform(40, 80, size=outlier_mask.sum())
value = np.round(value, 2)

revenue = np.random.lognormal(mean=7, sigma=0.6, size=n)
spike_mask = np.random.rand(n) < 0.01
revenue[spike_mask] *= np.random.uniform(30, 100, size=spike_mask.sum())
revenue = np.round(revenue, 2)

temperature = np.random.normal(25, 2, size=n)
sentinel_mask = np.random.rand(n) < 0.015
temperature[sentinel_mask] = np.random.choice([-999, 9999], size=sentinel_mask.sum())
temperature = np.round(temperature, 1)

duration_ms = np.random.gamma(shape=2, scale=150, size=n)
slow_mask = np.random.rand(n) < 0.03
duration_ms[slow_mask] = np.random.uniform(100_000, 500_000, size=slow_mask.sum())
duration_ms = duration_ms.astype(int)

clean = np.round(np.random.normal(100, 10, size=n), 2)

df = pd.DataFrame({
    "id": id_,
    "group": group,
    "value": value,
    "revenue": revenue,
    "temperature": temperature,
    "duration_ms": duration_ms,
    "clean": clean,
})

print(df.head())
print(df.shape)
print(df.dtypes)
print(df.describe())

output_path = Path(__file__).resolve().parent.parent / "outliers.csv"
df.to_csv(output_path, index=False)
print(f"Saved to: {output_path}")
