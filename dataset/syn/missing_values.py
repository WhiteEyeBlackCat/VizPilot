#| 欄位            | 型態          | 缺失比例 | 目的                  |
#| ------------- | ----------- | ---- | ------------------- |
#| `id`          | integer     | 0%   | 完整欄位                |
#| `age`         | integer     | 10%  | 少量缺失 → 應仍可畫圖         |
#| `income`      | numeric     | 30%  | 中度缺失                |
#| `score`       | numeric     | 60%  | 重度缺失 → 應被降權或警告      |
#| `gender`      | categorical | 15%  | categorical 缺失       |
#| `city`        | categorical | 40%  | categorical 重度缺失     |
#| `last_login`  | datetime    | 25%  | datetime 缺失          |
#| `notes`       | string      | 95%  | 幾乎全空                |
#| `empty_col`   | -           | 100% | 全空欄位 → 不應出現在任何圖表    |

import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)
n = 800


def with_missing(arr, ratio):
    arr = pd.Series(arr, dtype="object")
    mask = np.random.rand(n) < ratio
    arr[mask] = np.nan
    return arr


id_ = np.arange(1, n + 1)
age = with_missing(np.random.randint(18, 70, size=n), 0.10)
income = with_missing(np.round(np.random.lognormal(10.5, 0.5, size=n), 0), 0.30)
score = with_missing(np.round(np.random.normal(70, 12, size=n), 1), 0.60)
gender = with_missing(np.random.choice(["M", "F", "Other"], size=n, p=[0.48, 0.48, 0.04]), 0.15)
city = with_missing(np.random.choice(["Taipei", "Taichung", "Kaohsiung", "Tainan", "Hsinchu"], size=n), 0.40)
last_login = with_missing(
    (pd.to_datetime("2025-01-01") + pd.to_timedelta(np.random.randint(0, 365, size=n), unit="D")).strftime("%Y-%m-%d"),
    0.25,
)
notes = with_missing(np.random.choice(["VIP", "churn risk", "follow up", "new"], size=n), 0.95)
empty_col = [np.nan] * n

df = pd.DataFrame({
    "id": id_,
    "age": age,
    "income": income,
    "score": score,
    "gender": gender,
    "city": city,
    "last_login": last_login,
    "notes": notes,
    "empty_col": empty_col,
})

print(df.head(10))
print(df.shape)
print(df.dtypes)
print("missing ratio:")
print(df.isna().mean().round(3))

output_path = Path(__file__).resolve().parent.parent / "missing_values.csv"
df.to_csv(output_path, index=False)
print(f"Saved to: {output_path}")
