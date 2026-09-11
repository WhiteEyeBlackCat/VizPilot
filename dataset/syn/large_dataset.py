#| 欄位           | 型態          | 目的                          |
#| ------------ | ----------- | --------------------------- |
#| `event_id`   | integer     | 1 ~ 1,000,000                |
#| `event_time` | datetime    | 一年內隨機，秒級                   |
#| `user_id`    | integer     | 100,000 個 user (高 cardinality) |
#| `country`    | categorical | 12 國，不均勻                    |
#| `device`     | categorical | 3 種                         |
#| `category`   | categorical | 8 種                         |
#| `amount`     | numeric     | log-normal                   |
#| `latency_ms` | numeric     | gamma 右偏                    |
#| `is_success` | boolean     | ~95% True                    |
# 100 萬筆，輸出 parquet；用來測試 profile / render 的效能與降採樣

import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)
n = 1_000_000

event_id = np.arange(1, n + 1, dtype=np.int64)
event_time = pd.to_datetime("2025-01-01") + pd.to_timedelta(np.random.randint(0, 365 * 24 * 3600, size=n), unit="s")
event_time = event_time.sort_values()
user_id = np.random.randint(1, 100_001, size=n)

country_base = ["TW", "US", "JP", "KR", "DE", "FR", "GB", "IN", "BR", "AU", "CA", "SG"]
country_p = np.array([30, 20, 12, 8, 6, 5, 5, 5, 3, 3, 2, 1], dtype=float)
country = np.random.choice(country_base, size=n, p=country_p / country_p.sum())
device = np.random.choice(["mobile", "desktop", "tablet"], size=n, p=[0.6, 0.3, 0.1])
category = np.random.choice([f"cat_{i}" for i in range(1, 9)], size=n)

amount = np.round(np.random.lognormal(mean=4, sigma=1, size=n), 2)
device_latency = {"mobile": 1.3, "desktop": 1.0, "tablet": 1.15}
latency_ms = np.random.gamma(shape=2, scale=80, size=n) * np.array([device_latency[d] for d in device])
latency_ms = np.round(latency_ms, 1)
is_success = np.random.rand(n) < 0.95

df = pd.DataFrame({
    "event_id": event_id,
    "event_time": event_time,
    "user_id": user_id,
    "country": country,
    "device": device,
    "category": category,
    "amount": amount,
    "latency_ms": latency_ms,
    "is_success": is_success,
})

print(df.head())
print(df.shape)
print(df.dtypes)
print(df.memory_usage(deep=True).sum() / 1e6, "MB in memory")

output_path = Path(__file__).resolve().parent.parent / "large_dataset.parquet"
df.to_parquet(output_path, index=False, engine="pyarrow", compression="snappy")
print(f"Saved to: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
