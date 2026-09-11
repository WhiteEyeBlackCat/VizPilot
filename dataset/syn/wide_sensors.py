#| 欄位                    | 型態          | 目的                         |
#| --------------------- | ----------- | -------------------------- |
#| `timestamp`           | datetime    | 每分鐘一筆，共 2000 筆              |
#| `machine_id`          | categorical | 4 台機器                      |
#| `status`              | categorical | normal / warning / fault    |
#| `sensor_01`~`sensor_60`| numeric    | 60 個 sensor，分成幾種訊號型態：       |
#|                       |             |   01-15 正弦波 (不同週期/相位)        |
#|                       |             |   16-30 隨機漫步 (trend)          |
#|                       |             |   31-45 白噪音 (無結構)             |
#|                       |             |   46-55 與 sensor_01 高度相關 (冗餘) |
#|                       |             |   56-60 幾乎常數 (低變異)           |
# 總共 63 欄，測試 wide table 的欄位挑選與 UI 呈現

import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)
n = 2000

timestamp = pd.date_range(start="2025-03-01 00:00", periods=n, freq="min")
machine_id = np.random.choice(["M1", "M2", "M3", "M4"], size=n)
status = np.random.choice(["normal", "warning", "fault"], size=n, p=[0.9, 0.08, 0.02])

t = np.arange(n)
sensors = {}

for i in range(1, 16):
    period = np.random.uniform(60, 600)
    phase = np.random.uniform(0, 2 * np.pi)
    amp = np.random.uniform(5, 50)
    sensors[f"sensor_{i:02d}"] = 100 + amp * np.sin(2 * np.pi * t / period + phase) + np.random.normal(0, 1, size=n)

for i in range(16, 31):
    sensors[f"sensor_{i:02d}"] = 50 + np.cumsum(np.random.normal(0, 0.5, size=n))

for i in range(31, 46):
    sensors[f"sensor_{i:02d}"] = np.random.normal(np.random.uniform(0, 100), np.random.uniform(1, 20), size=n)

for i in range(46, 56):
    sensors[f"sensor_{i:02d}"] = sensors["sensor_01"] * np.random.uniform(0.5, 2) + np.random.normal(0, 2, size=n)

for i in range(56, 61):
    sensors[f"sensor_{i:02d}"] = np.random.uniform(0, 10) + np.random.normal(0, 0.01, size=n)

df = pd.DataFrame({"timestamp": timestamp, "machine_id": machine_id, "status": status})
for name, vals in sensors.items():
    df[name] = np.round(vals, 3)

print(df.head())
print(df.shape)
print(df.dtypes.value_counts())

output_path = Path(__file__).resolve().parent.parent / "wide_sensors.csv"
df.to_csv(output_path, index=False)
print(f"Saved to: {output_path}")
