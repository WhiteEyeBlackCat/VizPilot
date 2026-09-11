#| 欄位            | 型態          | 目的             |
#| ------------- | ----------- | -------------- |
#| `timestamp`   | datetime    | 時間軸            |
#| `station`     | categorical | 測站             |
#| `pm25`        | numeric     | 主要 time-series |
#| `temperature` | numeric     | 連續變數           |
#| `humidity`    | numeric     | 連續變數           |
#| `wind_speed`  | numeric     | 連續變數           |


import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

n = 30*24
timestamps = pd.date_range(start="2025-01-01", periods=n, freq="h")

hours = np.arange(n)
daily_cycle = np.sin(2*np.pi*hours/24)


temperature = 20 + 5 * daily_cycle + np.random.normal(loc=0,scale=1.5,size=n)
humidity = 70 - 1.2 * (temperature - 20) + np.random.normal(loc=0,scale=5,size=n)
humidity = np.clip(humidity, 30, 100)
wind_speed = np.random.gamma(shape=2, scale=1.5, size=n) #非負、右尾較長
pm25 = 30 + 8 * daily_cycle + 0.15 * humidity - 2.5 * wind_speed + np.random.normal(loc=0, scale=5, size=n)



temperature = np.round(temperature, 2)
humidity = np.round(humidity, 2)
wind_speed = np.round(wind_speed, 2)
pm25 = np.round(pm25, 2)

station_base = ["A", "B", "C"]
station = np.random.choice(station_base, size=n)
station_effect = {"A":0, "B":5, "C":12}
station_offset = np.array([station_effect[s] for s in station])
pm25 += station_offset
pm25 = np.clip(pm25, 0, None)

df = pd.DataFrame({
    "timestamp": timestamps,
    "station": station,
    "temperature": temperature,
    "humidity": humidity,
    "wind_speed": wind_speed,
    "pm25": pm25
})

print(df.head())
print(df.shape)
print(df.dtypes)
print(df.describe())

output_path = Path(__file__).resolve().parent.parent/"air_quality.csv"
df.to_csv(output_path, index=False)