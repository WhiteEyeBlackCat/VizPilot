# Bike-sharing-like hourly table (stage 14 fixture): the shape of the UCI
# hour.csv that exposed the temp / atemp duplication. Written with polars +
# the standard library so it runs in backend/.venv (no numpy / pandas there).
#
#| 欄位        | 型態        | 目的                                            |
#| ---------- | ----------- | ----------------------------------------------- |
#| dteday     | datetime    | 時間軸（每小時一列，90 天）                        |
#| season     | categorical | 1..3                                            |
#| mnth       | categorical | 月份（只出現 3 個月）                             |
#| hr         | numeric     | 0..23，用量的主要驅動（真實訊號）                   |
#| weathersit | categorical | 1..3                                            |
#| temp       | numeric     | 正規化氣溫，含日夜與季節週期                        |
#| atemp      | numeric     | 體感溫度 = 0.9·temp + 小雜訊 → 與 temp 近似複製      |
#| hum        | numeric     | 濕度，與 temp 負相關                               |
#| windspeed  | numeric     | 雜訊                                             |
#| casual     | numeric     | 臨時用戶，隨 hr 與 temp 變動                        |
#| registered | numeric     | 註冊用戶，通勤尖峰                                 |
#| cnt        | numeric     | casual + registered（衍生欄，stage 13 應偵測）      |

import math
import random
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

rng = random.Random(14)

n_days = 90
n = n_days * 24
start = datetime(2025, 3, 1)


def poisson(lam: float) -> int:
    # Knuth; lam stays small enough (< 400) for this to be exact and fast
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


rows = []
for i in range(n):
    day, hr = divmod(i, 24)
    ts = start + timedelta(hours=i)
    mnth = ts.month
    season = 1 if mnth <= 3 else 2 if mnth <= 5 else 3
    weathersit = rng.choices([1, 2, 3], weights=[0.65, 0.28, 0.07])[0]
    diurnal = math.sin(2 * math.pi * (hr - 6) / 24)
    temp = 0.35 + 0.25 * day / n_days + 0.12 * diurnal + rng.gauss(0, 0.03)
    temp = min(max(temp, 0.02), 1.0)
    atemp = 0.9 * temp + rng.gauss(0, 0.012)  # apparent temperature: rho ~0.99 with temp
    hum = min(max(0.7 - 0.4 * (temp - 0.5) + rng.gauss(0, 0.08), 0.1), 1.0)
    windspeed = rng.gammavariate(2.0, 0.08)
    commute = math.exp(-((hr - 8) ** 2) / 4) + math.exp(-((hr - 17.5) ** 2) / 5)
    leisure = math.exp(-((hr - 14) ** 2) / 18)
    penalty = {1: 1.0, 2: 0.8, 3: 0.4}[weathersit]
    registered = poisson(20 + 180 * commute * penalty * (0.6 + 0.8 * temp))
    casual = poisson(3 + 60 * leisure * penalty * (0.3 + 1.4 * temp))
    rows.append(
        {
            "instant": i + 1,
            "dteday": ts,
            "season": season,
            "mnth": mnth,
            "hr": hr,
            "weathersit": weathersit,
            "temp": round(temp, 4),
            "atemp": round(atemp, 4),
            "hum": round(hum, 4),
            "windspeed": round(windspeed, 4),
            "casual": casual,
            "registered": registered,
            "cnt": casual + registered,
        }
    )

df = pl.DataFrame(rows)
print(df.head())
print(df.shape)
print(df.select(pl.corr("temp", "atemp", method="spearman").alias("rho_temp_atemp"), pl.corr("hr", "cnt").alias("r_hr_cnt")))

output_path = Path(__file__).resolve().parent.parent / "hour_like.csv"
df.write_csv(output_path)
print(f"Saved to: {output_path}")
