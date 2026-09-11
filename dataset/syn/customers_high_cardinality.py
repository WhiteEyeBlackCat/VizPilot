#| 欄位              | 型態          | 目的                          |
#| --------------- | ----------- | --------------------------- |
#| `customer_id`   | string      | 全部唯一 (5000 種)               |
#| `email`         | string      | 全部唯一，看起來像 categorical 但不能畫圖 |
#| `full_name`     | string      | 幾乎唯一                        |
#| `city`          | categorical | 200 個城市，超過 group-by 上限 (20)  |
#| `zip_code`      | string      | 數字外觀的高 cardinality 字串        |
#| `segment`       | categorical | 只有 4 種，是唯一適合 group-by 的欄位    |
#| `signup_date`   | datetime    | 時間軸                         |
#| `num_orders`    | integer     | 數值                          |
#| `lifetime_value`| numeric     | 數值 (右偏)                     |

import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)
n = 5000

first_names = ["Wei", "Ming", "Yu", "Jia", "Hao", "Ling", "Chen", "Xin", "Ya", "Kai",
               "Anna", "Ben", "Cathy", "David", "Emma", "Frank", "Grace", "Henry", "Ivy", "Jack"]
last_names = ["Chen", "Lin", "Wang", "Huang", "Zhang", "Li", "Wu", "Liu", "Tsai", "Yang",
              "Smith", "Brown", "Lee", "Kim", "Park", "Garcia", "Miller", "Davis", "Lopez", "Wilson"]
domains = ["gmail.com", "yahoo.com", "outlook.com", "example.org", "mail.tw"]
segment_base = ["Bronze", "Silver", "Gold", "Platinum"]

customer_id = [f"C{i:06d}" for i in range(1, n + 1)]
first = np.random.choice(first_names, size=n)
last = np.random.choice(last_names, size=n)
full_name = [f"{f} {l}" for f, l in zip(first, last)]
email = [f"{f.lower()}.{l.lower()}{i}@{np.random.choice(domains)}"
         for f, l, i in zip(first, last, range(1, n + 1))]

# 200 個城市，長尾分佈 (少數大城市佔多數人)
cities = [f"City_{i:03d}" for i in range(1, 201)]
city_weights = np.random.pareto(1.5, size=200) + 1
city_weights /= city_weights.sum()
city = np.random.choice(cities, size=n, p=city_weights)

zip_code = [f"{np.random.randint(100, 999)}{np.random.randint(10, 99)}" for _ in range(n)]

segment = np.random.choice(segment_base, size=n, p=[0.5, 0.3, 0.15, 0.05])
segment_ltv = {"Bronze": 500, "Silver": 2000, "Gold": 6000, "Platinum": 20000}

signup_date = pd.to_datetime("2022-01-01") + pd.to_timedelta(np.random.randint(0, 3 * 365, size=n), unit="D")

num_orders = np.random.poisson(lam=[{"Bronze": 2, "Silver": 6, "Gold": 15, "Platinum": 40}[s] for s in segment])
lifetime_value = np.array([segment_ltv[s] for s in segment]) * np.random.lognormal(0, 0.5, size=n)
lifetime_value = np.round(lifetime_value, 2)

df = pd.DataFrame({
    "customer_id": customer_id,
    "email": email,
    "full_name": full_name,
    "city": city,
    "zip_code": zip_code,
    "segment": segment,
    "signup_date": signup_date,
    "num_orders": num_orders,
    "lifetime_value": lifetime_value,
})

print(df.head())
print(df.shape)
print(df.dtypes)
print(df.nunique())

output_path = Path(__file__).resolve().parent.parent / "customers_high_cardinality.csv"
df.to_csv(output_path, index=False)
print(f"Saved to: {output_path}")
