# TODO:
# 1. order_id       1~1000, int
# 2. order_date     2025-01-01 ～ 2025-12-31
# 3. region         ["North", "Central", "South", "East"]
# 4. category       ["Electronics", "Clothing", "Food", "Home"]
# 5. unit_price     50~5000, float
# 6. quantity       1~10, int
# 7. discount       [0, 0.05, 0.1, 0.2]
# 8. sales


import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

n = 1000

dates_range = pd.date_range(start="2025-01-01", end="2025-12-31")
region_base = ["North", "Central", "South", "East"]
category_base = ["Electronics", "Clothing", "Food", "Home"]
discount_base = [0, 0.05, 0.1, 0.2]

order_id = np.arange(1, n+1)
order_date = np.random.choice(dates_range, size=n)
region = np.random.choice(region_base, size=n)
category = np.random.choice(category_base, size=n)
unit_price = np.round(np.random.uniform(50,5000, size=n), 2)
quantity = np.random.randint(1,11,size=n)
discount = np.random.choice(discount_base, size=n)
sales = np.round(unit_price * quantity * (1-discount), 2)


df = pd.DataFrame({
    "order_id": order_id,
    "order_date": order_date,
    "region": region,
    "category": category,
    "unit_price": unit_price,
    "quantity": quantity,
    "discount": discount,
    "sales": sales,
})

print(df.head())

print(df.shape)

print(df.dtypes)

print(df.describe())


output_path = Path(__file__).resolve().parent.parent / "sales_basic.csv"
df.to_csv(output_path, index=False)
print(f"Saved to: {output_path}")