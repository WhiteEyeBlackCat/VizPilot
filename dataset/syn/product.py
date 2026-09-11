#| 欄位              | 類型          |
#| --------------- | ----------- |
#| `product_id`    | integer     |
#| `category`      | categorical |
#| `brand`         | categorical |
#| `price`         | numeric     |
#| `rating`        | numeric     |
#| `stock`         | integer     |
#| `monthly_sales` | integer     |

import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)
n = 1000

category_base = ["Electronics", "Clothing", "Food", "Home"]
brand_base = ["Alpha", "Beta", "Gamma", "Delta"]

# 各 category 的價格區間不同，讓 group-by 圖表有明顯差異
price_range = {
    "Electronics": (500, 5000),
    "Clothing": (100, 1500),
    "Food": (20, 300),
    "Home": (200, 3000),
}

product_id = np.arange(1, n + 1)
category = np.random.choice(category_base, size=n, p=[0.3, 0.3, 0.2, 0.2])
brand = np.random.choice(brand_base, size=n)

price = np.array([np.random.uniform(*price_range[c]) for c in category])
price = np.round(price, 2)

# rating 偏高 (多數落在 3.5~4.8)，brand 有小幅差異
brand_effect = {"Alpha": 0.3, "Beta": 0.0, "Gamma": -0.2, "Delta": 0.1}
rating = 4.0 + np.array([brand_effect[b] for b in brand]) + np.random.normal(0, 0.5, size=n)
rating = np.round(np.clip(rating, 1, 5), 1)

stock = np.random.randint(0, 500, size=n)

# 銷量與價格負相關、與 rating 正相關
monthly_sales = 200 - 0.03 * price + 40 * (rating - 4) + np.random.normal(0, 30, size=n)
monthly_sales = np.clip(np.round(monthly_sales), 0, None).astype(int)

df = pd.DataFrame({
    "product_id": product_id,
    "category": category,
    "brand": brand,
    "price": price,
    "rating": rating,
    "stock": stock,
    "monthly_sales": monthly_sales,
})

print(df.head())
print(df.shape)
print(df.dtypes)
print(df.describe())

output_path = Path(__file__).resolve().parent.parent / "products.csv"
df.to_csv(output_path, index=False)
print(f"Saved to: {output_path}")
