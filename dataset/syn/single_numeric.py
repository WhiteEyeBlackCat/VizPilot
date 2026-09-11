#| 欄位      | 型態      | 目的                                  |
#| ------- | ------- | ----------------------------------- |
#| `value` | numeric | 唯一欄位；雙峰分佈 (兩個常態混合) → 只能畫 histogram |

import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)
n = 2000

left = np.random.normal(loc=40, scale=6, size=int(n * 0.6))
right = np.random.normal(loc=75, scale=8, size=n - int(n * 0.6))
value = np.round(np.concatenate([left, right]), 2)
np.random.shuffle(value)

df = pd.DataFrame({"value": value})

print(df.head())
print(df.shape)
print(df.dtypes)
print(df.describe())

output_path = Path(__file__).resolve().parent.parent / "single_numeric.csv"
df.to_csv(output_path, index=False)
print(f"Saved to: {output_path}")
