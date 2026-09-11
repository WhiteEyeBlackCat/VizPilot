#| 欄位      | 型態          | 目的                              |
#| ------- | ----------- | ------------------------------- |
#| `name`  | string      |                                 |
#| `dept`  | categorical | 2 種                             |
#| `age`   | integer     |                                 |
#| `score` | numeric     |                                 |
# 只有 4 筆資料 → 統計量不可靠，系統應避免給出過度自信的 insight

import pandas as pd
from pathlib import Path

df = pd.DataFrame({
    "name": ["Alice", "Bob", "Carol", "Dave"],
    "dept": ["Sales", "Engineering", "Sales", "Engineering"],
    "age": [29, 41, 35, 52],
    "score": [88.5, 72.0, 91.3, 65.8],
})

print(df)
print(df.shape)
print(df.dtypes)

output_path = Path(__file__).resolve().parent.parent / "tiny_dataset.csv"
df.to_csv(output_path, index=False)
print(f"Saved to: {output_path}")
