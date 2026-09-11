#| 欄位              | 表面型態   | 實際內容                                  | 目的                       |
#| --------------- | ------ | ------------------------------------- | ------------------------ |
#| `id`            | string | "1", "2", ... 純數字字串                    | 應被推斷為 integer            |
#| `price`         | string | "$1,234.50", "1234.5", "N/A", " 99 "   | 帶貨幣符號/千分位/空白的數值          |
#| `quantity`      | string | "12", "twelve", "", "3.0"             | 數字混文字，不應被當成 numeric       |
#| `pct`           | string | "45%", "45.0%", "0.45"                | 百分比字串                    |
#| `is_active`     | string | "yes"/"no"/"True"/"False"/"1"/"0"     | 各種 boolean 寫法             |
#| `order_date`    | string | 混合 "2025-01-05", "05/01/2025", "Jan 5 2025" | 多種日期格式               |
#| `mixed_num`     | string | 大多數是數字，少數 "unknown"                   | 幾乎是 numeric 但有髒值          |
#| `int_as_float`  | float  | 1.0, 2.0, 3.0                         | 應被視為 integer / categorical |
#| `code`          | string | "001", "002" 前導零                       | 不應丟掉前導零變成 int             |
#| `category_num`  | integer| 1~5 的整數                               | 數字但語意是 categorical        |

import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)
n = 500

id_ = [str(i) for i in range(1, n + 1)]

raw_price = np.random.uniform(10, 5000, size=n)
price = []
for p in raw_price:
    style = np.random.choice(["currency", "plain", "na", "space"], p=[0.5, 0.35, 0.05, 0.10])
    if style == "currency":
        price.append(f"${p:,.2f}")
    elif style == "plain":
        price.append(f"{p:.2f}")
    elif style == "na":
        price.append("N/A")
    else:
        price.append(f" {p:.0f} ")

words = {1: "one", 2: "two", 3: "three", 5: "five", 10: "ten", 12: "twelve"}
quantity = []
for _ in range(n):
    q = int(np.random.randint(1, 15))
    style = np.random.choice(["int", "word", "empty", "float"], p=[0.7, 0.1, 0.1, 0.1])
    if style == "int":
        quantity.append(str(q))
    elif style == "word":
        quantity.append(words.get(q, str(q)))
    elif style == "empty":
        quantity.append("")
    else:
        quantity.append(f"{q}.0")

raw_pct = np.random.uniform(0, 100, size=n)
pct = [np.random.choice([f"{v:.0f}%", f"{v:.1f}%", f"{v/100:.2f}"]) for v in raw_pct]

is_active = np.random.choice(["yes", "no", "True", "False", "1", "0", "Y", "N"], size=n)

base_dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(np.random.randint(0, 365, size=n), unit="D")
order_date = []
for d in base_dates:
    style = np.random.choice(["iso", "dmy", "text", "slash"], p=[0.5, 0.2, 0.15, 0.15])
    if style == "iso":
        order_date.append(d.strftime("%Y-%m-%d"))
    elif style == "dmy":
        order_date.append(d.strftime("%d/%m/%Y"))
    elif style == "text":
        order_date.append(d.strftime("%b %d %Y"))
    else:
        order_date.append(d.strftime("%Y/%m/%d"))

mixed_num = [f"{v:.1f}" if np.random.rand() > 0.03 else "unknown" for v in np.random.normal(50, 10, size=n)]
int_as_float = np.random.randint(1, 6, size=n).astype(float)
code = [f"{i:03d}" for i in np.random.randint(1, 50, size=n)]
category_num = np.random.randint(1, 6, size=n)

df = pd.DataFrame({
    "id": id_,
    "price": price,
    "quantity": quantity,
    "pct": pct,
    "is_active": is_active,
    "order_date": order_date,
    "mixed_num": mixed_num,
    "int_as_float": int_as_float,
    "code": code,
    "category_num": category_num,
})

print(df.head(10))
print(df.shape)
print(df.dtypes)

output_path = Path(__file__).resolve().parent.parent / "dirty_types.csv"
df.to_csv(output_path, index=False)
print(f"Saved to: {output_path}")
