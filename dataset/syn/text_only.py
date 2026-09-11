#| 欄位         | 型態     | 目的                              |
#| ---------- | ------ | ------------------------------- |
#| `doc_id`   | string | "DOC-0001" 全部唯一                 |
#| `title`    | string | 短句，幾乎全部唯一                       |
#| `content`  | string | 長段落，全部唯一                        |
#| `author`   | string | 300 位不同作者 → cardinality 過高不能 group |
#| `tags`     | string | 逗號分隔的多值字串                       |
# 沒有 numeric、datetime，也沒有低 cardinality 的 categorical
# → 系統應回報「無適合圖表」而不是硬畫

import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)
n = 300

subjects = ["The system", "Our team", "A user", "The pipeline", "The model", "This report",
            "The dashboard", "The dataset", "The service", "The customer"]
verbs = ["analyzes", "describes", "handles", "improves", "reviews", "monitors",
         "summarizes", "validates", "processes", "explains"]
objects = ["quarterly results", "sensor readings", "the deployment plan", "user feedback",
           "the data schema", "a new feature", "performance issues", "the onboarding flow",
           "edge cases", "the migration"]
fillers = ["in detail", "without delay", "for the first time", "as expected", "with some caveats",
           "across all regions", "after the incident", "before release", "under load", "at scale"]

def sentence():
    return f"{np.random.choice(subjects)} {np.random.choice(verbs)} {np.random.choice(objects)} {np.random.choice(fillers)}."

doc_id = [f"DOC-{i:04d}" for i in range(1, n + 1)]
title = [sentence().rstrip(".") for _ in range(n)]
content = [" ".join(sentence() for _ in range(np.random.randint(3, 8))) for _ in range(n)]
author = [f"author_{i:03d}" for i in np.random.permutation(n)]
tag_pool = ["ops", "ml", "ui", "backend", "infra", "docs", "bug", "feature", "research", "meeting"]
tags = [",".join(np.random.choice(tag_pool, size=np.random.randint(1, 4), replace=False)) for _ in range(n)]

df = pd.DataFrame({
    "doc_id": doc_id,
    "title": title,
    "content": content,
    "author": author,
    "tags": tags,
})

print(df.head())
print(df.shape)
print(df.dtypes)
print(df.nunique())

output_path = Path(__file__).resolve().parent.parent / "text_only.csv"
df.to_csv(output_path, index=False)
print(f"Saved to: {output_path}")
