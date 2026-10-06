# Probe-pass benchmark table (stage 20c): device changes the SHAPE of
# latency, not its mean — group means are equal so group_difference sees
# ~0 effect; only a distribution_difference probe (always uncovered by
# evidence L1/L2 — see llm/coverage.py check_coverage, "no KS statistic in
# the evidence tables: always a probe") can find it. Deliberately minimal
# (no time column, no distractor categorical) so the planted signal is the
# single standout item in the evidence section shown to the LLM. Built to
# elicit and document a real-model (qwen2.5:14b) probe pass end to end.
#
#| 欄位        | 型態        | 目的                                                        |
#| ---------- | ----------- | ----------------------------------------------------------- |
#| device     | categorical | wifi / cellular，平均延遲相同，分布形狀不同（wifi 雙峰）           |
#| latency_ms | numeric     | 目標欄：wifi 雙峰（~20ms / ~80ms），cellular 在同範圍內均勻分布    |
#| n0..n1     | numeric     | 獨立高斯雜訊                                                  |

import random
from pathlib import Path

import polars as pl

rng = random.Random(2026)
n = 600

device = [rng.choice(["wifi", "cellular"]) for _ in range(n)]
latency_ms = []
for d in device:
    if d == "wifi":
        # two clean, separated modes (clear vs congested channel); they
        # average to ~50, same as cellular below, and the global histogram
        # shows two separated peaks (layer-2 multimodal_signal = true)
        latency_ms.append(rng.gauss(20, 3) if rng.random() < 0.5 else rng.gauss(80, 3))
    else:
        # cellular: flat between the same two bounds, same mean (~50), no
        # separated peaks of its own — group means are equal either way
        latency_ms.append(rng.uniform(20, 80))

rows = {
    "device": device,
    "latency_ms": [round(v, 2) for v in latency_ms],
}
for k in range(2):
    rows[f"n{k}"] = [rng.gauss(0, 1) for _ in range(n)]

df = pl.DataFrame(rows)
print(df.shape)

output_path = Path(__file__).resolve().parent.parent / "probe_pass.csv"
df.write_csv(output_path)
print(f"Saved to: {output_path}")
