"""Robust statistics, suspected sentinels and extreme values (stage 9 #6).

Everything here is computed on the finite non-null values of the profiled
sample and only ever ADDS flags: raw min/max/mean/std stay as they are, no row
is dropped from statistics, evidence or charts (decision A.6). Two labels:

- suspected_sentinel: a distinct value that is extremely far from the bulk AND
  shows at least two of three corroborating signals (repeated, common
  sentinel pattern, scale inconsistency). "suspected", never "invalid".
- extreme_value: rows beyond the Tukey far-out fences of the sentinel-free
  values. Legitimate data (long tails, spikes); readability, not validity.
"""

import math

import polars as pl

from .models import RobustRange, SentinelCandidate

# below this many finite values quantiles are not meaningful: no robust block
MIN_ROBUST_N = 5
# MAD -> sigma for a normal distribution; MAD has a 50% breakdown point, so
# the scale survives up to half the column being sentinels (IQR would not)
MAD_SCALE = 1.4826
# fallback when MAD is 0 (more than half the values identical): IQR -> sigma
IQR_TO_SIGMA = 1.349
# robust z beyond which a value is "extreme" for sentinel purposes: an order
# of magnitude past the usual 3.5 modified-z outlier cutoff, so genuine
# heavy-tail observations rarely qualify and never on this signal alone
Z_SENTINEL = 10.0
# "repeated": at least 2 rows and at least this share of the finite values —
# real extremes are distinct, sentinels recur
SENTINEL_MIN_REPEAT = 2
SENTINEL_MIN_REPEAT_SHARE = 0.002
# common fill values; -1 / 0 / 999-on-its-own are NOT patterns: they need the
# other signals (never treated as missing on sight)
SENTINEL_PATTERNS = frozenset({999.0, 9999.0, 99999.0, 999999.0})
# "scale": magnitude at least this many times the column's own scale
SENTINEL_SCALE_FACTOR = 100.0
# signals (besides the necessary "extreme") required for a suspected sentinel
SENTINEL_MIN_SIGNALS = 2
# Tukey far-out fences (q1 - k IQR, q3 + k IQR); the box render uses 1.5
FAR_OUT_IQR_MULTIPLIER = 3.0


def _quantiles(v: pl.Series) -> tuple[float, float, float]:
    q1 = float(v.quantile(0.25, interpolation="linear"))
    q3 = float(v.quantile(0.75, interpolation="linear"))
    return q1, q3, q3 - q1


def robust_quality(finite: pl.Series) -> dict:
    """ColumnQuality robust fields for the finite non-null values of a numeric
    column; {} when there are too few values."""
    v = finite.cast(pl.Float64)
    n = len(v)
    if n < MIN_ROBUST_N:
        return {}
    median = float(v.median())
    mad = float((v - median).abs().median())
    q1, q3, iqr = _quantiles(v)
    scale = MAD_SCALE * mad
    if scale == 0:
        scale = iqr / IQR_TO_SIGMA
    if scale == 0:
        # constant / overwhelmingly discrete column: nothing can be "far"
        return {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "mad": mad,
            "robust_z_max": None,
            "extreme_value_count": 0,
            "extreme_value_ratio": 0.0,
            "suspected_sentinels": [],
            "sentinel_row_count": 0,
            "robust_range": RobustRange(lo=float(v.min()), hi=float(v.max())),
        }

    z = (v - median).abs() / scale
    robust_z_max = float(z.max())

    # --- pass 1: suspected sentinels among the extreme distinct values -----
    sentinels: list[SentinelCandidate] = []
    candidates = v.filter(z > Z_SENTINEL)
    if len(candidates):
        scale_reference = SENTINEL_SCALE_FACTOR * max(abs(q1), abs(q3), scale)
        for value, count in candidates.value_counts().rows():
            signals = ["extreme"]
            if count >= SENTINEL_MIN_REPEAT and count >= SENTINEL_MIN_REPEAT_SHARE * n:
                signals.append("repeated")
            if value == math.floor(value) and abs(value) in SENTINEL_PATTERNS:
                signals.append("pattern")
            if abs(value) >= scale_reference:
                signals.append("scale")
            if len(signals) - 1 >= SENTINEL_MIN_SIGNALS:
                sentinels.append(SentinelCandidate(value=float(value), count=int(count), signals=signals))
    sentinels.sort(key=lambda s: (-s.count, s.value))

    # --- pass 2: fences and extremes on the sentinel-free values ------------
    clean = v.filter(~v.is_in([s.value for s in sentinels])) if sentinels else v
    sentinel_rows = n - len(clean)
    q1c, q3c, iqrc = _quantiles(clean)
    lo, hi = q1c - FAR_OUT_IQR_MULTIPLIER * iqrc, q3c + FAR_OUT_IQR_MULTIPLIER * iqrc
    extreme_count = int(((clean < lo) | (clean > hi)).sum())
    return {
        "q1": q1c,
        "q3": q3c,
        "iqr": iqrc,
        "mad": mad,
        "robust_z_max": robust_z_max,
        "extreme_value_count": extreme_count,
        "extreme_value_ratio": extreme_count / n,
        "suspected_sentinels": sentinels,
        "sentinel_row_count": sentinel_rows,
        "robust_range": RobustRange(lo=max(float(clean.min()), lo), hi=min(float(clean.max()), hi)),
    }
