"""Robust statistics, suspected sentinels and extreme values (stage 9 #6).

Everything here is computed on the finite non-null values of the profiled
sample and only ever ADDS flags: raw min/max/mean/std stay as they are, no row
is dropped from statistics, evidence or charts (decision A.6). Two labels:

- suspected_sentinel: a distinct value that is extremely far from the bulk AND
  shows at least two of three corroborating signals (repeated, common
  sentinel pattern, scale inconsistency). "suspected", never "invalid".
- extreme_value: rows beyond the Tukey far-out fences of the sentinel-free
  values. Legitimate data (long tails, spikes); readability, not validity.

Two regimes (audit 2026-10):

- MAD > 0 (the usual case): robust z = |x - median| / (1.4826 MAD); a value is
  "extreme" when z > 10; fences on the sentinel-free values.
- MAD == 0, i.e. at least half the values sit exactly on one value (the
  median, which is then the mode): z is undefined (and the old IQR fallback
  scored zero-inflated counts as 10% "extreme"). The column is described as
  mode + off-mode values T instead:
    * concentration = constant (T empty) / concentrated (mode share >= 0.9)
      / mode_dominated — informational only, never a warning;
    * sentinels: distinct off-mode values peeled from the outside in; a value
      is "extreme" when its distance to the mode exceeds 10x the median
      distance of the remaining off-mode values (always, when none remain);
      the "scale" signal compares |value| with 100x max(|mode|, that median
      distance) and is NOT available when both are 0 (an all-zero bulk gives
      no magnitude to be inconsistent with). Same 2-of-3 rule as above, so a
      zero-mode column needs the sentinel PATTERN plus repetition: 999999 /
      -9999 are flagged, a repeated -1 or 12345 next to zeros is not;
    * extremes: Tukey far-out fences on the sentinel-free off-mode values
      (>= MIN_ROBUST_N of them and IQR > 0), else none — the mode itself is
      never an extreme and a second mode never collapses the fences.
  The same off-mode fences are used when the sentinel-free IQR of the MAD > 0
  path is 0 (fences would otherwise collapse onto one value).
"""

import math

import polars as pl

from .models import RobustRange, SentinelCandidate

# below this many finite values quantiles are not meaningful: no robust block
MIN_ROBUST_N = 5
# MAD -> sigma for a normal distribution; MAD has a 50% breakdown point, so
# the scale survives up to half the column being sentinels (IQR would not)
MAD_SCALE = 1.4826
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
# mode-dominated columns (MAD == 0): share of the mode from which the column
# is reported "concentrated" (informational)
CONCENTRATED_SHARE = 0.9
# mode-dominated columns: at most this many distinct off-mode values are
# examined (outermost first) as sentinel candidates — bounds the cost
MAX_MODE_CANDIDATES = 32


def _quantiles(v: pl.Series) -> tuple[float, float, float]:
    q1 = float(v.quantile(0.25, interpolation="linear"))
    q3 = float(v.quantile(0.75, interpolation="linear"))
    return q1, q3, q3 - q1


def _signals(value: float, count: int, n: int, scale_reference: float | None) -> list[str]:
    """'extreme' plus the corroborating signals; scale_reference None means
    the scale signal is unavailable."""
    signals = ["extreme"]
    if count >= SENTINEL_MIN_REPEAT and count >= SENTINEL_MIN_REPEAT_SHARE * n:
        signals.append("repeated")
    if value == math.floor(value) and abs(value) in SENTINEL_PATTERNS:
        signals.append("pattern")
    if scale_reference is not None and abs(value) >= scale_reference:
        signals.append("scale")
    return signals


def _off_mode_window(clean: pl.Series, mode: float) -> tuple[int, float, float]:
    """(extreme count, lo, hi) for sentinel-free values dominated by `mode`:
    Tukey far-out fences on the off-mode values, widened to include the mode
    and clipped to the data range; no fences (0 extremes, full range) when
    there are too few off-mode values or they are themselves concentrated."""
    lo, hi = float(clean.min()), float(clean.max())
    off = clean.filter(clean != mode)
    if len(off) < MIN_ROBUST_N:
        return 0, lo, hi
    q1, q3, iqr = _quantiles(off)
    if iqr == 0:
        return 0, lo, hi
    f_lo, f_hi = q1 - FAR_OUT_IQR_MULTIPLIER * iqr, q3 + FAR_OUT_IQR_MULTIPLIER * iqr
    extreme_count = int(((off < f_lo) | (off > f_hi)).sum())
    return extreme_count, max(lo, min(mode, f_lo)), min(hi, max(mode, f_hi))


def _mode_sentinels(v: pl.Series, mode: float) -> list[SentinelCandidate]:
    """Suspected sentinels among the off-mode values of a MAD == 0 column,
    peeled from the outside in (see module docstring)."""
    n = len(v)
    off = v.filter(v != mode)
    distinct = sorted(off.value_counts().rows(), key=lambda r: (-abs(r[0] - mode), r[0]))
    sentinels: list[SentinelCandidate] = []
    flagged: list[float] = []
    for value, count in distinct[:MAX_MODE_CANDIDATES]:
        rest = off.filter(~off.is_in([*flagged, value]))
        ref = float((rest - mode).abs().median()) if len(rest) else 0.0
        if ref > 0 and abs(value - mode) <= Z_SENTINEL * ref:
            break  # reached the off-mode bulk: inner values are closer still
        magnitude = max(abs(mode), ref)
        scale_reference = SENTINEL_SCALE_FACTOR * magnitude if magnitude > 0 else None
        signals = _signals(float(value), int(count), n, scale_reference)
        if len(signals) - 1 >= SENTINEL_MIN_SIGNALS:
            sentinels.append(SentinelCandidate(value=float(value), count=int(count), signals=signals))
            flagged.append(value)
    return sentinels


def _mode_dominated(v: pl.Series, mad: float, mode: float) -> dict:
    """MAD == 0 regime: `mode` (== the median) holds at least half the rows."""
    n = len(v)
    mode_count = int((v == mode).sum())
    mode_share = mode_count / n
    if mode_count == n:
        concentration = "constant"
    elif mode_share >= CONCENTRATED_SHARE:
        concentration = "concentrated"
    else:
        concentration = "mode_dominated"
    sentinels = _mode_sentinels(v, mode) if mode_count < n else []
    sentinels.sort(key=lambda s: (-s.count, s.value))
    clean = v.filter(~v.is_in([s.value for s in sentinels])) if sentinels else v
    q1c, q3c, iqrc = _quantiles(clean)
    extreme_count, lo, hi = _off_mode_window(clean, mode)
    return {
        "q1": q1c,
        "q3": q3c,
        "iqr": iqrc,
        "mad": mad,
        "robust_z_max": None,  # undefined without dispersion around the median
        "extreme_value_count": extreme_count,
        "extreme_value_ratio": extreme_count / n,
        "suspected_sentinels": sentinels,
        "sentinel_row_count": n - len(clean),
        "robust_range": RobustRange(lo=lo, hi=hi),
        "mode_value": mode,
        "mode_share": mode_share,
        "concentration": concentration,
    }


def robust_quality(finite: pl.Series) -> dict:
    """ColumnQuality robust fields for the finite non-null values of a numeric
    column; {} when there are too few values."""
    v = finite.cast(pl.Float64)
    n = len(v)
    if n < MIN_ROBUST_N:
        return {}
    median = float(v.median())
    mad = float((v - median).abs().median())
    if mad == 0:
        return _mode_dominated(v, mad, median)
    q1, q3, _ = _quantiles(v)
    scale = MAD_SCALE * mad

    z = (v - median).abs() / scale
    robust_z_max = float(z.max())

    # --- pass 1: suspected sentinels among the extreme distinct values -----
    sentinels: list[SentinelCandidate] = []
    candidates = v.filter(z > Z_SENTINEL)
    if len(candidates):
        scale_reference = SENTINEL_SCALE_FACTOR * max(abs(q1), abs(q3), scale)
        for value, count in candidates.value_counts().rows():
            signals = _signals(value, count, n, scale_reference)
            if len(signals) - 1 >= SENTINEL_MIN_SIGNALS:
                sentinels.append(SentinelCandidate(value=float(value), count=int(count), signals=signals))
    sentinels.sort(key=lambda s: (-s.count, s.value))

    # --- pass 2: fences and extremes on the sentinel-free values ------------
    clean = v.filter(~v.is_in([s.value for s in sentinels])) if sentinels else v
    sentinel_rows = n - len(clean)
    q1c, q3c, iqrc = _quantiles(clean)
    if iqrc == 0:
        # the sentinel-free values are mode-dominated: Tukey fences would
        # collapse onto that value and call every other row extreme
        extreme_count, lo, hi = _off_mode_window(clean, q1c)
    else:
        lo, hi = q1c - FAR_OUT_IQR_MULTIPLIER * iqrc, q3c + FAR_OUT_IQR_MULTIPLIER * iqrc
        extreme_count = int(((clean < lo) | (clean > hi)).sum())
        lo, hi = max(float(clean.min()), lo), min(float(clean.max()), hi)
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
        "robust_range": RobustRange(lo=lo, hi=hi),
    }
