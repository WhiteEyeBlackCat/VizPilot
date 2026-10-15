"""Backend-verified claims and the semantic wording check (audit A1/A2).

Principle: the backend decides the facts, the LLM only phrases them.

After a hypothesis is validated, `build_claim` turns its evidence (never the
LLM's text) into a `Claim`: the facts a sentence about the finding may state —
which group is highest / lowest, each group's own numbers, the direction of a
relationship inside each group, the rise or fall of a level shift, the shape
of a curve. Coverage-path and probe-path evidence differ in shape (the probe's
group_difference carries top/bottom groups, coverage's does not; coverage's
slope table is a label -> corr dict, the probe's a list of rows ...); the
builder normalises both into one `Claim`.

`render_claim` writes the deterministic sentence for a claim (numbers from the
evidence, formatted so the number gate accepts them). `semantic_problem`
checks free text (LLM #2 text / headline / why_it_matters, LLM #1 reason,
chart captions) against the claim:

- causal language ("causes", "drives", "leads to", ...) is never supported:
  every test here is associational;
- a comparison between two group labels ("a higher than e", "e outsells a",
  "compared with a, e ...") must match the evidence order; a superlative
  bound to a label ("highest for e") must name the evidence's top / bottom;
- a number written next to a group label must be that group's own value
  (correct number, wrong group is rejected);
- direction words (positive / rises / falls / declines ...) must match the
  measured direction; "consistent / same direction / every / all / always"
  only when every group was estimated and all point the same way;
- a universal quantifier over something that is not the claim's grouping
  ("in every region") is an unsupported generalisation.

The regexes only tokenise; every decision is made against the claim.

What is NOT verifiable (rule 5 of the audit): for a claim kind whose facts
cannot bind a word (interaction cells, a grouped time pattern answered by an
interaction share, a group difference without per-group means, a curve
without a monotone direction ...), text is accepted only if it carries no
comparative / superlative / direction / quantifier / causal word; otherwise
the caller falls back to the claim template. Labels that are too ambiguous to
find safely in prose (stop words like "a", bare numbers like "1", labels that
are also lexicon words) are only matched in an explicit context ("segment a",
"'a'"); an unmatched label is not checked, which is why the template — the
only text the backend writes without the LLM — is always claim-built.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from ..probes.engine import NEAR_ZERO_CORR

# --- the claim --------------------------------------------------------------------


@dataclass
class GroupFact:
    label: str
    n: int | None = None
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    q25: float | None = None
    q75: float | None = None
    iqr: float | None = None
    corr: float | None = None
    slope: float | None = None

    def own_numbers(self) -> list[float]:
        return [
            float(v)
            for v in (self.n, self.mean, self.median, self.std, self.q25, self.q75, self.iqr, self.corr, self.slope)
            if _num(v)
        ]

    def sign(self) -> int | None:
        """+1 / -1 / 0 (near zero) / None (not estimated)."""
        if self.corr is None:
            return None
        if abs(self.corr) < NEAR_ZERO_CORR:
            return 0
        return 1 if self.corr > 0 else -1


@dataclass
class Claim:
    """What the backend verified for one hypothesis. `kind` is the probe type,
    except a grouped_relationship whose groups point in opposite directions,
    which is re-typed "sign_flip" (the finding is that the direction differs)."""

    kind: str
    test_type: str
    columns: dict[str, str]
    group_column: str | None = None
    groups: list[GroupFact] = field(default_factory=list)
    order_key: str | None = None  # "mean" / "median": what "higher" means for labels
    direction: str | None = None  # positive / negative / mixed / none (relationships, time, monotone curves)
    same_sign: bool | None = None
    unmeasured: list[str] = field(default_factory=list)
    overall_corr: float | None = None
    simpson: bool = False  # the pooled correlation points against every group
    shape: str | None = None
    change: dict[str, Any] | None = None  # time_pattern level shift: before / after / at
    ks_pair: list[str] | None = None
    bin_extremes: tuple[float, float] | None = None  # nonlinear: (max, min) bin mean
    universal_ok: bool = False  # "every / all / consistent" is supported
    comparisons_ok: bool = False  # label comparisons can be checked against an order
    signs_ok: bool = False  # direction words can be checked

    @property
    def labelled(self) -> list[GroupFact]:
        return [g for g in self.groups if g.label]

    def ordered(self, key: str) -> list[GroupFact]:
        rows = [g for g in self.groups if _num(_value(g, key))]
        return sorted(rows, key=lambda g: (-float(_value(g, key)), g.label))

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind, "columns": dict(self.columns)}
        if self.group_column:
            out["group_column"] = self.group_column
        if self.order_key and self.groups:
            ordered = self.ordered(self.order_key)
            if ordered:
                out["order_by"] = self.order_key
                out["top"] = ordered[0].label
                out["bottom"] = ordered[-1].label
        if self.direction is not None:
            out["direction"] = self.direction
        if self.same_sign is not None:
            out["same_sign"] = self.same_sign
        if self.kind in ("grouped_relationship", "sign_flip", "slope_difference"):
            out["signs"] = {g.label: g.sign() for g in self.groups}
        if self.unmeasured:
            out["unmeasured"] = list(self.unmeasured)
        if self.simpson:
            out["simpson"] = True
        if self.shape:
            out["shape"] = self.shape
        if self.change:
            out["change"] = dict(self.change)
        out["checks"] = {
            "comparisons": self.comparisons_ok,
            "signs": self.signs_ok,
            "universal": self.universal_ok,
        }
        return out


def _num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _value(g: GroupFact, key: str) -> float | None:
    if key == "abs_corr":
        return abs(g.corr) if _num(g.corr) else None
    if key == "iqr":
        if _num(g.iqr):
            return g.iqr
        return (g.q75 - g.q25) if _num(g.q75) and _num(g.q25) else None
    value = getattr(g, key, None)
    return value if _num(value) else None


def _group_facts(raw: Any) -> list[GroupFact]:
    """Both evidence shapes: a label -> corr dict (slope table) or a list of
    per-group rows (GroupStat / GroupCorrelation / probe entries)."""
    out: list[GroupFact] = []
    if isinstance(raw, dict):
        for label, corr in raw.items():
            out.append(GroupFact(label=str(label), corr=float(corr) if _num(corr) else None))
        return out
    if not isinstance(raw, list):
        return out
    for row in raw:
        if not isinstance(row, dict) or row.get("group") is None:
            continue
        fact = GroupFact(label=str(row["group"]))
        for key in ("mean", "median", "std", "q25", "q75", "iqr", "corr", "slope"):
            if _num(row.get(key)):
                setattr(fact, key, float(row[key]))
        if _num(row.get("n")):
            fact.n = int(row["n"])
        out.append(fact)
    return out


def _overall_from_profile(profile: Any, x: str, y: str) -> float | None:
    corr = getattr(profile, "correlations", None) if profile is not None else None
    if corr is None:
        return None
    try:
        i, j = corr.columns.index(x), corr.columns.index(y)
    except ValueError:
        return None
    value = corr.matrix[i][j]
    return float(value) if _num(value) else None


def build_claim(v: Any, profile: Any = None) -> Claim:
    """The claim for a validated hypothesis, from its evidence only."""
    ev = v.evidence if isinstance(v.evidence, dict) else {}
    cols = dict(v.columns)
    kind = v.test_type
    claim = Claim(kind=kind, test_type=kind, columns=cols)

    if kind in ("group_difference", "distribution_difference"):
        claim.group_column = cols.get("group")
        claim.groups = _group_facts(ev.get("groups"))
        claim.order_key = "mean" if kind == "group_difference" else "median"
        claim.comparisons_ok = len(claim.ordered(claim.order_key)) >= 2
        pair = ev.get("ks_pair")
        if isinstance(pair, list) and len(pair) == 2:
            claim.ks_pair = [str(p) for p in pair]
        return claim

    if kind in ("grouped_relationship", "slope_difference"):
        claim.group_column = cols.get("group")
        claim.groups = _group_facts(ev.get("groups"))
        measured = [g for g in claim.groups if g.corr is not None]
        claim.unmeasured = [g.label for g in claim.groups if g.corr is None]
        signs = {g.sign() for g in measured if g.sign()}
        if not signs:
            claim.direction = "none"
        elif len(signs) > 1:
            claim.direction = "mixed"
        else:
            claim.direction = "positive" if signs == {1} else "negative"
        claim.same_sign = bool(measured) and (
            all(g.corr > 0 for g in measured) or all(g.corr < 0 for g in measured)  # type: ignore[operator]
        )
        overall = ev.get("overall_corr")
        claim.overall_corr = float(overall) if _num(overall) else _overall_from_profile(profile, cols.get("x", ""), cols.get("y", ""))
        if claim.direction in ("positive", "negative") and claim.overall_corr is not None:
            if abs(claim.overall_corr) >= NEAR_ZERO_CORR and (claim.overall_corr > 0) != (claim.direction == "positive"):
                claim.simpson = True
        if kind == "grouped_relationship" and claim.direction == "mixed":
            # "the relationship holds inside every group" with opposite signs
            # is not one relationship: the finding is that the direction flips
            claim.kind = "sign_flip"
        # consistent in every group: every group estimated, all one sign, none near zero
        claim.universal_ok = (
            claim.direction in ("positive", "negative")
            and bool(claim.same_sign)
            and not claim.unmeasured
            and all(g.sign() for g in measured)
        )
        claim.signs_ok = claim.direction in ("positive", "negative", "mixed")
        claim.comparisons_ok = len(measured) >= 2  # strength comparisons only (|r|)
        return claim

    if kind == "nonlinear_relationship":
        shape = ev.get("shape")
        claim.shape = shape if isinstance(shape, str) else None
        bins = [b for b in ev.get("bins") or [] if isinstance(b, dict) and _num(b.get("y_mean"))]
        if bins:
            means = [float(b["y_mean"]) for b in bins]
            claim.bin_extremes = (max(means), min(means))
            if claim.shape in ("linear", "monotone_nonlinear"):
                claim.direction = "positive" if means[-1] > means[0] else "negative"
        if claim.direction is None and claim.shape in ("u_shape", "inverted_u", "multi_peak"):
            claim.direction = "nonmonotone"
        claim.signs_ok = claim.direction is not None
        return claim

    if kind == "time_pattern":
        if cols.get("group"):
            return claim  # answered by an interaction share: nothing to bind words to
        change = ev.get("change_point")
        if isinstance(change, dict) and _num(change.get("before_mean")) and _num(change.get("after_mean")):
            # coverage keeps only graded change points; the probe flags them
            if change.get("flagged", True) or change.get("strength") not in (None, "none"):
                before, after = float(change["before_mean"]), float(change["after_mean"])
                if before != after:
                    claim.change = {"before": before, "after": after, "at": str(change.get("change_at") or "")[:10]}
                    claim.direction = "positive" if after > before else "negative"
                    claim.signs_ok = True
        return claim

    # interaction and anything else: opaque — only word-free text is accepted
    return claim


# --- deterministic wording ----------------------------------------------------------


def fmt(value: float) -> str:
    """A backend number as the number gate accepts it: rounded, never to a
    bare integer unless it is one (the gate wants integers verbatim)."""
    if float(value).is_integer() and abs(value) < 1e15:
        return str(int(value))
    a = abs(value)
    digits = 1 if a >= 10 else 2 if a >= 0.1 else 3
    text = f"{value:.{digits}f}"
    return text if float(text) != 0 else f"{value:.4g}"


def _labels(groups: list[GroupFact], column: str | None, limit: int = 4) -> str:
    names = [f"{column} {g.label}" if column else g.label for g in groups[:limit]]
    more = len(groups) - limit
    if more > 0:
        names.append(f"{more} more")
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _corr_list(groups: list[GroupFact], column: str | None, limit: int = 4) -> str:
    shown = [f"{column} {g.label} (r = {fmt(g.corr)})" if column else f"{g.label} (r = {fmt(g.corr)})" for g in groups[:limit]]  # type: ignore[arg-type]
    if len(groups) > limit:
        shown.append(f"{len(groups) - limit} more")
    return shown[0] if len(shown) == 1 else ", ".join(shown[:-1]) + " and " + shown[-1]


_SHAPE_WORDS = {
    "u_shape": "U-shaped",
    "inverted_u": "inverted-U-shaped",
    "multi_peak": "with several peaks",
    "monotone_nonlinear": "curved but one-directional",
    "linear": "close to a straight line",
    "other": "curved",
    "flat": "flat",
}


def render_claim(claim: Claim) -> str | None:
    """The claim as one sentence (no trailing effect suffix), or None when the
    evidence carries nothing beyond the neutral description."""
    c = claim.columns
    g = claim.group_column
    if claim.kind in ("group_difference", "distribution_difference") and claim.comparisons_ok:
        key = claim.order_key or "mean"
        ordered = claim.ordered(key)
        top, bottom = ordered[0], ordered[-1]
        word = "Mean" if key == "mean" else "Median"
        if fmt(_value(top, key)) == fmt(_value(bottom, key)):  # type: ignore[arg-type]
            # an exact (or rounded) tie: no highest / lowest to name
            return (
                f"{word} {c['target']} is equal across the {len(ordered)} {g} groups ({fmt(_value(top, key))})"  # type: ignore[arg-type]
            )
        text = (
            f"{word} {c['target']} is highest for {g} {top.label} ({fmt(_value(top, key))}) and lowest for "
            f"{g} {bottom.label} ({fmt(_value(bottom, key))}) across {len(ordered)} {g} groups"
        )
        if claim.kind == "distribution_difference" and claim.ks_pair:
            text += f"; the distributions differ most between {g} {claim.ks_pair[0]} and {g} {claim.ks_pair[1]}"
        return text
    if claim.kind in ("grouped_relationship", "sign_flip", "slope_difference"):
        measured = [x for x in claim.groups if x.corr is not None]
        if not measured:
            return None
        pos = sorted((x for x in measured if x.sign() == 1), key=lambda x: (-abs(x.corr), x.label))  # type: ignore[arg-type]
        neg = sorted((x for x in measured if x.sign() == -1), key=lambda x: (-abs(x.corr), x.label))  # type: ignore[arg-type]
        flat = [x for x in measured if x.sign() == 0]
        if claim.direction == "mixed":
            text = (
                f"{c['x']} and {c['y']} move together in {_corr_list(pos, g)} but in opposite directions in "
                f"{_corr_list(neg, g)}"
            )
        elif claim.direction in ("positive", "negative"):
            verb = "rises" if claim.direction == "positive" else "falls"
            together = pos if claim.direction == "positive" else neg
            if claim.kind == "slope_difference":
                strongest = max(measured, key=lambda x: (abs(x.corr), x.label))  # type: ignore[arg-type]
                weakest = min(measured, key=lambda x: (abs(x.corr), x.label))  # type: ignore[arg-type]
                text = (
                    f"The {c['x']}-{c['y']} correlation differs by {g}: strongest in {g} {strongest.label} "
                    f"(r = {fmt(strongest.corr)}), weakest in {g} {weakest.label} (r = {fmt(weakest.corr)})"  # type: ignore[arg-type]
                )
            elif claim.universal_ok:
                lo = min(abs(x.corr) for x in together)  # type: ignore[arg-type]
                hi = max(abs(x.corr) for x in together)  # type: ignore[arg-type]
                sign = "" if claim.direction == "positive" else "-"
                text = (
                    f"{c['y']} {verb} with {c['x']} within each of the {len(together)} {g} groups "
                    f"(r from {sign}{fmt(lo)} to {sign}{fmt(hi)})"
                )
            else:
                text = f"{c['y']} {verb} with {c['x']} in {_corr_list(together, g)}"
        else:
            return None
        if flat:
            text += f"; no clear relationship in {_corr_list(flat, g)}"
        if claim.unmeasured:
            text += f"; {_labels([GroupFact(label=u) for u in claim.unmeasured], g)} had too few rows to estimate"
        if claim.simpson and claim.overall_corr is not None:
            text += f"; pooled over all rows the correlation is {fmt(claim.overall_corr)}, in the opposite direction"
        return text
    if claim.kind == "nonlinear_relationship" and claim.shape:
        text = f"{c['y']} varies non-linearly with {c['x']} ({_SHAPE_WORDS.get(claim.shape, 'curved')})"
        if claim.bin_extremes is not None:
            hi, lo = claim.bin_extremes
            text += f": mean {c['y']} ranges from {fmt(lo)} to {fmt(hi)} across {c['x']} bins"
        return text
    if claim.kind == "time_pattern" and claim.change:
        verb = "rises" if claim.direction == "positive" else "falls"
        at = f" around {claim.change['at']}" if claim.change.get("at") else ""
        return (
            f"Mean {c['target']} {verb} from {fmt(claim.change['before'])} to {fmt(claim.change['after'])}{at}"
        )
    return None


# --- the semantic check (fail closed) -------------------------------------------------
#
# The gate does not try to understand arbitrary prose. It recognises a small
# set of assertion forms it can verify against the claim and rejects every
# sentence that makes an assertion it cannot verify:
#
# - non-English text (any non-Latin letter once column names and group labels
#   are masked), comparison symbols (< > ≤ ≥ ≠ ≈, "=" other than "n = 600" /
#   "r = 0.95"), causal language, negated assertions -> reject;
# - every evaluative / ordering / magnitude word (a broad class: comparatives,
#   superlatives, ranks, "leads", "dominates", "twice", "half", "times", "high",
#   "low", any other -er / -est adjective, ...) must be one the verifier
#   recognises for this claim kind AND bound to the groups the evidence ranks
#   that way; an unrecognised one, one with nothing to bind to, or one next to
#   an unknown entity (a capitalised name that is not a label, "southern" for
#   "South") rejects the sentence;
# - direction words must match the measured direction; for a curve that is not
#   monotone only the shape's own name is accepted;
# - universal quantifiers only where every group was estimated with one sign.
#
# A rejected sentence costs the LLM's wording, never the finding: the caller
# falls back to the deterministic claim template.

_CAUSAL = re.compile(
    r"\b(?:cause[sd]?|causing|causal(?:ly)?|drive[sn]?|driving|drove|driver[s]?|lead(?:s|ing)?\s+to|led\s+to|"
    r"result(?:s|ed|ing)?\s+in|because|on\s+account\s+of|due\s+to|thanks\s+to|owing\s+to|as\s+a\s+result|"
    r"make[s]?\s+\S+\s+(?:higher|lower|more|less|rise|fall|increase|decrease|grow|go\s+up|go\s+down)|"
    r"affect(?:s|ed|ing)?|influenc(?:e|es|ed|ing)|impact(?:s|ed|ing)?|determin(?:e|es|ed|ing|ant)|"
    r"boost(?:s|ed|ing)?|trigger(?:s|ed|ing)?|produce[sd]?|responsible\s+for|consequence[s]?|"
    r"explain(?:s|ed|ing)?|shapes(?!\s+of\b)|shaped\s+by|shaping|come[s]?\s+from|came\s+from|stem(?:s|med|ming)?\s+from|"
    r"effect\s+(?:of|on)|effects\s+(?:of|on)|contribut(?:e|es|ed|ing)|depend(?:s|ed|ing)?\s+on|"
    r"generat(?:e|es|ed|ing)\s+\S+\s+as|yield(?:s|ed)?|leads?\s+\S+\s+to|"
    r"push(?:es|ed)?|rais(?:e|es|ed|ing)|lift(?:s|ed)?|lower(?:s|ed)(?=\s+(?:#|the\b)))(?![a-z])",
    re.IGNORECASE,
)
# a column doing something to a column: "z raises w", "z increases w"
_COLUMN_VERB = re.compile(r"#+\s+(?:increas|decreas|reduc|rais|lower|push|lift|boost|driv|chang|alter)\w*\s+#+")
_SYMBOL = re.compile(r"[<>≤≥≠≈]")
_STAT_EQ = re.compile(r"(?:\b(?:n|r|rho|r2|r²|eta|eta-squared|eta²|η²|ks|d|p|spread|effect)|#)\s*$", re.IGNORECASE)
_NEGATION = {"not", "no", "none", "never", "neither", "nor", "without", "nothing", "nobody", "nowhere", "cannot"}

# phrase normalisation (same length, padded): idioms -> one verifiable word
_PHRASES = [
    (r"\bnot\s+(?:the\s+)?(?:same|equal|identical|uniform)\b", "differs"),
    (r"\bfrom\s+(?:the\s+)?(?:highest|top|most)\s+to\s+(?:the\s+)?(?:lowest|bottom|least)\b", "descending"),
    (r"\bfrom\s+(?:the\s+)?(?:lowest|bottom|least)\s+to\s+(?:the\s+)?(?:highest|top|most)\b", "ascending"),
    (r"\b(?:falls?|fell|falling)\s+short\s+of\b", "trails"),
    (r"\blag(?:s|ged|ging)?\s+behind\b", "trails"),
    (r"\bbehind\b", "below"),
    (r"\b(?:edges?|edged)\s+out\b", "outsells"),
    (r"\b(?:comes?|came|coming)\s+out\s+on\s+top\b", "highest"),
    (r"\bon\s+top\b", "top"),
    (r"\bstands?\s+out\b", "highest"),
    (r"\bbottoms?\s+out\b", "lowest"),
    (r"\bahead\s+of\b", "above"),
    (r"\bout-?(?:earns?|paces?|strips?|ranks?)\b|\bovertakes?\b|\beclipses?\b|\boutstrips?\b|\boutpaces?\b", "exceeds"),
    (r"\b(?:goes|go|went|gone|going)\s+up\b|\bpick(?:s|ed)?\s+up\b|\bmoves?\s+together\b|\bin\s+step\b", "rises"),
    (r"\b(?:goes|go|went|gone|going)\s+down\b", "falls"),
    (r"\banti-?correlat\w*\b", "negative"),
    (r"\bmoves?\s+opposite\b|\b(?:opposite|different|contrary)\s+directions?\b|\bopposite\b", "negative"),
    (r"\bnon-?linear(?:ly)?\b|\bcurvilinear\b", "curved"),
]

_COMP_UP = {
    "higher", "more", "greater", "larger", "bigger", "above", "exceed", "exceeds", "exceeded", "exceeding",
    "outsell", "outsells", "outsold", "outselling", "outperform", "outperforms", "outperformed", "surpass",
    "surpasses", "surpassed", "beat", "beats", "tops", "topped",
}
_COMP_DOWN = {
    "lower", "less", "fewer", "smaller", "below", "trail", "trails", "trailed", "trailing", "lag", "lags",
    "lagged", "underperform", "underperforms", "underperformed",
}
_VERB_COMP = {"above", "below", "exceed", "exceeds", "exceeded", "exceeding", "outsell", "outsells", "outsold",
              "outselling", "outperform", "outperforms", "outperformed", "surpass", "surpasses", "surpassed", "beat",
              "beats", "tops", "topped", "trail", "trails", "trailed", "trailing", "lag", "lags", "lagged",
              "underperform", "underperforms", "underperformed"}
_SUP_UP = {"highest", "largest", "biggest", "greatest", "most", "top", "first", "leader", "leaders", "leads", "lead",
           "leading", "best", "maximum", "max", "peak", "peaks", "peaked", "dominates", "dominate", "dominated",
           "dominant", "dominating"}
_SUP_DOWN = {"lowest", "smallest", "least", "fewest", "bottom", "last", "worst", "minimum", "min", "trough"}
_STRENGTH = {"stronger": 1, "steeper": 1, "tighter": 1, "weaker": -1, "flatter": -1, "looser": -1}
_STRENGTH_SUP = {"strongest": 1, "steepest": 1, "tightest": 1, "weakest": -1, "flattest": -1, "loosest": -1}
_SPREAD_COMP = {"wider": 1, "narrower": -1}
_SPREAD_SUP = {"widest": 1, "narrowest": -1}
_RANK = {"second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5}
_MAG = {"double": 2.0, "doubles": 2.0, "doubled": 2.0, "twice": 2.0, "triple": 3.0, "tripled": 3.0, "triples": 3.0,
        "thrice": 3.0, "half": 0.5, "halved": 0.5, "halves": 0.5, "times": None, "fold": None}
_INTENSITY = {"high": 1, "large": 1, "big": 1, "strong": 1, "huge": 1, "massive": 1, "elevated": 1, "hefty": 1,
              "sizable": 1, "sizeable": 1, "low": -1, "small": -1, "weak": -1, "tiny": -1, "modest": -1, "poor": -1,
              "slight": -1, "limited": -1, "depressed": -1, "substantial": 1, "considerable": 1, "notable": 1,
              "pronounced": 1, "sharp": 1, "wide": 1}
_EQUAL = {"same", "equal", "similar", "identical", "comparable", "alike", "uniform", "equally", "similarly"}
_OTHER_EVAL = {"percent", "percentage", "ratio", "ranking", "outlier", "record",
               "premium", "advantage", "majority", "minority", "multiple", "%", "upper", "superior", "inferior",
               "standout", "winner", "loser", "excel", "excels", "excelled", "underdog", "laggard", "champion"}
_HEDGE = {"roughly", "about", "nearly", "almost", "around", "approximately", "some", "close", "circa", "approx"}
_GAP_WORDS = {"gap", "gaps", "difference", "differences", "differ", "differs", "differed", "distance", "contrast",
              "separation", "spread"}
# -er / -est words that are not comparatives (anything else ending so is a
# comparative the verifier does not know -> rejected)
_ER_EXCEPTIONS = {
    "other", "others", "another", "under", "over", "never", "ever", "however", "whether", "either", "neither",
    "rather", "after", "per", "number", "numbers", "water", "customer", "customers", "order", "orders", "member",
    "members", "quarter", "quarters", "winter", "summer", "center", "centre", "together", "enter", "cluster",
    "clusters", "filter", "register", "offer", "offers", "consider", "cover", "partner", "partners", "user", "users",
    "paper", "letter", "interest", "test", "tests", "rest", "forest", "suggest", "request", "contest", "invest",
    "harvest", "manager", "managers", "worker", "workers", "power", "tower", "owner", "owners", "provider",
    "providers", "buyer", "buyers", "seller", "sellers", "retailer", "retailers", "dealer", "dealers", "player",
    "players", "computer", "matter", "master", "chapter", "character", "parameter", "meter", "later", "earlier",
    "whatever", "whenever", "wherever", "remainder", "answer", "consumer", "consumers", "shopper", "shoppers",
    "subscriber", "subscribers", "visitor", "worker", "teacher", "number", "factor", "sector", "vector", "border",
    "corner", "weather", "temper", "transfer", "chamber", "december", "november", "october", "september", "proper",
    "super", "per", "hour", "her", "their", "water", "paper", "ber", "fever", "cancer", "gender", "tier", "server",
    "servers", "header", "folder", "marker", "scatter", "counter", "encounter", "barrier", "carrier", "supplier",
    "suppliers", "manufacturer", "driver", "drivers", "chester", "manchester", "hamster", "monster", "longer",
    "differ", "deliver", "delivers", "gather", "wonder", "discover", "remember", "render", "suffer", "prefer",
    "refer", "alter", "offer", "together", "further", "farther", "latest", "honest", "modest", "digest", "biggest", "interest", "pest", "chest", "nest", "west", "best", "least",
    "most", "against", "amongst", "midst", "est", "guest", "quest", "protest", "arrest", "behest", "conquest",
}
_SIGN_POS = {
    "positive", "positively", "increase", "increases", "increased", "increasing", "rise", "rises", "rising", "rose",
    "risen", "grow", "grows", "growing", "grew", "grown", "climb", "climbs", "climbed", "climbing", "surge",
    "surges", "surged", "surging", "jump", "jumps", "jumped", "spike", "spikes", "spiked", "soar", "soars", "soared",
    "upward", "upwards", "tracks", "track", "tracking", "gain", "gains", "gained", "accelerate", "accelerates",
    "accelerated", "rebound", "rebounds", "rebounded", "recover", "recovers", "recovered",
}
_SIGN_NEG = {
    "negative", "negatively", "decrease", "decreases", "decreased", "decreasing", "decline", "declines", "declined",
    "declining", "fall", "falls", "falling", "fell", "fallen", "drop", "drops", "dropped", "dropping", "dip",
    "dips", "dipped", "dipping", "plunge", "plunges", "plunged", "sink", "sinks", "sank", "slump", "slumps",
    "slumped", "shrink", "shrinks", "shrank", "downward", "downwards", "inverse", "inversely", "collapse",
    "collapsed", "tumble", "tumbled", "slide", "slid", "contract", "contracted", "slows", "slowed", "slowing",
}
_SHAPE_ONLY = {"monotone", "monotonic", "monotonically", "linear", "linearly", "steadily", "steady", "trough",
               "valley", "hump", "plateau", "plateaus", "flattens", "saturates", "minimal", "maximal", "extreme", "extremes", "minimised", "minimized",
               "maximised", "maximized", "minimum", "maximum", "bowl", "convex", "concave", "turning"}
_AFTER_WORDS = {"after", "since", "later", "afterwards", "recent", "recently", "now", "subsequently", "then"}
_BEFORE_WORDS = {"before", "earlier", "previously", "prior", "initially", "formerly"}
_SCOPE_ALL = re.compile(r"\b(?:overall|pooled|aggregate|combined|across\s+all\s+rows|whole\s+table|all\s+rows)\b")
_NUMBER_WORDS = r"(?:two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)"
_QUANT_NOUN = re.compile(
    rf"\b(?P<q>every|each|all)\s+(?:of\s+)?(?:the\s+|its\s+|their\s+|these\s+|those\s+)?(?:{_NUMBER_WORDS}\s+)?(?P<noun>[a-z][a-z0-9_]*)"
)
_QUANT_WORD = re.compile(
    r"\b(?:always|everywhere|without\s+exception|universal(?:ly)?|invariabl[ey]|consistent(?:ly)?|uniform(?:ly)?|"
    r"same\s+direction|regardless\s+of|irrespective\s+of|no\s+matter|across\s+the\s+board|in\s+all\s+cases|"
    r"hand\s+in\s+hand|hold[s]?\s+across|hold[s]?\s+in\s+(?:every|each|all)|same\s+way|similar\s+way|"
    r"(?:the\s+)?same\s+(?:in|for|across)|alike)\b"
)
_ACROSS = re.compile(r"\bacross\s+(?:the\s+)?(?:" + _NUMBER_WORDS + r"\s+)?(?P<noun>[a-z][a-z0-9_]*)")
_ALL_OTHER = re.compile(r"\b(?:every|all(?:\s+the)?|any)\s+other\b")
_GENERIC_GROUP_NOUNS = {"group", "groups", "category", "categories", "subgroup", "subgroups", "segment", "segments",
                        "level", "levels", "class", "classes"}
_SENTENCE_BREAK = re.compile(r"[.;!?:](?=\s|$)|\n")
_CLAUSE_BREAK = re.compile(r"[.;!?:](?=\s|$)|\n|\b(?:but|while|whereas|although|though|however|yet)\b")
_SEGMENT_BREAK = re.compile(r"[.;!?,](?=\s|$)|\n|\b(?:but|while|whereas|although|though|however|yet)\b")
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")
_WORDTOK = re.compile(r"[A-Za-z0-9_]+")
_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?|%")
_REFERENCE = re.compile(r"\b(?:than|compared\s+(?:to|with)|relative\s+to|versus|vs\.?)(?=\W|$)")
_SHAPE_PHRASES = {
    "u_shape": r"\bu[\s-]?shaped?\b|\bu[\s-]shape\b|\bu-?curve\b",
    "inverted_u": r"\binverted[\s-]?u(?:[\s-]?shaped?)?\b|\bhump[\s-]?shaped\b",
    "multi_peak": r"\b(?:several|multiple|many)\s+peaks\b",
}
# labels that are too ambiguous to find as bare words in prose
_STOP = {
    "a", "an", "i", "the", "of", "in", "on", "at", "to", "or", "and", "is", "it", "as", "be", "by", "for", "no",
    "not", "yes", "all", "any", "one", "two", "none", "other", "others", "rest", "high", "low", "higher", "lower",
    "more", "less", "most", "least", "same", "new", "old", "up", "down", "top", "bottom", "total", "average",
    "mean", "median", "group", "groups", "n", "r", "vs", "true", "false", "null", "na", "unknown", "each",
    "every", "than", "with", "per", "from", "this", "that", "these", "those", "s", "t", "d", "m", "first", "last",
    "best", "worst", "u",
}


@dataclass
class _Hit:
    start: int
    end: int
    label: str


@dataclass
class _Tok:
    word: str
    start: int
    end: int


def _lower(text: str) -> str:
    # length-preserving case fold (str.lower can change the length of a few
    # characters, which would shift every span)
    out = "".join(ch.lower() if len(ch.lower()) == 1 else ch for ch in text)
    # typographic quotes are punctuation, not content: fold to ASCII (1:1)
    out = out.translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'}))
    # "e.g." / "i.e." would otherwise read as the labels "e" / "i"
    return re.sub(r"\b(?:e\.g\.|i\.e\.)", lambda m: " " * len(m.group(0)), out)


def _is_lexicon(word: str) -> bool:
    return any(
        word in s
        for s in (_COMP_UP, _COMP_DOWN, _SUP_UP, _SUP_DOWN, _STRENGTH, _STRENGTH_SUP, _RANK, _MAG, _INTENSITY,
                  _EQUAL, _OTHER_EVAL, _SIGN_POS, _SIGN_NEG)
    )


def _ambiguous(label: str, column_names: set[str]) -> bool:
    low = label.lower()
    if low in _STOP or low in column_names:
        return True
    if re.fullmatch(r"-?\d+(?:\.\d+)?", low):
        return True
    return _is_lexicon(low) or _eval_class(low) is not None


def _find_labels(low: str, claim: Claim, column_names: set[str]) -> list[_Hit]:
    """Label occurrences: bare (unambiguous labels only), after the group
    column name ("segment a", "segment=a"), or quoted ('a', "a", `a`)."""
    hits: list[_Hit] = []
    gcol = (claim.group_column or "").lower()
    for g in claim.groups:
        label = g.label.strip()
        if not label or len(label) > 60:
            continue
        lab = re.escape(label.lower())
        patterns = [rf"[`'\"“‘]{lab}[`'\"”’]"]
        if gcol:
            patterns.append(rf"\b{re.escape(gcol)}s?\s*[=:]?\s*[`'\"“‘]?(?P<l>{lab})(?![\w])")
        if not _ambiguous(label, column_names):
            patterns.append(rf"(?<![\w]){lab}(?![\w])")
        for pattern in patterns:
            for m in re.finditer(pattern, low):
                if "l" in (m.groupdict() or {}) and m.group("l") is not None:
                    start, end = m.start("l"), m.end("l")
                else:
                    start, end = m.start(), m.end()
                hits.append(_Hit(start, end, g.label))
    # longest match wins on overlap; then one hit per span
    hits.sort(key=lambda h: (h.start, -(h.end - h.start)))
    out: list[_Hit] = []
    for h in hits:
        if out and h.start < out[-1].end:
            continue
        out.append(h)
    return out


def _spans(text: str, breaker: re.Pattern[str]) -> list[tuple[int, int]]:
    cuts = [0] + [m.end() for m in breaker.finditer(text)] + [len(text)]
    return [(a, b) for a, b in zip(cuts, cuts[1:]) if b > a]


def _word_gap(low: str, a: int, b: int) -> int:
    lo, hi = (a, b) if a <= b else (b, a)
    return len(_WORDTOK.findall(low[lo:hi]))


def _eval_class(word: str) -> tuple[str, Any] | None:
    """The evaluative class of one word, or None for a word that asserts no
    order / magnitude / level. Unknown -er / -est adjectives are ("unknown", w)."""
    if word in _COMP_UP:
        return ("comp", 1)
    if word in _COMP_DOWN:
        return ("comp", -1)
    if word in _SUP_UP:
        return ("sup", 1)
    if word in _SUP_DOWN:
        return ("sup", -1)
    if word in _STRENGTH:
        return ("strength_comp", _STRENGTH[word])
    if word in _STRENGTH_SUP:
        return ("strength_sup", _STRENGTH_SUP[word])
    if word in _SPREAD_COMP:
        return ("spread_comp", _SPREAD_COMP[word])
    if word in _SPREAD_SUP:
        return ("spread_sup", _SPREAD_SUP[word])
    if word in _RANK:
        return ("rank", _RANK[word])
    if word in _MAG:
        return ("mag", _MAG[word])
    if word in _INTENSITY:
        return ("intensity", _INTENSITY[word])
    if word in _EQUAL:
        return ("equal", 0)
    if word == "between":
        return ("between", 0)
    if word == "followed":
        return ("followed", 0)
    if word in _OTHER_EVAL:
        return ("unknown", word)
    if word not in _ER_EXCEPTIONS and len(word) > 4 and (word.endswith("er") or word.endswith("est")):
        return ("unknown", word)
    return None


def _sign_of(word: str) -> int | None:
    if word in _SIGN_POS:
        return 1
    if word in _SIGN_NEG:
        return -1
    return None


@dataclass
class _Text:
    """One text prepared for the gate: lower-cased, idioms normalised, column
    names masked with '#', labels with '§' (positions preserved)."""

    raw: str
    low: str  # lower-cased with idioms normalised (labels / columns intact)
    norm: str  # low with labels and columns masked
    labels: list[_Hit]
    tokens: list[_Tok]
    claim: Claim | None


def _prepare(text: str, claim: Claim | None, names: set[str]) -> _Text:
    low = _lower(text)
    if claim is not None and claim.kind == "nonlinear_relationship" and claim.shape in _SHAPE_PHRASES:
        # the shape's own name is the one shape wording a curve accepts
        low = re.sub(_SHAPE_PHRASES[claim.shape], lambda m: " " * len(m.group(0)), low)
    for pattern, word in _PHRASES:
        low = re.sub(pattern, lambda m, w=word: w.ljust(len(m.group(0))) if len(m.group(0)) >= len(w) else m.group(0), low)
    labels = _find_labels(low, claim, names) if claim is not None else []
    chars = list(low)
    for h in labels:
        for i in range(h.start, h.end):
            chars[i] = "§"
    norm = "".join(chars)
    for name in sorted((n for n in names if n), key=len, reverse=True):
        norm = re.sub(r"(?<![\w])" + re.escape(name.lower()) + r"(?![\w])", lambda m: "#" * len(m.group(0)), norm)
    if claim is not None and claim.kind in ("sign_flip", "slope_difference"):
        # "the sign of the z-w relationship depends on the region" IS the finding
        # (sign / strength by group), not a causal claim between two measures
        norm = re.sub(r"(\b(?:sign|direction|strength|slope)\b[^.;]{0,60}?)\bdepends\s+on(?=\s+(?:the\s+)?#)",
                      lambda m: m.group(1) + "differs by", norm)
    tokens = [_Tok(m.group(0), m.start(), m.end()) for m in _TOKEN.finditer(norm)]
    return _Text(text, low, norm, labels, tokens, claim)


def _unknown_entities(t: _Text, start: int, end: int) -> list[str]:
    """Names in a sentence that the claim does not know: a capitalised word
    (not the sentence's first word) that is neither a label nor a column, or a
    word derived from a label ("southern" for "South")."""
    out: list[str] = []
    labels = [g.label.lower() for g in t.claim.groups] if t.claim is not None else []
    all_words = re.findall(r"[A-Za-z][A-Za-z'-]*", t.raw)
    title_case = len(all_words) >= 3 and sum(w[0].isupper() for w in all_words) >= 0.6 * len(all_words)
    first = True
    for m in re.finditer(r"[A-Za-z][A-Za-z'-]*", t.raw[start:end]):
        s = start + m.start()
        if t.norm[s] in "#§":
            first = False
            continue
        word = m.group(0)
        lw = word.lower()
        derived = any(len(lab) >= 4 and lw != lab and lw.startswith(lab) for lab in labels)
        capitalised = not first and not title_case and word[0].isupper()
        if derived or (capitalised and not _is_lexicon(lw) and lw not in ("i", "ks", "sd")):
            out.append(word)
        first = False
    return out


def semantic_problem(text: str, claim: Claim | None, column_names: set[str] | None = None) -> str | None:
    """None when `text` asserts only what the claim supports (see above)."""
    if not text or not text.strip():
        return None
    names = {n.lower() for n in (column_names or set())} | (
        {c.lower() for c in claim.columns.values()} if claim is not None else set()
    )
    t = _prepare(text, claim, names)
    if any(ch.isalpha() and not ch.isascii() for ch in t.norm):
        return "text is not in English (only English wording can be verified)"
    if any(not ch.isascii() for ch in t.norm.replace("§", "").replace("#", "")):
        return "non-ASCII symbol (emoji, dash, sign) cannot be verified"
    m = _SYMBOL.search(t.norm)
    if m:
        return f"comparison symbol '{m.group(0)}' cannot be verified"
    for m in re.finditer("=", t.norm):
        if not _STAT_EQ.search(t.norm[: m.start()]):
            return "'=' outside a statistic ('n = ...', 'r = ...') cannot be verified"
    m = _CAUSAL.search(t.norm) or _COLUMN_VERB.search(t.norm)
    if m:
        return f"causal language '{m.group(0).strip()}' is not supported by an associational test"
    if claim is None:
        return _allowlist_problem(t, None) or _claimless_problem(t)
    problem = _quantifier_problem(t, claim) or _allowlist_problem(t, claim)
    if problem:
        return problem
    for start, end in _spans(t.norm, _SENTENCE_BREAK):
        problem = _sentence_problem(t, start, end, claim)
        if problem:
            return problem
    for start, end in _spans(t.norm, _SEGMENT_BREAK):
        problem = _binding_problem(t, start, end, claim)
        if problem:
            return problem
    return _time_number_problem(t, claim) or _sequence_problem(t, claim)


_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_PERIOD_AFTER = re.compile(r"\b(?:after|since|later|afterwards|subsequently|now|recent(?:ly)?|second\s+half)\b")
_PERIOD_BEFORE = re.compile(r"\b(?:before|earlier|previously|prior|initially|formerly|first\s+half)\b")


def _time_number_problem(t: _Text, claim: Claim) -> str | None:
    """A level shift's two means must be bound to the right periods: "from A
    to B" means A before and B after; a number next to "before" / "after"
    belongs to that period. A before/after mean that is not bound is
    unverifiable."""
    if claim.kind != "time_pattern" or not claim.change:
        return None
    from .service import _number_supported

    before, after = claim.change["before"], claim.change["after"]
    text = _ISO_DATE_RE.sub(lambda m: " " * len(m.group(0)), t.norm)
    numbers = [m for m in _NUMBER.finditer(text)]

    def which(token: str) -> str | None:
        b, a = _number_supported(token, {before}), _number_supported(token, {after})
        return "before" if b and not a else "after" if a and not b else None

    bound: set[int] = set()
    for m in re.finditer(r"\bfrom\s+(-?\d+(?:\.\d+)?)\s+to\s+(-?\d+(?:\.\d+)?)", text):
        first, second = which(m.group(1)), which(m.group(2))
        if first == "after" or second == "before":
            return f"'from {m.group(1)} to {m.group(2)}' reverses the measured change ({fmt(before)} to {fmt(after)})"
        bound |= {m.start(1), m.start(2)}
    for m in numbers:
        if m.start() in bound:
            continue
        period = which(m.group(0))
        if period is None:
            continue
        segment = next(((a, b) for a, b in _spans(text, _SEGMENT_BREAK) if a <= m.start() < b), (0, len(text)))
        words = [(w.start(), "after") for w in _PERIOD_AFTER.finditer(text, *segment)] + [
            (w.start(), "before") for w in _PERIOD_BEFORE.finditer(text, *segment)]
        if not words:
            return f"number {m.group(0)} is a period mean but no period is named for it"
        nearest = min(words, key=lambda w: (_word_gap(text, min(w[0], m.start()), max(w[0], m.start())), w[0] < m.start()))
        if nearest[1] != period:
            return f"number {m.group(0)} is the {period} mean, not the {nearest[1]} mean"
    return None


_UP_ORDER = re.compile(r"\b(?:highest|most|largest|biggest|greatest|top|leads|leader|best|higher|more|greater|larger|descending)\b")
_DOWN_ORDER = re.compile(r"\b(?:lowest|least|smallest|fewest|bottom|worst|lower|less|fewer|smaller|ascending)\b")


def _sequence_problem(t: _Text, claim: Claim) -> str | None:
    """A list of groups read as an order ("North, then North East, then West",
    "..., followed by ...", "by region: North, South, West") must follow the
    evidence order in the direction the text states; with no stated direction
    the order is unverifiable."""
    if claim.kind not in ("group_difference", "distribution_difference") or not claim.comparisons_ok:
        return None
    for start, end in _spans(t.norm, re.compile(r"[.;!?](?=\s|$)|\n")):
        seg = t.norm[start:end]
        hits = [h for h in t.labels if start <= h.start < end]
        order: list[str] = []
        for h in hits:
            if h.label not in order:
                order.append(h.label)
        listed = re.search(r":", seg) is not None and len(order) >= 3
        if len(order) < 2 or not (re.search(r"\b(?:then|followed)\b", seg) or listed):
            continue
        if re.search(r"\bfrom\s+(?:the\s+)?(?:highest|top|most)\b", seg):
            direction = 1
        elif re.search(r"\bfrom\s+(?:the\s+)?(?:lowest|bottom|least)\b", seg):
            direction = -1
        else:
            up, down = bool(_UP_ORDER.search(seg)), bool(_DOWN_ORDER.search(seg))
            if up == down:
                return "a list of groups without one stated direction cannot be verified as an order"
            direction = 1 if up else -1
        key = claim.order_key or "mean"
        values = {g.label: _value(g, key) for g in claim.groups}
        seq = [values.get(lab) for lab in order]
        if any(v is None for v in seq):
            return "a listed group has no value to order"
        if any((a - b) * direction < 0 for a, b in zip(seq, seq[1:])):
            return f"the listed order {', '.join(order)} contradicts the {key} order"
    return None


def _words(t: _Text, start: int, end: int) -> list[_Tok]:
    return [k for k in t.tokens if start <= k.start < end]


def _claimless_problem(t: _Text) -> str | None:
    """Free text with no claim behind it (a bare LLM chart suggestion's
    caption or title): nothing can be checked, so no word that asserts."""
    for k in t.tokens:
        if _eval_class(k.word) is not None or _sign_of(k.word) is not None:
            return f"unverifiable assertion '{k.word}' in unvalidated text"
    m = _QUANT_WORD.search(t.norm) or _QUANT_NOUN.search(t.norm)
    if m:
        return f"unverifiable generalisation '{m.group(0)}' in unvalidated text"
    return None


def _group_nouns(claim: Claim) -> set[str]:
    gcol = (claim.group_column or "").lower()
    nouns = set(_GENERIC_GROUP_NOUNS)
    if gcol:
        nouns |= {gcol, gcol + "s", gcol + "es"}
    return nouns


def _quantifier_problem(t: _Text, claim: Claim) -> str | None:
    text = t.low
    nouns = _group_nouns(claim)
    m = _QUANT_WORD.search(text)
    if m and not claim.universal_ok:
        return f"'{m.group(0)}' is not supported: the evidence does not show one consistent direction in every group"
    relationship = claim.kind in ("grouped_relationship", "sign_flip", "slope_difference")
    if relationship and not claim.universal_ok:
        m = _ACROSS.search(text)
        if m and (m.group("noun") in nouns or m.group("noun").rstrip("s") in nouns):
            return f"'{m.group(0)}' is not supported: the claim was not established in every group"
    for m in _QUANT_NOUN.finditer(text):
        noun = m.group("noun")
        before = text[max(0, m.start() - 4) : m.start()]
        if before.strip().endswith("of") and claim.comparisons_ok:
            continue  # "the highest of all segments": a superlative's scope, checked as a superlative
        if noun == "other" and claim.comparisons_ok:
            continue  # "outsells every other segment": checked as a top / bottom comparison
        if noun in nouns or noun.rstrip("s") in nouns:
            if claim.universal_ok:
                continue
            return f"'{m.group(0)}' is not supported: the claim was not established in every {claim.group_column or 'group'}"
        return f"unsupported generalisation '{m.group(0)}': '{noun}' is not a grouping the evidence covers"
    return None


def _sentence_problem(t: _Text, start: int, end: int, claim: Claim) -> str | None:
    words = _words(t, start, end)
    evals = [(k, _eval_class(k.word)) for k in words if _eval_class(k.word) is not None]
    if claim.kind == "nonlinear_relationship":
        # a curve accepts its shape name only (removed in _prepare); any other shape word is unverified
        evals += [(k, ("unknown", k.word)) for k in words if k.word in _SHAPE_ONLY and _eval_class(k.word) is None]
    signs = [(k, _sign_of(k.word)) for k in words if _sign_of(k.word) is not None]
    if claim.kind not in ("group_difference", "distribution_difference"):
        # "the relationship between z and w": "between" only orders groups
        evals = [(k, c) for k, c in evals if c[0] != "between"]
    if not evals and not signs:
        return None
    if any(k.word in _NEGATION or k.word.endswith("n't") for k in words) or "n't" in t.low[start:end]:
        return "a negated comparison or direction cannot be verified"
    labels = [h for h in t.labels if start <= h.start < end]
    unknown = _unknown_entities(t, start, end)
    if unknown:
        return f"assertion about '{unknown[0]}', which is not a group of this finding"
    for k, cls in evals:
        if cls[0] == "unknown":
            return f"'{cls[1]}' is not an assertion the backend can verify"
    if claim.kind in ("group_difference", "distribution_difference"):
        return _group_sentence(t, start, end, claim, evals, signs, labels)
    if claim.kind in ("grouped_relationship", "sign_flip", "slope_difference"):
        return _relationship_sentence(t, start, end, claim, evals, signs, labels)
    if claim.kind == "nonlinear_relationship":
        return _curve_sentence(t, start, end, claim, evals, signs)
    if claim.kind == "time_pattern":
        return _time_sentence(t, start, end, claim, evals, signs)
    word = (evals or signs)[0][0].word
    return f"'{word}' cannot be verified for a {claim.kind} finding"


# -- group differences ------------------------------------------------------------------


def _values(claim: Claim, key: str) -> dict[str, float]:
    return {g.label: float(_value(g, key)) for g in claim.ordered(key)}  # type: ignore[arg-type]


def _middle(values: dict[str, float]) -> float:
    ordered = sorted(values.values())
    n = len(ordered)
    return ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2


def _keys_for(claim: Claim, clause: str, dim: str) -> list[str]:
    if dim == "spread":
        return ["iqr"] if any(_num(_value(g, "iqr")) for g in claim.groups) else ["std"]
    if re.search(r"\bmedian\b", clause):
        return ["median"]
    if claim.kind == "distribution_difference":
        return ["median", "mean"]  # "higher" on a distribution: both locations must agree
    return [claim.order_key or "mean"]


def _group_sentence(t, start, end, claim, evals, signs, labels) -> str | None:
    if not claim.comparisons_ok:
        word = (evals or signs)[0][0].word
        return f"'{word}' cannot be verified: no per-group values for this finding"
    text = t.norm[start:end]
    # a direction word is only verifiable as "rises from A to B"
    for k, polarity in signs:
        m = re.match(r"\s+from\s+", t.norm[k.end : end])
        a = next((h for h in labels if h.start > k.end), None)
        b = next((h for h in labels if a is not None and h.start > a.end), None)
        if not m or a is None or b is None or not re.search(r"\bto\s*$", t.norm[a.end : b.start]):
            return f"direction word '{k.word}' cannot be verified for a {claim.kind} finding"
        for key in _keys_for(claim, text, "level"):
            values = _values(claim, key)
            if a.label in values and b.label in values and (values[b.label] - values[a.label]) * polarity <= 0:
                return f"'{k.word} from {a.label} to {b.label}' contradicts the {key} order"
    if not evals:
        return None
    if any(cls[0] == "equal" for _, cls in evals):
        return "a claim of equality between groups contradicts a validated difference"
    if any(cls[0] in ("strength_comp", "strength_sup") for _, cls in evals):
        evals = [(k, ("comp" if cls[0] == "strength_comp" else "sup", cls[1])) if cls[0].startswith("strength") else (k, cls)
                 for k, cls in evals]
    dim = "spread" if any(cls[0].startswith("spread") for _, cls in evals) or re.search(
        r"\b(?:spread|variab\w*|dispersion|iqr|std|standard\s+deviation|volatil\w*)\b", text) else "level"
    evals = [(k, (cls[0].replace("spread_", ""), cls[1])) for k, cls in evals]
    gap_context = any(k.word in _GAP_WORDS for k in _words(t, start, end))
    if not labels:
        # nothing to bind an order to: only the size of the (validated) gap may be described
        if gap_context and all(cls[0] in ("intensity", "sup") and cls[1] > 0 for _, cls in evals):
            return None
        return f"'{evals[0][0].word}' is not bound to a group, so it cannot be verified"
    for key in _keys_for(claim, text, dim):
        problem = _order_check(t, start, end, claim, evals, labels, key, gap_context)
        if problem:
            return problem
    return None


def _order_check(t, start, end, claim, evals, labels, key, gap_context) -> str | None:
    values = _values(claim, key)
    if len(values) < 2:
        return f"no per-group {key} to compare"
    ordered = sorted(values, key=lambda lab: -values[lab])
    middle = _middle(values)
    # "the gap is largest between A and B" / "a large gap between A and B"
    if gap_context:
        m = re.search(r"\bbetween\b", t.norm[start:end])
        if m:
            at = start + m.end()
            pair = [h for h in labels if h.start >= at][:2]
            if len(pair) == 2:
                if any(cls[0] == "sup" for _, cls in evals):
                    want = set(claim.ks_pair or []) if claim.kind == "distribution_difference" and claim.ks_pair else {ordered[0], ordered[-1]}
                    if {pair[0].label, pair[1].label} != want:
                        return f"the largest gap is not between {pair[0].label} and {pair[1].label}"
                labels = [h for h in labels if h not in pair]
                evals = [(k, c) for k, c in evals if c[0] not in ("sup", "intensity", "between")]
                if not labels or not evals:
                    return None
    clauses = _spans(t.norm[start:end], re.compile(r"\b(?:but|while|whereas|although|though|however|yet)\b|;"))
    prev_subjects: list[_Hit] = []
    for cs, ce in clauses:
        cs, ce = start + cs, start + ce
        c_labels = [h for h in labels if cs <= h.start < ce]
        c_evals = [(k, c) for k, c in evals if cs <= k.start < ce]
        if not c_evals:
            prev_subjects = c_labels[:1] or prev_subjects
            continue
        problem, prev_subjects = _clause_order(t, cs, ce, claim, c_evals, c_labels, values, ordered, middle, key, prev_subjects)
        if problem:
            return problem
    return None


def _clause_order(t, cs, ce, claim, evals, labels, values, ordered, middle, key, prev_subjects):
    comps = [(k, c) for k, c in evals if c[0] == "comp"]
    sups = [(k, c) for k, c in evals if c[0] == "sup"]
    ranks = [(k, c) for k, c in evals if c[0] == "rank"]
    mags = [(k, c) for k, c in evals if c[0] == "mag"]
    intens = [(k, c) for k, c in evals if c[0] == "intensity"]
    betweens = [(k, c) for k, c in evals if c[0] == "between"]
    followed = [(k, c) for k, c in evals if c[0] == "followed"]
    if any(lab.label not in values for lab in labels):
        return f"no {key} for every group named", prev_subjects
    if not labels and not prev_subjects:
        return f"'{evals[0][0].word}' is not bound to a group, so it cannot be verified", prev_subjects
    labels_or_prev = labels or prev_subjects

    def above(a: str, b: str, polarity: int) -> bool:
        return (values[a] - values[b]) * polarity > 0

    # ranks: "North East is second", "the second-lowest"
    for k, (_, rank) in ranks:
        nxt = next((c for kk, c in sups if kk.start > k.start and _word_gap(t.norm, k.end, kk.start) <= 1), None)
        polarity = nxt[1] if nxt else 1
        sups = [(kk, c) for kk, c in sups if not (kk.start > k.start and _word_gap(t.norm, k.end, kk.start) <= 1)]
        h = _nearest(t, labels_or_prev, k.start, prefer_before=True)
        if h is None:
            return f"'{k.word}' is not bound to a group", prev_subjects
        order = ordered if polarity > 0 else ordered[::-1]
        if rank > len(order) or values[order[rank - 1]] != values[h.label]:
            return f"{h.label} is not {k.word} in {key}", prev_subjects
    # superlatives
    if sups:
        distinct = []
        for h in labels_or_prev:
            if all(h.label != d.label for d in distinct):
                distinct.append(h)
        if len(distinct) == len(sups):
            pairs = list(zip(sups, distinct))
        else:
            pairs = []
            for s in sups:
                h = _nearest(t, labels_or_prev, s[0].start, prefer_before=False)
                if h is None or _word_gap(t.norm, s[0].start, h.start) > 8:
                    return f"'{s[0].word}' is not bound to a group", prev_subjects
                pairs.append((s, h))
        for (k, (_, polarity)), h in pairs:
            want = ordered[0] if polarity > 0 else ordered[-1]
            if values[h.label] != values[want]:
                return f"'{k.word}' is bound to {h.label}, but the {key} {'top' if polarity > 0 else 'bottom'} is {want}", prev_subjects
    # between: "North East sits between South and West"
    for k, _ in betweens:
        before = [h for h in labels_or_prev if h.start < k.start]
        after = [h for h in labels if h.start > k.start][:2]
        if not before or len(after) < 2:
            return "'between' is not bound to three groups", prev_subjects
        x, a, b = values[before[-1].label], values[after[0].label], values[after[1].label]
        if not min(a, b) < x < max(a, b):
            return f"{before[-1].label} is not between {after[0].label} and {after[1].label} in {key}", prev_subjects
    # "followed by": the groups listed from the one before it are in order
    for k, _ in followed:
        seq = [h for h in labels_or_prev if h.start < k.start][-1:] + [h for h in labels if h.start > k.start]
        polarity = -1 if any(c[1] < 0 for _, c in sups) else 1
        for a, b in zip(seq, seq[1:]):
            if a.label != b.label and not above(a.label, b.label, polarity):
                return f"{b.label} does not follow {a.label} in the {key} order", prev_subjects
    # comparatives and magnitudes: subjects vs references
    subjects, refs = _subjects_refs(t, cs, ce, labels, comps + mags)
    if (comps or mags) and not subjects:
        subjects = prev_subjects
    if comps:
        polarities = {c[1] for _, c in comps}
        if len(polarities) > 1:
            return "comparisons of opposite direction in one clause cannot be verified", prev_subjects
        polarity = polarities.pop()
        every_other = _ALL_OTHER.search(t.norm, cs, ce) is not None
        for s in subjects:
            for r in refs:
                if r.label != s.label and not above(s.label, r.label, polarity):
                    return f"comparison of {s.label} and {r.label} contradicts the {key} order", prev_subjects
            if not refs:
                if every_other:
                    want = ordered[0] if polarity > 0 else ordered[-1]
                    if values[s.label] != values[want]:
                        return f"{s.label} is not {'above' if polarity > 0 else 'below'} every other group in {key}", prev_subjects
                elif (values[s.label] - middle) * polarity <= 0:
                    return f"{s.label} is not {'above' if polarity > 0 else 'below'} the other groups in {key}", prev_subjects
        if not subjects:
            for r in refs:
                if (middle - values[r.label]) * polarity <= 0:
                    return f"the other groups are not {'above' if polarity > 0 else 'below'} {r.label} in {key}", prev_subjects
    for k, (_, factor) in mags:
        if factor is None:  # "N times" / "N-fold": the number before it
            m = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*$", t.norm[cs : k.start])
            if not m:
                return f"'{k.word}' without a factor cannot be verified", prev_subjects
            factor = float(m.group(1))
        as_as = re.search(r"\bas\s+\w+\s+(?:in|for)\s+§+\s+as\s+(?:in|for)\s+§", t.norm[k.end : ce])
        if not subjects and len(refs) == 2 and as_as:
            subjects, refs = refs[:1], refs[1:]  # "twice as high in South as in North"
        if len(subjects) != 1 or len(refs) != 1:
            return f"'{k.word}' needs exactly one group compared with one other", prev_subjects
        a, b = values[subjects[0].label], values[refs[0].label]
        if b == 0:
            return f"'{k.word}' cannot be verified against a zero {key}", prev_subjects
        ratio = a / b
        tolerance = 0.15 if any(w.word in _HEDGE for w in _words(t, cs, ce)) else 0.07
        if comps and next(iter({c[1] for _, c in comps}), 1) > 0:
            ok = ratio >= factor * (1 - tolerance)
        elif comps:
            ok = ratio <= factor * (1 + tolerance)
        else:
            ok = abs(ratio / factor - 1) <= tolerance
        if not ok:
            return f"{subjects[0].label} is not '{k.word}' {refs[0].label} in {key} (ratio {ratio:.2f})", prev_subjects
    # "high" / "low" bound to a group: above / below the middle
    for k, (_, polarity) in intens:
        following = next((w.word for w in _words(t, k.end, ce)), "")
        if following in _GAP_WORDS or following in ("margin", "margins"):
            continue  # "a wide margin", "a large gap": the size of the (validated) difference
        h = _nearest(t, labels_or_prev, k.start, prefer_before=True)
        if h is None:
            return f"'{k.word}' is not bound to a group", prev_subjects
        if (values[h.label] - middle) * polarity <= 0:
            return f"{h.label} is not '{k.word}' relative to the other groups in {key}", prev_subjects
    return None, (subjects or labels[:1] or prev_subjects)


def _nearest(t: _Text, labels: list[_Hit], at: int, prefer_before: bool) -> _Hit | None:
    """The label closest to a word (in words between them); ties go to the
    label before it when `prefer_before`."""
    if not labels:
        return None
    return min(labels, key=lambda h: (_word_gap(t.norm, h.end if h.start < at else at, at if h.start < at else h.start),
                                      (h.start > at) if prefer_before else (h.start < at)))


def _subjects_refs(t: _Text, cs: int, ce: int, labels: list[_Hit], items) -> tuple[list[_Hit], list[_Hit]]:
    if not items:
        return list(labels), []
    ref = _REFERENCE.search(t.norm, cs, ce)
    if ref is not None:
        marker = ref.group(0)
        if marker == "than":
            return [h for h in labels if h.start < ref.start()], [h for h in labels if h.start > ref.start()]
        after = [h for h in labels if h.start > ref.start()]
        refs = after[:1]
        return [h for h in labels if h not in refs], refs
    pivot_tok = next((k for k, c in items if k.word in _VERB_COMP or c[0] == "mag"), None)
    if pivot_tok is not None:
        return [h for h in labels if h.start < pivot_tok.start], [h for h in labels if h.start > pivot_tok.start]
    return list(labels), []


# -- relationships ----------------------------------------------------------------------


def _relationship_sentence(t, start, end, claim, evals, signs, labels) -> str | None:
    signs = [(k.start, p) for k, p in signs]
    level = [(k, c) for k, c in evals if c[0] == "comp"]
    strength = [(k, c) for k, c in evals if c[0] in ("strength_comp", "strength_sup")]
    intens = [(k, c) for k, c in evals if c[0] == "intensity"]
    other = [(k, c) for k, c in evals if c[0] not in ("comp", "strength_comp", "strength_sup", "intensity", "between")]
    if other:
        return f"'{other[0][0].word}' cannot be bound to a measured quantity of this relationship"
    if intens and labels:
        return f"'{intens[0][0].word}' about a single group's relationship cannot be verified"
    if level:
        # "higher z goes with higher w": two comparatives are one direction
        if len(level) == 2 and not _REFERENCE.search(t.norm, start, end):
            signs.append((level[1][0].start, level[0][1][1] * level[1][1][1]))
        else:
            return f"comparison '{level[0][0].word}' cannot be bound to a measured quantity of this relationship"
    signs = _collapse_pairs(t, start, end, sorted(signs), labels)
    if signs:
        problem = _sign_problem(t, start, end, signs, labels, claim)
        if problem:
            return problem
    for k, (cls, polarity) in strength:
        problem = _strength_check(t, start, end, k, cls, polarity, labels, claim)
        if problem:
            return problem
    return None


def _collapse_pairs(
    t: _Text, start: int, end: int, signs: list[tuple[int, int]], labels: list[_Hit] | None = None
) -> list[tuple[int, int]]:
    """Per clause, two direction words are one statement about the pair:
    "w falls as z rises" moves x one way and y the other -> negative. Not when
    groups sit between them ("positive in north and negative in east"): then
    each word speaks for the groups that follow it."""
    out: list[tuple[int, int]] = []
    for cs, ce in _clauses(t, start, end):
        c = [s for s in signs if cs <= s[0] < ce]
        between = len(c) == 2 and any(c[0][0] < h.start < c[1][0] for h in (labels or []))
        out += [(c[0][0], c[0][1] * c[1][1])] if len(c) == 2 and not between else c
    return out


def _clauses(t: _Text, start: int, end: int) -> list[tuple[int, int]]:
    return [(start + a, start + b) for a, b in _spans(t.norm[start:end], _CLAUSE_BREAK)]


def _sign_problem(t: _Text, start: int, end: int, signs: list[tuple[int, int]], labels: list[_Hit], claim: Claim) -> str | None:
    if not claim.signs_ok:
        return f"direction word cannot be verified for a {claim.kind} finding"
    if claim.simpson and _SCOPE_ALL.search(t.norm, start, end):
        return "direction stated for the pooled data, which runs against every group"
    if claim.direction in ("positive", "negative") and claim.kind != "sign_flip":
        expected = 1 if claim.direction == "positive" else -1
        for _, polarity in signs:
            if polarity != expected:
                return f"direction word contradicts the measured {claim.direction} direction"
        for h in labels:
            polarity = _nearest_sign(h, signs)
            fact = next((g for g in claim.groups if g.label == h.label), None)
            if fact is not None and fact.sign() is not None and fact.sign() != polarity:
                return f"direction stated for {h.label} does not match its correlation"
        return None
    # mixed directions: each direction must be bound, clause by clause, to the groups it holds for
    for cs, ce in _clauses(t, start, end):
        c_signs = [s for s in signs if cs <= s[0] < ce]
        c_labels = [h for h in labels if cs <= h.start < ce]
        if not c_signs:
            continue
        if not c_labels:
            return "a single direction is stated for a relationship whose direction differs between groups"
        for h in c_labels:
            # a direction word speaks for the groups after it, up to the next one;
            # groups before every direction word take the nearest
            preceding = [sg for sg in c_signs if sg[0] < h.start]
            polarity = preceding[-1][1] if preceding else _nearest_sign(h, c_signs)
            fact = next((g for g in claim.groups if g.label == h.label), None)
            if fact is None or fact.sign() != polarity:
                return f"direction stated for {h.label} does not match its correlation"
    return None


def _nearest_sign(h: _Hit, signs: list[tuple[int, int]]) -> int:
    return min(signs, key=lambda s: (abs(s[0] - h.start), s[0] > h.start))[1]


def _strength_check(t, start, end, k, cls, polarity, labels, claim) -> str | None:
    values = {g.label: abs(g.corr) for g in claim.groups if g.corr is not None}
    if len(values) < 2:
        return f"'{k.word}' cannot be verified: fewer than two estimated groups"
    c_labels = [h for h in labels if any(cs <= h.start < ce and cs <= k.start < ce for cs, ce in _clauses(t, start, end))]
    if not c_labels:
        return f"'{k.word}' is not bound to a group"
    if any(h.label not in values for h in c_labels):
        return f"'{k.word}' names a group without an estimated correlation"
    ordered = sorted(values, key=lambda lab: -values[lab])
    if cls == "strength_sup":
        h = _nearest(t, c_labels, k.start, prefer_before=False)
        want = ordered[0] if polarity > 0 else ordered[-1]
        if values[h.label] != values[want]:
            return f"'{k.word}' is bound to {h.label}, but it is {want}"
        return None
    ref = _REFERENCE.search(t.norm, start, end)
    subjects = [h for h in c_labels if ref is None or h.start < ref.start()]
    refs = [h for h in c_labels if ref is not None and h.start > ref.start()]
    if not refs:
        return f"'{k.word}' without a group to compare with cannot be verified"
    for s in subjects:
        for r in refs:
            if (values[s.label] - values[r.label]) * polarity <= 0:
                return f"the relationship in {s.label} is not '{k.word}' than in {r.label}"
    return None


# -- curves and time --------------------------------------------------------------------


def _curve_sentence(t, start, end, claim, evals, signs) -> str | None:
    evals = [(k, c) for k, c in evals if c[0] != "between"]
    if not evals and not signs:
        return None
    words = {k.word for k in _words(t, start, end)}
    if claim.direction == "nonmonotone":
        # only the shape's own name (removed in _prepare) describes a curve
        # that turns: any direction, extremum or order word is unverifiable
        word = (evals or signs)[0][0].word
        return f"'{word}' cannot be verified for a {claim.shape} curve; only its shape name can"
    if claim.direction not in ("positive", "negative"):
        word = (evals or signs)[0][0].word
        return f"'{word}' cannot be verified for a curve without a direction"
    if words & _SHAPE_ONLY:
        return f"shape word '{sorted(words & _SHAPE_ONLY)[0]}' does not match the measured {claim.shape} curve"
    level = [(k, c) for k, c in evals if c[0] == "comp"]
    other = [(k, c) for k, c in evals if c[0] != "comp"]
    if other:
        return f"'{other[0][0].word}' cannot be verified for this curve"
    s = [(k.start, p) for k, p in signs]
    if level:
        if len(level) == 2 and not _REFERENCE.search(t.norm, start, end):
            s.append((level[1][0].start, level[0][1][1] * level[1][1][1]))
        else:
            return f"comparison '{level[0][0].word}' cannot be verified for this curve"
    s.sort()
    if len(s) == 2:
        s = [(s[0][0], s[0][1] * s[1][1])]
    expected = 1 if claim.direction == "positive" else -1
    if any(p != expected for _, p in s):
        return f"direction word contradicts the measured {claim.direction} direction"
    return None


def _time_sentence(t, start, end, claim, evals, signs) -> str | None:
    evals = [(k, c) for k, c in evals if c[0] != "between"]
    if not evals and not signs:
        return None
    if not claim.signs_ok or claim.change is None:
        word = (evals or signs)[0][0].word
        return f"'{word}' cannot be verified: no measured direction over time"
    expected = 1 if claim.direction == "positive" else -1
    words = _words(t, start, end)
    s = [p for _, p in signs]
    for k, cls in evals:
        if cls[0] == "comp":
            # "higher after ... (than before)": the period named first after the comparative
            period = next((w.word for w in words if w.start > k.start and (w.word in _AFTER_WORDS or w.word in _BEFORE_WORDS)), None)
            if period is None:
                period = next((w.word for w in reversed(words) if w.start < k.start and (w.word in _AFTER_WORDS or w.word in _BEFORE_WORDS)), None)
            if period is None:
                return f"comparison '{k.word}' over time does not say which period is meant"
            s.append(cls[1] if period in _AFTER_WORDS else -cls[1])
        elif cls[0] == "mag" and cls[1] is not None:
            before, after = claim.change["before"], claim.change["after"]
            if before == 0 or abs((after / before) / cls[1] - 1) > 0.15:
                return f"'{k.word}' does not match the measured change ({fmt(before)} to {fmt(after)})"
            s.append(1 if cls[1] > 1 else -1)
        else:
            return f"'{k.word}' cannot be verified for a change over time"
    if any(p != expected for p in s):
        return f"direction word contradicts the measured {'rise' if expected > 0 else 'fall'}"
    return None


# -- numbers next to labels ---------------------------------------------------------------


def _binding_problem(t: _Text, start: int, end: int, claim: Claim) -> str | None:
    """A number next to a group label must be that group's own number; a
    number in parentheses belongs to the label before it."""
    labels = [h for h in t.labels if start <= h.start < end]
    if not labels:
        return None
    from .service import _number_supported  # the number gate's own rounding rule

    low = t.low
    inside = [(h.start, h.end) for h in t.labels]
    facts = {g.label: g for g in claim.groups}

    def others_of(label: str) -> set[float]:
        return {x for g in claim.groups if g.label != label for x in g.own_numbers()}

    def check(token: str, label: str) -> str | None:
        fact = facts.get(label)
        if fact is None or _number_supported(token, set(fact.own_numbers())):
            return None
        if _number_supported(token, others_of(label)):
            return f"number {token} belongs to another group, not {label}"
        return None

    paired: set[int] = set()
    for p in re.finditer(r"\(([^()]*)\)", low[start:end]):
        p_start, p_end = start + p.start(), start + p.end()
        numbers = [n for n in _NUMBER.finditer(low, p_start, p_end) if not any(a <= n.start() < b for a, b in inside)]
        if not numbers:
            continue
        before: list[str] = []
        for h in labels:
            if h.start < p_start and h.label not in before:
                before.append(h.label)
        inner = [h for h in labels if p_start <= h.start < p_end]
        if inner:
            continue  # "(South 200.0)": the nearest-label rule below decides
        if len(numbers) >= 2 and len(before) == len(numbers):
            pairs = list(zip(numbers, before))  # "e higher than a (79.1 vs 50.5)"
        elif len(numbers) == 1 and before:
            pairs = [(numbers[0], before[-1])]  # "South (200.0)", "the lowest (99.1)" -> the label before
        else:
            continue
        for n, label in pairs:
            paired.add(n.start())
            problem = check(n.group(0), label)
            if problem:
                return problem
    for m in _NUMBER.finditer(low, start, end):
        if m.start() in paired or any(a <= m.start() < b for a, b in inside):
            continue
        # "99.1 in North", "200.0 for South": the label the number is placed in
        placed = re.match(r"\s*(?:in|for|at)\s+", low[m.end():end])
        target = next((h for h in labels if placed and h.start == m.end() + placed.end()), None)
        if target is not None:
            problem = check(m.group(0), target.label)
            if problem:
                return problem
            continue
        best: tuple[int, bool, _Hit] | None = None
        for h in labels:
            gap = _word_gap(low, h.end, m.start()) if h.start < m.start() else _word_gap(low, m.end(), h.start)
            after = h.start > m.start()
            if gap <= 4 and (best is None or (gap, after) < (best[0], best[1])):
                best = (gap, after, h)
        if best is None:
            continue
        problem = check(m.group(0), best[2].label)
        if problem:
            return problem
    return None


# --- token allowlist (review round 3): fail closed by construction --------------------
#
# LLM text is accepted only if EVERY word, after masking column names, group
# labels and numbers, is in one of these closed lists: function words, neutral
# statistical / descriptive words, a few plain advisory words (for
# why_it_matters), the verifier's lexicon (each such word must then bind and
# match the evidence, see above) and the words of the claim's own template.
# Anything else - an idiom ("takes the crown"), a causal verb ("fuels"), a
# foreign word, a spelled-out number - rejects the text, so no list of bad
# words is needed and an unknown phrasing can never pass.

_FUNCTION_WORDS = frozenset("""
a an the and or but while whereas although though however yet so as than that which who whose whom where when
this these those it its they them their there here is are was were be been being has have had having does do did
can could will would may might should of in on at by for with within without across between among from to into
onto over per each both either neither also only about roughly approximately around nearly almost some all any
other others rest another every s versus vs compared relative then after before later earlier during throughout
up out off again further same what how than such very more most less least well far much slightly
""".split()) - {"neither", "without", "throughout", "up", "out", "off", "again", "such"}

_NEUTRAL_WORDS = frozenset("""
mean means average averages averaged averaging median medians value values group groups grouping category
categories subgroup subgroups segment segments level levels class classes relationship relationships
correlation correlations correlated correlate correlates related relate relates associated association
associations link links linked connection connected trend trends change changes changed shift shifts shifted
shifting point points row rows record records sample samples data dataset table distribution distributions
distributed spread spreads variance variation variations varies vary varied varying differs differ differed
differing difference differences different distinct gap gaps margin margins shape shapes pattern patterns column
columns observation observations period periods time times over bin bins range ranges ranged ranging strength
sign signs direction directions finding findings evidence measured measure overall pooled whole total inside
across among statistic statistics adjusted eta squared r n ks pearson coefficient sd standard deviation iqr
quartile quartiles effect size share proportion count counts number numbers observed seen shows show showed shown
sits sit lies lie has have records posts sees earns earn earned earning sells sell sold selling generates
generate generated tends tend tended goes go went follows follow followed pairs pattern holds is are was were
remains remain stays stay appears appear looks look moves move moving together jointly along alongside clearly
clear markedly marked sharply sharp strongly notably noticeably substantially considerably distinctly modestly
slightly mildly moderately moderate somewhat generally typically consistently noticeable visible apparent
regional by case cases part whereas whole entire mid middle centre center end ends edge range wide
""".split())

_ADVISORY_WORDS = frozenset("""
matters matter important useful worth helps help helpful understand understanding knowing know pricing price
prices market markets separate behaves behave behaviour behavior like customers customer demand investment
investments invest budget budgets planning plan plans strategy strategies targeting target targets decisions
decision monitor monitoring explore exploring consider considering focus focusing resources resource priorities
priority prioritise prioritize risk risks opportunity opportunities performance operations operational inventory
staffing capacity forecasting forecast forecasts design marketing allocation allocate allocating attention review
reviewing check checking compare comparing comparison comparisons tailor tailoring expectations
expectation interpret interpreting reporting report reports analysis analyses further deeper why need needs
needed worthwhile relevant guide guides guiding should could may might
llm ai model says suggests suggested suggestion chart charts plot plots view axis bar bars line lines scatter
histogram box boxes heatmap own month months year years day days week weeks date dates mid early late start
beginning january february march april may june july august september october november december
hour hours weekday weekdays weekend weekends level timeline reason reasons descending ascending order
""".split())

_LEXICON_WORDS = frozenset(
    set(_COMP_UP) | set(_COMP_DOWN) | set(_SUP_UP) | set(_SUP_DOWN) | set(_STRENGTH) | set(_STRENGTH_SUP)
    | set(_SPREAD_COMP) | set(_SPREAD_SUP) | set(_RANK) | {k for k in _MAG if k not in ("fold",)}
    | set(_INTENSITY) | set(_SIGN_POS) | set(_SIGN_NEG) | _AFTER_WORDS | _BEFORE_WORDS
    | {"between", "followed", "rank", "ranks", "ranked", "differs", "curved", "positive", "negative"}
)
_SPELLED = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "eleven": 11, "twelve": 12}


def _claim_words(claim: Claim | None) -> set[str]:
    if claim is None:
        return set()
    words: set[str] = set()
    sentence = render_claim(claim)
    if sentence:
        words |= set(re.findall(r"[a-z]+", sentence.lower()))
    gcol = (claim.group_column or "").lower()
    if gcol:
        words |= {gcol, gcol + "s", gcol + "es", gcol + "al", gcol + "ally"}
    for name in claim.columns.values():
        words |= set(re.findall(r"[a-z]+", name.lower()))
    # the words of the group labels themselves (a label may be several words)
    for g in claim.groups:
        words |= set(re.findall(r"[a-z]+", g.label.lower()))
    return words


def _allowlist_problem(t: _Text, claim: Claim | None) -> str | None:
    allowed = _FUNCTION_WORDS | _NEUTRAL_WORDS | _ADVISORY_WORDS | _LEXICON_WORDS | _claim_words(claim)
    words = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[a-z]+", t.norm)]
    n_groups = len(claim.groups) if claim is not None else 0
    for i, (word, start, end) in enumerate(words):
        if word in allowed:
            continue
        if word in _SPELLED:
            # "all four regions": a count of the claim's own groups, verified
            nxt = words[i + 1][0] if i + 1 < len(words) else ""
            nouns = _group_nouns(claim) if claim is not None else set()
            if nxt in nouns and _SPELLED[word] == n_groups:
                continue
        return f"'{word}' is not a word the backend can verify"
    return None
