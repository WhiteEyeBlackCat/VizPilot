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


# --- the semantic check -------------------------------------------------------------

_CAUSAL = re.compile(
    r"\b(?:cause[sd]?|causing|causal(?:ly)?|drive[sn]?|driving|drove|driven\s+by|lead(?:s|ing)?\s+to|led\s+to|"
    r"result(?:s|ed|ing)?\s+in|because\s+of|due\s+to|thanks\s+to|owing\s+to|as\s+a\s+result|"
    r"make[s]?\s+\w+\s+(?:higher|lower|more|less|rise|fall|increase|decrease|grow|go\s+up|go\s+down)|"
    r"affect(?:s|ed|ing)?|influenc(?:e|es|ed|ing)|impact(?:s|ed|ing)?|determin(?:e|es|ed|ing)|"
    r"boost(?:s|ed|ing)?|trigger(?:s|ed|ing)?|produce[sd]?|responsible\s+for|consequence)\b",
    re.IGNORECASE,
)

# comparatives: polarity +1 (higher) / -1 (lower); "verb" forms compare their
# subject to their object without "than"
_COMPARATIVE = re.compile(
    r"\b(?P<word>higher|more|greater|larger|bigger|above|exceed(?:s|ed|ing)?|outsell(?:s|ing)?|outsold|"
    r"outperform(?:s|ed|ing)?|surpass(?:es|ed|ing)?|beat(?:s)?|tops|"
    r"lower|less|fewer|smaller|below|trail(?:s|ed|ing)?|lag(?:s|ged|ging)?|underperform(?:s|ed|ing)?|"
    r"stronger|weaker|steeper|flatter|tighter|looser|wider|narrower)\b",
    re.IGNORECASE,
)
_HIGH_WORDS = {
    "higher", "more", "greater", "larger", "bigger", "above", "stronger", "steeper", "tighter", "wider",
}
_VERB_COMPARATIVES = ("above", "below", "exceed", "outsell", "outsold", "outperform", "surpass", "beat", "tops", "trail", "lag", "underperform")
_LOW_PREFIXES = ("lower", "less", "fewer", "smaller", "below", "trail", "lag", "underperform", "weaker", "flatter", "looser", "narrower")
_STRENGTH_WORDS = ("stronger", "weaker", "steeper", "flatter", "tighter", "looser")
_SPREAD = re.compile(r"\b(?:spread|variab\w*|dispersion|wider|narrower|iqr|std|standard\s+deviation|volatil\w*|range)\b", re.IGNORECASE)
_MEDIAN = re.compile(r"\bmedian\b", re.IGNORECASE)
_SUPERLATIVE = re.compile(
    r"\b(?P<word>highest|largest|biggest|greatest|the\s+most|maximum|lowest|smallest|fewest|the\s+least|minimum|"
    r"strongest|weakest|steepest|flattest|widest|narrowest)\b",
    re.IGNORECASE,
)
_SUPER_LOW = ("lowest", "smallest", "fewest", "least", "minimum", "weakest", "flattest", "narrowest")
_REFERENCE = re.compile(r"\b(?:than|compared\s+(?:to|with)|relative\s+to|versus|vs\.?)(?=\W|$)", re.IGNORECASE)
_POS = re.compile(
    r"\b(?:positive(?:ly)?|increas(?:e|es|ed|ing)|ris(?:e|es|ing)|rose|grow(?:s|ing|n)?|grew|climb(?:s|ed|ing)?|"
    r"go(?:es)?\s+up|went\s+up|move[s]?\s+together|upward|in\s+step)\b",
    re.IGNORECASE,
)
_NEG = re.compile(
    r"\b(?:negative(?:ly)?|decreas(?:e|es|ed|ing)|declin(?:e|es|ed|ing)|fall(?:s|ing)?|fell|drop(?:s|ped|ping)?|"
    r"go(?:es)?\s+down|went\s+down|downward|invers(?:e|ely)|shrink(?:s|ing)?|shrank)\b",
    re.IGNORECASE,
)
_AFTER = re.compile(r"\b(?:after|since|later|afterwards|recent(?:ly)?|from\s+then\s+on|now)\b", re.IGNORECASE)
_BEFORE = re.compile(r"\b(?:before|earlier|previously|prior)\b", re.IGNORECASE)
_SCOPE_ALL = re.compile(r"\b(?:overall|pooled|aggregate|combined|across\s+all\s+rows)\b", re.IGNORECASE)
_QUANT_NOUN = re.compile(
    r"\b(?P<q>every|each|all)\s+(?:of\s+)?(?:the\s+|its\s+|their\s+)?(?:\d+\s+)?(?P<noun>[a-z][a-z0-9_]*)", re.IGNORECASE
)
_QUANT_WORD = re.compile(
    r"\b(?:always|everywhere|without\s+exception|universal(?:ly)?|invariabl[ey]|consistent(?:ly)?|uniform(?:ly)?|"
    r"same\s+direction|regardless\s+of|across\s+the\s+board|in\s+all\s+cases)\b",
    re.IGNORECASE,
)
_ALL_OTHER = re.compile(r"\b(?:every|all(?:\s+the)?|any)\s+other\b", re.IGNORECASE)
_GENERIC_GROUP_NOUNS = {"group", "groups", "category", "categories", "subgroup", "subgroups", "segment", "segments", "level", "levels", "class", "classes"}
_CLAUSE_BREAK = re.compile(r"[.;!?](?=\s|$)|\b(?:but|while|whereas|although|though|however|yet)\b", re.IGNORECASE)
_SEGMENT_BREAK = re.compile(r"[.;!?,](?=\s|$)|\b(?:but|while|whereas|although|though|however|yet)\b", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")
_WORDTOK = re.compile(r"[A-Za-z0-9_]+")
# labels that are too ambiguous to find as bare words in prose
_STOP = {
    "a", "an", "i", "the", "of", "in", "on", "at", "to", "or", "and", "is", "it", "as", "be", "by", "for", "no",
    "not", "yes", "all", "any", "one", "two", "none", "other", "others", "rest", "high", "low", "higher", "lower",
    "more", "less", "most", "least", "same", "new", "old", "up", "down", "top", "bottom", "total", "average",
    "mean", "median", "group", "groups", "n", "r", "vs", "true", "false", "null", "na", "unknown", "each",
    "every", "than", "with", "per", "from", "this", "that", "these", "those", "s", "t", "d", "m",
}


@dataclass
class _Hit:
    start: int
    end: int
    label: str


def _lower(text: str) -> str:
    # length-preserving case fold (str.lower can change the length of a few
    # characters, which would shift every span)
    out = "".join(ch.lower() if len(ch.lower()) == 1 else ch for ch in text)
    # "e.g." / "i.e." would otherwise read as the labels "e" / "i"
    return re.sub(r"\b(?:e\.g\.|i\.e\.)", lambda m: " " * len(m.group(0)), out)


def _ambiguous(label: str, column_names: set[str]) -> bool:
    low = label.lower()
    if low in _STOP or low in column_names:
        return True
    if re.fullmatch(r"-?\d+(?:\.\d+)?", low):
        return True
    if _COMPARATIVE.fullmatch(low) or _SUPERLATIVE.fullmatch(low) or _POS.fullmatch(low) or _NEG.fullmatch(low):
        return True
    return False


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


def _spans(low: str, breaker: re.Pattern[str]) -> list[tuple[int, int]]:
    cuts = [0] + [m.end() for m in breaker.finditer(low)] + [len(low)]
    return [(a, b) for a, b in zip(cuts, cuts[1:]) if b > a]


def _negated(low: str, start: int) -> bool:
    window = low[max(0, start - 25) : start]
    words = _WORDTOK.findall(window)[-3:]
    return any(w in ("not", "no", "never", "neither", "nor", "without") for w in words) or "n't" in window[-12:]


def _word_gap(low: str, a: int, b: int) -> int:
    lo, hi = (a, b) if a <= b else (b, a)
    return len(_WORDTOK.findall(low[lo:hi]))


def semantic_problem(text: str, claim: Claim | None, column_names: set[str] | None = None) -> str | None:
    """None when every comparative / direction / quantifier / causal word and
    every label-bound number in `text` agrees with the claim."""
    if not text or not text.strip():
        return None
    names = {n.lower() for n in (column_names or set())}
    low = _lower(text)
    m = _CAUSAL.search(low)
    if m:
        return f"causal language '{m.group(0)}' is not supported by an associational test"
    if claim is None:
        return _claimless_problem(low)

    labels = _find_labels(low, claim, names)
    problem = _quantifier_problem(low, claim, names)
    if problem:
        return problem
    for start, end in _spans(low, _CLAUSE_BREAK):
        clause_labels = [h for h in labels if start <= h.start < end]
        problem = _clause_problem(low, start, end, clause_labels, claim)
        if problem:
            return problem
    for start, end in _spans(low, _SEGMENT_BREAK):
        problem = _binding_problem(low, start, end, [h for h in labels if start <= h.start < end], labels, claim)
        if problem:
            return problem
    return None


def _claimless_problem(low: str) -> str | None:
    """Free text with no claim behind it (a bare LLM chart suggestion's
    caption): nothing can be checked, so no word that asserts a fact."""
    for pattern, what in ((_COMPARATIVE, "comparison"), (_SUPERLATIVE, "superlative"), (_QUANT_WORD, "generalisation")):
        m = pattern.search(low)
        if m:
            return f"unverifiable {what} '{m.group(0)}' in unvalidated text"
    m = _QUANT_NOUN.search(low)
    if m:
        return f"unverifiable generalisation '{m.group(0)}' in unvalidated text"
    return None


def _quantifier_problem(low: str, claim: Claim, names: set[str]) -> str | None:
    gcol = (claim.group_column or "").lower()
    group_nouns = set(_GENERIC_GROUP_NOUNS)
    if gcol:
        group_nouns |= {gcol, gcol + "s", gcol + "es"}
    m = _QUANT_WORD.search(low)
    if m and not claim.universal_ok:
        return f"'{m.group(0)}' is not supported: the evidence does not show one consistent direction in every group"
    for m in _QUANT_NOUN.finditer(low):
        noun = m.group("noun").lower()
        before = low[max(0, m.start() - 4) : m.start()]
        if before.strip().endswith("of") and claim.comparisons_ok:
            continue  # "the highest of all segments": a superlative's scope, checked as a superlative
        if noun == "other" and claim.comparisons_ok:
            continue  # "outsells every other segment": checked as a top / bottom comparison
        if noun in group_nouns or noun.rstrip("s") in group_nouns:
            if claim.universal_ok:
                continue
            return f"'{m.group(0)}' is not supported: the claim was not established in every {claim.group_column or 'group'}"
        return f"unsupported generalisation '{m.group(0)}': '{noun}' is not a grouping the evidence covers"
    return None


def _comparatives(low: str, start: int, end: int) -> list[tuple[int, int, int, str]]:
    """(position, end, polarity, word) for the comparatives in a clause."""
    out = []
    for m in _COMPARATIVE.finditer(low, start, end):
        word = m.group("word").lower()
        polarity = -1 if word.startswith(_LOW_PREFIXES) else 1
        out.append((m.start(), m.end(), polarity, word))
    return out


def _dimension(clause: str, words: list[str]) -> str:
    if any(w.startswith(_STRENGTH_WORDS) for w in words):
        return "strength"
    if _SPREAD.search(clause):
        return "spread"
    return "level"


def _clause_problem(low: str, start: int, end: int, labels: list[_Hit], claim: Claim) -> str | None:
    clause = low[start:end]
    comps = _comparatives(low, start, end)
    supers = [
        (m.start(), m.end(), -1 if m.group("word").lower().split()[-1] in _SUPER_LOW else 1, m.group("word").lower())
        for m in _SUPERLATIVE.finditer(low, start, end)
    ]
    pos = [(m.start(), 1) for m in _POS.finditer(low, start, end)]
    neg = [(m.start(), -1) for m in _NEG.finditer(low, start, end)]
    for at, _end, _p, word in comps + supers:
        if _negated(low, at):
            return f"negated comparison '{word}' cannot be verified"
    for at, _p in pos + neg:
        if _negated(low, at):
            return "negated direction word cannot be verified"

    relationship = claim.kind in ("grouped_relationship", "sign_flip", "slope_difference")
    signs = sorted(pos + neg)
    if len(signs) == 2 and claim.direction != "nonmonotone" and (relationship or claim.kind == "nonlinear_relationship"):
        # "w falls as z rises": one word moves x, the other y — the
        # relationship's sign is their product
        signs = [(signs[0][0], signs[0][1] * signs[1][1])]
    # comparatives that are really direction statements: "higher z comes with
    # higher w" (two comparatives, no reference) on a relationship / curve, or
    # "higher after ... than before" on a time pattern
    if comps and claim.kind in ("grouped_relationship", "sign_flip", "slope_difference", "nonlinear_relationship"):
        level = [c for c in comps if not c[3].startswith(_STRENGTH_WORDS)]
        if level:
            if len(level) == 2 and not _REFERENCE.search(clause):
                signs.append((level[1][0], level[0][2] * level[1][2]))
                comps = [c for c in comps if c not in level]
            else:
                return f"comparison '{level[0][3]}' cannot be bound to a measured quantity of this relationship"
    if comps and claim.kind == "time_pattern":
        after, before = bool(_AFTER.search(clause)), bool(_BEFORE.search(clause))
        if after == before:
            return f"comparison '{comps[0][3]}' over time does not say which period is meant"
        for at, _e, polarity, _w in comps:
            signs.append((at, polarity if after else -polarity))
        comps = []

    if signs:
        problem = _sign_problem(low, clause, signs, labels, claim)
        if problem:
            return problem

    if comps or supers:
        if not claim.comparisons_ok:
            word = (comps or supers)[0][3]
            return f"comparison '{word}' cannot be verified for a {claim.kind} finding"
        words = [c[3] for c in comps] + [s[3] for s in supers]
        dim = _dimension(clause, words)
        if relationship:
            if dim != "strength":
                return f"comparison '{words[0]}' between groups of a relationship is not a measured quantity"
            keys = ["abs_corr"]
        elif dim == "strength":
            return f"'{words[0]}' does not apply to a {claim.kind} finding"
        elif dim == "spread":
            keys = ["iqr"] if any(_num(_value(g, "iqr")) for g in claim.groups) else ["std"]
        elif _MEDIAN.search(clause):
            keys = ["median"]
        elif claim.kind == "distribution_difference":
            keys = ["median", "mean"]  # "higher" on a distribution: both locations must agree
        else:
            keys = [claim.order_key or "mean"]
        for key in keys:
            problem = _order_problem(low, start, end, comps, supers, labels, claim, key)
            if problem:
                return problem
    return None


def _sign_problem(low: str, clause: str, signs: list[tuple[int, int]], labels: list[_Hit], claim: Claim) -> str | None:
    if not claim.signs_ok:
        return f"direction word cannot be verified for a {claim.kind} finding"
    if claim.direction == "nonmonotone":
        polarities = {p for _, p in signs}
        if len(polarities) == 1:
            return f"a single direction is stated for a {claim.shape} curve"
        return None
    if claim.simpson and _SCOPE_ALL.search(clause):
        return "direction stated for the pooled data, which runs against every group"
    if claim.direction in ("positive", "negative") and claim.kind not in ("sign_flip",):
        expected = 1 if claim.direction == "positive" else -1
        bound = {h.label: _nearest_sign(h, signs) for h in labels}
        for at, polarity in signs:
            if polarity != expected:
                return f"direction word contradicts the measured {claim.direction} direction"
        for label, polarity in bound.items():
            fact = next((g for g in claim.groups if g.label == label), None)
            if fact is not None and fact.sign() is not None and fact.sign() != polarity:
                return f"direction stated for {label} does not match its correlation"
        return None
    # mixed directions: each direction word must be bound to the groups it holds for
    if not labels:
        return "a single direction is stated for a relationship whose direction differs between groups"
    for h in labels:
        polarity = _nearest_sign(h, signs)
        fact = next((g for g in claim.groups if g.label == h.label), None)
        if fact is None or fact.sign() != polarity:
            return f"direction stated for {h.label} does not match its correlation"
    return None


def _nearest_sign(h: _Hit, signs: list[tuple[int, int]]) -> int:
    return min(signs, key=lambda s: (abs(s[0] - h.start), s[0] > h.start))[1]


def _order_problem(
    low: str,
    start: int,
    end: int,
    comps: list[tuple[int, int, int, str]],
    supers: list[tuple[int, int, int, str]],
    labels: list[_Hit],
    claim: Claim,
    key: str,
) -> str | None:
    ordered = claim.ordered(key)
    if len(ordered) < 2:
        return f"no per-group {key} to compare"
    value = {g.label: float(_value(g, key)) for g in ordered}  # type: ignore[arg-type]
    middle = sorted(value.values())[len(value) // 2] if len(value) % 2 else (
        sum(sorted(value.values())[len(value) // 2 - 1 : len(value) // 2 + 1]) / 2
    )
    if comps:
        polarities = {c[2] for c in comps}
        if len(polarities) > 1:
            return "comparisons of opposite direction in one clause cannot be verified"
        polarity = polarities.pop()
        ref = _REFERENCE.search(low, start, end)
        if ref is not None:
            marker = ref.group(0).lower()
            if marker == "than":
                subjects = [h for h in labels if h.start < ref.start()]
                refs = [h for h in labels if h.start > ref.start()]
            else:
                after = [h for h in labels if h.start > ref.start()]
                refs = after[:1]
                subjects = [h for h in labels if h not in refs]
        elif any(c[3].startswith(_VERB_COMPARATIVES) for c in comps):
            pivot = comps[0][0]
            subjects = [h for h in labels if h.start < pivot]
            refs = [h for h in labels if h.start > pivot]
        else:
            subjects, refs = list(labels), []
        for s in subjects:
            if s.label not in value:
                return f"{s.label} has no {key} to compare"
            for r in refs:
                if r.label == s.label or r.label not in value:
                    continue
                if (value[s.label] - value[r.label]) * polarity <= 0:
                    return f"comparison of {s.label} and {r.label} contradicts the {key} order"
            if not refs and _ALL_OTHER.search(low, start, end):
                # "e outsells every other segment" is a top / bottom claim
                want = ordered[0].label if polarity > 0 else ordered[-1].label
                if value[s.label] != value[want]:
                    return f"{s.label} is not {'above' if polarity > 0 else 'below'} every other group in {key}"
            elif not refs and (value[s.label] - middle) * polarity <= 0:
                return f"{s.label} is not {'above' if polarity > 0 else 'below'} the other groups in {key}"
        if not subjects:
            for r in refs:
                if r.label in value and (middle - value[r.label]) * polarity <= 0:
                    return f"the other groups are not {'above' if polarity > 0 else 'below'} {r.label} in {key}"
    if supers and labels:
        top, bottom = ordered[0].label, ordered[-1].label
        distinct: list[_Hit] = []
        for h in labels:
            if all(h.label != d.label for d in distinct):
                distinct.append(h)
        if len(distinct) == len(supers):
            # "highest in e and lowest in b" / "e has the highest, b the lowest":
            # as many superlatives as labels pair up in order of mention
            pairs = list(zip(supers, distinct))
        else:
            # otherwise each superlative binds the label nearest to it
            pairs = []
            for s in supers:
                h = min(labels, key=lambda h: (_word_gap(low, s[0], h.start), h.start < s[0]))
                if _word_gap(low, s[0], h.start) <= 8:
                    pairs.append((s, h))
        for s, h in pairs:
            want = top if s[2] > 0 else bottom
            if h.label != want and value.get(h.label) != value.get(want):
                return f"'{s[3]}' is bound to {h.label}, but the {key} {'top' if s[2] > 0 else 'bottom'} is {want}"
    return None


def _binding_problem(
    low: str, start: int, end: int, labels: list[_Hit], all_labels: list[_Hit], claim: Claim
) -> str | None:
    """A number next to a group label must be that group's own number."""
    if not labels:
        return None
    from .service import _number_supported  # the number gate's own rounding rule

    inside = [(h.start, h.end) for h in all_labels]
    facts = {g.label: g for g in claim.groups}
    others_of = lambda label: {x for g in claim.groups if g.label != label for x in g.own_numbers()}  # noqa: E731
    paired: set[int] = set()
    # "e is higher than a (79.1 vs 50.5)": a parenthesis with several numbers
    # pairs them, in order, with the labels mentioned before it
    for p in re.finditer(r"\(([^()]*)\)", low[start:end]):
        p_start = start + p.start()
        numbers = [n for n in _NUMBER.finditer(low, p_start, start + p.end()) if not any(a <= n.start() < b for a, b in inside)]
        if len(numbers) < 2:
            continue
        before: list[str] = []
        for h in labels:
            if h.start < p_start and h.label not in before:
                before.append(h.label)
        if len(before) != len(numbers):
            continue  # no one-to-one pairing: the nearest-label rule below decides
        for n, label in zip(numbers, before):
            paired.add(n.start())
            token = n.group(0)
            if label in facts and not _number_supported(token, set(facts[label].own_numbers())) and _number_supported(token, others_of(label)):
                return f"number {token} belongs to another group, not {label}"
    for m in _NUMBER.finditer(low, start, end):
        if m.start() in paired or any(a <= m.start() < b for a, b in inside):
            continue  # paired above, or the number is the label itself ("segment 2")
        best: tuple[int, bool, _Hit] | None = None
        for h in labels:
            gap = _word_gap(low, h.end, m.start()) if h.start < m.start() else _word_gap(low, m.end(), h.start)
            after = h.start > m.start()
            if gap <= 4 and (best is None or (gap, after) < (best[0], best[1])):
                best = (gap, after, h)
        if best is None:
            continue
        label = best[2].label
        fact = facts.get(label)
        if fact is None:
            continue
        token = m.group(0)
        if _number_supported(token, set(fact.own_numbers())):
            continue
        if _number_supported(token, others_of(label)):
            return f"number {token} belongs to another group, not {label}"
    return None
