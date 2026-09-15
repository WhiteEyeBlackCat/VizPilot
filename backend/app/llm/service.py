"""Two-stage insight workflow (stage 17.3) on top of the rule engine.

    rules -> LLM #1 (hypotheses) -> gate -> evidence coverage check
          -> targeted probes for what the tables do not answer
          -> validated hypotheses -> LLM #2 (wording; conditional) -> insights

Division of labour (confirmed with the user): the LLM interprets, proposes
and communicates; the backend measures, validates and decides. No number the
LLM writes is ever used; every claim is re-derived from the profile, the
evidence tables or a deterministic probe. A hypothesis that fails is dropped,
never softened. Zero insights is a legitimate answer. With the LLM disabled or
failing, the output is exactly the rule-engine result (SPEC §13).
"""

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import polars as pl
from pydantic import ValidationError

from ..charts.confidence import Warning, excluded_column_warnings, with_caution
from ..charts.rules import (
    TOP_SCORE_FLOOR,
    Recommendation,
    Tier,
    VerificationLevel,
    apply_confidence,
    apply_derived_caps,
    apply_diversity_caps,
    assign_tiers,
    canonicalize_spec,
    dedup_equivalent,
    definitional_reason,
    derived_column_warnings,
    evaluate_llm_spec,
    near_duplicate_substituted_warning,
    recommend_charts,
    stricter_cap,
)
from ..charts.spec import ChartSpec, validate_spec
from ..probes import MAX_PROBES, ProbeCache, ProbeRejected, ProbeRequest, ProbeResult, run_probes
from ..probes.schemas import PROBE_ROLES
from ..profiling.models import DatasetProfile
from ..probes.engine import MIN_ROWS
from .coverage import (
    CoverageHit,
    ValidatedHypothesis,
    apply_confidence_cap,
    check_coverage,
    corroborated,
    definitional_conflict,
    infer_probe,
    required_roles,
    suggest_chart,
)
from .provider import LLMError, LLMProvider
from .schemas import FinalInsight, FinalResponse, Hypothesis, HypothesisResponse, LLMChartSuggestion, LLMUsage

logger = logging.getLogger(__name__)

MAX_INSIGHTS = 5
MAX_HYPOTHESES = 5  # LLM #1 items considered, by importance
MAX_INSIGHT_CHARS = 300
UNVERIFIED_NOTE = "hypothesis not verified against the data"
FALLBACK_MESSAGE = "AI suggestions unavailable ({reason}); showing rule-based recommendations."
NO_PATTERNS_MESSAGE = "No strong non-definitional and analytically useful patterns were found."
_SNAKE_TOKEN = re.compile(r"\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+\b")
# numbers the LLM writes are never evidence: "(eta² = 0.99)", "r=0.8", "n = 12", "(0.45)"
_STAT_NUMBER = re.compile(
    r"\b(?:adjusted\s+)?(?:eta[²2]?|eta-squared|r[²2]?|rho|corr(?:elation)?|n|p|z|d|ks|spread|effect(?:\s+size)?)"
    r"\s*[=:≈]\s*-?\d+(?:\.\d+)?%?",
    re.IGNORECASE,
)
_PAREN_NUMBER = re.compile(r"\s*\([^()]*\d[^()]*\)")
_EMPTY_PAREN = re.compile(r"\s*\(\s*[,;:\s]*\)")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]*")

SpecKey = tuple[str | None, ...]
Supported = Literal["strong", "weak", "unverified"]

_TIER_CAP: dict[VerificationLevel, Tier | None] = {
    "strong": None,
    "neutral": None,  # no testable hypothesis -> rules fixed score, no cap
    "weak": "secondary",
    "unverified": "exploratory",
}


def _key(spec: ChartSpec) -> SpecKey:
    return (spec.type, spec.x, spec.y, spec.group_by)


def _supported_label(level: VerificationLevel) -> Supported:
    # "neutral" charts (heatmap/histogram) carry nothing to verify; the badge
    # must not read "unverified" (critique #3) — "weak" is the honest middle
    return {"strong": "strong", "weak": "weak", "neutral": "weak", "unverified": "unverified"}[level]


class RecommendationService:
    def __init__(self, provider: LLMProvider, probe_cache: ProbeCache | None = None) -> None:
        self._provider = provider
        self._probe_cache = probe_cache or ProbeCache()
        # merged results per (dataset_id, llm_used); profiles are immutable so
        # entries never expire (D5). Per-key locks prevent duplicate LLM calls
        # for the same key without serializing unrelated requests.
        self._cache: dict[tuple[str, bool], dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._key_locks: dict[tuple[str, bool], threading.Lock] = {}

    def get(
        self,
        profile: DatasetProfile,
        use_llm: bool,
        df: pl.DataFrame | None = None,
        include_debug: bool = False,
    ) -> dict[str, Any]:
        """`df` is only needed for targeted probes (the LLM path); without it,
        hypotheses that need a probe are dropped. The cached payload carries
        the workflow trace under "debug"; it is stripped unless asked for."""
        cache_key = (profile.dataset_id, use_llm)
        cached = self._cache.get(cache_key)
        if cached is None:
            # per-key lock: an in-flight LLM build must not block the llm=false
            # escape hatch or other datasets
            with self._lock:
                key_lock = self._key_locks.setdefault(cache_key, threading.Lock())
            with key_lock:
                if cache_key not in self._cache:
                    self._cache[cache_key] = self._build(profile, use_llm, df)
                cached = self._cache[cache_key]
        if include_debug:
            return cached
        return {k: v for k, v in cached.items() if k != "debug"}

    def _build(self, profile: DatasetProfile, use_llm: bool, df: pl.DataFrame | None) -> dict[str, Any]:
        rules = recommend_charts(profile)
        # dataset-level notes (columns the rule engine left out for
        # missingness, derived columns it demoted) ride along on every path
        dataset_warnings = excluded_column_warnings(profile) + derived_column_warnings(profile)
        if not use_llm:
            return _shape(rules, [], None, dataset_warnings, None)

        trace = _Trace()
        started = time.perf_counter()
        try:
            response, usage = self._provider.generate_hypotheses(profile, rules)
        except LLMError as exc:
            logger.warning("LLM #1 call failed (%s): %s", exc.category, exc)
            trace.llm_calls = 1
            trace.errors.append(f"llm1: {exc.category}")
            trace.latency_ms["total"] = _ms(started)
            return _shape(rules, [], FALLBACK_MESSAGE.format(reason=exc.category), dataset_warnings, trace)
        trace.record_call("llm1", usage)

        workflow = _Workflow(profile, rules, df, self._probe_cache, trace)
        validated = workflow.validate(response)

        final: FinalResponse | None = None
        if workflow.needs_final_call(validated):
            try:
                final, usage2 = self._provider.finalize_insights(profile, validated)
                trace.record_call("llm2", usage2)
            except LLMError as exc:  # LLM #2 is communication only: fall back to the template
                logger.warning("LLM #2 call failed (%s): %s — using template wording", exc.category, exc)
                trace.llm_calls += 1
                trace.errors.append(f"llm2: {exc.category}")
                final = None

        merged, insights = workflow.finish(validated, final, response)
        message = None
        if not insights:
            if workflow.hypotheses_seen == 0 and workflow.merger.attempted > 0 and workflow.merger.kept == 0:
                message = FALLBACK_MESSAGE.format(reason="all suggestions were invalid")
            else:
                message = NO_PATTERNS_MESSAGE
        trace.latency_ms["total"] = _ms(started)
        trace.validated = len(validated)
        trace.insights = len(insights)
        return _shape(merged, insights, message, dataset_warnings, trace)


# --- workflow trace (debug payload) -------------------------------------------------


@dataclass
class _Trace:
    llm_calls: int = 0
    tokens: dict[str, dict[str, int | None]] = field(default_factory=dict)
    latency_ms: dict[str, float] = field(default_factory=dict)
    prompt_chars: dict[str, int] = field(default_factory=dict)
    hypotheses_proposed: int = 0
    dropped: list[dict[str, Any]] = field(default_factory=list)
    covered: list[dict[str, Any]] = field(default_factory=list)
    probes: list[dict[str, Any]] = field(default_factory=list)
    validated: int = 0
    insights: int = 0
    final_call: bool = False
    wording: list[dict[str, Any]] = field(default_factory=list)  # per validated hypothesis: LLM #1 statement -> final text
    errors: list[str] = field(default_factory=list)

    def record_call(self, name: str, usage: LLMUsage) -> None:
        self.llm_calls += 1
        self.tokens[name] = {"prompt": usage.prompt_tokens, "completion": usage.completion_tokens}
        self.latency_ms[name] = usage.latency_ms
        self.prompt_chars[name] = usage.prompt_chars

    def as_dict(self) -> dict[str, Any]:
        return {
            "llm_calls": self.llm_calls,
            "final_call": self.final_call,
            "tokens": self.tokens,
            "prompt_chars": self.prompt_chars,
            "latency_ms": self.latency_ms,
            "hypotheses_proposed": self.hypotheses_proposed,
            "validated": self.validated,
            "insights": self.insights,
            "covered_by_existing_evidence": self.covered,
            "probes": self.probes,
            "wording": self.wording,
            "dropped": self.dropped,
            "errors": self.errors,
        }


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


def _shape(
    recs: list[Recommendation],
    insights: list[dict[str, Any]],
    message: str | None,
    dataset_warnings: list[Warning],
    trace: _Trace | None,
) -> dict[str, Any]:
    charts = [rec.model_dump() for rec in recs]
    if message is None and not charts:
        message = "No charts could be recommended for this dataset."
    payload = {
        "charts": charts,
        "insights": insights,
        "message": message,
        "warnings": [w.model_dump() for w in dataset_warnings],
    }
    if trace is not None:
        payload["debug"] = trace.as_dict()
    return payload


# --- tolerant parsing -----------------------------------------------------------------


def _parse_suggestion(item: Any) -> LLMChartSuggestion | None:
    if not isinstance(item, dict):
        logger.warning("dropping non-object LLM chart item %r", item)
        return None
    try:
        return LLMChartSuggestion.model_validate(item)
    except ValidationError as exc:
        logger.warning("dropping malformed LLM chart item %r: %s", item, exc)
        return None


def _parse_hypotheses(raw: Any) -> list[Hypothesis]:
    """Tolerant per-item parse. Accepts the stage-8 insight shape too
    (`text` + `chart`) so a model that answers in the older format still
    gets its claims verified; plain strings carry no testable claim and are
    dropped."""
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        if raw:
            logger.warning("dropping malformed hypotheses payload %r", raw)
        return []
    items: list[Hypothesis] = []
    for item in raw:
        if not isinstance(item, dict):
            logger.warning("dropping non-object hypothesis item %r", item)
            continue
        data = dict(item)
        if not data.get("statement") and isinstance(data.get("text"), str):
            data["statement"] = data.pop("text")
        try:
            hypothesis = Hypothesis.model_validate(data)
        except ValidationError as exc:
            logger.warning("dropping malformed hypothesis item %r: %s", item, exc)
            continue
        if hypothesis.statement.strip():
            items.append(hypothesis)
    return items


def _parse_final(raw: Any) -> list[FinalInsight]:
    if not isinstance(raw, list):
        if raw:
            logger.warning("dropping malformed final insights payload %r", raw)
        return []
    items: list[FinalInsight] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            insight = FinalInsight.model_validate(item)
        except ValidationError as exc:
            logger.warning("dropping malformed final insight %r: %s", item, exc)
            continue
        if insight.text.strip():
            items.append(insight)
    return items


# --- chart merge (stage 8, unchanged semantics) ----------------------------------------


class _Merger:
    def __init__(self, profile: DatasetProfile, rules: list[Recommendation]) -> None:
        self.profile = profile
        self.by_key = {_key(rec.spec): rec for rec in rules}
        self.added: dict[SpecKey, Recommendation] = {}
        self.llm_priority: dict[SpecKey, int] = {}
        self.attempted = 0
        self.kept = 0

    def integrate(self, suggestion: LLMChartSuggestion) -> tuple[SpecKey, VerificationLevel] | None:
        """Validates, evidence-scores and merges one LLM chart. Returns its
        dedup key and verification level, or None when dropped."""
        self.attempted += 1
        if not suggestion.title:
            logger.warning("dropping LLM chart without a title: %r", suggestion)
            return None
        try:
            spec = ChartSpec(
                title=suggestion.title,
                type=suggestion.type,  # type: ignore[arg-type]  # Literal rejects bad values
                x=suggestion.x,
                y=suggestion.y,
                group_by=suggestion.group_by,
                aggregation=suggestion.aggregation,  # type: ignore[arg-type]
                reason=suggestion.reason,
            )
        except ValidationError as exc:
            logger.warning("dropping LLM chart '%s': %s", suggestion.title, exc)
            return None
        return self.integrate_spec(spec, suggestion.priority, suggestion.reason)

    def integrate_spec(self, spec: ChartSpec, priority: int, reason: str) -> tuple[SpecKey, VerificationLevel] | None:
        # stage 14: a near-duplicate column (atemp) is mapped to its
        # representative (temp) before validation, so the chart dedups with
        # the rules chart it restates; x == y after the mapping means the
        # chart only showed the duplication and is dropped
        canonical, substitutions = canonicalize_spec(spec, self.profile)
        if canonical is None:
            logger.warning("dropping LLM chart '%s': near-duplicate pair %s", spec.title, substitutions)
            return None
        spec = canonical
        errors = validate_spec(spec, self.profile)
        if errors:
            logger.warning("dropping LLM chart '%s': %s", spec.title, "; ".join(errors))
            return None

        score, level = evaluate_llm_spec(spec, self.profile)
        key = _key(spec)
        if key in self.by_key:
            # duplicate of a rules chart: keep the rules spec (and its
            # evidence score), adopt the LLM reason — re-applying the
            # confidence caution the rules chart already carries
            target = self.by_key[key]
            if reason:
                # stage 13: the definitional note survives the wording swap
                wording = definitional_reason(spec, self.profile, reason) or reason
                target.spec.reason = with_caution(wording, target.warnings)
        else:
            if key not in self.added:
                if level == "unverified":
                    note = f" ({UNVERIFIED_NOTE})"
                    spec.reason = (spec.reason + note) if spec.reason else UNVERIFIED_NOTE
                rec = Recommendation(spec=spec, score=score, source="llm", tier_cap=_TIER_CAP[level])
                apply_confidence([rec], self.profile)  # same chain as the rules charts
                apply_derived_caps([rec], self.profile)
                if substitutions:
                    rec.warnings = rec.warnings + [near_duplicate_substituted_warning(substitutions)]
                self.added[key] = rec
            target = self.added[key]
        # a hypothesis is only "strong" if it still clears the top floor after
        # the confidence discount (tiny / heavily missing data -> weak)
        if level == "strong" and target.score < TOP_SCORE_FLOOR:
            level = "weak"
            target.tier_cap = stricter_cap(target.tier_cap, _TIER_CAP[level])
        self.llm_priority[key] = min(priority, self.llm_priority.get(key, priority))
        self.kept += 1
        return key, level


# --- the workflow ----------------------------------------------------------------------


@dataclass
class _Candidate:
    """A hypothesis between the gate and validation."""

    id: int
    hypothesis: Hypothesis
    test_type: str | None  # None: chart-only claim, verified through evaluate_llm_spec
    columns: dict[str, str]
    chart: ChartSpec | None  # canonicalised, validated LLM chart (may be None)
    chart_priority: int


class _Workflow:
    def __init__(
        self,
        profile: DatasetProfile,
        rules: list[Recommendation],
        df: pl.DataFrame | None,
        probe_cache: ProbeCache,
        trace: _Trace,
    ) -> None:
        self.profile = profile
        self.df = df
        self.probe_cache = probe_cache
        self.trace = trace
        self.merger = _Merger(profile, rules)
        self.columns = {c.name: c for c in profile.columns}
        self.hypotheses_seen = 0

    # -- stage 1: gate ------------------------------------------------------------

    def validate(self, response: HypothesisResponse) -> list[ValidatedHypothesis]:
        hypotheses = _parse_hypotheses(response.hypotheses)
        self.hypotheses_seen = len(hypotheses)
        self.trace.hypotheses_proposed = len(hypotheses)
        ranked = sorted(enumerate(hypotheses), key=lambda ih: (-int(ih[1].importance or 0), ih[0]))
        if len(ranked) > MAX_HYPOTHESES:
            for i, h in ranked[MAX_HYPOTHESES:]:
                self._drop(i + 1, h, "beyond the hypothesis cap")
            ranked = ranked[:MAX_HYPOTHESES]

        candidates: list[_Candidate] = []
        for i, h in ranked:
            candidate = self._gate(i + 1, h)
            if candidate is not None:
                candidates.append(candidate)
        rows = self.profile.profiled_rows or self.profile.n_rows
        if rows < MIN_ROWS:
            # a handful of rows cannot support any finding: nothing to
            # verify, nothing to probe, no second LLM call
            for c in candidates:
                self._drop(c.id, c.hypothesis, f"fewer than {MIN_ROWS} rows ({rows})")
            return []

        validated: list[ValidatedHypothesis] = []
        probe_queue: list[_Candidate] = []
        for c in candidates:
            if c.test_type is None:
                # a chart-only claim is verified through the probe the chart
                # implies (bar -> group difference, scatter -> non-linear
                # dependence, ...) with exactly the thresholds a test_needed
                # hypothesis gets; heatmaps, histograms and count bars imply
                # nothing testable and are dropped
                probe = infer_probe(c.chart, self.profile) if c.chart is not None else None
                if probe is None:
                    self._drop(c.id, c.hypothesis, "no testable claim: chart carries nothing to verify")
                    continue
                c.test_type, c.columns = probe
                conflict = definitional_conflict(list(c.columns.values()), self.profile)
                if conflict:
                    self._drop(c.id, c.hypothesis, conflict)
                    continue
            hit = check_coverage(c.test_type, c.columns, self.profile)

            if hit is None:
                probe_queue.append(c)
                continue
            hit = apply_confidence_cap(hit, c.chart or suggest_chart(c.test_type or "", c.columns, self.profile), self.profile)
            if hit.n < MIN_ROWS:
                self._drop(c.id, c.hypothesis, f"fewer than {MIN_ROWS} rows ({hit.n})")
                continue
            if hit.verdict == "fail":
                self._drop(c.id, c.hypothesis, f"existing evidence fails: {hit.effect_label} = {hit.effect_value:.3f}")
                continue
            reason = corroborated(c.test_type or "", c.columns, hit, self.profile)
            if reason:
                self._drop(c.id, c.hypothesis, reason)
                continue
            self.trace.covered.append(
                {"id": c.id, "test": c.test_type, "columns": c.columns, "verdict": hit.verdict, "effect": hit.effect_value}
            )
            validated.append(self._validated(c, hit, "existing_evidence", probe_chart=None))

        validated += self._run_probes(probe_queue)
        validated.sort(key=lambda v: v.id)
        return validated

    def _gate(self, hid: int, h: Hypothesis) -> _Candidate | None:
        """Hallucination gate: every column must exist (near-duplicates are
        mapped to their representative), the probe type must be one the
        backend offers, and the claim must not restate a definition."""
        try:
            roles = self._columns(h)
        except ValueError as exc:
            self._drop(hid, h, str(exc))
            return None
        # the claim text: LLM-written numbers are stripped (they are never
        # evidence), near-duplicate names are mapped to the representative,
        # and a column-like token outside the dataset is a hallucination
        statement = self._clean_statement(h.statement)
        foreign = [t for t in _SNAKE_TOKEN.findall(statement) if t not in self.columns]
        if foreign:
            self._drop(hid, h, f"statement names unknown columns {foreign}")
            return None
        h.statement = statement
        if h.test_needed is not None and h.test_needed not in PROBE_ROLES:
            self._drop(hid, h, f"unknown probe type '{h.test_needed}'")
            return None
        for v in h.variables:
            name = v if isinstance(v, str) else None
            if name is None or self._canonical(name) is None:
                self._drop(hid, h, f"variable '{v}' is not a column of this dataset")
                return None

        chart: ChartSpec | None = None
        chart_priority = 50
        if h.chart is not None:
            suggestion = _parse_suggestion(h.chart)
            chart = self._chart_spec(suggestion) if suggestion is not None else None
            if suggestion is not None:
                chart_priority = suggestion.priority
                if chart is None:
                    # a chart the profile cannot draw — the claim may still be
                    # testable through its probe type; only the chart is lost
                    logger.warning("hypothesis %d: supporting chart rejected", hid)

        test_type = h.test_needed
        columns = roles
        if test_type is not None:
            required, optional = PROBE_ROLES[test_type]
            extra = [r for r in columns if r not in required and r not in optional]
            if extra:
                self._drop(hid, h, f"{test_type} does not accept roles {extra}")
                return None
            missing = [r for r in required_roles(test_type) if r not in columns]
            if missing:
                inferred = infer_probe(chart, self.profile) if chart is not None else None
                if inferred is not None and inferred[0] == test_type:
                    columns = inferred[1]
                else:
                    self._drop(hid, h, f"{test_type} needs roles {list(required_roles(test_type))}; missing {missing}")
                    return None
        axes = (chart.x, chart.y) if chart is not None else ()
        names = list(dict.fromkeys(list(columns.values()) + [n for n in axes if n]))
        conflict = definitional_conflict(names, self.profile)
        if conflict:
            self._drop(hid, h, conflict)
            return None
        if chart is None and test_type is None:
            self._drop(hid, h, "no chart and no probe type: nothing to verify")
            return None
        return _Candidate(hid, h, test_type, columns, chart, chart_priority)

    def _columns(self, h: Hypothesis) -> dict[str, str]:
        raw = h.columns
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise ValueError("columns must be an object of role -> column")
        out: dict[str, str] = {}
        for role, name in raw.items():
            if not isinstance(role, str) or not isinstance(name, str):
                raise ValueError("columns must map role names to column names")
            canonical = self._canonical(name)
            if canonical is None:
                raise ValueError(f"column '{name}' is not a column of this dataset")
            out[role] = canonical
        if len(set(out.values())) != len(out):
            raise ValueError("the same column fills two roles (after near-duplicate mapping)")
        return out

    def _clean_statement(self, text: str) -> str:
        text = _STAT_NUMBER.sub("", text)
        text = _PAREN_NUMBER.sub("", text)
        text = _EMPTY_PAREN.sub("", text)
        for g in self.profile.evidence.near_duplicate_groups:
            for dup in g.duplicates:
                text = re.sub(rf"\b{re.escape(dup)}\b", g.representative, text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        return re.sub(r"\s+([.,;:])", r"\1", text)

    def _canonical(self, name: str) -> str | None:
        if name not in self.columns:
            return None
        for g in self.profile.evidence.near_duplicate_groups:
            if name in g.duplicates:
                return g.representative
        return name

    def _chart_spec(self, suggestion: LLMChartSuggestion) -> ChartSpec | None:
        if not suggestion.title:
            return None
        try:
            spec = ChartSpec(
                title=suggestion.title,
                type=suggestion.type,  # type: ignore[arg-type]
                x=suggestion.x,
                y=suggestion.y,
                group_by=suggestion.group_by,
                aggregation=suggestion.aggregation,  # type: ignore[arg-type]
                reason=suggestion.reason,
            )
        except ValidationError:
            return None
        canonical, _ = canonicalize_spec(spec, self.profile)
        if canonical is None or validate_spec(canonical, self.profile):
            return None
        return canonical

    def _drop(self, hid: int, h: Hypothesis, reason: str) -> None:
        logger.info("dropping hypothesis %d (%r): %s", hid, h.statement[:80], reason)
        self.trace.dropped.append({"id": hid, "statement": h.statement[:120], "reason": reason})

    # -- stage 2: coverage -----------------------------------------------------------

    def _validated(
        self, c: _Candidate, hit: CoverageHit, validation: str, probe_chart: ChartSpec | None
    ) -> ValidatedHypothesis:
        candidates: list[tuple[str, ChartSpec]] = []
        backend_chart = probe_chart or suggest_chart(c.test_type or "", c.columns, self.profile)
        if backend_chart is not None:
            candidates.append((f"h{c.id}c1", backend_chart))
        if c.chart is not None and all(_key(c.chart) != _key(s) for _, s in candidates):
            candidates.append((f"h{c.id}c{len(candidates) + 1}", c.chart))
        return ValidatedHypothesis(
            id=c.id,
            statement=c.hypothesis.statement.strip(),
            reason=c.hypothesis.reason.strip(),
            importance=int(c.hypothesis.importance or 3),
            test_type=c.test_type or "chart",
            columns=dict(c.columns),
            validation=validation,  # type: ignore[arg-type]
            verdict=hit.verdict,  # type: ignore[arg-type]
            effect_label=hit.effect_label,
            effect_value=round(float(hit.effect_value), 6),
            n=int(hit.n),
            evidence_lines=list(hit.evidence_lines),
            evidence=dict(hit.evidence),
            chart_candidates=candidates,
        )

    # -- stage 3: targeted probes ----------------------------------------------------------

    def _run_probes(self, queue: list[_Candidate]) -> list[ValidatedHypothesis]:
        if not queue:
            return []
        runnable: list[_Candidate] = []
        for c in queue:
            if c.test_type == "time_pattern" and c.columns.get("group"):
                self._drop(c.id, c.hypothesis, "a grouped time pattern is an interaction claim the tables do not cover and no probe can test")
                continue
            runnable.append(c)
        if self.df is None:
            for c in runnable:
                self._drop(c.id, c.hypothesis, "needs a probe but no data frame is available")
            return []
        queue = sorted(runnable, key=lambda c: (-int(c.hypothesis.importance or 0), c.id))
        requests: list[tuple[_Candidate, ProbeRequest]] = []
        for c in queue:
            try:
                requests.append((c, ProbeRequest(type=c.test_type, columns=c.columns)))  # type: ignore[arg-type]
            except ValidationError as exc:
                self._drop(c.id, c.hypothesis, f"rejected probe request: {exc.errors()[0].get('msg', exc)}")
        if not requests:
            return []
        outcomes = run_probes(
            self.df, self.profile, [r for _, r in requests], max_probes=MAX_PROBES, cache=self.probe_cache
        )
        validated: list[ValidatedHypothesis] = []
        for (c, _), outcome in zip(requests, outcomes):
            if isinstance(outcome, ProbeRejected):
                self._drop(c.id, c.hypothesis, f"probe rejected: {outcome.reason}")
                self.trace.probes.append({"id": c.id, "type": outcome.type, "columns": outcome.columns, "rejected": outcome.reason})
                continue
            result: ProbeResult = outcome
            self.trace.probes.append(
                {
                    "id": c.id,
                    "type": result.type,
                    "columns": result.columns,
                    "verdict": result.verdict,
                    "effect": result.effect_size,
                    "n": result.n,
                    "cached": result.cached,
                }
            )
            if result.n < MIN_ROWS:
                self._drop(c.id, c.hypothesis, f"fewer than {MIN_ROWS} rows ({result.n})")
                continue
            if result.verdict == "fail":
                self._drop(c.id, c.hypothesis, f"probe failed: {result.effect_label} = {result.effect_size:.3f}")
                continue
            hit = CoverageHit(
                verdict=result.verdict,
                effect_label=result.effect_label,
                effect_value=result.effect_size,
                n=result.n,
                evidence_lines=_probe_lines(result),
                evidence=dict(result.evidence),
                n_min_group=result.n_min_group,
            )
            reason = corroborated(c.test_type or "", c.columns, hit, self.profile)
            if reason:
                self._drop(c.id, c.hypothesis, reason)
                continue
            validated.append(self._validated(c, hit, "probe", probe_chart=result.chart))
        return validated

    # -- stage 4: final wording and charts ------------------------------------------------

    def needs_final_call(self, validated: list[ValidatedHypothesis]) -> bool:
        """LLM #2 only when something new was measured (a probe) or several
        validated findings need one coherent narrative; a single finding
        already answered by the evidence tables is worded by the template."""
        if not validated:
            return False
        return any(v.validation == "probe" for v in validated) or len(validated) >= 2

    def finish(
        self,
        validated: list[ValidatedHypothesis],
        final: FinalResponse | None,
        response: HypothesisResponse,
    ) -> tuple[list[Recommendation], list[dict[str, Any]]]:
        wording = self._final_wording(validated, final) if final is not None else {}
        self.trace.final_call = final is not None

        rows: list[dict[str, Any]] = []
        for v in validated:
            chart_key = self._integrate_candidates(v)
            if chart_key is None:
                self._drop_validated(v, "no supporting chart could be drawn")
                continue
            words = wording.get(v.id)
            text = words["text"] if words else _template(v)
            why = words["why_it_matters"] if words and words["why_it_matters"] else v.reason
            priority = max(1, min(5, int(words["priority"] if words else v.importance)))
            self.trace.wording.append(
                {"id": v.id, "statement": v.statement, "final_text": text, "source": "llm2" if words else "template"}
            )
            rows.append(
                {
                    "text": text[:MAX_INSIGHT_CHARS],
                    "supported": "strong" if v.verdict == "pass" else "weak",
                    "key": chart_key,
                    "validation": v.validation,
                    "why_it_matters": why[:MAX_INSIGHT_CHARS] if why else None,
                    "priority": int(priority),
                    "effect": {"label": v.effect_label, "value": v.effect_value, "n": v.n},
                    "id": v.id,
                }
            )

        for item in response.charts:  # bare chart suggestions, per-item tolerant
            suggestion = _parse_suggestion(item)
            if suggestion is not None:
                self.merger.integrate(suggestion)

        merged, final_priority = self._merge()
        rows.sort(key=lambda r: (-r["priority"], r["id"]))
        insights = [
            {
                "text": r["text"],
                "supported": r["supported"],
                # None when the chart fell to the caps/MAX_CHARTS cut — the
                # insight survives, it just has no chart to link to
                "chart_priority": final_priority.get(r["key"]),
                "validation": r["validation"],
                "why_it_matters": r["why_it_matters"],
                "priority": r["priority"],
                "effect": r["effect"],
            }
            for r in rows[:MAX_INSIGHTS]
        ]
        return merged, insights

    def _integrate_candidates(self, v: ValidatedHypothesis) -> SpecKey | None:
        """The backend chart first, the LLM's own chart as fallback; both go
        through the stage-8 merge (evidence score, caps, canonicalisation)."""
        priority = 6 - max(1, min(5, v.importance))
        for _, spec in v.chart_candidates:
            spec = spec.model_copy()
            reason = spec.reason or v.reason or v.statement
            result = self.merger.integrate_spec(spec, priority, reason)
            if result is not None:
                return result[0]
        return None

    def _drop_validated(self, v: ValidatedHypothesis, reason: str) -> None:
        logger.info("dropping validated hypothesis %d: %s", v.id, reason)
        self.trace.dropped.append({"id": v.id, "statement": v.statement[:120], "reason": reason})

    def _final_wording(
        self, validated: list[ValidatedHypothesis], final: FinalResponse
    ) -> dict[int, dict[str, Any]]:
        """LLM #2 output through its own gate: known hypothesis id, known
        chart id, no column-like token outside the dataset."""
        by_id = {v.id: v for v in validated}
        chart_ids = {cid for v in validated for cid, _ in v.chart_candidates}
        names = set(self.columns)
        out: dict[int, dict[str, Any]] = {}
        for item in _parse_final(final.insights):
            hid = _as_int(item.hypothesis_id)
            if hid is None or hid not in by_id or hid in out:
                logger.warning("LLM #2 insight with unknown or repeated hypothesis_id %r dropped", item.hypothesis_id)
                continue
            if item.chart_id is not None and str(item.chart_id) not in chart_ids:
                logger.warning("LLM #2 insight %d names unknown chart %r — chart id ignored", hid, item.chart_id)
            v = by_id[hid]
            problem = _wording_problem(item.text + " " + item.why_it_matters, v, names)
            if problem:
                logger.warning("LLM #2 insight %d rejected: %s — template used", hid, problem)
                self.trace.errors.append(f"llm2 insight {hid}: {problem}")
                continue
            out[hid] = {
                "text": item.text.strip(),
                "why_it_matters": item.why_it_matters.strip(),
                "priority": max(1, min(5, int(item.priority or by_id[hid].importance))),
            }
        return out

    def _merge(self) -> tuple[list[Recommendation], dict[SpecKey, int]]:
        # merge -> equivalence dedup (stage 14) -> re-apply diversity caps
        # (critique #4) -> display order -> score-based tiers with LLM caps
        # (blocking #2)
        rules = list(self.merger.by_key.values())
        pool, redirect = dedup_equivalent(rules + list(self.merger.added.values()), self.profile)
        capped = apply_diversity_caps(pool)
        llm_priority = self.merger.llm_priority

        def display_key(rec: Recommendation) -> tuple:
            key = _key(rec.spec)
            if key in llm_priority:
                return (0, llm_priority[key], -rec.score, rec.spec.title)
            return (1, -rec.score, rec.spec.type, rec.spec.x or "", rec.spec.y or "")

        merged = sorted(capped, key=display_key)
        for i, rec in enumerate(merged):
            rec.spec.priority = i + 1
        assign_tiers(merged)
        final_priority = {_key(rec.spec): rec.spec.priority for rec in merged}
        return merged, {key: final_priority.get(redirect.get(key, key)) for key in set(llm_priority) | set(final_priority)}


def _wording_problem(text: str, v: ValidatedHypothesis, names: set[str]) -> str | None:
    """LLM #2 may only restate: every number must be one the backend measured
    for this hypothesis (rounded to the digits written), every column name
    must be one the hypothesis is about, and no column-like token may come
    from outside the dataset."""
    foreign = [t for t in _SNAKE_TOKEN.findall(text) if t not in names]
    if foreign:
        return f"unknown tokens {foreign}"
    allowed = set(v.columns.values())
    mentioned = {w for w in _WORD.findall(text) if w in names and w not in allowed}
    if mentioned:
        return f"names columns outside the hypothesis {sorted(mentioned)}"
    evidence_numbers = _numbers_in(v.evidence) | {float(v.effect_value), float(v.n)}
    for token in _NUMBER.findall(text):
        if not _number_supported(token, evidence_numbers):
            return f"number {token} is not in the validated evidence"
    return None


def _numbers_in(value: Any) -> set[float]:
    out: set[float] = set()
    if isinstance(value, bool):
        return out
    if isinstance(value, (int, float)):
        if value == value and abs(value) != float("inf"):
            out.add(float(value))
    elif isinstance(value, dict):
        for item in value.values():
            out |= _numbers_in(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            out |= _numbers_in(item)
    elif isinstance(value, str):
        # ISO dates in change points: their year / month / day are quotable
        for part in _NUMBER.findall(value):
            try:
                out.add(float(part))
            except ValueError:
                pass
    return out


def _number_supported(token: str, evidence: set[float]) -> bool:
    try:
        written = float(token)
    except ValueError:
        return True
    decimals = len(token.split(".")[1]) if "." in token else 0
    for value in evidence:
        for candidate in (value, value * 100.0):  # a ratio may be quoted as a percentage
            if abs(round(candidate, decimals) - written) < 10 ** (-decimals) / 2 + 1e-9:
                return True
    return False


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("hH").isdigit():
        return int(value.strip().lstrip("hH"))
    return None


def _template(v: ValidatedHypothesis) -> str:
    """Deterministic wording when LLM #2 is not called (or failed): the
    statement plus the backend's number."""
    statement = v.statement.rstrip(".")
    return f"{statement} ({v.effect_label}: {v.effect_value:.2f}, n={v.n})."


def _probe_lines(result: ProbeResult) -> list[str]:
    ev = result.evidence
    lines = [f"{result.effect_label} = {result.effect_size:.3f} (n={result.n}, verdict {result.verdict})"]
    groups = ev.get("groups")
    if isinstance(groups, list) and groups and isinstance(groups[0], dict):
        if "mean" in groups[0]:
            listed = sorted(groups, key=lambda g: -g["mean"])[:8]
            lines.append("mean per group: " + ", ".join(f"{g['group']}={g['mean']:.3g} (n={g['n']})" for g in listed))
        elif "corr" in groups[0]:
            lines.append(
                "correlation per group: "
                + ", ".join(f"{g['group']}={g['corr']:.2f} (n={g['n']})" for g in groups if g.get("corr") is not None)
            )
    if isinstance(ev.get("bins"), list) and ev["bins"]:
        lines.append(
            f"shape {ev.get('shape')}; y mean per x bin (low to high): "
            + ", ".join(f"{b['y_mean']:.3g}" for b in ev["bins"])
        )
    if isinstance(ev.get("ks_pair"), list):
        lines.append(f"largest distribution gap between {ev['ks_pair'][0]} and {ev['ks_pair'][1]} (KS {ev.get('max_ks')})")
    change = ev.get("change_point")
    if isinstance(change, dict) and change.get("flagged"):
        lines.append(
            f"level shift at {str(change.get('change_at'))[:10]}: {change.get('before_mean'):.3g} before, "
            f"{change.get('after_mean'):.3g} after"
        )
    if isinstance(ev.get("cells"), list) and ev["cells"]:
        cells = ev["cells"][:8]
        lines.append(
            "cell means: " + ", ".join(f"{c['factor1']}/{c['factor2']}={c['mean']:.3g} (n={c['n']})" for c in cells)
        )
    return lines + [note for note in result.notes[:2]]
