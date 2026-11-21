"""
Validation, Security, Sanity, and Deduplication Filters for Generated TestCases.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Set, Tuple

from .models import RESERVED_TEST_CASE_KEYS, CheckSpec, TestCase

if TYPE_CHECKING:
    from ..evaluation.check_suite import CheckSuiteMetric

# Conservative static detector for nested quantifiers prone to catastrophic backtracking (ReDoS),
# e.g., `(a+)+`, `([a-z]*)*`, `(a|aa)+` nested inside a repeated group.
_NESTED_QUANTIFIER_RE = re.compile(r"\((?:[^()\\]|\\.)*[+*](?:[^()\\]|\\.)*\)[+*{]")


def is_safe_regex(pattern: str, *, max_length: int = 512) -> Tuple[bool, str]:
    """Validate that a regex compiles cleanly and avoids obvious ReDoS patterns."""
    if not isinstance(pattern, str) or not pattern:
        return False, "empty regex pattern"
    if len(pattern) > max_length:
        return False, f"regex exceeds max_length={max_length}"
    if _NESTED_QUANTIFIER_RE.search(pattern):
        return False, "regex contains nested quantifiers (potential ReDoS)"
    try:
        re.compile(pattern)
    except re.error as exc:
        return False, f"regex compile error: {exc}"
    return True, ""


def normalize_input_text(inputs: Dict[str, Any]) -> str:
    """Normalize a case's inputs dict into lowercase single-spaced text for deduplication."""
    parts = [f"{k}={inputs[k]}" for k in sorted(inputs.keys())]
    joined = " ".join(parts).lower()
    joined = re.sub(r"[^\w\s]", " ", joined)
    return re.sub(r"\s+", " ", joined).strip()


def tokenize_for_jaccard(text: str) -> Set[str]:
    """Tokenize normalized text into a set of alphanumeric tokens for Jaccard similarity."""
    return set(re.findall(r"\w+", text.lower()))


def token_jaccard(tokens_a: Set[str], tokens_b: Set[str]) -> float:
    """Compute Jaccard similarity between two token sets in `[0.0, 1.0]`."""
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    union = tokens_a | tokens_b
    return float(len(tokens_a & tokens_b) / len(union))


@dataclass
class FilterStats:
    """Audit counts of candidate cases accepted or rejected by each filter stage."""

    candidates_seen: int = 0
    accepted: int = 0
    dropped_schema: int = 0
    dropped_template_vars: int = 0
    dropped_security: int = 0
    dropped_judge_disabled: int = 0
    dropped_sanity: int = 0
    dropped_duplicate: int = 0
    reasons: Dict[str, int] = field(default_factory=dict)

    def record_drop(self, bucket: str, reason: str = "") -> None:
        setattr(self, bucket, getattr(self, bucket) + 1)
        if reason:
            self.reasons[reason] = self.reasons.get(reason, 0) + 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidates_seen": self.candidates_seen,
            "accepted": self.accepted,
            "dropped_schema": self.dropped_schema,
            "dropped_template_vars": self.dropped_template_vars,
            "dropped_security": self.dropped_security,
            "dropped_judge_disabled": self.dropped_judge_disabled,
            "dropped_sanity": self.dropped_sanity,
            "dropped_duplicate": self.dropped_duplicate,
        }


def _stringify_reference(val: Any) -> Optional[str]:
    """Serialize a reference output cleanly (preserving valid JSON for dict/list values)."""
    if val is None:
        return None
    if isinstance(val, (dict, list)):
        return json.dumps(val, ensure_ascii=False)
    return str(val)


def check_deterministic_sanity(
    case: TestCase,
    *,
    metric: Optional[CheckSuiteMetric] = None,
) -> Tuple[bool, str]:
    """Verify deterministic checks on `case` are both passable and failable.

    - Pure `llm_judge` items are skipped here (never invokes LLM judges during filtering).
    - Passability: requires `reference_pass` (or `target` fallback) to score 1.0 on all
      deterministic checks.
    - Failability:
      - If `reference_fail` is provided, it must score < 1.0.
      - If any positive check exists (`kind != 'not_contains'`), empty string `""` and
        an input echo (when distinct from the valid reference answer) must score < 1.0.
      - If the deterministic suite contains ONLY `not_contains` checks, `reference_fail`
        is mandatory and must score < 1.0 (so `not_contains` is never erroneously dropped
        for passing on `""`).
    """
    det_checks = [c for c in case.checks if c.kind != "llm_judge"]
    if not det_checks:
        return True, ""

    if metric is not None:
        evaluator = metric
    else:
        from ..evaluation.check_suite import CheckSuiteMetric

        evaluator = CheckSuiteMetric(allow_custom_checks=False)
    probe_ctx: Dict[str, Any] = {
        "checks": [c.model_dump(mode="json") for c in det_checks]
    }

    # 1. Passability probe
    pass_candidate: Optional[str] = (
        case.reference_pass
        if case.reference_pass is not None
        else _stringify_reference(case.target)
    )
    if pass_candidate is None:
        return False, "missing reference_pass/target to verify passability"

    try:
        pass_score = evaluator.compute(
            pass_candidate, target=case.target, input_data=probe_ctx
        )
    except Exception as exc:
        return False, f"passability probe raised: {exc}"
    if pass_score < 1.0 - 1e-9:
        return (
            False,
            f"reference_pass failed deterministic checks (score={pass_score:.3f})",
        )

    # 2. Failability probe
    has_positive_check = any(c.kind != "not_contains" for c in det_checks)
    if case.reference_fail is not None:
        fail_score = evaluator.compute(
            case.reference_fail, target=case.target, input_data=probe_ctx
        )
        if fail_score >= 1.0 - 1e-9:
            return False, "reference_fail passed all deterministic checks"
    elif not has_positive_check:
        return False, "not_contains-only check suite requires reference_fail"

    if has_positive_check:
        # Empty string must never satisfy a positive check suite (unless valid answer is empty)
        if pass_candidate.strip() != "":
            empty_score = evaluator.compute(
                "", target=case.target, input_data=probe_ctx
            )
            if empty_score >= 1.0 - 1e-9:
                return False, "deterministic checks pass empty string"

        input_echo = " ".join(str(v) for v in case.inputs.values()).strip()
        if input_echo and input_echo.lower() != pass_candidate.strip().lower():
            echo_score = evaluator.compute(
                input_echo, target=case.target, input_data=probe_ctx
            )
            if echo_score >= 1.0 - 1e-9:
                return False, "deterministic checks pass raw input echo"

    return True, ""


class CaseFilterPipeline:
    """Stateful filter pipeline enforcing template validity, security, sanity, and deduplication."""

    def __init__(
        self,
        *,
        required_vars: Sequence[str],
        judge_fallback: bool = True,
        dedup_jaccard_threshold: float = 0.9,
        existing_cases: Optional[Sequence[TestCase]] = None,
    ):
        if not (0.0 < dedup_jaccard_threshold <= 1.0):
            raise ValueError(
                f"dedup_jaccard_threshold must lie in (0, 1], got {dedup_jaccard_threshold}"
            )
        from ..evaluation.check_suite import CheckSuiteMetric

        self.required_vars: Set[str] = {str(v) for v in required_vars}
        self.judge_fallback = bool(judge_fallback)
        self.dedup_jaccard_threshold = float(dedup_jaccard_threshold)
        self._metric = CheckSuiteMetric(allow_custom_checks=False)
        self.stats = FilterStats()

        self._seen_ids: Set[str] = set()
        self._seen_norm_texts: Set[str] = set()
        self._seen_token_sets: List[Set[str]] = []

        if existing_cases:
            for c in existing_cases:
                self._register_seen(c)

    def _register_seen(self, case: TestCase) -> None:
        self._seen_ids.add(case.id)
        norm = normalize_input_text(case.inputs)
        self._seen_norm_texts.add(norm)
        self._seen_token_sets.append(tokenize_for_jaccard(norm))

    def _is_duplicate(self, case: TestCase) -> Tuple[bool, str]:
        if case.id in self._seen_ids:
            return True, "duplicate case id"
        norm = normalize_input_text(case.inputs)
        if norm in self._seen_norm_texts:
            return True, "exact normalized input duplicate"
        tokens = tokenize_for_jaccard(norm)
        for prev_tokens in self._seen_token_sets:
            if token_jaccard(tokens, prev_tokens) >= self.dedup_jaccard_threshold:
                return (
                    True,
                    f"near-duplicate input (Jaccard >= {self.dedup_jaccard_threshold})",
                )
        return False, ""

    def filter_candidate(
        self,
        raw_case: Dict[str, Any],
        *,
        kind: str,
        intended_factor: Optional[str] = None,
        source: str = "synthetic",
    ) -> Optional[TestCase]:
        """Validate, sanitize, sanity-check, and deduplicate one raw candidate dict."""
        self.stats.candidates_seen += 1

        if not isinstance(raw_case, dict):
            self.stats.record_drop("dropped_schema", "candidate is not a dict")
            return None

        inputs = raw_case.get("inputs")
        if not isinstance(inputs, dict):
            self.stats.record_drop("dropped_schema", "missing or non-dict inputs")
            return None

        # 1. Reserved keys & required template variables
        if set(inputs.keys()) & RESERVED_TEST_CASE_KEYS:
            self.stats.record_drop("dropped_security", "reserved input key")
            return None
        missing_vars = [
            v
            for v in self.required_vars
            if v not in inputs
            or inputs[v] is None
            or (isinstance(inputs[v], str) and not inputs[v].strip())
        ]
        if missing_vars:
            self.stats.record_drop(
                "dropped_template_vars", f"missing vars: {sorted(missing_vars)}"
            )
            return None

        # 2. Check parsing, security, ReDoS, and judge_fallback enforcement
        raw_checks = raw_case.get("checks") or []
        target = raw_case.get("target")
        if not raw_checks and target is not None:
            raw_checks = [{"kind": "exact", "value": None}]
        if not isinstance(raw_checks, list) or not raw_checks:
            self.stats.record_drop("dropped_schema", "missing checks and target")
            return None

        validated_checks: List[CheckSpec] = []
        for raw_chk in raw_checks:
            if not isinstance(raw_chk, dict):
                self.stats.record_drop("dropped_schema", "non-dict check entry")
                return None
            chk_kind = raw_chk.get("kind")
            if chk_kind == "custom" and source == "synthetic":
                self.stats.record_drop(
                    "dropped_security", "custom check forbidden on synthetic"
                )
                return None
            if chk_kind == "llm_judge" and not self.judge_fallback:
                self.stats.record_drop("dropped_judge_disabled", "llm_judge disabled")
                return None
            if chk_kind == "regex":
                ok_re, re_reason = is_safe_regex(str(raw_chk.get("value") or ""))
                if not ok_re:
                    self.stats.record_drop("dropped_security", re_reason)
                    return None
            try:
                validated_checks.append(CheckSpec.model_validate(raw_chk))
            except Exception as exc:
                self.stats.record_drop("dropped_schema", f"invalid CheckSpec: {exc}")
                return None

        ref_pass = _stringify_reference(raw_case.get("reference_pass"))
        ref_fail = _stringify_reference(raw_case.get("reference_fail"))
        try:
            case = TestCase(
                inputs=inputs,
                target=target,
                checks=validated_checks,
                reference_pass=ref_pass,
                reference_fail=ref_fail,
                kind=kind,  # type: ignore[arg-type]
                intended_factor=intended_factor if kind == "targeted" else None,
                source=source,  # type: ignore[arg-type]
            )
        except Exception as exc:
            self.stats.record_drop("dropped_schema", f"invalid TestCase: {exc}")
            return None

        # 3. Passable / failable deterministic sanity filter
        ok_sanity, sanity_reason = check_deterministic_sanity(case, metric=self._metric)
        if not ok_sanity:
            self.stats.record_drop("dropped_sanity", sanity_reason)
            return None

        # 4. Exact & near-duplicate deduplication
        is_dup, dup_reason = self._is_duplicate(case)
        if is_dup:
            self.stats.record_drop("dropped_duplicate", dup_reason)
            return None

        self._register_seen(case)
        self.stats.accepted += 1
        return case
