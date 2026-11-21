"""
Scored Test Set Generator: Clause-Role Classification, Baseline & Targeted Generation,
Independent Coverage Tagging, Filtering, Splitting, and Confirmation Refresh.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from datetime import datetime, timezone
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from ..core.factors import Factor, FactorSet, FactorType
from ..core.models import _canonical_json_sha256
from ..runner.client import CallableLLM, LLMClient
from ..template.composer import PromptTemplate
from .filters import CaseFilterPipeline
from .models import ClauseRole, TestCase, TestSet

_CLASSIFY_ROLES_TEMPLATE = """[PROMPTPRISM_TASK:CLASSIFY_ROLES]
Classify each prompt clause into one of two roles:
- "conditional_rule": defines a verifiable output requirement, format contract, edge-case trigger, refusal, or abstention rule that changes what a correct answer looks like on triggered inputs.
- "global_scaffold": a general performance/accuracy scaffold (persona, step-by-step reasoning cue, few-shot demonstrations, self-verification) that does NOT change the definition of a correct answer.

Return ONLY valid JSON:
{"roles": {"<factor_id>": "conditional_rule" | "global_scaffold", ...}}
"""

_BASELINE_GEN_TEMPLATE = """[PROMPTPRISM_TASK:GENERATE_BASELINE]
Generate {n} realistic, diverse test inputs for the task below WITHOUT referencing any optional prompt clause.
For each item, attach deterministic checks first ("exact", "regex", "contains", "not_contains", "json_schema").
{judge_instruction}
Never use "custom" checks. Include "reference_pass" (a valid output that passes every check) and "reference_fail" (a realistic wrong output that fails at least one check).

Return ONLY valid JSON:
{{"cases": [{{"inputs": {{{vars_hint}}}, "target": "<optional reference answer>", "checks": [{{"kind": "exact|regex|contains|not_contains|json_schema{judge_kind}", "value": "...", "weight": 1.0, "description": "..."}}], "reference_pass": "...", "reference_fail": "..."}}]}}
"""

_TARGETED_RULE_TEMPLATE = """[PROMPTPRISM_TASK:GENERATE_TARGETED_RULE]
Generate {n} diverse targeted test inputs for conditional clause `{factor_id}` ({factor_name}) where following Level 1 instead of Level 0 changes what a correct response looks like.
Include a spread of obvious triggers, subtle triggers, and near-misses that look relevant but should NOT change the output.
Attach deterministic checks first encoding the clause's intended specification; {judge_instruction}
Never use "custom" checks. Include "reference_pass" and "reference_fail".

Return ONLY valid JSON:
{{"cases": [{{"inputs": {{{vars_hint}}}, "target": "<optional reference answer>", "checks": [{{"kind": "exact|regex|contains|not_contains|json_schema{judge_kind}", "value": "...", "weight": 1.0, "description": "..."}}], "reference_pass": "...", "reference_fail": "..."}}]}}
"""

_TARGETED_SCAFFOLD_TEMPLATE = """[PROMPTPRISM_TASK:GENERATE_TARGETED_SCAFFOLD]
Generate {n} hard, subtle, or ambiguous task inputs where global scaffold `{factor_id}` ({factor_name}) is hypothesized to improve task accuracy.
CRITICAL: Score strictly on baseline TASK CORRECTNESS. Do NOT create checks that test the scaffold's surface form (never check for reasoning text, persona voice, or verification commentary).
Attach deterministic task checks first; {judge_instruction}
Never use "custom" checks. Include "reference_pass" and "reference_fail".

Return ONLY valid JSON:
{{"cases": [{{"inputs": {{{vars_hint}}}, "target": "<optional reference answer>", "checks": [{{"kind": "exact|regex|contains|not_contains|json_schema{judge_kind}", "value": "...", "weight": 1.0, "description": "..."}}], "reference_pass": "...", "reference_fail": "..."}}]}}
"""

_TAG_COVERAGE_TEMPLATE = """[PROMPTPRISM_TASK:TAG_COVERAGE]
For each test case below, independently identify which prompt clauses could change the correct response or task accuracy on that input.
Return ONLY valid JSON mapping each case id to a list of matching factor_ids:
{"tags": {"<case_id>": ["<factor_id>", ...]}}
"""

GENERATOR_PROMPT_HASH: str = _canonical_json_sha256(
    {
        "classify": _CLASSIFY_ROLES_TEMPLATE,
        "baseline": _BASELINE_GEN_TEMPLATE,
        "targeted_rule": _TARGETED_RULE_TEMPLATE,
        "targeted_scaffold": _TARGETED_SCAFFOLD_TEMPLATE,
        "tagger": _TAG_COVERAGE_TEMPLATE,
    }
)

_SCAFFOLD_NAME_HINTS = (
    "persona",
    "role_framing",
    "few_shot",
    "demo",
    "exemplar",
    "reasoning",
    "cot",
    "chain_of_thought",
    "step_by_step",
    "decomposition",
    "verification",
    "self_verify",
    "self_critique",
)
_RULE_HINTS = (
    "json",
    "schema",
    "format",
    "contract",
    "null",
    "abstain",
    "out_of_scope",
    "refuse",
    "guardrail",
    "grounding",
    "forbid",
    "never ",
    "if no ",
    "when missing",
)


def _heuristic_clause_role(factor: Factor) -> ClauseRole:
    """Deterministic fallback classifier for `conditional_rule` vs. `global_scaffold`."""
    name_desc = f"{factor.name} {factor.description or ''}".lower()
    lvl1 = str(factor.get_level(1).content).lower()

    looks_scaffold_name = any(h in name_desc for h in _SCAFFOLD_NAME_HINTS)
    looks_rule_name = any(h in name_desc for h in _RULE_HINTS)

    # Explicit scaffold names (e.g. verification_pass, few_shot_demos, specialist_persona)
    # remain global_scaffold even if their body mentions verifying format/JSON examples.
    if looks_scaffold_name and not looks_rule_name:
        return "global_scaffold"
    if looks_rule_name or any(h in lvl1 for h in _RULE_HINTS):
        return "conditional_rule"
    if any(
        h in lvl1 for h in ("you are an expert", "think step by step", "example 1:")
    ):
        return "global_scaffold"
    return "conditional_rule"


def _extract_json_payload(raw_text: str) -> Any:
    """Extract a JSON object or array from raw LLM text (handling fences and prose)."""
    text = str(raw_text or "").strip()
    if not text:
        raise ValueError("Empty LLM response")

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    candidate = fenced.group(1).strip() if fenced else text

    try:
        return json.loads(candidate)
    except Exception:
        pass

    # Find first balanced JSON object or array
    start_positions = [idx for idx, ch in enumerate(candidate) if ch in "{["]
    for start in start_positions:
        opener = candidate[start]
        closer = "}" if opener == "{" else "]"
        depth = 0
        in_str = False
        escaped = False
        for end in range(start, len(candidate)):
            ch = candidate[end]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        snippet = candidate[start : end + 1]
                        try:
                            return json.loads(snippet)
                        except Exception:
                            break
    raise ValueError(f"Could not parse JSON payload from LLM response: {text[:200]!r}")


def _extract_cases_list(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("cases", "items", "tests", "examples"):
            val = payload.get(key)
            if isinstance(val, list):
                return [x for x in val if isinstance(x, dict)]
    return []


def _extract_tag_map(payload: Any) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    if isinstance(payload, dict):
        if isinstance(payload.get("tags"), dict):
            for cid, tags in payload["tags"].items():
                if isinstance(tags, list):
                    out[str(cid)] = [str(t) for t in tags]
            return out
        if isinstance(payload.get("results"), list):
            payload = payload["results"]
        else:
            # Flat dict {case_id: [...]}
            for cid, tags in payload.items():
                if isinstance(tags, list):
                    out[str(cid)] = [str(t) for t in tags]
            return out

    if isinstance(payload, list):
        for entry in payload:
            if isinstance(entry, dict) and "id" in entry:
                raw_tags = entry.get("factors") or entry.get("tags") or []
                if isinstance(raw_tags, list):
                    out[str(entry["id"])] = [str(t) for t in raw_tags]
    return out


class TestSetGenerator:
    """Generates, filters, independently tags, splits, and freezes PromptPrism TestSets."""

    __test__ = False

    def __init__(
        self,
        client: Union[LLMClient, Callable[..., Any]],
        *,
        tagger_client: Optional[Union[LLMClient, Callable[..., Any]]] = None,
        n_baseline_discovery: int = 100,
        n_baseline_confirmation: int = 40,
        n_per_factor_discovery: int = 35,
        n_per_factor_confirmation: int = 20,
        max_oversample_factor: float = 2.0,
        batch_size: int = 15,
        tagger_batch_size: int = 20,
        dedup_jaccard_threshold: float = 0.9,
        seed: int = 0,
        judge_fallback: bool = True,
    ):
        for attr_name, val in (
            ("n_baseline_discovery", n_baseline_discovery),
            ("n_baseline_confirmation", n_baseline_confirmation),
            ("n_per_factor_discovery", n_per_factor_discovery),
            ("n_per_factor_confirmation", n_per_factor_confirmation),
        ):
            if val < 0:
                raise ValueError(f"{attr_name} must be >= 0, got {val}")
        if max_oversample_factor < 1.0:
            raise ValueError(
                f"max_oversample_factor must be >= 1.0, got {max_oversample_factor}"
            )
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if tagger_batch_size < 1:
            raise ValueError(f"tagger_batch_size must be >= 1, got {tagger_batch_size}")
        if not (0.0 < dedup_jaccard_threshold <= 1.0):
            raise ValueError(
                f"dedup_jaccard_threshold must lie in (0, 1], got {dedup_jaccard_threshold}"
            )

        self.client: LLMClient = (
            client if isinstance(client, LLMClient) else CallableLLM(client)
        )
        self.tagger_client: LLMClient = (
            self.client
            if tagger_client is None
            else (
                tagger_client
                if isinstance(tagger_client, LLMClient)
                else CallableLLM(tagger_client)
            )
        )
        self.n_baseline_discovery = int(n_baseline_discovery)
        self.n_baseline_confirmation = int(n_baseline_confirmation)
        self.n_per_factor_discovery = int(n_per_factor_discovery)
        self.n_per_factor_confirmation = int(n_per_factor_confirmation)
        self.max_oversample_factor = float(max_oversample_factor)
        self.batch_size = int(batch_size)
        self.tagger_batch_size = int(tagger_batch_size)
        self.dedup_jaccard_threshold = float(dedup_jaccard_threshold)
        self.seed = int(seed)
        self.judge_fallback = bool(judge_fallback)

    @staticmethod
    def _validate_factors(factors: Union[FactorSet, Sequence[Factor]]) -> FactorSet:
        factor_set = factors if isinstance(factors, FactorSet) else FactorSet(factors)
        if len(factor_set) == 0:
            raise ValueError("TestSetGenerator requires at least one Factor.")
        for factor in factor_set:
            if factor.factor_type == FactorType.PARAM:
                raise ValueError(
                    f"Factor {factor.id!r} ({factor.name!r}) has factor_type=PARAM; "
                    "TestSetGenerator only supports text/section prompt clauses."
                )
            if len(factor.levels) != 2:
                raise ValueError(
                    f"Factor {factor.id!r} ({factor.name!r}) has {len(factor.levels)} levels; "
                    "TestSetGenerator currently requires binary (2-level) factors."
                )
        return factor_set

    @staticmethod
    def _factor_alias_lookup(factor_set: FactorSet) -> Dict[str, str]:
        lookup: Dict[str, str] = {}
        for f in factor_set:
            lookup[f.id] = f.id
            lookup[f.id.lower()] = f.id
        for f in factor_set:
            if f.name not in lookup:
                lookup[f.name] = f.id
            if f.name.lower() not in lookup:
                lookup[f.name.lower()] = f.id
        return lookup

    def classify_factor_roles(
        self,
        factor_set: FactorSet,
        *,
        task_description: str = "",
        explicit_roles: Optional[Mapping[str, ClauseRole]] = None,
    ) -> Dict[str, ClauseRole]:
        """Classify each factor as `conditional_rule` or `global_scaffold`."""
        valid_roles = {"conditional_rule", "global_scaffold"}
        alias_lookup = self._factor_alias_lookup(factor_set)
        roles: Dict[str, ClauseRole] = {}

        # 1. Explicit caller overrides
        if explicit_roles:
            for k, role in explicit_roles.items():
                if role not in valid_roles:
                    raise ValueError(
                        f"Invalid clause role {role!r} for {k!r}; expected {sorted(valid_roles)}."
                    )
                fid = alias_lookup.get(str(k)) or alias_lookup.get(str(k).lower())
                if fid is None:
                    raise ValueError(
                        f"factor_roles key {k!r} does not match any factor in FactorSet."
                    )
                roles[fid] = role

        # 2. Factor.metadata["clause_role"]
        for f in factor_set:
            if f.id in roles:
                continue
            meta_role = f.metadata.get("clause_role")
            if meta_role is not None:
                if meta_role not in valid_roles:
                    raise ValueError(
                        f"Invalid Factor.metadata['clause_role']={meta_role!r} on {f.id!r}."
                    )
                roles[f.id] = meta_role

        unclassified = [f for f in factor_set if f.id not in roles]
        if not unclassified:
            return roles

        # 3. Single LLM pre-pass for remaining factors
        clause_block = "\n".join(
            f"- {f.id} ({f.name}): description={f.description or ''!r}, "
            f"level_0={str(f.get_level(0).content)!r}, level_1={str(f.get_level(1).content)!r}"
            for f in unclassified
        )
        prompt = (
            f"{_CLASSIFY_ROLES_TEMPLATE}\n"
            f"Task: {task_description or '(unspecified)'}\n"
            f"Clauses:\n{clause_block}\n"
        )
        try:
            resp = self.client.generate(prompt)
            if not resp.error and resp.content:
                payload = _extract_json_payload(resp.content)
                raw_roles = (
                    payload.get("roles", payload) if isinstance(payload, dict) else {}
                )
                if isinstance(raw_roles, dict):
                    for k, val in raw_roles.items():
                        fid = alias_lookup.get(str(k)) or alias_lookup.get(
                            str(k).lower()
                        )
                        if fid and fid not in roles and val in valid_roles:
                            roles[fid] = val  # type: ignore[assignment]
        except Exception:
            pass

        # 4. Deterministic fallback for any factor still unclassified
        for f in unclassified:
            if f.id not in roles:
                roles[f.id] = _heuristic_clause_role(f)

        return roles

    def _build_prompt_context(
        self,
        template: PromptTemplate,
        required_vars: Sequence[str],
        task_description: str,
        seed_examples: Optional[Sequence[Dict[str, Any]]],
    ) -> Tuple[str, str, str]:
        vars_hint = (
            ", ".join(f'"{v}": "..."' for v in required_vars) or '"input": "..."'
        )
        if self.judge_fallback:
            judge_inst = 'Use "llm_judge" ONLY when deterministic checks cannot express correctness.'
            judge_kind = "|llm_judge"
        else:
            judge_inst = 'Do NOT use "llm_judge"; use deterministic checks only.'
            judge_kind = ""

        ctx_parts = [f"Task: {task_description or '(see template)'}"]
        ctx_parts.append(f"Required input variables: {list(required_vars)}")
        if template.master_template:
            ctx_parts.append(f"Template:\n{template.master_template}")
        else:
            static_secs = [
                f"[{s.id}] {s.content}"
                for s in template.sections
                if s.is_active and not s.factor_id and s.content
            ]
            if static_secs:
                ctx_parts.append("Static prompt sections:\n" + "\n".join(static_secs))
        if seed_examples:
            shown = list(seed_examples)[:10]
            ctx_parts.append(
                "Real seed examples (match style & domain):\n"
                + json.dumps(shown, ensure_ascii=False, indent=2)
            )
        return vars_hint, judge_inst + "||" + judge_kind, "\n\n".join(ctx_parts)

    def _generate_batch_loop(
        self,
        *,
        n_target: int,
        kind: str,
        intended_factor: Optional[str],
        prompt_builder: Callable[[int, int, List[str]], str],
        filter_pipeline: CaseFilterPipeline,
    ) -> List[TestCase]:
        if n_target <= 0:
            return []

        max_requests = max(
            n_target, int(math.ceil(n_target * self.max_oversample_factor))
        )
        collected: List[TestCase] = []
        requested = 0
        batch_idx = 0
        consecutive_zero_yield = 0

        while len(collected) < n_target and requested < max_requests:
            remaining = n_target - len(collected)
            budget_left = max_requests - requested
            ask_n = min(
                self.batch_size,
                budget_left,
                max(remaining, int(math.ceil(remaining * 1.25))),
            )
            recent_inputs = [c.canonical_inputs_text() for c in collected[-5:]]

            prompt = prompt_builder(ask_n, batch_idx, recent_inputs)
            requested += ask_n
            batch_idx += 1

            resp = self.client.generate(prompt)
            if resp.error or not resp.content:
                consecutive_zero_yield += 1
                if consecutive_zero_yield >= 2:
                    break
                continue

            try:
                payload = _extract_json_payload(resp.content)
                raw_cases = _extract_cases_list(payload)
            except Exception:
                consecutive_zero_yield += 1
                if consecutive_zero_yield >= 2:
                    break
                continue

            accepted_in_batch = 0
            for raw_case in raw_cases:
                if len(collected) >= n_target:
                    break
                case = filter_pipeline.filter_candidate(
                    raw_case,
                    kind=kind,
                    intended_factor=intended_factor,
                    source="synthetic",
                )
                if case is not None:
                    collected.append(case)
                    accepted_in_batch += 1

            if accepted_in_batch == 0:
                consecutive_zero_yield += 1
                if consecutive_zero_yield >= 2:
                    break
            else:
                consecutive_zero_yield = 0

        return collected

    def tag_cases(
        self,
        cases: Sequence[TestCase],
        factor_set: FactorSet,
        *,
        task_description: str = "",
    ) -> int:
        """Independently tag every case (blind to `intended_factor`) using `tagger_client`.

        Retries a failed/unparseable tagger batch once. Distinguishes explicit tagger
        disagreement (`case.id in tag_map` without `intended_factor` -> `tagger_agrees=False`)
        from a failed/omitted tagger response (`case.id not in tag_map` -> `tagger_agrees=None`).

        Returns the number of cases omitted or untagged due to tagger failure.
        """
        if not cases:
            return 0

        alias_lookup = self._factor_alias_lookup(factor_set)
        clauses_summary = "\n".join(
            f"- {f.id} ({f.name}): level_0={str(f.get_level(0).content)!r} -> "
            f"level_1={str(f.get_level(1).content)!r}"
            for f in factor_set
        )
        untagged_count = 0

        for start in range(0, len(cases), self.tagger_batch_size):
            batch = cases[start : start + self.tagger_batch_size]
            # Blind payload: omit `intended_factor` and `kind` completely!
            blind_items = [
                {
                    "id": c.id,
                    "inputs": c.inputs,
                    "target": c.target,
                    "check_descriptions": [
                        chk.description for chk in c.checks if chk.description
                    ],
                }
                for c in batch
            ]
            prompt = (
                f"{_TAG_COVERAGE_TEMPLATE}\n"
                f"Task: {task_description or '(unspecified)'}\n"
                f"Clauses:\n{clauses_summary}\n\n"
                f"Cases:\n{json.dumps(blind_items, ensure_ascii=False)}\n"
            )
            tag_map: Dict[str, List[str]] = {}
            for _attempt in range(2):
                try:
                    resp = self.tagger_client.generate(prompt)
                    if not resp.error and resp.content:
                        payload = _extract_json_payload(resp.content)
                        parsed_map = _extract_tag_map(payload)
                        if parsed_map:
                            tag_map = parsed_map
                            break
                except Exception:
                    continue

            for case in batch:
                was_tagged = case.id in tag_map
                if not was_tagged:
                    untagged_count += 1
                    case.tagger_factors = []
                    case.tagger_agrees = None
                    if case.kind == "targeted":
                        if case.intended_factor:
                            canon_int = (
                                alias_lookup.get(case.intended_factor)
                                or alias_lookup.get(case.intended_factor.lower())
                                or case.intended_factor
                            )
                            case.intended_factor = canon_int
                        case.targets_factors = case.effective_targets_factors(
                            require_tag_agreement=True
                        )
                    else:
                        case.targets_factors = []
                    continue

                raw_tags = tag_map[case.id]
                canonical_tags: Set[str] = set()
                for t in raw_tags:
                    fid = alias_lookup.get(str(t)) or alias_lookup.get(str(t).lower())
                    if fid is not None:
                        canonical_tags.add(fid)
                case.tagger_factors = sorted(canonical_tags)

                if case.kind == "targeted":
                    if case.intended_factor:
                        canon_int = (
                            alias_lookup.get(case.intended_factor)
                            or alias_lookup.get(case.intended_factor.lower())
                            or case.intended_factor
                        )
                        case.intended_factor = canon_int
                        case.tagger_agrees = canon_int in canonical_tags
                    else:
                        case.tagger_agrees = None
                    case.targets_factors = case.effective_targets_factors(
                        require_tag_agreement=True
                    )
                else:
                    case.tagger_agrees = None
                    case.targets_factors = list(case.tagger_factors)

        if untagged_count > 0:
            warnings.warn(
                f"[PromptPrism] Independent coverage tagger failed or omitted {untagged_count} "
                f"of {len(cases)} case(s); their `tagger_agrees` remains None.",
                UserWarning,
                stacklevel=2,
            )
        return untagged_count

    def _split_generated_cases(
        self,
        test_set: TestSet,
        *,
        n_baseline_disc: int,
        n_baseline_conf: int,
        n_factor_disc: int,
        n_factor_conf: int,
        seed: int,
    ) -> None:
        """Assign discovery/confirmation splits per group.

        Groups that hit their full target receive exact `(n_disc, n_conf)` counts;
        groups that fell short split proportionally by `n_disc / (n_disc + n_conf)`
        so confirmation is not starved when `n_conf > 0`.
        """
        import numpy as np

        rng = np.random.default_rng(seed)

        def _assign(group: List[TestCase], target_d: int, target_c: int) -> None:
            if not group:
                return
            order = list(rng.permutation(len(group)))
            total_target = target_d + target_c
            if target_c == 0:
                n_d = len(group)
            elif target_d == 0:
                n_d = 0
            elif len(group) >= total_target:
                n_d = target_d
            elif len(group) == 1:
                n_d = 1
            else:
                ratio = target_d / float(total_target)
                n_d = max(1, min(len(group) - 1, int(round(len(group) * ratio))))
            for pos, idx in enumerate(order):
                group[idx].split = "discovery" if pos < n_d else "confirmation"

        active_baseline = [
            c for c in test_set.cases if c.review != "rejected" and c.kind == "baseline"
        ]
        _assign(active_baseline, n_baseline_disc, n_baseline_conf)

        targeted_groups: Dict[str, List[TestCase]] = {}
        for c in test_set.cases:
            if c.review == "rejected" or c.kind != "targeted":
                continue
            key = test_set._canonical_factor_key(c.intended_factor or "", key_by="id")
            targeted_groups.setdefault(key, []).append(c)

        for key in sorted(targeted_groups.keys()):
            _assign(targeted_groups[key], n_factor_disc, n_factor_conf)

        test_set.recompute_metadata()

    def _generate_cases(
        self,
        *,
        factor_set: FactorSet,
        template: PromptTemplate,
        roles: Mapping[str, ClauseRole],
        required_vars: Sequence[str],
        task_description: str,
        seed_examples: Optional[Sequence[Dict[str, Any]]],
        n_baseline: int,
        n_per_factor: int,
        gen_seed: int,
        filter_pipeline: CaseFilterPipeline,
    ) -> Tuple[List[TestCase], Dict[str, Dict[str, int]]]:
        """Generate and filter baseline + per-factor targeted cases, tracking any shortfalls."""
        vars_hint, judge_spec, shared_ctx = self._build_prompt_context(
            template, required_vars, task_description, seed_examples
        )
        judge_instruction, judge_kind = judge_spec.split("||", 1)
        shortfalls: Dict[str, Dict[str, int]] = {}

        def _baseline_prompt(n: int, b_idx: int, recent: List[str]) -> str:
            avoid = (
                f"\nAvoid repeating these recent inputs:\n{json.dumps(recent, ensure_ascii=False)}"
                if recent
                else ""
            )
            return (
                _BASELINE_GEN_TEMPLATE.format(
                    n=n,
                    vars_hint=vars_hint,
                    judge_instruction=judge_instruction,
                    judge_kind=judge_kind,
                )
                + f"\n{shared_ctx}\nSeed: {gen_seed}, Batch: {b_idx}{avoid}\n"
            )

        baseline_cases = self._generate_batch_loop(
            n_target=n_baseline,
            kind="baseline",
            intended_factor=None,
            prompt_builder=_baseline_prompt,
            filter_pipeline=filter_pipeline,
        )
        if len(baseline_cases) < n_baseline:
            shortfalls["baseline"] = {
                "requested": n_baseline,
                "accepted": len(baseline_cases),
            }

        targeted_cases: List[TestCase] = []
        for factor in factor_set:
            role = roles.get(factor.id, "conditional_rule")
            tmpl_str = (
                _TARGETED_RULE_TEMPLATE
                if role == "conditional_rule"
                else _TARGETED_SCAFFOLD_TEMPLATE
            )
            clause_detail = (
                f"Factor ID: {factor.id}\n"
                f"Factor Name: {factor.name}\n"
                f"Clause Role: {role}\n"
                f"Description: {factor.description or ''}\n"
                f"Level 0 (OFF): {str(factor.get_level(0).content)!r}\n"
                f"Level 1 (ON):  {str(factor.get_level(1).content)!r}"
            )

            def _targeted_prompt(
                n: int,
                b_idx: int,
                recent: List[str],
                _tmpl: str = tmpl_str,
                _f: Factor = factor,
                _detail: str = clause_detail,
            ) -> str:
                avoid = (
                    f"\nAvoid repeating these recent inputs:\n{json.dumps(recent, ensure_ascii=False)}"
                    if recent
                    else ""
                )
                return (
                    _tmpl.format(
                        n=n,
                        factor_id=_f.id,
                        factor_name=_f.name,
                        vars_hint=vars_hint,
                        judge_instruction=judge_instruction,
                        judge_kind=judge_kind,
                    )
                    + f"\n{_detail}\n\n{shared_ctx}\nSeed: {gen_seed}, Batch: {b_idx}{avoid}\n"
                )

            f_cases = self._generate_batch_loop(
                n_target=n_per_factor,
                kind="targeted",
                intended_factor=factor.id,
                prompt_builder=_targeted_prompt,
                filter_pipeline=filter_pipeline,
            )
            if len(f_cases) < n_per_factor:
                shortfalls[factor.id] = {
                    "requested": n_per_factor,
                    "accepted": len(f_cases),
                }
            targeted_cases.extend(f_cases)

        return baseline_cases + targeted_cases, shortfalls

    def _resolved_model_ids(self) -> Tuple[str, str]:
        gen_model_id = str(self.client.model_id or "unknown")
        if self.tagger_client is self.client:
            tagger_model_id = gen_model_id
        else:
            tagger_model_id = str(self.tagger_client.model_id or "unknown")
        return gen_model_id, tagger_model_id

    def generate(
        self,
        factors: Union[FactorSet, Sequence[Factor]],
        template: PromptTemplate,
        *,
        task_description: str = "",
        seed_examples: Optional[Sequence[Dict[str, Any]]] = None,
        factor_roles: Optional[Mapping[str, ClauseRole]] = None,
        freeze: bool = True,
    ) -> TestSet:
        """Run the full generation pipeline: classify roles -> generate -> filter -> tag -> split -> freeze."""
        from .. import __version__

        factor_set = self._validate_factors(factors)
        required_vars = sorted(template.required_data_variables(factor_set))
        roles = self.classify_factor_roles(
            factor_set,
            task_description=task_description,
            explicit_roles=factor_roles,
        )

        filter_pipeline = CaseFilterPipeline(
            required_vars=required_vars,
            judge_fallback=self.judge_fallback,
            dedup_jaccard_threshold=self.dedup_jaccard_threshold,
        )

        all_cases, shortfalls = self._generate_cases(
            factor_set=factor_set,
            template=template,
            roles=roles,
            required_vars=required_vars,
            task_description=task_description,
            seed_examples=seed_examples,
            n_baseline=self.n_baseline_discovery + self.n_baseline_confirmation,
            n_per_factor=self.n_per_factor_discovery + self.n_per_factor_confirmation,
            gen_seed=self.seed,
            filter_pipeline=filter_pipeline,
        )

        # Independent blind coverage tagging
        untagged_count = self.tag_cases(
            all_cases, factor_set, task_description=task_description
        )

        if shortfalls:
            warnings.warn(
                f"[PromptPrism] TestSetGenerator fell short of post-filter target counts: {shortfalls}",
                UserWarning,
                stacklevel=2,
            )

        gen_model_id, tagger_model_id = self._resolved_model_ids()
        generator_meta: Dict[str, Any] = {
            "model_id": gen_model_id,
            "tagger_model_id": tagger_model_id,
            "prompt_hash": GENERATOR_PROMPT_HASH,
            "seed": self.seed,
            "prompt_prism_version": __version__,
            "filter_stats": filter_pipeline.stats.to_dict(),
        }
        if shortfalls:
            generator_meta["shortfalls"] = shortfalls
        if untagged_count > 0:
            generator_meta["tagger_untagged_cases"] = untagged_count

        ts = TestSet.create(
            all_cases,
            factors=factor_set,
            template=template,
            factor_roles=roles,
            generator=generator_meta,
            freeze=False,
        )

        # Split into discovery / confirmation
        if all_cases:
            self._split_generated_cases(
                ts,
                n_baseline_disc=self.n_baseline_discovery,
                n_baseline_conf=self.n_baseline_confirmation,
                n_factor_disc=self.n_per_factor_discovery,
                n_factor_conf=self.n_per_factor_confirmation,
                seed=self.seed,
            )

        if freeze:
            ts.freeze()
        return ts

    def refresh_confirmation(
        self,
        existing: TestSet,
        factors: Union[FactorSet, Sequence[Factor]],
        template: PromptTemplate,
        *,
        task_description: str = "",
        seed_examples: Optional[Sequence[Dict[str, Any]]] = None,
        seed_offset: int = 1,
        freeze: bool = True,
    ) -> TestSet:
        """Regenerate ONLY the `confirmation` split while preserving `discovery` cases untouched.

        Verifies factor/template hash consistency and deduplicates new confirmation items
        against BOTH existing discovery and retired confirmation items to prevent overlap.
        """
        from .. import __version__

        if existing.is_frozen:
            existing.verify_integrity()

        factor_set = self._validate_factors(factors)
        expected_factors_sha = _canonical_json_sha256(factor_set.to_dict())
        if (
            existing.manifest.factors_sha256
            and existing.manifest.factors_sha256 != expected_factors_sha
        ):
            raise ValueError(
                "Cannot refresh confirmation: supplied factors do not match existing TestSet "
                f"factors_sha256 ({existing.manifest.factors_sha256!r} != {expected_factors_sha!r})."
            )
        expected_template_sha = template.template_sha256()
        if (
            existing.manifest.template_sha256
            and existing.manifest.template_sha256 != expected_template_sha
        ):
            raise ValueError(
                "Cannot refresh confirmation: supplied template does not match existing TestSet "
                f"template_sha256 ({existing.manifest.template_sha256!r} != {expected_template_sha!r})."
            )

        required_vars = sorted(template.required_data_variables(factor_set))
        roles = self.classify_factor_roles(
            factor_set,
            task_description=task_description,
            explicit_roles=existing.manifest.factor_roles or None,
        )

        # Preserve all discovery cases (active and rejected) untouched
        discovery_cases = [
            c.model_copy(deep=True) for c in existing.cases if c.split == "discovery"
        ]

        # Seed filter with ALL existing cases (discovery + retired confirmation) so refreshed
        # confirmation items are strictly disjoint from both.
        filter_pipeline = CaseFilterPipeline(
            required_vars=required_vars,
            judge_fallback=self.judge_fallback,
            dedup_jaccard_threshold=self.dedup_jaccard_threshold,
            existing_cases=existing.cases,
        )

        refresh_seed = self.seed + int(seed_offset)
        new_conf, shortfalls = self._generate_cases(
            factor_set=factor_set,
            template=template,
            roles=roles,
            required_vars=required_vars,
            task_description=task_description,
            seed_examples=seed_examples,
            n_baseline=self.n_baseline_confirmation,
            n_per_factor=self.n_per_factor_confirmation,
            gen_seed=refresh_seed,
            filter_pipeline=filter_pipeline,
        )

        untagged_count = self.tag_cases(
            new_conf, factor_set, task_description=task_description
        )
        for c in new_conf:
            c.split = "confirmation"

        if shortfalls:
            warnings.warn(
                f"[PromptPrism] TestSetGenerator.refresh_confirmation fell short of target counts: {shortfalls}",
                UserWarning,
                stacklevel=2,
            )

        gen_meta = dict(existing.manifest.generator)
        gen_meta.update(
            {
                "confirmation_refreshed_at": datetime.now(timezone.utc).isoformat(),
                "confirmation_refresh_seed": refresh_seed,
                "confirmation_filter_stats": filter_pipeline.stats.to_dict(),
                "prompt_prism_version": __version__,
            }
        )
        if shortfalls:
            gen_meta["confirmation_shortfalls"] = shortfalls
        else:
            gen_meta.pop("confirmation_shortfalls", None)
        if untagged_count > 0:
            gen_meta["confirmation_tagger_untagged_cases"] = untagged_count
        else:
            gen_meta.pop("confirmation_tagger_untagged_cases", None)

        refreshed = TestSet.create(
            discovery_cases + new_conf,
            factors=factor_set,
            template=template,
            factor_roles=roles,
            generator=gen_meta,
            freeze=freeze,
        )
        return refreshed
