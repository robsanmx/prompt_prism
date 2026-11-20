"""Frozen semantic-preserving stability perturbation panel, canonical prompt renderer, and deterministic scorers."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from research.real_v4.schemas import (
    CURRENT_SCHEMA_VERSION,
    EnvironmentFamily,
    EnvironmentSpec,
    canonical_json,
    canonical_sha256,
)
from research.real_v4.selection import FACTOR_IDS

ROOT = Path(__file__).resolve().parents[2]
PERTURBATION_REGISTRY_PATH = ROOT / "research" / "protocol" / "real_v4" / "perturbation_registry.json"

REGISTERED_TASKS: Tuple[str, ...] = (
    "re3d_entity_extraction",
    "squad2_grounded_qa",
    "banking77_intent",
    "gsm8k_math",
    "mbpp_python",
    "sciq_verification",
)


class OutOfEnvelopePerturbationError(ValueError):
    """Raised when a perturbation alters task meaning, targets, options, factors, parser, or model version."""


@dataclass(frozen=True)
class RenderedPromptInstance:
    task_id: str
    sample_id: str
    env_id: str
    family: EnvironmentFamily
    factor_levels: Dict[str, int]
    system_text: str
    user_text: str
    parent_prompt_sha256: str
    rendered_sha256: str
    factual_context_sha256: str
    options_sha256: str
    target_sha256: str
    parser_hash: str


# ==============================================================================
# Deterministic Task Parsers & Primary Scorers ([0.0, 1.0], failure -> 0.0)
# ==============================================================================

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^\w\s]")


def _normalize_text(text: str) -> str:
    lowered = text.lower()
    no_punct = _PUNCT_RE.sub(" ", lowered)
    no_art = _ARTICLES_RE.sub(" ", no_punct)
    return " ".join(no_art.split())


def score_re3d_entity_extraction(raw_output: str, item: Mapping[str, Any]) -> float:
    """Deterministic micro set F1 over 5 entity types; malformed JSON -> 0.0."""
    if not raw_output or not raw_output.strip():
        return 0.0
    text = raw_output.strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return 0.0
    try:
        parsed = json.loads(match.group(0))
    except Exception:
        return 0.0
    if not isinstance(parsed, dict):
        return 0.0

    keys = ("person", "organization", "location", "money", "temporal")
    gold_target = item["target"]
    pred_set = set()
    gold_set = set()
    for k in keys:
        vals = parsed.get(k, [])
        if not isinstance(vals, list):
            return 0.0
        for v in vals:
            norm = _normalize_text(str(v))
            if norm:
                pred_set.add((k, norm))
        for g in gold_target.get(k, []):
            norm_g = _normalize_text(str(g))
            if norm_g:
                gold_set.add((k, norm_g))

    if not gold_set and not pred_set:
        return 1.0
    if not gold_set or not pred_set:
        return 0.0
    tp = len(pred_set & gold_set)
    if tp == 0:
        return 0.0
    prec = tp / len(pred_set)
    rec = tp / len(gold_set)
    return float(2.0 * prec * rec / (prec + rec))


def score_squad2_grounded_qa(raw_output: str, item: Mapping[str, Any]) -> float:
    """Deterministic SQuAD 2.0 token F1 supporting explicit UNANSWERABLE abstention."""
    if not raw_output or not raw_output.strip():
        return 0.0
    text = raw_output.strip()
    ans_match = re.search(r"(?:FINAL_ANSWER|ANSWER)\s*:\s*(.+)", text, flags=re.IGNORECASE)
    candidate = ans_match.group(1).strip() if ans_match else text.splitlines()[-1].strip()
    if not candidate:
        return 0.0

    is_unans_pred = candidate.upper().strip(" .\"'") in {"UNANSWERABLE", "NO_ANSWER", "CANNOT_ANSWER"}
    gold_answers: Sequence[str] = item["target"]
    if not gold_answers:  # Unanswerable item
        return 1.0 if is_unans_pred else 0.0
    if is_unans_pred:
        return 0.0

    pred_toks = _normalize_text(candidate).split()
    if not pred_toks:
        return 0.0
    best_f1 = 0.0
    for gold in gold_answers:
        gold_toks = _normalize_text(str(gold)).split()
        if not gold_toks:
            continue
        common = {}
        for t in set(pred_toks) & set(gold_toks):
            common[t] = min(pred_toks.count(t), gold_toks.count(t))
        num_same = sum(common.values())
        if num_same == 0:
            continue
        p = num_same / len(pred_toks)
        r = num_same / len(gold_toks)
        best_f1 = max(best_f1, 2.0 * p * r / (p + r))
    return float(best_f1)


def score_banking77_intent(raw_output: str, item: Mapping[str, Any]) -> float:
    """Deterministic exact label accuracy against allowed Banking77 label inventory."""
    if not raw_output or not raw_output.strip():
        return 0.0
    text = raw_output.strip()
    lbl_match = re.search(r"(?:INTENT|LABEL|FINAL_ANSWER)\s*:\s*([a-z0-9_]+)", text, flags=re.IGNORECASE)
    cand = lbl_match.group(1).lower().strip() if lbl_match else text.splitlines()[-1].strip().lower().strip(" .\"'`")
    allowed = {str(x).lower() for x in item.get("options", ())}
    if allowed and cand not in allowed:
        return 0.0
    return 1.0 if cand == str(item["target"]).lower() else 0.0


def score_gsm8k_math(raw_output: str, item: Mapping[str, Any]) -> float:
    """Deterministic exact numeric match on canonicalized final numerical answer."""
    if not raw_output or not raw_output.strip():
        return 0.0
    text = raw_output.strip()
    tagged = re.search(r"(?:####|FINAL_ANSWER\s*:)\s*([-+]?\d[\d,]*(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if tagged:
        raw_num = tagged.group(1)
    else:
        nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
        if not nums:
            return 0.0
        raw_num = nums[-1]
    try:
        pred_val = float(raw_num.replace(",", ""))
        gold_val = float(str(item["target"]).replace(",", ""))
    except ValueError:
        return 0.0
    return 1.0 if abs(pred_val - gold_val) <= 1e-6 else 0.0


def _extract_python_code(raw_output: str) -> str:
    fenced = re.findall(r"```(?:python)?\s*\n(.*?)```", raw_output, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced[-1].strip()
    return raw_output.strip()


def score_mbpp_python(raw_output: str, item: Mapping[str, Any], timeout_sec: float = 2.5) -> float:
    """Deterministic assertion pass rate in an isolated Python subprocess with syntax check."""
    if not raw_output or not raw_output.strip():
        return 0.0
    code = _extract_python_code(raw_output)
    if not code:
        return 0.0
    try:
        ast.parse(code)
    except SyntaxError:
        return 0.0

    assertions: Sequence[str] = item["target"]
    if not assertions:
        return 0.0

    harness_lines = [
        code,
        "",
        "_passed = 0",
        f"_tests = {json.dumps(list(assertions))}",
        "for _stmt in _tests:",
        "    try:",
        "        exec(_stmt, globals())",
        "        _passed += 1",
        "    except Exception:",
        "        pass",
        "print(f'MBPP_SCORE:{_passed}/{len(_tests)}')",
    ]
    script = "\n".join(harness_lines)
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return 0.0
    if proc.returncode != 0:
        return 0.0
    m = re.search(r"MBPP_SCORE:(\d+)/(\d+)", proc.stdout)
    if not m:
        return 0.0
    passed, total = int(m.group(1)), int(m.group(2))
    return float(passed / total) if total > 0 else 0.0


def score_sciq_verification(raw_output: str, item: Mapping[str, Any]) -> float:
    """Deterministic four-way choice accuracy (A/B/C/D)."""
    if not raw_output or not raw_output.strip():
        return 0.0
    text = raw_output.strip()
    m = re.search(r"(?:CHOICE|FINAL_ANSWER|ANSWER)\s*:\s*\(?([ABCD])\)?\b", text, flags=re.IGNORECASE)
    if m:
        letter = m.group(1).upper()
    else:
        last_line = text.splitlines()[-1].strip()
        m2 = re.match(r"^\(?([ABCD])\)?[.):]?$", last_line, flags=re.IGNORECASE)
        if not m2:
            return 0.0
        letter = m2.group(1).upper()
    return 1.0 if letter == str(item["target"]).upper() else 0.0


TASK_SCORERS = {
    "re3d_entity_extraction": score_re3d_entity_extraction,
    "squad2_grounded_qa": score_squad2_grounded_qa,
    "banking77_intent": score_banking77_intent,
    "gsm8k_math": score_gsm8k_math,
    "mbpp_python": score_mbpp_python,
    "sciq_verification": score_sciq_verification,
}


def parser_hashes_by_task() -> Dict[str, str]:
    """Deterministic SHA-256 digest of each task's primary parser/scorer source bytecode & name."""
    import inspect

    out: Dict[str, str] = {}
    for task_id, fn in sorted(TASK_SCORERS.items()):
        src = inspect.getsource(fn)
        out[task_id] = hashlib.sha256(f"{task_id}:{src}".encode("utf-8")).hexdigest()
    return out


def score_task_response(task_id: str, raw_output: str, item: Mapping[str, Any]) -> float:
    if task_id not in TASK_SCORERS:
        raise ValueError(f"Unknown task_id={task_id!r}")
    return float(TASK_SCORERS[task_id](raw_output, item))


# ==============================================================================
# Registered Perturbation Environments & Canonical Rendering
# ==============================================================================

_PREAMBLE_BY_ENV: Dict[str, str] = {
    "ref": "Read the task input below and produce your response.",
    "repeat_r2": "Read the task input below and produce your response.",
    "repeat_r3": "Read the task input below and produce your response.",
    "fmt_whitespace": "Read the task input below and produce your response.",
    "fmt_delimiter": "Read the task input below and produce your response.",
    "word_paraphrase_1": "Carefully review the provided input below and supply your response.",
    "word_paraphrase_2": "Examine the input material below and provide the requested response.",
    "item_holdout_group": "Read the task input below and produce your response.",
    "temporal_wave_2": "Read the task input below and produce your response.",
}

_TASK_DEMOS: Dict[str, str] = {
    "re3d_entity_extraction": (
        'Example 1:\nSentence: NATO Secretary General Jens Stoltenberg met in Brussels on Monday.\n'
        'Output: {"person": ["Jens Stoltenberg"], "organization": ["NATO"], "location": ["Brussels"], "money": [], "temporal": ["Monday"]}\n'
        'Example 2:\nSentence: The agency allocated $5 million in 2024 for Geneva operations.\n'
        'Output: {"person": [], "organization": [], "location": ["Geneva"], "money": ["$5 million"], "temporal": ["2024"]}'
    ),
    "squad2_grounded_qa": (
        "Example 1:\nContext: Oxygen was isolated by Joseph Priestley in 1774.\nQuestion: In what year did Priestley isolate oxygen?\nOutput: FINAL_ANSWER: 1774\n"
        "Example 2:\nContext: Oxygen was isolated by Joseph Priestley in 1774.\nQuestion: Where was Priestley born?\nOutput: FINAL_ANSWER: UNANSWERABLE"
    ),
    "banking77_intent": (
        "Example 1:\nQuery: My new card still has not arrived in the mail.\nOutput: FINAL_ANSWER: card_arrival\n"
        "Example 2:\nQuery: Why did the ATM refuse to dispense my cash?\nOutput: FINAL_ANSWER: declined_cash_withdrawal"
    ),
    "gsm8k_math": (
        "Example 1:\nProblem: Mia buys 3 notebooks at $4 each and 2 pens at $2 each. How much does she spend?\nOutput: FINAL_ANSWER: 16\n"
        "Example 2:\nProblem: A train travels 120 miles in 2 hours. What is its average speed in mph?\nOutput: FINAL_ANSWER: 60"
    ),
    "mbpp_python": (
        "Example 1:\nSpecification: Write a Python function `add_pair(a, b)` that returns the sum of two numbers.\nOutput:\n```python\ndef add_pair(a, b):\n    return a + b\n```"
    ),
    "sciq_verification": (
        "Example 1:\nSupport: Chlorophyll absorbs red and blue light inside chloroplasts.\nQuestion: Which organelle performs photosynthesis?\nOptions:\n(A) Mitochondria\n(B) Chloroplast\n(C) Ribosome\n(D) Nucleus\nOutput: FINAL_ANSWER: B"
    ),
}

_TASK_OUTPUT_CONTRACTS: Dict[str, Tuple[str, str]] = {
    "re3d_entity_extraction": (
        "Return a JSON object with keys person, organization, location, money, and temporal.",
        'STRICT CONTRACT: Return ONLY a valid single-line JSON object with exact keys {"person": [...], "organization": [...], "location": [...], "money": [...], "temporal": [...]}.',
    ),
    "squad2_grounded_qa": (
        "Answer the question concisely from the context, or reply UNANSWERABLE if unsupported.",
        "STRICT CONTRACT: End your response with `FINAL_ANSWER: <exact span>` or `FINAL_ANSWER: UNANSWERABLE` if the context does not answer the question.",
    ),
    "banking77_intent": (
        "Return the matching intent label from the allowed list.",
        "STRICT CONTRACT: End your response with `FINAL_ANSWER: <exact_intent_label>` using only one label from the allowed inventory.",
    ),
    "gsm8k_math": (
        "Provide the final numeric answer.",
        "STRICT CONTRACT: End your response with `FINAL_ANSWER: <number>` containing only digits (and optional decimal/minus sign).",
    ),
    "mbpp_python": (
        "Write the requested Python function.",
        "STRICT CONTRACT: Return ONLY executable Python code inside a ```python ... ``` block implementing the exact requested function signature.",
    ),
    "sciq_verification": (
        "Select the correct option letter A, B, C, or D.",
        "STRICT CONTRACT: End your response with `FINAL_ANSWER: <letter>` where `<letter>` is A, B, C, or D.",
    ),
}


def build_perturbation_environments() -> Tuple[EnvironmentSpec, ...]:
    """Return the frozen Real-v4 perturbation environment specifications."""
    defs = [
        ("ref", EnvironmentFamily.REFERENCE, "Canonical clean confirmation condition", None, True, False),
        ("repeat_r2", EnvironmentFamily.REPEAT, "Identical-request repeat pass 2", "ref", True, False),
        ("repeat_r3", EnvironmentFamily.REPEAT, "Identical-request repeat pass 3", "ref", True, False),
        ("fmt_whitespace", EnvironmentFamily.FORMATTING, "Expanded vertical whitespace between structural blocks", "ref", True, False),
        ("fmt_delimiter", EnvironmentFamily.FORMATTING, "Banner-style section delimiters (=== HEADER ===)", "ref", True, False),
        ("word_paraphrase_1", EnvironmentFamily.WORDING, "Controlled semantic-preserving wrapper paraphrase A", "ref", True, False),
        ("word_paraphrase_2", EnvironmentFamily.WORDING, "Controlled semantic-preserving wrapper paraphrase B", "ref", True, False),
        ("item_holdout_group", EnvironmentFamily.ITEM_GROUP, "Group-disjoint secondary confirmation item stratum", "ref", True, True),
        ("temporal_wave_2", EnvironmentFamily.TEMPORAL, "Second execution wave under pinned provider version", "ref", True, False),
    ]
    specs: List[EnvironmentSpec] = []
    for env_id, fam, desc, parent, req_ver, disj in defs:
        rule_digest = hashlib.sha256(f"real_v4:{env_id}:{fam.value}:{_PREAMBLE_BY_ENV[env_id]}".encode("utf-8")).hexdigest()[:16]
        specs.append(
            EnvironmentSpec(
                schema_version=CURRENT_SCHEMA_VERSION,
                env_id=env_id,
                family=fam,
                description=desc,
                parent_env_id=parent,
                transformation_hash=rule_digest,
                semantic_equivalence_audited=True,
                requires_identical_model_version=req_ver,
                uses_disjoint_item_group=disj,
            )
        )
    return tuple(specs)


def _format_factual_block(task_id: str, item: Mapping[str, Any]) -> Tuple[str, str, str]:
    """Extract canonical factual payload, factual digest, and options digest without touching target."""
    if task_id == "re3d_entity_extraction":
        factual = f"Sentence: {item['input_text']}"
        options_str = "person|organization|location|money|temporal"
    elif task_id == "squad2_grounded_qa":
        factual = f"Context: {item['context']}\nQuestion: {item['question']}"
        options_str = "span_or_UNANSWERABLE"
    elif task_id == "banking77_intent":
        opts = list(item["options"])
        options_str = ", ".join(opts)
        factual = f"Allowed Intents: {options_str}\nQuery: {item['input_text']}"
    elif task_id == "gsm8k_math":
        factual = f"Problem: {item['question']}"
        options_str = "numeric"
    elif task_id == "mbpp_python":
        factual = f"Specification: {item['prompt']}\nRequired Signature Hint: {item['signature_hint']}"
        options_str = str(item["signature_hint"])
    elif task_id == "sciq_verification":
        opts = item["options"]
        opt_lines = "\n".join(f"({k}) {opts[k]}" for k in ("A", "B", "C", "D"))
        factual = f"Support: {item['support']}\nQuestion: {item['question']}\nOptions:\n{opt_lines}"
        options_str = canonical_json(opts)
    else:
        raise ValueError(f"Unsupported task_id={task_id!r}")

    f_sha = hashlib.sha256(factual.encode("utf-8")).hexdigest()
    o_sha = hashlib.sha256(options_str.encode("utf-8")).hexdigest()
    return factual, f_sha, o_sha


def render_prompt_instance(
    task_id: str,
    item: Mapping[str, Any],
    factor_levels: Mapping[str, int],
    *,
    env_id: str = "ref",
) -> RenderedPromptInstance:
    """Render a task prompt under exact factor levels A..F and a registered environment."""
    env_specs = {e.env_id: e for e in build_perturbation_environments()}
    if env_id not in env_specs:
        raise OutOfEnvelopePerturbationError(f"Unregistered environment env_id={env_id!r}")
    if task_id not in REGISTERED_TASKS:
        raise OutOfEnvelopePerturbationError(f"Unregistered task_id={task_id!r}")
    if set(factor_levels.keys()) != set(FACTOR_IDS) or any(factor_levels[f] not in (0, 1) for f in FACTOR_IDS):
        raise OutOfEnvelopePerturbationError(f"Invalid factor_levels={dict(factor_levels)!r}")

    levels_clean = {f: int(factor_levels[f]) for f in FACTOR_IDS}
    p_hashes = parser_hashes_by_task()

    # Factor A: system role
    system_text = (
        f"You are a precision {task_id.replace('_', ' ')} specialist. Follow all task rules accurately."
        if levels_clean["A"] == 1
        else "You are a helpful and accurate assistant."
    )

    # Build instructional sections from Factors B, C, D, E, F
    inst_lines: List[str] = [_PREAMBLE_BY_ENV[env_id]]
    if levels_clean["B"] == 1:
        inst_lines.append("Decompose the problem concisely in 1-2 brief lines before emitting the final answer.")
    else:
        inst_lines.append("Respond directly without unnecessary commentary.")

    if levels_clean["E"] == 1:
        inst_lines.append("Rely strictly on the provided input and never invent unsupported facts, labels, or assumptions.")
    if levels_clean["F"] == 1:
        inst_lines.append("Before finishing, silently verify that every constraint and formatting requirement is satisfied.")

    c0, c1 = _TASK_OUTPUT_CONTRACTS[task_id]
    inst_lines.append(c1 if levels_clean["D"] == 1 else c0)

    factual_block, f_sha, o_sha = _format_factual_block(task_id, item)
    demo_block = _TASK_DEMOS[task_id] if levels_clean["C"] == 1 else ""

    if env_id == "fmt_delimiter":
        h_inst, h_demo, h_inp = "=== INSTRUCTIONS ===", "=== EXAMPLES ===", "=== TASK INPUT ==="
    else:
        h_inst, h_demo, h_inp = "[INSTRUCTIONS]", "[EXAMPLES]", "[TASK INPUT]"

    sep = "\n\n\n" if env_id == "fmt_whitespace" else "\n\n"
    pad = "  " if env_id == "fmt_whitespace" else ""

    blocks = [f"{h_inst}{pad}\n" + "\n".join(inst_lines)]
    if demo_block:
        blocks.append(f"{h_demo}{pad}\n{demo_block}")
    blocks.append(f"{h_inp}{pad}\n{factual_block}")
    user_text = sep.join(blocks)

    rendered_sha = hashlib.sha256(f"{system_text}\n---\n{user_text}".encode("utf-8")).hexdigest()
    if env_id == "ref":
        parent_sha = rendered_sha
    else:
        ref_inst = render_prompt_instance(task_id, item, levels_clean, env_id="ref")
        parent_sha = ref_inst.rendered_sha256

    t_sha = canonical_sha256(item["target"])
    return RenderedPromptInstance(
        task_id=task_id,
        sample_id=str(item["sample_id"]),
        env_id=env_id,
        family=env_specs[env_id].family,
        factor_levels=levels_clean,
        system_text=system_text,
        user_text=user_text,
        parent_prompt_sha256=parent_sha,
        rendered_sha256=rendered_sha,
        factual_context_sha256=f_sha,
        options_sha256=o_sha,
        target_sha256=t_sha,
        parser_hash=p_hashes[task_id],
    )


def audit_perturbation_invariance(
    ref_instance: RenderedPromptInstance,
    perturbed_instance: RenderedPromptInstance,
    *,
    ref_item: Mapping[str, Any],
    perturbed_item: Mapping[str, Any],
    pinned_model_version: str,
    observed_model_version: str,
    sample_outputs: Sequence[str],
    allow_disjoint_item: bool = False,
) -> Dict[str, Any]:
    """Audit that a perturbation preserves factors, targets, options, context, model version, and parser scoring."""
    if observed_model_version != pinned_model_version:
        raise OutOfEnvelopePerturbationError(
            f"Provider model version drift: expected {pinned_model_version!r}, got {observed_model_version!r}"
        )
    if ref_instance.task_id != perturbed_instance.task_id:
        raise OutOfEnvelopePerturbationError("Task ID changed across perturbation")
    if ref_instance.factor_levels != perturbed_instance.factor_levels:
        raise OutOfEnvelopePerturbationError(
            f"Factor levels mutated: {ref_instance.factor_levels} -> {perturbed_instance.factor_levels}"
        )
    if ref_instance.parser_hash != perturbed_instance.parser_hash:
        raise OutOfEnvelopePerturbationError("Parser hash mutated across perturbation")

    if not allow_disjoint_item:
        if ref_instance.target_sha256 != perturbed_instance.target_sha256:
            raise OutOfEnvelopePerturbationError("Target answer mutated across perturbation")
        if ref_instance.options_sha256 != perturbed_instance.options_sha256:
            raise OutOfEnvelopePerturbationError("Answer options inventory mutated across perturbation")
        if ref_instance.factual_context_sha256 != perturbed_instance.factual_context_sha256:
            raise OutOfEnvelopePerturbationError("Factual context mutated across perturbation")
        if perturbed_instance.parent_prompt_sha256 != ref_instance.rendered_sha256:
            raise OutOfEnvelopePerturbationError("Parent prompt SHA-256 mismatch")

        # Verify metric invariance when model output is held constant
        for raw_out in sample_outputs:
            s_ref = score_task_response(ref_instance.task_id, raw_out, ref_item)
            s_pert = score_task_response(perturbed_instance.task_id, raw_out, perturbed_item)
            if abs(s_ref - s_pert) > 1e-12:
                raise OutOfEnvelopePerturbationError(
                    f"Parser/metric invariance violated on output {raw_out!r}: {s_ref} vs {s_pert}"
                )

    if perturbed_instance.family in (
        EnvironmentFamily.REPEAT,
        EnvironmentFamily.TEMPORAL,
    ):
        if perturbed_instance.rendered_sha256 != ref_instance.rendered_sha256:
            raise OutOfEnvelopePerturbationError(
                f"{perturbed_instance.env_id} must render byte-identically to ref"
            )

    return {
        "env_id": perturbed_instance.env_id,
        "family": perturbed_instance.family.value,
        "parent_prompt_sha256": perturbed_instance.parent_prompt_sha256,
        "rendered_sha256": perturbed_instance.rendered_sha256,
        "semantic_invariant": True,
        "metric_invariant_checked_outputs": len(sample_outputs),
    }


def write_perturbation_registry(output_path: Path = PERTURBATION_REGISTRY_PATH) -> Dict[str, Any]:
    """Generate and persist `research/protocol/real_v4/perturbation_registry.json`."""
    specs = build_perturbation_environments()
    p_hashes = parser_hashes_by_task()

    sample_items: Dict[str, Dict[str, Any]] = {
        "re3d_entity_extraction": {
            "sample_id": "canon_re3d_0",
            "input_text": "Prime Minister Sanae Takaichi visited Tokyo on Tuesday with $10 million.",
            "target": {
                "person": ["Sanae Takaichi"],
                "organization": [],
                "location": ["Tokyo"],
                "money": ["$10 million"],
                "temporal": ["Tuesday"],
            },
        },
        "squad2_grounded_qa": {
            "sample_id": "canon_squad_0",
            "context": "The James Webb Space Telescope launched on December 25, 2021.",
            "question": "When did the James Webb Space Telescope launch?",
            "target": ["December 25, 2021"],
        },
        "banking77_intent": {
            "sample_id": "canon_bank_0",
            "input_text": "I tried to pay by card at the store and it was rejected.",
            "options": ["card_arrival", "declined_card_payment", "pending_cash_withdrawal"],
            "target": "declined_card_payment",
        },
        "gsm8k_math": {
            "sample_id": "canon_gsm_0",
            "question": "A box holds 12 apples. How many apples are in 4 boxes?",
            "target": "48",
        },
        "mbpp_python": {
            "sample_id": "canon_mbpp_0",
            "prompt": "Write a function `square_num(x)` that returns the square of `x`.",
            "signature_hint": "def square_num(x):",
            "target": ["assert square_num(3) == 9", "assert square_num(-4) == 16"],
        },
        "sciq_verification": {
            "sample_id": "canon_sciq_0",
            "support": "Sound waves require a mechanical medium and cannot travel through a vacuum.",
            "question": "Through which environment can sound waves NOT travel?",
            "options": {"A": "Water", "B": "Steel", "C": "Vacuum", "D": "Air"},
            "target": "C",
        },
    }

    default_levels = {f: 1 for f in FACTOR_IDS}
    golden_renders: Dict[str, Dict[str, str]] = {}
    for task_id, item in sorted(sample_items.items()):
        golden_renders[task_id] = {}
        for spec in specs:
            inst = render_prompt_instance(task_id, item, default_levels, env_id=spec.env_id)
            golden_renders[task_id][spec.env_id] = inst.rendered_sha256

    payload: Dict[str, Any] = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "study_id": "prompt-prism-real-v4-selection-stability-2026",
        "environments": [spec.model_dump(mode="json") for spec in specs],
        "parser_hashes": p_hashes,
        "golden_render_sha256": golden_renders,
    }
    payload["registry_sha256"] = canonical_sha256(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    write_perturbation_registry()
