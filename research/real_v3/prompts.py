"""Task-specific prompt rendering for the real-v3 factorial."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class RenderedPrompt:
    system_prompt: str
    user_prompt: str


def _task_input(task_id: str, item: dict) -> str:
    if task_id == "schema_extraction":
        return item["text"]
    if task_id == "grounded_qa":
        return f"Context:\n{item['context']}\n\nQuestion:\n{item['question']}"
    if task_id == "intent_classification":
        return item["utterance"]
    raise ValueError(f"Unknown task: {task_id}")


def _task_instruction(task_id: str, strict_contract: bool, labels: Sequence[str]) -> str:
    if task_id == "schema_extraction":
        base = (
            "Extract every annotated mention of person, organization, location, money, and temporal "
            "entities. Copy exact spans from the passage. Return a JSON object with these five keys; "
            "each value must be an array, including empty arrays when no entity is present."
        )
        if strict_contract:
            base += (
                " Output only valid JSON with exactly the keys person, organization, location, money, "
                "and temporal. Do not add Markdown or commentary."
            )
        return base
    if task_id == "grounded_qa":
        base = (
            "Answer from the context using the shortest sufficient answer span. If the context does not "
            "answer the question, return unanswerable."
        )
        if strict_contract:
            base += (
                ' Output only one JSON object with exactly the key "answer". Its value must be the answer '
                "string or null when unanswerable. Do not add Markdown or commentary."
            )
        return base
    if task_id == "intent_classification":
        inventory = ", ".join(labels)
        base = f"Classify the inquiry into exactly one of these intent labels: {inventory}."
        if strict_contract:
            base += (
                ' Output only one JSON object with exactly the key "intent" and one listed label as its '
                "value. Do not add Markdown or commentary."
            )
        else:
            base += " Return the label only, without explanation."
        return base
    raise ValueError(f"Unknown task: {task_id}")


def _render_examples(task_id: str, examples: List[Tuple[dict, dict]]) -> str:
    blocks = []
    for item, expected in examples:
        blocks.append(
            "Example input:\n"
            + _task_input(task_id, item)
            + "\nExample output:\n"
            + json.dumps(expected, ensure_ascii=False, sort_keys=True)
        )
    return "\n\n".join(blocks)


def render_prompt(
    task_id: str,
    item: dict,
    factors: Dict[str, int],
    examples_by_task: Dict[str, List[Tuple[dict, dict]]],
    intent_labels: Sequence[str],
) -> RenderedPrompt:
    """Render one factorial prompt while changing only declared factor sections."""
    system_prompt = ""
    if factors.get("A", -1) == 1:
        system_prompt = (
            "You are a careful information extraction and classification specialist. "
            "Follow the task definition exactly."
        )

    parts = []
    if factors.get("B", -1) == 1:
        parts.append("Analyze the requirements and evidence silently before returning only the final answer.")

    parts.append(_task_instruction(task_id, factors.get("D", -1) == 1, intent_labels))

    if factors.get("C", -1) == 1:
        examples = examples_by_task.get(task_id, [])
        if not examples:
            raise ValueError(f"Factor C is active but no pilot examples exist for {task_id}")
        parts.append(_render_examples(task_id, examples))

    parts.append("Evaluation input:\n" + _task_input(task_id, item))

    if factors.get("E", -1) == 1:
        parts.append(
            "Use only the supplied input. Do not infer facts or labels that are unsupported by it."
        )
    if factors.get("F", -1) == 1:
        parts.append(
            "Before responding, silently verify that the answer follows the requested format and is supported by the input."
        )

    return RenderedPrompt(system_prompt=system_prompt, user_prompt="\n\n".join(parts))
