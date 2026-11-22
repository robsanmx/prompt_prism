"""
Modular Prompt Composition & Templating Engine for Design of Experiments.
"""

from __future__ import annotations

import string
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import jinja2
from jinja2 import meta as jinja2_meta
from pydantic import BaseModel, Field

from ..core.factors import Factor, FactorSet
from ..core.models import RunConfig, _canonical_json_sha256


def _extract_template_variables(raw_text: str) -> Set[str]:
    """Extract undeclared placeholder variable names from Jinja2 or `string.Template` text."""
    if not raw_text:
        return set()
    if "{{" in raw_text or "{%" in raw_text:
        env = jinja2.Environment()
        parsed = env.parse(raw_text)
        return set(jinja2_meta.find_undeclared_variables(parsed))
    if "$" in raw_text:
        return set(string.Template(raw_text).get_identifiers())
    return set()


class PromptSection(BaseModel):
    """
    A modular section of a prompt that can be toggled, varied, or conditioned on a factor.

    Attributes:
        id: Section identifier (e.g. 'persona', 'task', 'examples', 'constraints').
        factor_id: Optional ID or name of the Factor controlling this section.
        content: Default content or template string if not overridden by factor levels.
        role: Message role ('system', 'user', 'assistant') for chat completion models.
        position: Ordering position (lower numbers appear earlier in prompt).
        prefix: Optional prefix (e.g., '### INSTRUCTIONS:\n').
        suffix: Optional suffix (e.g., '\n').
        delimiter: Separator after this section (default newline).
        is_active: Whether section is active regardless of factors.
    """

    id: str
    factor_id: Optional[str] = None
    content: str = ""
    role: Optional[str] = (
        None  # 'system', 'user', 'assistant', or None for text template
    )
    position: int = 0
    prefix: str = ""
    suffix: str = ""
    delimiter: str = "\n\n"
    is_active: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def render(
        self,
        factor_level_content: Optional[Any] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Render this section with variable interpolation."""
        raw_text = (
            str(factor_level_content)
            if factor_level_content is not None
            else self.content
        )
        if not raw_text.strip():
            return ""

        # Interpolate sample data if present
        if data:
            # 1. Try Jinja2 if template markers present
            if "{{" in raw_text or "{%" in raw_text:
                try:
                    tmpl = jinja2.Template(raw_text)
                    raw_text = tmpl.render(**data)
                except Exception:
                    pass
            # 2. Try Python string.Template / format
            elif "{" in raw_text or "$" in raw_text:
                try:
                    raw_text = string.Template(raw_text).safe_substitute(data)
                except Exception:
                    pass

        rendered = raw_text.strip()
        if rendered:
            if self.prefix:
                rendered = f"{self.prefix}{rendered}"
            if self.suffix:
                rendered = f"{rendered}{self.suffix}"
        return rendered


class PromptTemplate:
    """
    A template containing multiple modular sections or a Jinja2 master template.
    """

    def __init__(
        self,
        sections: Optional[Sequence[PromptSection]] = None,
        master_template: Optional[str] = None,
        default_role: str = "user",
        delimiter: str = "\n\n",
    ):
        self.sections: List[PromptSection] = list(sections or [])
        self.master_template: Optional[str] = master_template
        self.default_role = default_role
        self.delimiter = delimiter

    def add_section(
        self,
        id: str,
        factor_id: Optional[str] = None,
        content: str = "",
        role: Optional[str] = None,
        position: Optional[int] = None,
        prefix: str = "",
        suffix: str = "",
        delimiter: str = "\n\n",
    ) -> PromptSection:
        """
        Add a section to the prompt template.

        Note: Setting an explicit `role` (or `system_prompt`) on any section switches
        `PromptComposer.compose` to output structured chat messages (`List[Dict[str, str]]`),
        which is passed directly to the LLM client. Leaving `role=None` preserves plain text output.
        """
        pos = position if position is not None else len(self.sections)
        sec = PromptSection(
            id=id,
            factor_id=factor_id,
            content=content,
            role=role,
            position=pos,
            prefix=prefix,
            suffix=suffix,
            delimiter=delimiter,
        )
        self.sections.append(sec)
        self.sections.sort(key=lambda s: s.position)
        return sec

    @classmethod
    def from_factors(
        cls,
        factors: Union[FactorSet, Sequence[Factor]],
        system_prompt: Optional[str] = None,
        data_template: Optional[str] = None,
    ) -> PromptTemplate:
        """
        Convenience constructor: creates a PromptTemplate directly from a set of Factors.
        Each Factor becomes a PromptSection controlled by that Factor.
        """
        template = cls()
        pos = 0
        if system_prompt:
            template.add_section(
                id="system_static", content=system_prompt, role="system", position=pos
            )
            pos += 1

        for factor in factors:
            template.add_section(
                id=factor.name,
                factor_id=factor.id or factor.name,
                position=factor.position or pos,
            )
            pos += 1

        if data_template:
            template.add_section(id="data_payload", content=data_template, position=pos)

        return template

    @classmethod
    def structured_chat(
        cls,
        factors: Union[FactorSet, Sequence[Factor]],
        *,
        system_prompt: Optional[str] = None,
        input_fields: Optional[Sequence[str]] = None,
        data_template: Optional[str] = None,
        field_style: str = "xml",
    ) -> PromptTemplate:
        """Build a perturbation-resilient structured chat PromptTemplate.

        Places `system_prompt` and non-demo instruction factors in `role='system'`,
        emits demo factors (`metadata['is_demo']=True` or list-of-turns content) as
        native alternating `user`/`assistant` chat turns, and frames task inputs in
        explicit XML (`<field>{{ field }}</field>`) or bracket (`[[ ## field ## ]]`)
        delimiters in the final `role='user'` turn.
        """
        if field_style not in {"xml", "brackets"}:
            raise ValueError(
                f"field_style must be 'xml' or 'brackets', got {field_style!r}"
            )
        template = cls(default_role="user")
        pos = 0
        if system_prompt:
            template.add_section(
                id="system_static", content=system_prompt, role="system", position=pos
            )
            pos += 1

        demo_factors: List[Factor] = []
        inst_factors: List[Factor] = []
        for factor in factors:
            is_demo = bool(factor.metadata.get("is_demo")) or any(
                isinstance(lvl.content, list) for lvl in factor.levels
            )
            if is_demo:
                demo_factors.append(factor)
            else:
                inst_factors.append(factor)

        for factor in inst_factors:
            sec_role = str(factor.metadata.get("role") or "system")
            template.add_section(
                id=factor.name,
                factor_id=factor.id or factor.name,
                role=sec_role,
                position=pos,
            )
            pos += 1

        for factor in demo_factors:
            template.add_section(
                id=factor.name,
                factor_id=factor.id or factor.name,
                role="user",
                position=pos,
            )
            pos += 1

        if data_template is not None:
            payload_tmpl = data_template
        elif input_fields:
            if field_style == "xml":
                payload_tmpl = "\n".join(
                    f"<{fld}>{{{{ {fld} }}}}</{fld}>" for fld in input_fields
                )
            else:
                payload_tmpl = "\n\n".join(
                    f"[[ ## {fld} ## ]]\n{{{{ {fld} }}}}" for fld in input_fields
                )
        else:
            payload_tmpl = None

        if payload_tmpl:
            template.add_section(
                id="data_payload",
                content=payload_tmpl,
                role="user",
                position=pos,
            )

        return template

    def to_dict(self) -> Dict[str, Any]:
        """Serialize template structure to a canonical JSON-compatible dictionary."""
        return {
            "master_template": self.master_template,
            "default_role": self.default_role,
            "delimiter": self.delimiter,
            "sections": [sec.model_dump(mode="json") for sec in self.sections],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PromptTemplate:
        """Reconstruct a PromptTemplate from a dictionary produced by `to_dict()`."""
        sections = [PromptSection(**s) for s in data.get("sections", [])]
        return cls(
            sections=sections,
            master_template=data.get("master_template"),
            default_role=data.get("default_role", "user"),
            delimiter=data.get("delimiter", "\n\n"),
        )

    def template_sha256(self) -> str:
        """Compute canonical-JSON SHA-256 hash of this PromptTemplate."""
        return _canonical_json_sha256(self.to_dict())

    def required_data_variables(
        self,
        factors: Optional[Union[FactorSet, Sequence[Factor]]] = None,
    ) -> Set[str]:
        """Extract all dataset variable names required to render this template.

        Handles both `master_template` (Jinja2, subtracting factor context keys) and
        section-based templates (Jinja2 and `string.Template` `$var` placeholders).
        """
        factor_set = (
            factors
            if isinstance(factors, FactorSet)
            else (FactorSet(factors) if factors else FactorSet())
        )
        factor_context_keys: Set[str] = set()
        for factor in factor_set:
            factor_context_keys.add(factor.name)
            factor_context_keys.add(f"{factor.name}_level")
            factor_context_keys.add(f"{factor.name}_name")
            if factor.id:
                factor_context_keys.add(factor.id)
                factor_context_keys.add(f"{factor.id}_level")

        if self.master_template:
            return (
                _extract_template_variables(self.master_template) - factor_context_keys
            )

        required: Set[str] = set()
        for sec in self.sections:
            if not sec.is_active:
                continue
            if sec.factor_id:
                factor = factor_set.get(sec.factor_id)
                if factor is not None:
                    for lvl in factor.levels:
                        if isinstance(lvl.content, str):
                            required.update(_extract_template_variables(lvl.content))
                elif sec.content:
                    required.update(_extract_template_variables(sec.content))
            elif sec.content:
                required.update(_extract_template_variables(sec.content))
        return required


class PromptComposer:
    """
    Compiles full prompts or chat message arrays for specific DoE RunConfigs and input datasets.
    """

    def __init__(
        self,
        template: PromptTemplate,
        factors: Optional[Union[FactorSet, Sequence[Factor]]] = None,
    ):
        self.template = template
        self.factors = (
            FactorSet(factors)
            if isinstance(factors, (list, tuple))
            else (factors or FactorSet())
        )

    def compose_text(
        self,
        run_config: RunConfig,
        data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Compose a single formatted prompt string for a given run configuration.
        """
        data = data or {}

        # If master Jinja2 template is provided
        if self.template.master_template:
            # Build context dict with factor level contents and data
            context: Dict[str, Any] = dict(data)
            for factor in self.factors:
                level_code = run_config.factor_levels.get(
                    factor.id, run_config.factor_names.get(factor.name, 0)
                )
                lvl = factor.get_level(level_code)
                context[factor.name] = lvl.content
                context[f"{factor.name}_level"] = level_code
                context[f"{factor.name}_name"] = lvl.name
                if factor.id:
                    context[factor.id] = lvl.content
                    context[f"{factor.id}_level"] = level_code

            tmpl = jinja2.Template(self.template.master_template)
            return tmpl.render(**context).strip()

        # Section-based composition
        sorted_sections = sorted(self.template.sections, key=lambda s: s.position)
        rendered_parts = []

        for sec in sorted_sections:
            if not sec.is_active:
                continue

            level_content = None
            if sec.factor_id:
                factor = self.factors.get(sec.factor_id)
                if factor:
                    lvl_code = run_config.factor_levels.get(
                        factor.id, run_config.factor_names.get(factor.name, 0)
                    )
                    level = factor.get_level(lvl_code)
                    level_content = level.content

            if isinstance(level_content, list):
                turn_lines = []
                for turn in level_content:
                    if isinstance(turn, dict) and str(turn.get("content", "")).strip():
                        r_lbl = str(turn.get("role", "user")).capitalize()
                        t_txt = sec.render(
                            factor_level_content=str(turn["content"]), data=data
                        )
                        if t_txt:
                            turn_lines.append(f"{r_lbl}: {t_txt}")
                rendered = "\n".join(turn_lines)
            else:
                rendered = sec.render(factor_level_content=level_content, data=data)
            if rendered:
                rendered_parts.append(rendered)

        return self.template.delimiter.join(rendered_parts).strip()

    def compose_messages(
        self,
        run_config: RunConfig,
        data: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        """
        Compose structured chat messages preserving multi-turn ordering:
        e.g. [{'role': 'system', ...}, {'role': 'user', ...}, {'role': 'assistant', ...}, {'role': 'user', ...}].

        Consecutive sections sharing the same role are merged with `template.delimiter`;
        alternating `user`/`assistant` turns (including list-of-turn demo factors) are preserved intact.
        """
        data = data or {}
        sorted_sections = sorted(self.template.sections, key=lambda s: s.position)

        raw_turns: List[Tuple[str, str]] = []
        for sec in sorted_sections:
            if not sec.is_active:
                continue

            level_content = None
            if sec.factor_id:
                factor = self.factors.get(sec.factor_id)
                if factor:
                    lvl_code = run_config.factor_levels.get(
                        factor.id, run_config.factor_names.get(factor.name, 0)
                    )
                    level = factor.get_level(lvl_code)
                    level_content = level.content

            # Support multi-turn chat payloads (e.g. Factor.demo(..., multi_turn=True))
            if isinstance(level_content, list):
                for turn in level_content:
                    if not isinstance(turn, dict):
                        continue
                    t_role = str(
                        turn.get("role") or sec.role or self.template.default_role
                    )
                    t_rendered = sec.render(
                        factor_level_content=str(turn.get("content", "")), data=data
                    )
                    if t_rendered:
                        raw_turns.append((t_role, t_rendered))
                continue

            rendered = sec.render(factor_level_content=level_content, data=data)
            if rendered:
                role = sec.role or self.template.default_role
                raw_turns.append((role, rendered))

        # Merge ONLY consecutive same-role turns so alternating user/assistant turns stay intact
        messages: List[Dict[str, str]] = []
        for role, text in raw_turns:
            if messages and messages[-1]["role"] == role:
                messages[-1][
                    "content"
                ] = f"{messages[-1]['content']}{self.template.delimiter}{text}".strip()
            else:
                messages.append({"role": role, "content": text.strip()})

        if not messages:
            # Fallback
            full_text = self.compose_text(run_config, data)
            if full_text:
                messages.append(
                    {"role": self.template.default_role, "content": full_text}
                )

        return messages
