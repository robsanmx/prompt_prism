"""
Modular Prompt Composition & Templating Engine for Design of Experiments.
"""

from __future__ import annotations

import string
from typing import Any, Dict, List, Optional, Sequence, Union

import jinja2
from pydantic import BaseModel, Field

from ..core.factors import Factor, FactorSet
from ..core.models import RunConfig


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
        Compose structured chat messages: [{'role': 'system', 'content': '...'}, {'role': 'user', 'content': '...'}]
        """
        data = data or {}
        sorted_sections = sorted(self.template.sections, key=lambda s: s.position)

        # Group rendered sections by role
        roles_order = []
        messages_by_role: Dict[str, List[str]] = {}

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

            rendered = sec.render(factor_level_content=level_content, data=data)
            if rendered:
                role = sec.role or self.template.default_role
                if role not in messages_by_role:
                    messages_by_role[role] = []
                    roles_order.append(role)
                messages_by_role[role].append(rendered)

        # Build messages
        messages = []
        for role in roles_order:
            content = self.template.delimiter.join(messages_by_role[role]).strip()
            if content:
                messages.append({"role": role, "content": content})

        if not messages:
            # Fallback
            full_text = self.compose_text(run_config, data)
            if full_text:
                messages.append(
                    {"role": self.template.default_role, "content": full_text}
                )

        return messages
