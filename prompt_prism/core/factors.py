"""
Core data structures for representing Prompt Factors and Levels in Design of Experiments.
"""

from __future__ import annotations

import enum
from typing import Any, Dict, List, Optional, Sequence, Union

from pydantic import BaseModel, Field, field_validator


class FactorType(str, enum.Enum):
    """Types of prompt factors."""

    TEXT = "text"  # Raw text insertion / replacement
    SECTION = "section"  # Modular prompt section (toggleable or multi-variant)
    TEMPLATE = "template"  # Dynamic template string
    PARAM = "param"  # Model generation parameter (e.g. temperature, top_p)
    BOOLEAN = "boolean"  # Simple on/off toggle
    CATEGORICAL = "categorical"  # Multi-level categorical variant
    NUMERIC = "numeric"  # Numeric scale factor


class Level(BaseModel):
    """
    Represents a specific level (variant/state) of a Factor.

    Attributes:
        code: The coded representation (e.g., 0, 1, or -1, +1, or custom index).
        name: Human-readable name (e.g., "Zero-Shot", "3-Shot Examples").
        content: The actual prompt content, template fragment, or parameter value.
        description: Optional description of this level.
        metadata: Optional dictionary of additional metadata.
    """

    code: int = 0
    name: str = ""
    content: Any = ""
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def __init__(
        self,
        code: int = 0,
        name: str = "",
        content: Any = "",
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(
            code=code,
            name=name
            or (
                str(content)
                if isinstance(content, str) and len(str(content)) < 30
                else f"Level {code}"
            ),
            content=content,
            description=description,
            metadata=metadata or {},
            **kwargs,
        )

    def __repr__(self) -> str:
        return f"Level(code={self.code}, name='{self.name}', content={repr(self.content)[:30]})"


class Factor(BaseModel):
    """
    Represents a variable factor in a Prompt Design of Experiments.

    Attributes:
        id: Standard single-letter or short identifier (e.g. 'A', 'B', 'C', 'X1').
        name: Full descriptive name of the factor (e.g. 'persona', 'few_shot_examples').
        factor_type: Type of factor (TEXT, SECTION, TEMPLATE, PARAM, etc.).
        levels: List of available levels (usually 2 for binary fractional factorial designs).
        description: Detailed explanation of what this factor tests.
        position: Optional ordering position when assembling the prompt sections.
        default_level_code: Default level code if not specified.
        metadata: Extra metadata dictionary.
    """

    id: str = ""
    name: str
    factor_type: FactorType = FactorType.SECTION
    levels: List[Level] = Field(default_factory=list)
    description: Optional[str] = None
    position: int = 0
    default_level_code: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("levels", mode="before")
    @classmethod
    def normalize_levels(cls, v: Any) -> List[Level]:
        if not v:
            return [
                Level(code=0, name="Disabled", content=""),
                Level(code=1, name="Enabled", content=""),
            ]
        if isinstance(v, list):
            res = []
            for idx, item in enumerate(v):
                if isinstance(item, Level):
                    res.append(item)
                elif isinstance(item, tuple) and len(item) == 2:
                    res.append(Level(code=idx, name=str(item[0]), content=item[1]))
                elif isinstance(item, dict):
                    res.append(Level(**item))
                else:
                    res.append(Level(code=idx, name=f"Level {idx}", content=item))
            return res
        return v

    def get_level(self, code_or_name: Union[int, str]) -> Level:
        """Find a level by its code or name."""
        if isinstance(code_or_name, int):
            for level in self.levels:
                if level.code == code_or_name:
                    return level
        elif isinstance(code_or_name, str):
            for level in self.levels:
                if (
                    level.name.lower() == code_or_name.lower()
                    or str(level.code) == code_or_name
                ):
                    return level
        # Default fallback to first level or level 0
        for level in self.levels:
            if level.code == self.default_level_code:
                return level
        return self.levels[0] if self.levels else Level(code=0, content="")

    @classmethod
    def binary(
        cls,
        name: str,
        level_0_content: Any = "",
        level_1_content: Any = "",
        id: str = "",
        description: Optional[str] = None,
        level_0_name: str = "Off",
        level_1_name: str = "On",
        position: int = 0,
        factor_type: FactorType = FactorType.SECTION,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Factor:
        """Convenience constructor for 2-level (binary) factors."""
        return cls(
            id=id,
            name=name,
            factor_type=factor_type,
            levels=[
                Level(code=0, name=level_0_name, content=level_0_content),
                Level(code=1, name=level_1_name, content=level_1_content),
            ],
            description=description,
            position=position,
            metadata=metadata or {},
        )

    @classmethod
    def demo(
        cls,
        name: str,
        user_content: str,
        assistant_content: str,
        *,
        id: str = "",
        description: Optional[str] = None,
        multi_turn: bool = True,
        position: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Factor:
        """Construct a binary few-shot demonstration Factor (Off vs. native chat turn or text block).

        When `multi_turn=True`, Level 1 stores a 2-turn chat payload
        `[{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]`
        which `PromptComposer.compose_messages()` emits as genuine alternating chat turns.
        """
        lvl1: Any = (
            [
                {"role": "user", "content": str(user_content)},
                {"role": "assistant", "content": str(assistant_content)},
            ]
            if multi_turn
            else f"Input:\n{user_content}\nOutput:\n{assistant_content}"
        )
        meta = {"clause_role": "global_scaffold", "is_demo": True}
        if metadata:
            meta.update(metadata)
        return cls.binary(
            name=name,
            level_0_content="",
            level_1_content=lvl1,
            id=id,
            description=description or f"Candidate demonstration '{name}'",
            level_0_name="Omit",
            level_1_name="Include",
            position=position,
            factor_type=FactorType.SECTION,
            metadata=meta,
        )

    def __repr__(self) -> str:
        return f"Factor(id='{self.id}', name='{self.name}', levels={len(self.levels)})"


class FactorSet:
    """
    A collection of factors for an experiment, with automatic ID assignment and lookup.
    """

    def __init__(self, factors: Optional[Sequence[Factor]] = None):
        self._factors: List[Factor] = []
        self._by_id: Dict[str, Factor] = {}
        self._by_name: Dict[str, Factor] = {}
        if factors:
            for f in factors:
                self.add(f)

    def add(self, factor: Factor) -> Factor:
        """Add a factor to the set, assigning an ID if missing."""
        if not factor.id:
            factor.id = self._generate_next_id()
        self._factors.append(factor)
        self._by_id[factor.id] = factor
        self._by_name[factor.name] = factor
        return factor

    def _generate_next_id(self) -> str:
        """Generate alphabetical IDs: A, B, ..., Z, AA, AB, ..."""
        idx = len(self._factors)
        alphabet = "ABCDEFGHJKLMNOPQRSTUVWXYZ"  # Standard DoE skips I to avoid confusion with Identity I
        if idx < len(alphabet):
            return alphabet[idx]
        first = alphabet[(idx // len(alphabet)) - 1]
        second = alphabet[idx % len(alphabet)]
        return f"{first}{second}"

    def get(self, key: Union[str, int]) -> Optional[Factor]:
        """Get factor by ID, name, or index."""
        if isinstance(key, int):
            return self._factors[key] if 0 <= key < len(self._factors) else None
        return self._by_id.get(key) or self._by_name.get(key)

    def __getitem__(self, key: Union[str, int]) -> Factor:
        f = self.get(key)
        if f is None:
            raise KeyError(f"Factor '{key}' not found in FactorSet")
        return f

    def __iter__(self):
        return iter(self._factors)

    def __len__(self) -> int:
        return len(self._factors)

    @property
    def ids(self) -> List[str]:
        return [f.id for f in self._factors]

    @property
    def names(self) -> List[str]:
        return [f.name for f in self._factors]

    def to_dict(self) -> List[Dict[str, Any]]:
        return [f.model_dump() for f in self._factors]

    @classmethod
    def from_dict(cls, data: Sequence[Dict[str, Any]]) -> FactorSet:
        return cls([Factor(**item) for item in data])

    @classmethod
    def from_candidate_demos(
        cls,
        demos: Sequence[Any],
        *,
        input_fields: Optional[Sequence[str]] = None,
        output_key: str = "target",
        field_style: str = "xml",
        output_prefix: str = "",
        multi_turn: bool = True,
        name_prefix: str = "demo",
        extra_factors: Optional[Sequence[Factor]] = None,
    ) -> FactorSet:
        """Build a FactorSet containing binary Factors for each candidate in-domain demonstration.

        Each item in `demos` may be either:
          - a `(user_text, assistant_text)` 2-tuple, or
          - a dict containing `input_fields` (or all keys except `output_key`/metadata) and `output_key`.

        `field_style` controls how dict inputs are framed:
          - `"xml"` (default): `<field>value</field>`
          - `"brackets"`: `[[ ## field ## ]]\\nvalue`
          - `"plain"`: `field: value`
        """
        if field_style not in {"xml", "brackets", "plain"}:
            raise ValueError(
                f"field_style must be 'xml', 'brackets', or 'plain', got {field_style!r}"
            )
        factors: List[Factor] = list(extra_factors or [])
        reserved_meta = {"sample_id", "group_id", "source", "task_id", "split"}

        for idx, raw_demo in enumerate(demos, start=1):
            if isinstance(raw_demo, tuple) and len(raw_demo) == 2:
                user_text = str(raw_demo[0])
                assistant_text = f"{output_prefix}{raw_demo[1]}"
            elif isinstance(raw_demo, dict):
                if output_key not in raw_demo:
                    raise ValueError(
                        f"Candidate demo #{idx} missing output_key={output_key!r}"
                    )
                if input_fields is not None:
                    fields = list(input_fields)
                else:
                    fields = [
                        k
                        for k in raw_demo.keys()
                        if k != output_key and k not in reserved_meta
                    ]
                parts: List[str] = []
                for fld in fields:
                    val = raw_demo.get(fld, "")
                    if isinstance(val, list):
                        val_str = ", ".join(str(x) for x in val)
                    else:
                        val_str = str(val)
                    if field_style == "xml":
                        parts.append(f"<{fld}>{val_str}</{fld}>")
                    elif field_style == "brackets":
                        parts.append(f"[[ ## {fld} ## ]]\n{val_str}")
                    else:
                        parts.append(f"{fld}: {val_str}")
                user_text = "\n".join(parts)
                raw_out = raw_demo[output_key]
                out_str = (
                    str(raw_out[0])
                    if isinstance(raw_out, list) and raw_out
                    else str(raw_out)
                )
                assistant_text = f"{output_prefix}{out_str}"
            else:
                raise TypeError(
                    f"Candidate demo #{idx} must be a (user, assistant) tuple or dict, got {type(raw_demo).__name__}"
                )

            factors.append(
                Factor.demo(
                    name=f"{name_prefix}_{idx}",
                    user_content=user_text,
                    assistant_content=assistant_text,
                    multi_turn=multi_turn,
                )
            )
        return cls(factors)
