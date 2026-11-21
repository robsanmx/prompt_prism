"""
YAML/JSON configuration and LLM client loaders for PromptPrism test set generation.

Supports:
- Loading binary/general Factor definitions (`factors.yaml` / `factors.json`)
- Loading PromptTemplate definitions (`template.yaml` / `template.json`)
- Loading optional seed example files (`.jsonl`, `.json`, `.yaml`, `.csv`)
- Resolving `--client module:callable` (core) and optional `--model <name>` via LiteLLM (`cloud` extra)
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd
import yaml

from ..core.factors import Factor, FactorSet, FactorType
from ..runner.client import CallableLLM, LLMClient, LLMResponse
from ..template.composer import PromptTemplate

_VALID_CLAUSE_ROLES = {"conditional_rule", "global_scaffold"}
_BINARY_SHORTHAND_KEYS = {
    "level_0",
    "level_1",
    "level_0_content",
    "level_1_content",
    "level_0_name",
    "level_1_name",
}
_ALLOWED_BINARY_FACTOR_KEYS = {
    "id",
    "name",
    "description",
    "position",
    "factor_type",
    "metadata",
    "default_level_code",
    "clause_role",
    *_BINARY_SHORTHAND_KEYS,
}
_ALLOWED_FULL_FACTOR_KEYS = {
    "id",
    "name",
    "description",
    "position",
    "factor_type",
    "levels",
    "metadata",
    "default_level_code",
    "clause_role",
}
_ALLOWED_TEMPLATE_DICT_KEYS = {
    "sections",
    "master_template",
    "default_role",
    "delimiter",
}
_ALLOWED_TEMPLATE_SHORTHAND_KEYS = {
    "system_prompt",
    "data_template",
    "default_role",
    "delimiter",
}


def _load_yaml_or_json(path: Union[str, Path]) -> Any:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    text = file_path.read_text(encoding="utf-8")
    try:
        if file_path.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            data = yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ValueError(
            f"Failed to parse configuration file '{file_path}': {exc}"
        ) from exc
    if data is None:
        raise ValueError(f"Configuration file is empty: {file_path}")
    return data


def _coerce_factor_type(raw: Any) -> FactorType:
    if isinstance(raw, FactorType):
        return raw
    if isinstance(raw, str):
        try:
            return FactorType(raw.strip().lower())
        except ValueError as exc:
            valid = sorted(ft.value for ft in FactorType)
            raise ValueError(
                f"Invalid factor_type {raw!r}; expected one of {valid}."
            ) from exc
    raise ValueError(f"Invalid factor_type {raw!r}; expected string or FactorType.")


def load_factors_file(path: Union[str, Path]) -> FactorSet:
    """Load a `FactorSet` from a YAML or JSON file.

    Accepts either a top-level list of factor definitions or a mapping with a
    `"factors"` list. Each factor maps 1:1 to `Factor.binary(...)` (supporting
    `level_0`/`level_0_content`, `level_1`/`level_1_content`, `level_0_name`,
    `level_1_name`, and optional `clause_role`) or full `Factor(levels=[...])`.
    """
    raw = _load_yaml_or_json(path)
    if isinstance(raw, dict):
        if "factors" not in raw:
            raise ValueError(
                f"Factors file '{path}' must be a list or a mapping containing a 'factors' key."
            )
        extra_top = sorted(set(raw.keys()) - {"factors"})
        if extra_top:
            raise ValueError(
                f"Factors file '{path}' contains unexpected top-level keys: {extra_top}."
            )
        items = raw["factors"]
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError(
            f"Factors file '{path}' must contain a list of factors or a mapping with 'factors'."
        )

    if not isinstance(items, list) or not items:
        raise ValueError(f"Factors file '{path}' must define at least one factor.")

    factors: List[Factor] = []
    seen_ids: set[str] = set()
    seen_names: set[str] = set()

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(
                f"Factor entry #{idx} in '{path}' must be a mapping, got {type(item).__name__}."
            )
        entry = dict(item)
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"Factor entry #{idx} in '{path}' requires a non-empty 'name'."
            )
        name = name.strip()
        if name in seen_names:
            raise ValueError(f"Duplicate factor name {name!r} in '{path}'.")
        seen_names.add(name)

        raw_id = entry.get("id")
        fid = "" if raw_id is None else str(raw_id).strip()
        if fid:
            if fid in seen_ids:
                raise ValueError(f"Duplicate factor id {fid!r} in '{path}'.")
            seen_ids.add(fid)

        raw_meta = entry.get("metadata")
        if raw_meta is None:
            meta: Dict[str, Any] = {}
        elif isinstance(raw_meta, dict):
            meta = dict(raw_meta)
        else:
            raise ValueError(f"Factor {name!r} 'metadata' must be a mapping.")

        clause_role = entry.get("clause_role")
        if clause_role is not None:
            role_str = str(clause_role).strip()
            if role_str not in _VALID_CLAUSE_ROLES:
                raise ValueError(
                    f"Factor {name!r} has invalid clause_role {clause_role!r}; "
                    f"expected one of {sorted(_VALID_CLAUSE_ROLES)}."
                )
            meta["clause_role"] = role_str
        elif "clause_role" in meta:
            role_str = str(meta["clause_role"]).strip()
            if role_str not in _VALID_CLAUSE_ROLES:
                raise ValueError(
                    f"Factor {name!r} has invalid metadata.clause_role {meta['clause_role']!r}; "
                    f"expected one of {sorted(_VALID_CLAUSE_ROLES)}."
                )
            meta["clause_role"] = role_str

        has_levels = "levels" in entry
        has_shorthand = bool(set(entry.keys()) & _BINARY_SHORTHAND_KEYS)
        if has_levels and has_shorthand:
            raise ValueError(
                f"Factor {name!r} specifies both 'levels' and binary shorthand keys "
                f"({sorted(set(entry.keys()) & _BINARY_SHORTHAND_KEYS)})."
            )

        if has_levels:
            unknown = sorted(set(entry.keys()) - _ALLOWED_FULL_FACTOR_KEYS)
            if unknown:
                raise ValueError(f"Factor {name!r} has unknown key(s): {unknown}.")
            entry.pop("clause_role", None)
            entry["name"] = name
            entry["id"] = fid
            entry["metadata"] = meta
            if entry.get("factor_type") is None:
                entry.pop("factor_type", None)
            else:
                entry["factor_type"] = _coerce_factor_type(entry["factor_type"])
            if entry.get("position") is None:
                entry.pop("position", None)
            if entry.get("default_level_code") is None:
                entry.pop("default_level_code", None)
            factors.append(Factor(**entry))
        else:
            unknown = sorted(set(entry.keys()) - _ALLOWED_BINARY_FACTOR_KEYS)
            if unknown:
                raise ValueError(f"Factor {name!r} has unknown key(s): {unknown}.")
            if "level_0" in entry and "level_0_content" in entry:
                raise ValueError(
                    f"Factor {name!r} specifies both 'level_0' and 'level_0_content'."
                )
            if "level_1" in entry and "level_1_content" in entry:
                raise ValueError(
                    f"Factor {name!r} specifies both 'level_1' and 'level_1_content'."
                )
            raw_l0 = (
                entry["level_0_content"]
                if "level_0_content" in entry
                else entry.get("level_0", "")
            )
            raw_l1 = (
                entry["level_1_content"]
                if "level_1_content" in entry
                else entry.get("level_1", "")
            )
            l0 = "" if raw_l0 is None else raw_l0
            l1 = "" if raw_l1 is None else raw_l1
            raw_l0_name = entry.get("level_0_name")
            raw_l1_name = entry.get("level_1_name")
            l0_name = "Off" if raw_l0_name is None else str(raw_l0_name)
            l1_name = "On" if raw_l1_name is None else str(raw_l1_name)
            raw_ftype = entry.get("factor_type")
            ftype = (
                FactorType.SECTION
                if raw_ftype is None
                else _coerce_factor_type(raw_ftype)
            )
            raw_pos = entry.get("position")
            pos = idx if raw_pos is None else int(raw_pos)
            factor = Factor.binary(
                name=name,
                level_0_content=l0,
                level_1_content=l1,
                id=fid,
                description=entry.get("description"),
                level_0_name=l0_name,
                level_1_name=l1_name,
                position=pos,
                factor_type=ftype,
                metadata=meta,
            )
            if entry.get("default_level_code") is not None:
                factor.default_level_code = int(entry["default_level_code"])
            factors.append(factor)

    fset = FactorSet(factors)
    if len(set(fset.ids)) != len(fset):
        counts: Dict[str, int] = {}
        for f in fset:
            counts[f.id] = counts.get(f.id, 0) + 1
        dups = sorted(k for k, v in counts.items() if v > 1)
        raise ValueError(
            f"Factor ID collision in '{path}' (explicit and auto-assigned IDs overlap): {dups}. "
            "Specify explicit unique 'id' values for all factors or omit 'id' on all factors."
        )
    return fset


def load_template_file(
    path: Union[str, Path],
    *,
    factors: Optional[Union[FactorSet, Sequence[Factor]]] = None,
) -> PromptTemplate:
    """Load a `PromptTemplate` from a YAML or JSON file.

    Supports three schemas:
    1. Canonical `PromptTemplate.to_dict()` / `from_dict()` format (`sections`, `default_role`, `delimiter`).
    2. Jinja2 `master_template` format (`master_template`, optional `default_role`, `delimiter`).
    3. Convenience `PromptTemplate.from_factors()` shorthand (`system_prompt`, `data_template`),
       which requires `factors` to be provided.
    """
    raw = _load_yaml_or_json(path)
    if not isinstance(raw, dict):
        raise ValueError(
            f"Template file '{path}' must contain a YAML/JSON mapping, got {type(raw).__name__}."
        )

    keys = set(raw.keys())
    has_direct = bool(keys & {"sections", "master_template"})
    has_shorthand = bool(keys & {"system_prompt", "data_template"})

    if has_direct and has_shorthand:
        raise ValueError(
            f"Template file '{path}' mixes direct ('sections'/'master_template') and "
            "shorthand ('system_prompt'/'data_template') fields."
        )

    if has_direct:
        unknown = sorted(keys - _ALLOWED_TEMPLATE_DICT_KEYS)
        if unknown:
            raise ValueError(f"Template file '{path}' has unknown key(s): {unknown}.")
        if "sections" in raw:
            if not isinstance(raw["sections"], list):
                raise ValueError(
                    f"Template file '{path}' field 'sections' must be a list."
                )
            for idx, sec in enumerate(raw["sections"]):
                if not isinstance(sec, dict):
                    raise ValueError(
                        f"Template file '{path}' section #{idx} must be a mapping, "
                        f"got {type(sec).__name__}."
                    )
        return PromptTemplate.from_dict(raw)

    if has_shorthand:
        unknown = sorted(keys - _ALLOWED_TEMPLATE_SHORTHAND_KEYS)
        if unknown:
            raise ValueError(f"Template file '{path}' has unknown key(s): {unknown}.")
        if factors is None:
            raise ValueError(
                f"Template file '{path}' uses 'system_prompt'/'data_template' shorthand, "
                "which requires factors to be supplied."
            )
        tmpl = PromptTemplate.from_factors(
            factors,
            system_prompt=raw.get("system_prompt"),
            data_template=raw.get("data_template"),
        )
        if "default_role" in raw and raw["default_role"] is not None:
            tmpl.default_role = str(raw["default_role"])
        if "delimiter" in raw and raw["delimiter"] is not None:
            tmpl.delimiter = str(raw["delimiter"])
        return tmpl

    raise ValueError(
        f"Template file '{path}' must specify 'sections', 'master_template', or "
        "'system_prompt'/'data_template'."
    )


def load_seed_examples_file(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load optional seed examples from `.jsonl`, `.json`, `.yaml`/`.yml`, or `.csv`."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Seed examples file not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        try:
            rows: List[Any] = [
                json.loads(line)
                for line in file_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Failed to parse JSONL seed examples file '{file_path}': {exc}"
            ) from exc
    elif suffix == ".csv":
        df = pd.read_csv(file_path)
        rows = [
            {k: (None if pd.isna(v) else v) for k, v in rec.items()}
            for rec in df.to_dict(orient="records")
        ]
    else:
        raw = _load_yaml_or_json(file_path)
        if isinstance(raw, list):
            rows = raw
        elif isinstance(raw, dict):
            if isinstance(raw.get("examples"), list):
                rows = raw["examples"]
            elif isinstance(raw.get("seeds"), list):
                rows = raw["seeds"]
            else:
                raise ValueError(
                    f"Seed examples file '{file_path}' mapping must contain 'examples' or 'seeds' list."
                )
        else:
            raise ValueError(
                f"Seed examples file '{file_path}' must contain a list of example dicts."
            )

    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(
                f"Seed example #{idx} in '{file_path}' must be an object/mapping, got {type(row).__name__}."
            )
        out.append(dict(row))
    return out


class LiteLLMClient(LLMClient):
    """LLMClient backed by LiteLLM (`prompt-prism[cloud]` optional extra)."""

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("LiteLLMClient requires a non-empty model string.")
        try:
            import litellm  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "Using --model requires the optional 'cloud' dependency (litellm). "
                "Install it with: pip install 'prompt-prism[cloud]' "
                "(or pass --client module:callable instead)."
            ) from exc

        self._litellm = litellm
        self.model_id = model.strip()
        self.temperature = float(temperature)
        self.max_tokens = max_tokens
        self.extra_kwargs = dict(extra_kwargs or {})

    def generate(
        self, prompt: Union[str, List[Dict[str, str]]], **kwargs: Any
    ) -> LLMResponse:
        start_t = time.perf_counter()
        messages: List[Dict[str, str]] = (
            prompt
            if isinstance(prompt, list)
            else [{"role": "user", "content": str(prompt)}]
        )
        call_kwargs: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            **self.extra_kwargs,
        }
        if self.max_tokens is not None:
            call_kwargs["max_tokens"] = self.max_tokens
        call_kwargs.update(kwargs)

        try:
            resp = self._litellm.completion(**call_kwargs)
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            content = self._extract_content(resp)
            usage = self._extract_usage(resp)
            return LLMResponse(
                content=content,
                raw=resp,
                latency_ms=elapsed_ms,
                token_usage=usage,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            return LLMResponse(
                content="",
                raw=None,
                latency_ms=elapsed_ms,
                error=str(exc),
            )

    @staticmethod
    def _extract_content(resp: Any) -> str:
        choices = (
            resp.get("choices")
            if isinstance(resp, dict)
            else getattr(resp, "choices", None)
        )
        if not choices:
            return ""
        first = choices[0]
        msg = (
            first.get("message")
            if isinstance(first, dict)
            else getattr(first, "message", None)
        )
        if msg is None:
            return ""
        text = (
            msg.get("content")
            if isinstance(msg, dict)
            else getattr(msg, "content", None)
        )
        return "" if text is None else str(text)

    @staticmethod
    def _extract_usage(resp: Any) -> Dict[str, int]:
        usage = (
            resp.get("usage")
            if isinstance(resp, dict)
            else getattr(resp, "usage", None)
        )
        if usage is None:
            return {}
        out: Dict[str, int] = {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            val = (
                usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None)
            )
            if isinstance(val, int) and not isinstance(val, bool):
                out[key] = val
        return out


def load_client_from_spec(spec: str) -> LLMClient:
    """Resolve a `'module.path:callable_or_client'` reference into an `LLMClient`."""
    if not isinstance(spec, str) or ":" not in spec:
        raise ValueError(
            f"Invalid client spec {spec!r}; expected 'module.path:callable' "
            "(e.g. 'mypkg.clients:generator')."
        )
    module_name, attr_path = spec.split(":", 1)
    module_name = module_name.strip()
    attr_path = attr_path.strip()
    if not module_name or not attr_path:
        raise ValueError(
            f"Invalid client spec {spec!r}; expected non-empty 'module:callable'."
        )

    cwd = os.getcwd()
    added_cwd = False
    top_pkg = module_name.split(".", 1)[0]
    try:
        try:
            mod = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == top_pkg and cwd not in sys.path and "" not in sys.path:
                sys.path.insert(0, cwd)
                added_cwd = True
                mod = importlib.import_module(module_name)
            else:
                raise
    finally:
        if added_cwd and sys.path and sys.path[0] == cwd:
            sys.path.pop(0)

    obj: Any = mod
    for part in attr_path.split("."):
        if not hasattr(obj, part):
            raise AttributeError(
                f"Attribute '{attr_path}' not found in module '{module_name}' (failed at '{part}')."
            )
        obj = getattr(obj, part)

    if isinstance(obj, LLMClient):
        return obj
    if isinstance(obj, type) and issubclass(obj, LLMClient):
        return obj()
    if callable(obj):
        return CallableLLM(obj)
    raise TypeError(
        f"Resolved client '{spec}' is neither an LLMClient instance nor a callable "
        f"(got {type(obj).__name__})."
    )


def resolve_cli_client(
    *,
    client_spec: Optional[str] = None,
    model: Optional[str] = None,
    role: str = "generator",
) -> LLMClient:
    """Resolve either `--client module:callable` or `--model <name>` into an `LLMClient`."""
    client_flag = "--tagger-client" if role == "tagger" else "--client"
    model_flag = "--tagger-model" if role == "tagger" else "--model"
    if client_spec and model:
        raise ValueError(
            f"Specify either {role} client ({client_flag}) or model ({model_flag}), not both."
        )
    if not client_spec and not model:
        raise ValueError(
            f"Must specify a {role} backend: pass {client_flag} module:callable "
            f"or {model_flag} <name> (requires prompt-prism[cloud])."
        )
    if client_spec:
        return load_client_from_spec(client_spec)
    return LiteLLMClient(model=str(model))
