"""Versioned schemas and protocol contracts for Real-v4 selection and stability."""

from __future__ import annotations

from enum import Enum
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_MAJOR_VERSION: int = 1
CURRENT_SCHEMA_VERSION: str = "1.0.0"

_SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
_PLACEHOLDER_PATTERNS = (
    re.compile(r"\bTODO\b"),
    re.compile(r"\bTBD\b"),
    re.compile(r"\bFIXME\b"),
    re.compile(r"\bXXX\b"),
    re.compile(r"<placeholder>", re.IGNORECASE),
)


class IncompatibleSchemaVersionError(ValueError):
    """Raised when a schema payload carries an unknown or incompatible major version."""


class ProtocolLintError(ValueError):
    """Raised when protocol documents miss required contracts or retain placeholders."""


def validate_schema_version(
    version: str,
    expected_major: int = SCHEMA_MAJOR_VERSION,
) -> Tuple[int, int, int]:
    """Validate semantic schema version and reject incompatible major versions.

    Forward-versioning contract: minor/patch increments within the same major version
    are accepted; major version 0 or any major != ``expected_major`` is rejected.
    """
    if not isinstance(version, str):
        raise IncompatibleSchemaVersionError(f"Schema version must be a string, got {type(version).__name__}")
    match = _SEMVER_RE.match(version.strip())
    if not match:
        raise IncompatibleSchemaVersionError(
            f"Malformed schema_version {version!r}; expected 'MAJOR.MINOR.PATCH'"
        )
    major, minor, patch = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    if major != expected_major:
        raise IncompatibleSchemaVersionError(
            f"Incompatible schema_version {version!r}: supported major={expected_major}, got major={major}"
        )
    return major, minor, patch


def _to_canonical_primitive(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _to_canonical_primitive(value.model_dump(mode="json"))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _to_canonical_primitive(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_to_canonical_primitive(v) for v in value]
    if isinstance(value, float):
        if value == 0.0:
            return 0.0
        return round(value, 12)
    return value


def canonical_json(value: Any) -> str:
    """Serialize any schema or JSON-compatible object to canonical deterministic JSON."""
    primitive = _to_canonical_primitive(value)
    return json.dumps(primitive, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def canonical_sha256(value: Any) -> str:
    """Compute canonical SHA-256 hex digest of a schema or primitive structure."""
    payload = canonical_json(value)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CertificateStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INVALID_OUT_OF_ENVELOPE = "INVALID/OUT_OF_ENVELOPE"


class OverallVerdictCode(str, Enum):
    PASS_SELECTION_AND_STABILITY = "PASS_SELECTION_AND_STABILITY"
    PASS_SELECTION_ONLY = "PASS_SELECTION_ONLY"
    PASS_STABILITY_ONLY = "PASS_STABILITY_ONLY"
    FAIL_BOTH = "FAIL_BOTH"
    INVALID_OUT_OF_ENVELOPE = "INVALID_OUT_OF_ENVELOPE"


ALLOWED_CONCLUSION_MAP: Dict[OverallVerdictCode, str] = {
    OverallVerdictCode.PASS_SELECTION_AND_STABILITY: (
        "PromptPrism improves prompt selection and provides bounded stability within the registered envelope."
    ),
    OverallVerdictCode.PASS_SELECTION_ONLY: (
        "PromptPrism selects better prompts, but stability is not established."
    ),
    OverallVerdictCode.PASS_STABILITY_ONLY: (
        "The evaluated prompt is stable, but PromptPrism's selection advantage is not established."
    ),
    OverallVerdictCode.FAIL_BOTH: "Neither claim is established.",
    OverallVerdictCode.INVALID_OUT_OF_ENVELOPE: (
        "Valid within-envelope certification could not be completed; fails closed."
    ),
}


class EnvironmentFamily(str, Enum):
    REFERENCE = "reference"
    REPEAT = "repeat"
    FORMATTING = "formatting"
    WORDING = "wording"
    ITEM_GROUP = "item_group"
    TEMPORAL = "temporal"


class _VersionedBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default=CURRENT_SCHEMA_VERSION)

    def __init__(self, **data: Any) -> None:
        if "schema_version" in data:
            validate_schema_version(data["schema_version"])
        super().__init__(**data)

    @classmethod
    def model_validate(cls, obj: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(obj, dict) and "schema_version" in obj:
            validate_schema_version(obj["schema_version"])
        return super().model_validate(obj, *args, **kwargs)

    @classmethod
    def model_validate_json(cls, json_data: str | bytes | bytearray, *args: Any, **kwargs: Any) -> Any:
        parsed = json.loads(json_data)
        if isinstance(parsed, dict) and "schema_version" in parsed:
            validate_schema_version(parsed["schema_version"])
        return super().model_validate_json(json_data, *args, **kwargs)

    @field_validator("schema_version")
    @classmethod
    def _check_version(cls, value: str) -> str:
        validate_schema_version(value)
        return value

    def canonical_json(self) -> str:
        return canonical_json(self)

    def canonical_sha256(self) -> str:
        return canonical_sha256(self)


class EnvironmentSpec(_VersionedBaseModel):
    env_id: str = Field(min_length=1)
    family: EnvironmentFamily
    description: str = Field(min_length=1)
    parent_env_id: Optional[str] = None
    transformation_hash: str = Field(min_length=8)
    semantic_equivalence_audited: bool = True
    requires_identical_model_version: bool = True
    uses_disjoint_item_group: bool = False

    @model_validator(mode="after")
    def _validate_reference_contract(self) -> "EnvironmentSpec":
        if self.family == EnvironmentFamily.REFERENCE:
            if self.parent_env_id is not None:
                raise ValueError("Reference environment must have parent_env_id=None")
        else:
            if not self.parent_env_id:
                raise ValueError(f"Perturbation environment {self.env_id!r} must declare parent_env_id")
        if not self.semantic_equivalence_audited:
            raise ValueError(f"Environment {self.env_id!r} has not passed semantic-equivalence audit")
        return self


class OperatingEnvelope(_VersionedBaseModel):
    study_id: str = Field(min_length=1)
    tasks: Tuple[str, ...] = Field(min_length=6)
    models: Tuple[str, ...] = Field(min_length=3)
    model_versions_pinned: Dict[str, str]
    factor_ids: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
    primary_metrics: Dict[str, str]
    parser_hashes: Dict[str, str]
    decoding_config: Dict[str, Any]
    partition_hashes: Dict[str, str]
    registered_environments: Tuple[EnvironmentSpec, ...] = Field(min_length=2)
    repeat_count: int = Field(ge=1)
    nondeterminism_detected: bool = False
    local_runtime_locks: Dict[str, str] = Field(default_factory=dict)
    code_hash: str = Field(min_length=8)

    @model_validator(mode="after")
    def _validate_envelope_completeness(self) -> "OperatingEnvelope":
        if len(set(self.tasks)) != len(self.tasks):
            raise ValueError("Duplicate task IDs in OperatingEnvelope.tasks")
        if len(set(self.models)) != len(self.models):
            raise ValueError("Duplicate model IDs in OperatingEnvelope.models")
        if len(set(self.factor_ids)) != len(self.factor_ids):
            raise ValueError("Duplicate factor IDs in OperatingEnvelope.factor_ids")

        missing_versions = set(self.models) - set(self.model_versions_pinned.keys())
        if missing_versions:
            raise ValueError(f"Missing pinned versions for models: {sorted(missing_versions)}")
        if any(not v.strip() for v in self.model_versions_pinned.values()):
            raise ValueError("Pinned model versions must be non-empty")

        missing_metrics = set(self.tasks) - set(self.primary_metrics.keys())
        if missing_metrics:
            raise ValueError(f"Missing primary_metrics for tasks: {sorted(missing_metrics)}")
        missing_parsers = set(self.tasks) - set(self.parser_hashes.keys())
        if missing_parsers:
            raise ValueError(f"Missing parser_hashes for tasks: {sorted(missing_parsers)}")

        env_ids = [e.env_id for e in self.registered_environments]
        if len(set(env_ids)) != len(env_ids):
            raise ValueError("Duplicate env_id in registered_environments")
        ref_envs = [e for e in self.registered_environments if e.family == EnvironmentFamily.REFERENCE]
        if len(ref_envs) != 1 or ref_envs[0].env_id != "ref":
            raise ValueError("OperatingEnvelope must declare exactly one reference environment with env_id='ref'")

        if self.nondeterminism_detected and self.repeat_count < 3:
            raise ValueError("When nondeterminism_detected=True, repeat_count must be >= 3")
        return self

    @property
    def perturbation_env_ids(self) -> Tuple[str, ...]:
        return tuple(e.env_id for e in self.registered_environments if e.family != EnvironmentFamily.REFERENCE)

    @property
    def all_env_ids(self) -> Tuple[str, ...]:
        return tuple(e.env_id for e in self.registered_environments)

    def envelope_sha256(self) -> str:
        return canonical_sha256(self)


class CandidateProvenance(_VersionedBaseModel):
    run_id: int = Field(ge=0, lt=64)
    factor_levels: Dict[str, int]
    source_rule: str = Field(min_length=1)
    screen_score: float = Field(ge=0.0, le=1.0)
    rank_in_screen: int = Field(ge=1)

    @field_validator("factor_levels")
    @classmethod
    def _validate_binary_levels(cls, levels: Dict[str, int]) -> Dict[str, int]:
        if not levels:
            raise ValueError("factor_levels cannot be empty")
        for key, val in levels.items():
            if val not in (0, 1):
                raise ValueError(f"Factor {key!r} level must be 0 or 1, got {val!r}")
        return levels


class MethodBudgetSpec(_VersionedBaseModel):
    method_id: str = Field(min_length=1)
    cell_id: str = Field(min_length=1)
    screen_configs: int = Field(ge=1)
    screen_items: int = Field(ge=1)
    screen_repeats: int = Field(ge=1)
    validation_candidates: int = Field(ge=1, le=4)
    validation_items: int = Field(ge=1)
    validation_envs: int = Field(ge=1)
    validation_repeats: int = Field(ge=1)
    oracle_audit_calls: int = Field(default=0, ge=0)

    @property
    def total_screen_calls(self) -> int:
        return self.screen_configs * self.screen_items * self.screen_repeats

    @property
    def total_validation_calls(self) -> int:
        return (
            self.validation_candidates
            * self.validation_items
            * self.validation_envs
            * self.validation_repeats
        )

    @property
    def total_operational_discovery_calls(self) -> int:
        return self.total_screen_calls + self.total_validation_calls

    def operational_signature(self) -> Tuple[int, int, int, int, int, int, int]:
        return (
            self.screen_configs,
            self.screen_items,
            self.screen_repeats,
            self.validation_candidates,
            self.validation_items,
            self.validation_envs,
            self.validation_repeats,
        )


class LockedSelectionRecord(_VersionedBaseModel):
    study_id: str = Field(min_length=1)
    cell_id: str = Field(min_length=1)
    method_id: str = Field(min_length=1)
    trajectory_id: str = Field(min_length=1)
    algorithm_version: str = Field(min_length=1)
    candidates: Tuple[CandidateProvenance, ...] = Field(min_length=1, max_length=4)
    selected_run_id: int = Field(ge=0, lt=64)
    selected_factor_levels: Dict[str, int]
    validation_robust_score: float
    validation_worst_env_score: float = Field(ge=0.0, le=1.0)
    validation_env_scores: Dict[str, float]
    budget: MethodBudgetSpec
    data_hash: str = Field(min_length=8)
    model_hash: str = Field(min_length=8)
    screen_manifest_hash: str = Field(min_length=8)
    validation_manifest_hash: str = Field(min_length=8)

    @model_validator(mode="after")
    def _validate_candidate_membership(self) -> "LockedSelectionRecord":
        cand_ids = [c.run_id for c in self.candidates]
        if len(set(cand_ids)) != len(cand_ids):
            raise ValueError(f"Duplicate candidate run_ids in LockedSelectionRecord: {cand_ids}")
        if self.selected_run_id not in cand_ids:
            raise ValueError(
                f"selected_run_id={self.selected_run_id} must belong to locked candidates {cand_ids}"
            )
        matching = next(c for c in self.candidates if c.run_id == self.selected_run_id)
        if matching.factor_levels != self.selected_factor_levels:
            raise ValueError("selected_factor_levels does not match selected candidate provenance")
        return self


class ComparatorContrastSummary(_VersionedBaseModel):
    comparator_id: str = Field(min_length=1)
    estimate: float
    se: float = Field(ge=0.0)
    ci95_lower: float
    p_value_one_sided: float = Field(ge=0.0, le=1.0)
    practical_margin: float = Field(ge=0.0)
    passes_statistical_gate: bool
    passes_practical_gate: bool


class StabilitySummary(_VersionedBaseModel):
    reference_mean: float = Field(ge=0.0, le=1.0)
    d_stability_estimate: float
    d_stability_upper_95_sim: float
    epsilon_stability: float = Field(gt=0.0)
    worst_perturbation_env_id: str = Field(min_length=1)
    per_env_means: Dict[str, float]
    per_env_drops: Dict[str, float]
    passes_absolute_noninferiority: bool
    comparator_worst_env_checks: Dict[str, ComparatorContrastSummary]
    passes_comparator_worst_env: bool


class SupportingDiagnostics(_VersionedBaseModel):
    effect_recovery_rmse_prism: Optional[float] = Field(default=None, ge=0.0)
    effect_recovery_rmse_random: Optional[float] = Field(default=None, ge=0.0)
    effect_recovery_gain: Optional[float] = None
    identical_repeat_exact_match_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    score_variance_across_repeats: Optional[float] = Field(default=None, ge=0.0)
    selection_resample_agreement: Optional[float] = Field(default=None, ge=0.0, le=1.0)


REQUIRED_PROTOCOL_FILES_AND_TOKENS: Dict[str, Tuple[str, ...]] = {
    "study_plan.md": (
        "Claim P",
        "Claim S",
        "G_{\\text{select},b}",
        "D_{\\text{stability}}",
        "G_{\\text{robust},b}",
        "delta_select = 0.01",
        "epsilon_stability = 0.02",
        "INVALID_OUT_OF_ENVELOPE",
        "Four-Outcome Interpretation Matrix",
        "universal guarantee",
        "Tie breaking",
        "full-cube",
    ),
    "preregistration.md": (
        "prompt-prism-real-v4-selection-stability-2026",
        "Claim P",
        "Claim S",
        "balanced_random",
        "bayesian_opt",
        "re3d_entity_extraction",
        "squad2_grounded_qa",
        "banking77_intent",
        "gsm8k_math",
        "mbpp_python",
        "sciq_verification",
        "gemini-2.5-flash",
        "Mistral-7B-Instruct-v0.3-4bit",
        "Qwen3-8B-4bit",
        "temporal_wave_2",
        "OSF",
    ),
    "statistical_analysis_plan.md": (
        "B = 2,000",
        "alpha = 0.05",
        "U_{95,\\text{sim}}(D_{\\text{stability}})",
        "G_{\\text{robust},b}",
        "lowest `run_id`",
        "PROVIDER_VERSION_DRIFT",
        "score = 0.0",
    ),
}


def lint_protocol_documents(protocol_dir: Path) -> Dict[str, str]:
    """Validate Real-v4 protocol documents for completeness, zero placeholders, and SHA-256 digests."""
    hashes: Dict[str, str] = {}
    for filename, required_tokens in REQUIRED_PROTOCOL_FILES_AND_TOKENS.items():
        path = protocol_dir / filename
        if not path.exists():
            raise ProtocolLintError(f"Missing required protocol document: {path}")
        content = path.read_text(encoding="utf-8")
        for pat in _PLACEHOLDER_PATTERNS:
            match = pat.search(content)
            if match:
                raise ProtocolLintError(
                    f"Protocol document {filename} contains forbidden placeholder token {match.group(0)!r}"
                )
        for token in required_tokens:
            if token not in content:
                raise ProtocolLintError(
                    f"Protocol document {filename} is missing required contract token {token!r}"
                )
        hashes[filename] = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return hashes
