"""
PromptPrism Test Set Schema, Content Hashing, Stratified Splitting, and Freezing.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Collection,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Union,
)

import jsonschema
import numpy as np
from pydantic import BaseModel, Field, model_validator

from ..core.factors import Factor, FactorSet
from ..core.models import _canonical_json_sha256
from ..template.composer import PromptTemplate

RESERVED_TEST_CASE_KEYS: frozenset[str] = frozenset(
    {
        "id",
        "target",
        "checks",
        "targets_factors",
        "intended_factor",
        "tagger_factors",
        "tagger_agrees",
        "kind",
        "split",
        "source",
        "review",
        "__prompt__",
        "__run_id__",
        "__sample_id__",
        "__test_inputs__",
    }
)

CheckKind = Literal[
    "exact",
    "regex",
    "contains",
    "not_contains",
    "json_schema",
    "custom",
    "llm_judge",
]
ClauseRole = Literal["conditional_rule", "global_scaffold"]
SplitName = Literal["discovery", "confirmation"]
SplitSelector = Literal["discovery", "confirmation", "all"]


class CheckSpec(BaseModel):
    """Declarative scoring check attached to a TestCase."""

    kind: CheckKind
    value: Any = None
    weight: float = Field(default=1.0, gt=0.0)
    description: str = ""

    @model_validator(mode="after")
    def _validate_kind_and_value(self) -> CheckSpec:
        if self.kind == "regex":
            if not isinstance(self.value, str) or not self.value:
                raise ValueError(
                    "CheckSpec(kind='regex') requires a non-empty pattern string."
                )
            try:
                re.compile(self.value)
            except re.error as exc:
                raise ValueError(
                    f"Invalid regex pattern {self.value!r}: {exc}"
                ) from exc
        elif self.kind in ("contains", "not_contains", "llm_judge"):
            if not isinstance(self.value, str) or not self.value.strip():
                raise ValueError(
                    f"CheckSpec(kind={self.kind!r}) requires a non-empty string value."
                )
        elif self.kind == "custom":
            if not isinstance(self.value, str) or ":" not in self.value:
                raise ValueError(
                    "CheckSpec(kind='custom') requires a 'module.path:callable_name' string."
                )
            mod_part, fn_part = self.value.split(":", 1)
            if not mod_part.strip() or not fn_part.strip():
                raise ValueError(
                    f"Invalid custom check reference {self.value!r}; expected 'module:callable'."
                )
        elif self.kind == "json_schema":
            if not isinstance(self.value, dict):
                raise ValueError(
                    "CheckSpec(kind='json_schema') requires a JSON Schema dict."
                )
            validator_cls = jsonschema.validators.validator_for(self.value)
            try:
                validator_cls.check_schema(self.value)
            except jsonschema.SchemaError as exc:
                raise ValueError(f"Invalid JSON Schema: {exc.message}") from exc
        return self


def compute_case_id(
    inputs: Dict[str, Any],
    target: Optional[Any],
    checks: Sequence[Union[CheckSpec, Dict[str, Any]]],
) -> str:
    """Compute a stable content hash over immutable test-case fields only."""
    norm_checks = [
        c.model_dump(mode="json") if isinstance(c, CheckSpec) else dict(c)
        for c in checks
    ]
    digest = _canonical_json_sha256(
        {"inputs": inputs, "target": target, "checks": norm_checks}
    )
    return f"tc_{digest[:16]}"


class TestCase(BaseModel):
    """One scored, coverage-tagged test item belonging to a PromptPrism TestSet."""

    __test__: ClassVar[bool] = False

    id: str = ""
    inputs: Dict[str, Any]
    target: Optional[Any] = None
    checks: List[CheckSpec] = Field(default_factory=list)
    reference_pass: Optional[str] = None
    reference_fail: Optional[str] = None
    kind: Literal["baseline", "targeted"]
    intended_factor: Optional[str] = None
    tagger_factors: List[str] = Field(default_factory=list)
    tagger_agrees: Optional[bool] = None
    targets_factors: List[str] = Field(default_factory=list)
    source: Literal["synthetic", "log", "human"] = "synthetic"
    split: Optional[SplitName] = None
    review: Literal["pending", "accepted", "rejected"] = "pending"

    @model_validator(mode="after")
    def _validate_case(self) -> TestCase:
        collisions = sorted(set(self.inputs.keys()) & RESERVED_TEST_CASE_KEYS)
        if collisions:
            raise ValueError(
                f"TestCase.inputs contains reserved key(s) {collisions} that would collide with dataset fields."
            )
        if self.source == "synthetic" and any(c.kind == "custom" for c in self.checks):
            raise ValueError(
                "CheckSpec(kind='custom') is forbidden on synthetic test cases (arbitrary code execution risk)."
            )
        if self.kind == "baseline" and self.intended_factor is not None:
            raise ValueError(
                "Baseline test cases cannot specify intended_factor (use kind='targeted')."
            )
        if self.target is None and any(
            c.kind == "exact" and c.value is None for c in self.checks
        ):
            raise ValueError(
                "CheckSpec(kind='exact', value=None) requires TestCase.target to be set."
            )
        if not self.id:
            self.id = compute_case_id(self.inputs, self.target, self.checks)
        return self

    def canonical_inputs_text(self) -> str:
        """Canonical JSON representation of test inputs only (isolated from prompt clauses)."""
        return json.dumps(self.inputs, sort_keys=True, ensure_ascii=False)

    def effective_targets_factors(
        self, *, require_tag_agreement: bool = True
    ) -> List[str]:
        """Return effective factor IDs/names covered by this case.

        For targeted cases where the independent tagger disagreed (`tagger_agrees is False`),
        `intended_factor` is excluded when `require_tag_agreement=True` unless explicitly
        accepted in human review (`review == "accepted"`).
        """
        tags: Set[str] = {str(x) for x in self.tagger_factors if str(x).strip()}
        if self.kind == "baseline":
            tags.update(str(x) for x in self.targets_factors if str(x).strip())
            return sorted(tags)

        if (
            self.tagger_agrees is False
            and require_tag_agreement
            and self.review != "accepted"
        ):
            # Exclude unverified intended_factor; keep only tagger-verified factors
            return sorted(tags)

        tags.update(str(x) for x in self.targets_factors if str(x).strip())
        if self.intended_factor:
            tags.add(str(self.intended_factor))
        return sorted(tags)


class TestSetManifest(BaseModel):
    """Frozen provenance and integrity manifest stored alongside `cases.jsonl`."""

    __test__: ClassVar[bool] = False

    schema_version: str = "1.0.0"
    factors_sha256: str = ""
    template_sha256: str = ""
    factor_id_to_name: Dict[str, str] = Field(default_factory=dict)
    factor_roles: Dict[str, ClauseRole] = Field(default_factory=dict)
    generator: Dict[str, Any] = Field(default_factory=dict)
    counts: Dict[str, Any] = Field(default_factory=dict)
    cases_sha256: str = ""
    active_cases_sha256: str = ""
    frozen_at: Optional[str] = None


def compute_cases_sha256(
    cases: Sequence[TestCase], *, active_only: bool = False
) -> str:
    """Canonical-JSON SHA-256 over cases sorted deterministically (OS line-ending independent)."""
    filtered = [c for c in cases if (not active_only or c.review != "rejected")]
    ordered = sorted(
        filtered,
        key=lambda c: (c.id, c.kind, c.intended_factor or "", c.split or "", c.review),
    )
    payload = [c.model_dump(mode="json") for c in ordered]
    return _canonical_json_sha256(payload)


class TestSet:
    """Collection of scored TestCase items bound to a SHA-256-verifiable TestSetManifest."""

    __test__: ClassVar[bool] = False

    CASES_FILENAME = "cases.jsonl"
    MANIFEST_FILENAME = "manifest.json"

    def __init__(self, cases: Sequence[TestCase], manifest: TestSetManifest):
        self.cases: List[TestCase] = list(cases)
        self.manifest: TestSetManifest = manifest
        self._assert_unique_ids()

    def _assert_unique_ids(self) -> None:
        seen: Set[str] = set()
        dups: List[str] = []
        for c in self.cases:
            if c.id in seen:
                dups.append(c.id)
            seen.add(c.id)
        if dups:
            raise ValueError(f"Duplicate TestCase IDs in TestSet: {sorted(set(dups))}")

    @property
    def is_frozen(self) -> bool:
        return self.manifest.frozen_at is not None

    @property
    def provenance(self) -> str:
        """Aggregate source provenance over active cases: synthetic | human | log | mixed | none."""
        sources = {c.source for c in self.cases if c.review != "rejected"}
        if not sources:
            return "none"
        if len(sources) == 1:
            return next(iter(sources))
        return "mixed"

    @classmethod
    def create(
        cls,
        cases: Sequence[TestCase],
        *,
        factors: Optional[Union[FactorSet, Sequence[Factor]]] = None,
        template: Optional[PromptTemplate] = None,
        factor_roles: Optional[Dict[str, ClauseRole]] = None,
        generator: Optional[Dict[str, Any]] = None,
        freeze: bool = False,
    ) -> TestSet:
        """Create a new draft (or immediately frozen) TestSet with computed hashes and counts."""
        factor_set: Optional[FactorSet] = None
        if factors is not None:
            factor_set = (
                factors if isinstance(factors, FactorSet) else FactorSet(factors)
            )

        factors_sha = _canonical_json_sha256(factor_set.to_dict()) if factor_set else ""
        template_sha = template.template_sha256() if template else ""
        id_to_name = {f.id: f.name for f in factor_set} if factor_set else {}

        manifest = TestSetManifest(
            factors_sha256=factors_sha,
            template_sha256=template_sha,
            factor_id_to_name=id_to_name,
            factor_roles=dict(factor_roles or {}),
            generator=dict(generator or {}),
        )
        ts = cls(cases=cases, manifest=manifest)
        ts.recompute_metadata()
        if freeze:
            ts.freeze()
        return ts

    def _canonical_factor_key(
        self, tag: str, *, key_by: Literal["id", "name"] = "id"
    ) -> str:
        """Resolve a factor ID or name tag consistently using `manifest.factor_id_to_name`."""
        id_to_name = self.manifest.factor_id_to_name
        s = str(tag)
        if not id_to_name:
            return s
        if key_by == "name":
            return id_to_name.get(s, s)
        if s in id_to_name:
            return s
        for fid, fname in id_to_name.items():
            if fname == s:
                return fid
        return s

    def recompute_counts(self) -> Dict[str, Any]:
        """Compute structured counts per split, kind, source, factor, and judge usage."""
        active = [c for c in self.cases if c.review != "rejected"]
        by_split: Dict[str, int] = {}
        by_kind: Dict[str, int] = {}
        by_source: Dict[str, int] = {}
        by_factor: Dict[str, int] = {
            fid: 0 for fid in self.manifest.factor_id_to_name.keys()
        }
        by_factor_split: Dict[str, Dict[str, int]] = {
            fid: {"discovery": 0, "confirmation": 0, "unassigned": 0}
            for fid in self.manifest.factor_id_to_name.keys()
        }
        judge_items = 0

        for c in active:
            split_key = c.split or "unassigned"
            by_split[split_key] = by_split.get(split_key, 0) + 1
            by_kind[c.kind] = by_kind.get(c.kind, 0) + 1
            by_source[c.source] = by_source.get(c.source, 0) + 1
            if any(chk.kind == "llm_judge" for chk in c.checks):
                judge_items += 1
            canonical_fids = sorted(
                {
                    self._canonical_factor_key(fid, key_by="id")
                    for fid in c.effective_targets_factors(require_tag_agreement=True)
                }
            )
            for fid in canonical_fids:
                by_factor[fid] = by_factor.get(fid, 0) + 1
                if fid not in by_factor_split:
                    by_factor_split[fid] = {
                        "discovery": 0,
                        "confirmation": 0,
                        "unassigned": 0,
                    }
                by_factor_split[fid][split_key] = (
                    by_factor_split[fid].get(split_key, 0) + 1
                )

        return {
            "total": len(self.cases),
            "active": len(active),
            "rejected": len(self.cases) - len(active),
            "by_split": by_split,
            "by_kind": by_kind,
            "by_source": by_source,
            "by_factor": by_factor,
            "by_factor_split": by_factor_split,
            "judge_check_items": judge_items,
        }

    def recompute_metadata(self) -> None:
        """Refresh manifest counts and canonical hashes (draft sets only)."""
        self.manifest.counts = self.recompute_counts()
        self.manifest.cases_sha256 = compute_cases_sha256(self.cases, active_only=False)
        self.manifest.active_cases_sha256 = compute_cases_sha256(
            self.cases, active_only=True
        )

    def freeze(self, *, force: bool = False) -> TestSet:
        """Freeze this TestSet with canonical-JSON SHA-256 hashes and a UTC timestamp."""
        if self.is_frozen and not force:
            raise ValueError(
                "TestSet is already frozen. Pass force=True to explicitly re-freeze."
            )
        self._assert_unique_ids()
        self.recompute_metadata()
        self.manifest.frozen_at = datetime.now(timezone.utc).isoformat()
        return self

    def verify_integrity(self) -> None:
        """Verify that current cases match the manifest's canonical SHA-256 hashes."""
        actual_all = compute_cases_sha256(self.cases, active_only=False)
        if actual_all != self.manifest.cases_sha256:
            raise ValueError(
                "Frozen TestSet integrity check failed (cases_sha256 mismatch): "
                f"expected {self.manifest.cases_sha256!r}, got {actual_all!r}."
            )
        actual_active = compute_cases_sha256(self.cases, active_only=True)
        if (
            self.manifest.active_cases_sha256
            and actual_active != self.manifest.active_cases_sha256
        ):
            raise ValueError(
                "Frozen TestSet integrity check failed (active_cases_sha256 mismatch): "
                f"expected {self.manifest.active_cases_sha256!r}, got {actual_active!r}."
            )

    def split_cases(
        self,
        *,
        discovery_ratio: float = 0.7,
        n_baseline_discovery: Optional[int] = None,
        n_per_factor_discovery: Optional[int] = None,
        seed: int = 0,
    ) -> TestSet:
        """Stratify cases into `discovery` and `confirmation` splits.

        Baseline cases are split randomly; targeted cases are stratified by `intended_factor`
        (falling back to their sorted effective factor tuple) so every factor with >= 2
        targeted items has representation in both splits.
        """
        if self.is_frozen:
            raise ValueError("Cannot split an already-frozen TestSet.")
        if not (0.0 < discovery_ratio < 1.0):
            raise ValueError(
                f"discovery_ratio must lie in (0, 1), got {discovery_ratio}"
            )
        if n_baseline_discovery is not None and n_baseline_discovery < 0:
            raise ValueError(
                f"n_baseline_discovery must be >= 0, got {n_baseline_discovery}"
            )
        if n_per_factor_discovery is not None and n_per_factor_discovery < 0:
            raise ValueError(
                f"n_per_factor_discovery must be >= 0, got {n_per_factor_discovery}"
            )

        rng = np.random.default_rng(seed)

        def _assign_group(group: List[TestCase], target_disc: Optional[int]) -> None:
            if not group:
                return
            order = list(rng.permutation(len(group)))
            if target_disc is not None:
                n_disc = max(0, min(len(group), int(target_disc)))
            elif len(group) == 1:
                n_disc = 1
            else:
                n_disc = int(round(len(group) * discovery_ratio))
                n_disc = max(1, min(len(group) - 1, n_disc))
            for pos, idx in enumerate(order):
                group[idx].split = "discovery" if pos < n_disc else "confirmation"

        active_baseline = [
            c for c in self.cases if c.review != "rejected" and c.kind == "baseline"
        ]
        _assign_group(active_baseline, n_baseline_discovery)

        targeted_groups: Dict[str, List[TestCase]] = {}
        for c in self.cases:
            if c.review == "rejected" or c.kind != "targeted":
                continue
            if c.intended_factor:
                key = self._canonical_factor_key(c.intended_factor, key_by="id")
            else:
                key = ",".join(
                    sorted(
                        {
                            self._canonical_factor_key(f, key_by="id")
                            for f in c.effective_targets_factors(
                                require_tag_agreement=False
                            )
                        }
                    )
                )
            targeted_groups.setdefault(key, []).append(c)

        for key in sorted(targeted_groups.keys()):
            _assign_group(targeted_groups[key], n_per_factor_discovery)

        self.recompute_metadata()
        return self

    def save(self, directory: Union[str, Path]) -> Path:
        """Write `cases.jsonl` and `manifest.json` into `directory`."""
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._assert_unique_ids()

        if self.is_frozen:
            self.verify_integrity()
        else:
            self.recompute_metadata()

        cases_path = out_dir / self.CASES_FILENAME
        manifest_path = out_dir / self.MANIFEST_FILENAME

        ordered = sorted(self.cases, key=lambda c: c.id)
        lines = [
            json.dumps(
                c.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            for c in ordered
        ]
        cases_path.write_text(
            ("\n".join(lines) + "\n") if lines else "", encoding="utf-8"
        )
        manifest_path.write_text(
            json.dumps(
                self.manifest.model_dump(mode="json"),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return out_dir

    @classmethod
    def load(cls, directory: Union[str, Path]) -> TestSet:
        """Load a TestSet directory, enforcing SHA-256 verification whenever frozen."""
        in_dir = Path(directory)
        cases_path = in_dir / cls.CASES_FILENAME
        manifest_path = in_dir / cls.MANIFEST_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"TestSet manifest not found: {manifest_path}")
        if not cases_path.exists():
            raise FileNotFoundError(f"TestSet cases file not found: {cases_path}")

        manifest = TestSetManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        cases: List[TestCase] = []
        for raw_line in cases_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            cases.append(TestCase.model_validate_json(line))

        ts = cls(cases=cases, manifest=manifest)
        if ts.is_frozen:
            ts.verify_integrity()
        return ts

    def cases_for_split(
        self,
        split: SplitSelector = "discovery",
        *,
        include_rejected: bool = False,
    ) -> List[TestCase]:
        """Return cases matching `split` (defaults to `discovery` to protect confirmation)."""
        if split not in ("discovery", "confirmation", "all"):
            raise ValueError(
                f"split must be 'discovery', 'confirmation', or 'all', got {split!r}"
            )
        pool = (
            list(self.cases)
            if include_rejected
            else [c for c in self.cases if c.review != "rejected"]
        )
        if split == "all":
            return pool

        has_any_split = any(c.split is not None for c in pool)
        if not has_any_split and pool:
            raise ValueError(
                f"TestSet has no split assignments yet; call split_cases() or pass split='all' (requested split={split!r})."
            )
        return [c for c in pool if c.split == split]

    def to_dataset(
        self,
        split: SplitSelector = "discovery",
        *,
        require_tag_agreement: bool = True,
        key_by: Literal["id", "name"] = "id",
    ) -> List[Dict[str, Any]]:
        """Convert active cases in `split` into a dataset list ready for `Experiment.run()`."""
        selected = self.cases_for_split(split=split, include_rejected=False)
        rows: List[Dict[str, Any]] = []
        for c in selected:
            row: Dict[str, Any] = dict(c.inputs)
            row["id"] = c.id
            row["target"] = c.target
            row["checks"] = [chk.model_dump(mode="json") for chk in c.checks]
            row["targets_factors"] = sorted(
                {
                    self._canonical_factor_key(fid, key_by=key_by)
                    for fid in c.effective_targets_factors(
                        require_tag_agreement=require_tag_agreement
                    )
                }
            )
            row["kind"] = c.kind
            row["split"] = c.split
            row["__test_inputs__"] = c.canonical_inputs_text()
            rows.append(row)
        return rows

    def coverage(
        self,
        split: SplitSelector = "discovery",
        *,
        factor_ids: Optional[Sequence[str]] = None,
        key_by: Literal["id", "name"] = "id",
        require_tag_agreement: bool = True,
    ) -> Dict[str, Set[str]]:
        """Build `factor_key -> set(case_ids)` coverage mapping for `diagnose`.

        Factor tags are normalized through `manifest.factor_id_to_name` (defaulting to
        canonical factor IDs, or factor names when `key_by='name'`) so ID/name aliases
        never produce duplicate ghost keys.
        """
        selected = self.cases_for_split(split=split, include_rejected=False)
        cov: Dict[str, Set[str]] = {}
        if factor_ids is not None:
            init_keys = [
                self._canonical_factor_key(str(fid), key_by=key_by)
                for fid in factor_ids
            ]
        elif key_by == "name":
            init_keys = list(self.manifest.factor_id_to_name.values())
        else:
            init_keys = list(self.manifest.factor_id_to_name.keys())
        for fid in init_keys:
            cov[str(fid)] = set()

        for c in selected:
            for raw_fid in c.effective_targets_factors(
                require_tag_agreement=require_tag_agreement
            ):
                key = self._canonical_factor_key(raw_fid, key_by=key_by)
                cov.setdefault(key, set()).add(c.id)
        return cov

    def baseline_sample_ids(
        self, split: SplitSelector = "discovery"
    ) -> Optional[Set[str]]:
        """Return active baseline case IDs in `split`, or None if no baseline cases exist."""
        selected = self.cases_for_split(split=split, include_rejected=False)
        ids = {c.id for c in selected if c.kind == "baseline"}
        return ids if ids else None

    def judge_scored_fraction(self, split: SplitSelector = "discovery") -> float:
        """Fraction of active cases in `split` that include at least one `llm_judge` check."""
        selected = self.cases_for_split(split=split, include_rejected=False)
        if not selected:
            return 0.0
        n_judge = sum(
            1 for c in selected if any(chk.kind == "llm_judge" for chk in c.checks)
        )
        return float(n_judge / len(selected))

    def detect_split_leakage(self, sample_ids: Collection[Any]) -> bool:
        """Return True if `sample_ids` spans both `discovery` and `confirmation` splits."""
        id_set = {str(x) for x in sample_ids}
        splits_seen = {
            c.split for c in self.cases if c.id in id_set and c.split is not None
        }
        return "discovery" in splits_seen and "confirmation" in splits_seen
