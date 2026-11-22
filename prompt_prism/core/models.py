"""
Core data structures for representing Design Matrices, Run Configurations, Trials, and Experiment Results.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    """
    Configuration for a single experimental run (a specific combination of factor levels).

    Attributes:
        run_id: Sequential or unique identifier of the run.
        factor_levels: Mapping of factor identifier (or name) to the assigned level code.
        factor_names: Optional mapping of factor name to level code.
        params: Model parameters (e.g. temperature, max_tokens) configured for this run.
        metadata: Extra metadata for tracking (e.g. plan_id, repetition).
    """

    run_id: int
    factor_levels: Dict[str, int] = Field(default_factory=dict)
    factor_names: Dict[str, int] = Field(default_factory=dict)
    combination_string: str = ""
    params: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def get_level(self, factor_id_or_name: str) -> int:
        """Get level code for a factor by ID or name."""
        if factor_id_or_name in self.factor_levels:
            return self.factor_levels[factor_id_or_name]
        if factor_id_or_name in self.factor_names:
            return self.factor_names[factor_id_or_name]
        raise KeyError(f"Factor '{factor_id_or_name}' not found in RunConfig")

    @property
    def level_string(self) -> str:
        """Compact string representation of levels (e.g. '01001')."""
        if self.combination_string:
            return self.combination_string
        return "".join(
            str(self.factor_levels[k]) for k in sorted(self.factor_levels.keys())
        )


class DesignMatrix(BaseModel):
    """
    A full Design of Experiments (DoE) matrix.

    Attributes:
        plan_id: Name or code of the design (e.g. '2(5-1)V', 'PB-12', 'Full-2^4').
        factor_ids: Ordered list of factor IDs (e.g. ['A', 'B', 'C', 'D', 'E']).
        factor_names: Optional descriptive names corresponding to factor_ids.
        runs: List of RunConfig objects representing each experimental condition.
        resolution: Design resolution (e.g., 3 for Res III, 4 for Res IV, 5 for Res V, None for PB/custom).
        generators: Generator formulas (e.g. ['E=ABCD']).
        metadata: Additional details (number of factors, fraction, design properties).
    """

    plan_id: str
    factor_ids: List[str]
    factor_names: List[str] = Field(default_factory=list)
    runs: List[RunConfig] = Field(default_factory=list)
    resolution: Optional[int] = None
    generators: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def num_factors(self) -> int:
        return len(self.factor_ids)

    @property
    def num_runs(self) -> int:
        return len(self.runs)

    def to_dataframe(self, coded: bool = True) -> pd.DataFrame:
        """
        Convert the design matrix to a pandas DataFrame.

        Args:
            coded: If True, values are in {0, 1} (or {-1, +1} if specified).
        """
        records = []
        for run in self.runs:
            row: Dict[str, Any] = {"run_id": run.run_id}
            for fid in self.factor_ids:
                val = run.factor_levels.get(fid, 0)
                row[fid] = val if coded else val
            row["combination"] = run.level_string
            records.append(row)
        return pd.DataFrame(records)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        factor_ids: Optional[List[str]] = None,
        plan_id: str = "custom",
        run_id_col: str = "run_id",
    ) -> DesignMatrix:
        """Create a DesignMatrix from a pandas DataFrame."""
        if factor_ids is None:
            # Exclude metadata columns
            exclude = {
                run_id_col,
                "combination",
                "order",
                "run",
                "EXPERIMENT_ORDER",
                "PLAN_ID",
                "FACTORS_COMBINATION",
            }
            factor_ids = [c for c in df.columns if c not in exclude]

        runs = []
        for idx, row in df.iterrows():
            run_id = int(row[run_id_col]) if run_id_col in row else int(idx) + 1
            f_levels = {fid: int(row[fid]) for fid in factor_ids}
            runs.append(RunConfig(run_id=run_id, factor_levels=f_levels))

        return cls(plan_id=plan_id, factor_ids=factor_ids, runs=runs)


class Trial(BaseModel):
    """
    The result of executing one RunConfig on a single sample dataset item.
    """

    trial_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id: int
    sample_id: Union[str, int]
    factor_levels: Dict[str, int] = Field(default_factory=dict)
    prompt: Union[str, List[Dict[str, str]]] = ""
    raw_response: Any = None
    parsed_response: Any = None
    metrics: Dict[str, float] = Field(default_factory=dict)
    latency_ms: Optional[float] = None
    token_usage: Dict[str, int] = Field(default_factory=dict)
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExperimentResults(BaseModel):
    """
    Collection of all trials and metadata for an experiment run.
    """

    experiment_id: str
    design: DesignMatrix
    trials: List[Trial] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Flatten all trial executions and metrics into a pandas DataFrame ready for ANOVA and statistical analysis.
        """
        records = []
        for t in self.trials:
            row: Dict[str, Any] = {
                "trial_id": t.trial_id,
                "run_id": t.run_id,
                "sample_id": t.sample_id,
                "latency_ms": t.latency_ms,
                "error": t.error,
            }
            # Trial metadata (e.g. combination, judge_reasons.*)
            if t.metadata:
                for meta_k, meta_v in t.metadata.items():
                    row[meta_k] = meta_v

            # Factor levels (precedence over metadata on key collision)
            for fid, lvl in t.factor_levels.items():
                row[fid] = lvl

            # Metric scores (precedence over metadata/factors on key collision)
            for m_name, m_val in t.metrics.items():
                row[m_name] = m_val

            records.append(row)
        return pd.DataFrame(records)

    def summary_by_run(self, metric_name: Optional[str] = None) -> pd.DataFrame:
        """Compute mean and std of metrics grouped by experimental run."""
        df = self.to_dataframe()
        if metric_name:
            metric_cols = [metric_name]
        else:
            all_metric_keys = set()
            for t in self.trials:
                all_metric_keys.update(t.metrics.keys())
            metric_cols = [c for c in df.columns if c in all_metric_keys]
            if not metric_cols:
                metric_cols = [
                    c
                    for c in df.columns
                    if c
                    not in {
                        "trial_id",
                        "run_id",
                        "sample_id",
                        "latency_ms",
                        "error",
                    }
                    and c not in self.design.factor_ids
                    and pd.api.types.is_numeric_dtype(df[c])
                ]
        return df.groupby("run_id")[metric_cols].agg(["mean", "std"])


class PromptEvidenceState(str):
    """Explicit evidence tier for a prompt configuration report."""

    PREDICTED = "predicted"
    CONFIRMATION_VALIDATED = "confirmation-validated"
    STABILITY_CERTIFIED = "stability-certified"

    VALID_STATES = (
        "predicted",
        "confirmation-validated",
        "stability-certified",
    )


def _canonical_json_sha256(payload: Any) -> str:
    import hashlib
    import json

    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class OperatingEnvelopeSpec(BaseModel):
    """Explicit, SHA-256-hashed operating envelope bounding any confirmation or stability claim."""

    schema_version: str = "1.0.0"
    task_id: str
    model_id: str
    model_version: str
    dataset_partition_hash: str
    parser_hash: str
    perturbation_env_ids: List[str] = Field(default_factory=list)
    algorithm_version: str = "promptprism_experimental_two_stage_v4.0.0"
    decoding_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "temperature": 0.0,
            "max_output_tokens": 512,
            "seed": 42,
        }
    )

    def envelope_sha256(self) -> str:
        payload = self.model_dump(mode="json")
        payload["perturbation_env_ids"] = sorted(
            set(payload.get("perturbation_env_ids", []))
        )
        return _canonical_json_sha256(payload)


class EnvelopeValidationResult(BaseModel):
    """Result of validating a runtime or deployment request against a certificate's operating envelope."""

    valid: bool
    evidence_state: str
    can_display_passing_stability_badge: bool
    badge_label: str
    mismatch_reasons: List[str] = Field(default_factory=list)


class PromptStabilityCertificate(BaseModel):
    """Machine-readable, fail-closed stability certificate bound to a named OperatingEnvelopeSpec.

    Note: Because the prospective Real-v4 promotion gate (`PASS_BOTH`) was not met (`FAIL_BOTH`),
    certificates are issued only in experimental opt-in workflows and always fail closed when any
    model, version, dataset, parser, perturbation, decoding, or algorithm boundary differs.
    """

    schema_version: str = "1.0.0"
    certificate_id: str = Field(default_factory=lambda: f"cert-{uuid.uuid4().hex[:12]}")
    experimental: bool = True
    envelope: OperatingEnvelopeSpec
    status: str = "FAIL"  # One of: PASS, FAIL, INVALID/OUT_OF_ENVELOPE
    reference_mean: Optional[float] = None
    d_stability_estimate: Optional[float] = None
    d_stability_upper_95_sim: Optional[float] = None
    epsilon_stability: float = 0.02
    missing_cells_count: int = 0
    drift_events: List[str] = Field(default_factory=list)
    worst_env_id: Optional[str] = None
    per_env_drops: Dict[str, float] = Field(default_factory=dict)
    per_env_se: Dict[str, float] = Field(default_factory=dict)
    repeat_noise_se: Optional[float] = None
    recommended_min_n: Optional[int] = None
    diagnostic_note: Optional[str] = None

    def certificate_sha256(self) -> str:
        return _canonical_json_sha256(self.model_dump(mode="json"))

    @property
    def is_passing_within_envelope(self) -> bool:
        return (
            self.status == "PASS"
            and self.missing_cells_count == 0
            and len(self.drift_events) == 0
            and self.d_stability_upper_95_sim is not None
            and float(self.d_stability_upper_95_sim) < float(self.epsilon_stability)
        )

    def validate_request(
        self,
        request_envelope: Optional[OperatingEnvelopeSpec],
        *,
        has_confirmation_validation: bool = False,
        strict: bool = False,
    ) -> EnvelopeValidationResult:
        """Validate a caller's operating envelope against this certificate (fail-closed)."""
        reasons: List[str] = []
        if request_envelope is None:
            reasons.append("missing_request_envelope")
        else:
            if request_envelope.task_id != self.envelope.task_id:
                reasons.append(
                    f"task_id_mismatch:{request_envelope.task_id}!={self.envelope.task_id}"
                )
            if request_envelope.model_id != self.envelope.model_id:
                reasons.append(
                    f"model_id_mismatch:{request_envelope.model_id}!={self.envelope.model_id}"
                )
            if request_envelope.model_version != self.envelope.model_version:
                reasons.append(
                    f"model_version_mismatch:{request_envelope.model_version}!={self.envelope.model_version}"
                )
            if (
                request_envelope.dataset_partition_hash
                != self.envelope.dataset_partition_hash
            ):
                reasons.append("dataset_partition_hash_mismatch")
            if request_envelope.parser_hash != self.envelope.parser_hash:
                reasons.append("parser_hash_mismatch")
            if sorted(set(request_envelope.perturbation_env_ids)) != sorted(
                set(self.envelope.perturbation_env_ids)
            ):
                reasons.append("perturbation_environments_mismatch")
            if request_envelope.algorithm_version != self.envelope.algorithm_version:
                reasons.append(
                    f"algorithm_version_mismatch:{request_envelope.algorithm_version}!={self.envelope.algorithm_version}"
                )
            if request_envelope.decoding_config != self.envelope.decoding_config:
                reasons.append("decoding_config_mismatch")

        if self.missing_cells_count > 0:
            reasons.append(f"missing_cells:{self.missing_cells_count}")
        if self.drift_events:
            reasons.extend(f"drift:{ev}" for ev in self.drift_events)

        if reasons:
            if strict:
                raise ValueError(
                    "Certificate validation refused (out of envelope): "
                    + "; ".join(reasons)
                )
            fallback_state = (
                PromptEvidenceState.CONFIRMATION_VALIDATED
                if (
                    has_confirmation_validation
                    and not any("mismatch" in r for r in reasons)
                )
                else PromptEvidenceState.PREDICTED
            )
            return EnvelopeValidationResult(
                valid=False,
                evidence_state=fallback_state,
                can_display_passing_stability_badge=False,
                badge_label="OUT-OF-ENVELOPE (NO STABILITY ASSURANCE)",
                mismatch_reasons=reasons,
            )

        if self.is_passing_within_envelope:
            return EnvelopeValidationResult(
                valid=True,
                evidence_state=PromptEvidenceState.STABILITY_CERTIFIED,
                can_display_passing_stability_badge=True,
                badge_label=(
                    f"STABILITY-CERTIFIED WITHIN ENVELOPE {self.envelope.envelope_sha256()[:12]} "
                    f"(U95={self.d_stability_upper_95_sim:.4f} < {self.epsilon_stability:.4f})"
                ),
                mismatch_reasons=[],
            )

        return EnvelopeValidationResult(
            valid=False,
            evidence_state=(
                PromptEvidenceState.CONFIRMATION_VALIDATED
                if has_confirmation_validation
                else PromptEvidenceState.PREDICTED
            ),
            can_display_passing_stability_badge=False,
            badge_label=f"STABILITY NOT CERTIFIED ({self.status})",
            mismatch_reasons=[f"certificate_status:{self.status}"],
        )


class PromptSelectionReport(BaseModel):
    """Public selection report explicitly distinguishing predicted, confirmation-validated, and stability-certified states."""

    schema_version: str = "1.0.0"
    strategy: str
    evidence_state: str = PromptEvidenceState.PREDICTED
    experimental: bool = False
    selected_factor_levels: Dict[str, int]
    selected_run_id: Optional[int] = None
    predicted_or_screen_score: float
    confirmation_score: Optional[float] = None
    can_display_passing_stability_badge: bool = False
    stability_badge_label: str = "UNVERIFIED (SINGLE-STAGE PREDICTED ONLY)"
    envelope: Optional[OperatingEnvelopeSpec] = None
    certificate: Optional[PromptStabilityCertificate] = None
    notes: List[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            "### PromptPrism Selection & Evidence State Report",
            f"- **Strategy:** `{self.strategy}`"
            + (" *(experimental opt-in)*" if self.experimental else ""),
            f"- **Evidence State:** `{self.evidence_state}`",
            f"- **Stability Badge:** `{self.stability_badge_label}`",
            f"- **Selected Factor Levels:** `{self.selected_factor_levels}`",
            f"- **Screen / Model-Predicted Score:** `{self.predicted_or_screen_score:.4f}`",
        ]
        if self.confirmation_score is not None:
            lines.append(
                f"- **Held-Out Confirmation Score:** `{self.confirmation_score:.4f}`"
            )
        if self.envelope is not None:
            lines.append(
                f"- **Operating Envelope:** task=`{self.envelope.task_id}`, "
                f"model=`{self.envelope.model_id}@{self.envelope.model_version}`, "
                f"sha256=`{self.envelope.envelope_sha256()[:16]}`"
            )
        for note in self.notes:
            lines.append(f"- *Note:* {note}")
        return "\n".join(lines)
