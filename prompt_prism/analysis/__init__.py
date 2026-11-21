"""
Statistical Analysis & ANOVA engine module.
"""

from .anova import ANOVAEngine, ANOVAResult, ANOVARow
from .diagnostics import (
    PRUNING_VERDICTS,
    CalibratedAttributionAudit,
    CalibratedFactorDiagnostic,
    NoiseFloorSummary,
    run_calibrated_attribution_audit,
)
from .effects import EffectAnalyzer, FactorEffect, InteractionEffect
from .optimizer import OptimalPromptFinder, OptimalPromptRecommendation
from .selection import (
    PROMOTED_TO_DEFAULT_PUBLIC_API,
    REAL_V4_REGISTERED_VERDICT,
    ExperimentalTwoStageSelector,
    build_selection_report_from_recommendation,
    evaluate_simultaneous_perturbation_certificate,
)

__all__ = [
    "ANOVAEngine",
    "ANOVAResult",
    "ANOVARow",
    "CalibratedAttributionAudit",
    "CalibratedFactorDiagnostic",
    "EffectAnalyzer",
    "ExperimentalTwoStageSelector",
    "FactorEffect",
    "InteractionEffect",
    "NoiseFloorSummary",
    "OptimalPromptFinder",
    "OptimalPromptRecommendation",
    "PROMOTED_TO_DEFAULT_PUBLIC_API",
    "PRUNING_VERDICTS",
    "REAL_V4_REGISTERED_VERDICT",
    "build_selection_report_from_recommendation",
    "evaluate_simultaneous_perturbation_certificate",
    "run_calibrated_attribution_audit",
]
