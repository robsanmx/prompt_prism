"""
prompt_prism 2.0: Prompt Diagnostics, Token Pruning, Calibrated Attribution,
and Envelope-Bounded Stability Certification for LLMs.

Applies Fractional Factorial Designs (2^(k-p)), Plackett-Burman screening,
contrast-specific item-clustered inference, TOST token-bloat pruning, and
fail-closed operating-envelope stability certificates.
"""

from .analysis.anova import ANOVAEngine, ANOVAResult, ANOVARow
from .analysis.diagnostics import (
    PRUNING_VERDICTS,
    CalibratedAttributionAudit,
    CalibratedFactorDiagnostic,
    NoiseFloorSummary,
    run_calibrated_attribution_audit,
)
from .analysis.effects import EffectAnalyzer, FactorEffect, InteractionEffect
from .analysis.optimizer import OptimalPromptFinder, OptimalPromptRecommendation
from .analysis.selection import (
    PROMOTED_TO_DEFAULT_PUBLIC_API,
    REAL_V4_REGISTERED_VERDICT,
    ExperimentalTwoStageSelector,
    build_selection_report_from_recommendation,
    evaluate_simultaneous_perturbation_certificate,
)
from .core.factors import Factor, FactorSet, FactorType, Level
from .core.models import (
    DesignMatrix,
    EnvelopeValidationResult,
    ExperimentResults,
    OperatingEnvelopeSpec,
    PromptEvidenceState,
    PromptSelectionReport,
    PromptStabilityCertificate,
    RunConfig,
    Trial,
)
from .design.aliasing import AliasStructure
from .design.catalog import CATALOG_DESIGNS, get_catalog_entry, list_available_plans
from .design.generators import (
    FractionalFactorialGenerator,
    FullFactorialGenerator,
    PlackettBurmanGenerator,
)
from .design.recommender import recommend_design
from .evaluation.check_suite import CheckSuiteMetric
from .evaluation.evaluator import Evaluator
from .evaluation.metrics import (
    CustomMetric,
    ExactMatch,
    F1Score,
    JSONValidation,
    KeyValuesExtractionOverlap,
    LevenshteinSimilarity,
    Metric,
    RegexMatch,
)
from .experiment import Experiment
from .reporting.reporter import AnalysisReport
from .runner.cache import ResponseCache
from .runner.client import CallableLLM, LLMClient, LLMResponse, MockLLM
from .runner.runner import ExperimentRunner
from .template.composer import PromptComposer, PromptSection, PromptTemplate
from .testgen.generator import TestSetGenerator
from .testgen.models import (
    RESERVED_TEST_CASE_KEYS,
    CheckSpec,
    TestCase,
    TestSet,
    TestSetManifest,
)
from .visualization.plots import (
    generate_ascii_pareto,
    plot_interaction_effects,
    plot_main_effects,
    plot_pareto_effects,
)

__version__ = "2.0.0"

__all__ = [
    # Core & v2.0 Envelope / Certificate Contracts
    "Factor",
    "Level",
    "FactorType",
    "FactorSet",
    "DesignMatrix",
    "RunConfig",
    "Trial",
    "ExperimentResults",
    "PromptEvidenceState",
    "OperatingEnvelopeSpec",
    "EnvelopeValidationResult",
    "PromptStabilityCertificate",
    "PromptSelectionReport",
    # Experiment Orchestrator
    "Experiment",
    # Design of Experiments
    "FractionalFactorialGenerator",
    "PlackettBurmanGenerator",
    "FullFactorialGenerator",
    "AliasStructure",
    "recommend_design",
    "CATALOG_DESIGNS",
    "get_catalog_entry",
    "list_available_plans",
    # Templating
    "PromptSection",
    "PromptTemplate",
    "PromptComposer",
    # Runner & LLM
    "LLMClient",
    "CallableLLM",
    "MockLLM",
    "LLMResponse",
    "ResponseCache",
    "ExperimentRunner",
    # Evaluation
    "Metric",
    "CheckSuiteMetric",
    "ExactMatch",
    "F1Score",
    "JSONValidation",
    "KeyValuesExtractionOverlap",
    "LevenshteinSimilarity",
    "RegexMatch",
    "CustomMetric",
    "Evaluator",
    "DeepEvalMetric",
    "deepeval_metric",
    "JudgeCache",
    # Statistical Analysis, Calibrated Attribution & v2.0 Selection
    "ANOVAEngine",
    "ANOVAResult",
    "ANOVARow",
    "EffectAnalyzer",
    "FactorEffect",
    "InteractionEffect",
    "OptimalPromptFinder",
    "OptimalPromptRecommendation",
    "CalibratedAttributionAudit",
    "CalibratedFactorDiagnostic",
    "NoiseFloorSummary",
    "PRUNING_VERDICTS",
    "run_calibrated_attribution_audit",
    "ExperimentalTwoStageSelector",
    "PROMOTED_TO_DEFAULT_PUBLIC_API",
    "REAL_V4_REGISTERED_VERDICT",
    "build_selection_report_from_recommendation",
    "evaluate_simultaneous_perturbation_certificate",
    # Test Set Schema, Generator & Freezing
    "RESERVED_TEST_CASE_KEYS",
    "CheckSpec",
    "TestCase",
    "TestSet",
    "TestSetGenerator",
    "TestSetManifest",
    # Visualization & Reporting
    "plot_main_effects",
    "plot_pareto_effects",
    "plot_interaction_effects",
    "generate_ascii_pareto",
    "AnalysisReport",
]


def __getattr__(name: str):
    if name in {"DeepEvalMetric", "deepeval_metric"}:
        from .evaluation.deepeval_metrics import DeepEvalMetric, deepeval_metric

        if name == "DeepEvalMetric":
            return DeepEvalMetric
        return deepeval_metric
    if name == "JudgeCache":
        from .evaluation.judge_cache import JudgeCache

        return JudgeCache
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
