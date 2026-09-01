"""
prompt_prism: Statistical Prompt Optimization & Factorial Analysis for LLMs.

A universal, statistically rigorous framework for optimizing LLM prompts using
Fractional Factorial Designs (2^(k-p)), Plackett-Burman screening, and ANOVA.
"""

from .analysis.anova import ANOVAEngine, ANOVAResult, ANOVARow
from .analysis.effects import EffectAnalyzer, FactorEffect, InteractionEffect
from .analysis.optimizer import OptimalPromptFinder, OptimalPromptRecommendation
from .core.factors import Factor, FactorSet, FactorType, Level
from .core.models import DesignMatrix, ExperimentResults, RunConfig, Trial
from .design.aliasing import AliasStructure
from .design.catalog import CATALOG_DESIGNS, get_catalog_entry, list_available_plans
from .design.generators import (
    FractionalFactorialGenerator,
    FullFactorialGenerator,
    PlackettBurmanGenerator,
)
from .design.recommender import recommend_design
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
from .visualization.plots import (
    generate_ascii_pareto,
    plot_interaction_effects,
    plot_main_effects,
    plot_pareto_effects,
)

__version__ = "1.0.0"

__all__ = [
    # Core
    "Factor",
    "Level",
    "FactorType",
    "FactorSet",
    "DesignMatrix",
    "RunConfig",
    "Trial",
    "ExperimentResults",
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
    # Statistical Analysis & ANOVA
    "ANOVAEngine",
    "ANOVAResult",
    "ANOVARow",
    "EffectAnalyzer",
    "FactorEffect",
    "InteractionEffect",
    "OptimalPromptFinder",
    "OptimalPromptRecommendation",
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
