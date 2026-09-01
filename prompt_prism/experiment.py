"""
Top-Level Experiment Orchestrator for Prompt Optimization using Fractional Factorial DoE & ANOVA.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, Optional, Sequence, Union

import pandas as pd

from .core.factors import Factor, FactorSet
from .core.models import DesignMatrix, ExperimentResults
from .design.generators import FractionalFactorialGenerator
from .design.recommender import recommend_design
from .evaluation.evaluator import Evaluator
from .evaluation.metrics import ExactMatch, F1Score, Metric
from .reporting.reporter import AnalysisReport
from .runner.client import LLMClient
from .runner.runner import ExperimentRunner
from .template.composer import PromptComposer, PromptTemplate


class Experiment:
    """
    All-in-one Experiment Manager for Prompt Optimization using Fractional Factorial Design of Experiments & ANOVA.
    """

    def __init__(
        self,
        factors: Union[FactorSet, Sequence[Factor]],
        template: PromptTemplate,
        design: Optional[Union[DesignMatrix, str]] = None,
        max_runs: Optional[int] = None,
        evaluator: Optional[Union[Evaluator, Sequence[Metric]]] = None,
        target_metric: str = "f1_score",
        title: str = "PromptPrism Experiment",
    ):
        self.factors = (
            FactorSet(factors) if isinstance(factors, (list, tuple)) else factors
        )
        self.template = template
        self.composer = PromptComposer(template=self.template, factors=self.factors)
        self.title = title
        self.target_metric = target_metric

        # Set up design matrix
        if isinstance(design, DesignMatrix):
            self.design = design
        elif isinstance(design, str):
            self.design = FractionalFactorialGenerator.from_plan_id(
                design, factor_names=self.factors.names
            )
        else:
            self.design = recommend_design(factors=self.factors, max_runs=max_runs)

        if isinstance(evaluator, Evaluator):
            self.evaluator = evaluator
        elif evaluator:
            self.evaluator = Evaluator(evaluator)
        else:
            self.evaluator = Evaluator([ExactMatch(), F1Score()])

        self.last_results: Optional[ExperimentResults] = None
        self.last_report: Optional[AnalysisReport] = None

    @classmethod
    def from_factors(
        cls,
        factors: Sequence[Factor],
        design: Optional[Union[DesignMatrix, str]] = None,
        max_runs: Optional[int] = None,
        system_prompt: Optional[str] = None,
        data_template: Optional[str] = None,
        metrics: Optional[Sequence[Metric]] = None,
        target_metric: str = "f1_score",
        title: str = "PromptPrism Experiment",
    ) -> Experiment:
        """
        Create an Experiment directly from a list of Factors and templates.

        Note: Supplying `system_prompt` assigns `role="system"` and configures the runner
        to pass structured chat messages (`List[Dict[str, str]]`) to the LLM client rather
        than a single formatted text string.
        """
        factor_set = FactorSet(factors)
        template = PromptTemplate.from_factors(
            factors=factor_set,
            system_prompt=system_prompt,
            data_template=data_template,
        )
        return cls(
            factors=factor_set,
            template=template,
            design=design,
            max_runs=max_runs,
            evaluator=metrics,
            target_metric=target_metric,
            title=title,
        )

    def estimate_judge_calls(
        self, dataset: Union[pd.DataFrame, Sequence[Dict[str, Any]]]
    ) -> int:
        """Estimate the total number of LLM judge calls needed for this experiment."""
        n_samples = (
            len(dataset)
            if isinstance(dataset, (list, tuple, pd.DataFrame))
            else len(list(dataset))
        )
        n_runs = len(self.design.runs)
        n_judge_metrics = sum(
            1 for m in self.evaluator.metrics if getattr(m, "is_llm_judge", False)
        )
        return n_runs * n_samples * n_judge_metrics

    def run(
        self,
        dataset: Union[pd.DataFrame, Sequence[Dict[str, Any]]],
        client: Union[LLMClient, Callable[..., Any]],
        max_workers: int = 4,
        cache_db: Optional[str] = None,
        target_col: str = "target",
        id_col: str = "id",
        experiment_id: str = "exp_01",
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> ExperimentResults:
        """
        Execute the experiment over the test dataset.
        """
        from .runner.cache import ResponseCache

        judge_calls = self.estimate_judge_calls(dataset)
        if judge_calls > 0:
            warnings.warn(
                f"[PromptPrism] Notice: This experiment includes LLM Judge metric(s). "
                f"Estimated judge calls: {judge_calls} ({len(self.design.runs)} runs × {len(dataset)} samples). "
                "Ensure caching is enabled to minimize repeated judge costs.",
                UserWarning,
                stacklevel=2,
            )

        cache = ResponseCache(db_path=cache_db) if cache_db else None
        runner = ExperimentRunner(
            composer=self.composer,
            client=client,
            evaluator=self.evaluator,
            cache=cache,
            max_workers=max_workers,
            target_col=target_col,
            id_col=id_col,
        )

        self.last_results = runner.run(
            design=self.design,
            dataset=dataset,
            experiment_id=experiment_id,
            progress_callback=progress_callback,
        )
        return self.last_results

    def analyze(
        self,
        results: Optional[ExperimentResults] = None,
        target_metric: Optional[str] = None,
        block_by: Optional[str] = "sample_id",
        include_interactions: bool = False,
        alpha: float = 0.05,
        maximize: Optional[bool] = None,
    ) -> AnalysisReport:
        """
        Perform ANOVA and Main Effects analysis on the experiment results using RCBD blocking.
        """
        exp_res = results or self.last_results
        if exp_res is None:
            raise ValueError(
                "No experiment results provided or available. Run the experiment first."
            )

        metric_name = target_metric or self.target_metric
        name_map = dict(zip(self.factors.ids, self.factors.names))

        if maximize is None:
            maximize = True
            for m in self.evaluator.metrics:
                if m.name == metric_name:
                    maximize = getattr(m, "higher_is_better", True)
                    break

        self.last_report = AnalysisReport.generate(
            experiment_results=exp_res,
            target_metric=metric_name,
            factor_name_map=name_map,
            block_col=block_by,
            include_interactions=include_interactions,
            alpha=alpha,
            title=f"{self.title} - ANOVA Analysis ({metric_name})",
            maximize=maximize,
        )
        return self.last_report

    def suggest_confirmation_design(
        self,
        report: Optional[AnalysisReport] = None,
        max_runs: Optional[int] = 16,
    ) -> DesignMatrix:
        """
        Given a screening experiment result, suggests an unaliased Resolution V or Full Factorial
        confirmation design over the surviving/candidate factors.
        """
        rep = report or self.last_report
        if rep is None:
            rep = self.analyze()

        opt = rep.optimal_recommendation
        if not opt:
            raise ValueError("No optimal recommendation available in report.")

        # Identify candidate factors that had non-zero/significant effects
        candidate_fids = set()
        for d in opt.significant_positive_drivers:
            candidate_fids.add(d.factor_id)
        for d in opt.harmful_negative_factors:
            candidate_fids.add(d.factor_id)

        # If too few, include top neutral factors
        if len(candidate_fids) < 3 and rep.anova_result:
            for eff in rep.anova_result.main_effects:
                candidate_fids.add(eff.factor_id)
                if len(candidate_fids) >= 3:
                    break

        surviving_factors = [
            self.factors[fid] for fid in candidate_fids if fid in self.factors.ids
        ]
        if not surviving_factors:
            surviving_factors = list(self.factors)

        # Recommends Resolution V or Full Factorial
        return recommend_design(
            factors=surviving_factors,
            max_runs=max_runs,
            min_resolution=5,
        )

    def get_optimal_prompt_template(
        self,
        report: Optional[AnalysisReport] = None,
        target_metric: Optional[str] = None,
        maximize: Optional[bool] = None,
    ) -> PromptTemplate:
        """
        Returns a new PromptTemplate pre-configured with the statistically optimal factor levels.

        Args:
            report: An existing report to read. Defaults to the last one analyzed.
            target_metric: Re-analyze for this metric. Ignored when `report` is passed
                explicitly, since that report is the caller's chosen input.
            maximize: Re-analyze in this direction. As above.

        Passing `target_metric` or `maximize` re-runs `analyze()` when they disagree with the
        cached report, rather than silently returning a recommendation computed for a
        different metric or the opposite direction.
        """
        rep = report or self.last_report

        # An explicit override must not be silently dropped just because a report is cached.
        if (
            report is None
            and rep is not None
            and (target_metric or maximize is not None)
        ):
            wanted_metric = target_metric or rep.anova_result.target_metric
            wanted_maximize = (
                maximize
                if maximize is not None
                else rep.anova_result.metadata.get("maximize", True)
            )
            if (
                wanted_metric != rep.anova_result.target_metric
                or wanted_maximize != rep.anova_result.metadata.get("maximize", True)
            ):
                rep = self.analyze(target_metric=target_metric, maximize=maximize)

        if rep is None:
            rep = self.analyze(target_metric=target_metric, maximize=maximize)

        opt_rec = rep.optimal_recommendation
        if not opt_rec:
            raise ValueError("Could not find optimal recommendation in report.")

        # Create configured template
        configured_template = PromptTemplate(
            sections=list(self.template.sections),
            master_template=self.template.master_template,
            default_role=self.template.default_role,
            delimiter=self.template.delimiter,
        )

        for sec in configured_template.sections:
            if sec.factor_id:
                factor = self.factors.get(sec.factor_id)
                if factor:
                    opt_level_code = opt_rec.optimal_factor_levels.get(factor.id, 0)
                    lvl = factor.get_level(opt_level_code)
                    sec.content = str(lvl.content)

        return configured_template
