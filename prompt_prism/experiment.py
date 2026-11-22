"""
Top-Level Experiment Orchestrator for Prompt Optimization using Fractional Factorial DoE & ANOVA.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Collection, Dict, Mapping, Optional, Sequence, Union

import pandas as pd

from .core.factors import Factor, FactorSet
from .core.models import (
    DesignMatrix,
    ExperimentResults,
    OperatingEnvelopeSpec,
    PromptSelectionReport,
    PromptStabilityCertificate,
)
from .design.generators import FractionalFactorialGenerator
from .design.recommender import recommend_design
from .evaluation.evaluator import Evaluator
from .evaluation.metrics import ExactMatch, F1Score, Metric
from .reporting.reporter import AnalysisReport
from .runner.client import LLMClient
from .runner.runner import ExperimentRunner
from .template.composer import PromptComposer, PromptTemplate
from .testgen.models import SplitSelector, TestSet


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
        self._last_test_set: Optional[TestSet] = None
        self._last_split: Optional[SplitSelector] = None

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
        self,
        dataset: Union[pd.DataFrame, Sequence[Dict[str, Any]], TestSet],
        *,
        split: SplitSelector = "discovery",
    ) -> int:
        """Estimate the total number of LLM judge calls needed for this experiment."""
        raw_items: Union[pd.DataFrame, Sequence[Dict[str, Any]]] = (
            dataset.to_dataset(split=split) if isinstance(dataset, TestSet) else dataset
        )
        items: Union[pd.DataFrame, Sequence[Dict[str, Any]]] = (
            raw_items
            if isinstance(raw_items, (list, tuple, pd.DataFrame))
            else list(raw_items)
        )
        n_samples = len(items)
        n_runs = len(self.design.runs)
        per_run_calls = 0
        for m in self.evaluator.metrics:
            if hasattr(m, "estimate_judge_calls"):
                per_run_calls += int(m.estimate_judge_calls(items))
            elif getattr(m, "is_llm_judge", False):
                per_run_calls += n_samples
        return n_runs * per_run_calls

    def run(
        self,
        dataset: Union[pd.DataFrame, Sequence[Dict[str, Any]], TestSet],
        client: Union[LLMClient, Callable[..., Any]],
        max_workers: int = 4,
        cache_db: Optional[str] = None,
        target_col: str = "target",
        id_col: str = "id",
        experiment_id: str = "exp_01",
        progress_callback: Optional[Callable[[int, int], None]] = None,
        *,
        split: SplitSelector = "discovery",
        model_id: Optional[str] = None,
    ) -> ExperimentResults:
        """
        Execute the experiment over a dataset or frozen `TestSet` (defaults to `split='discovery'`).
        """
        from .runner.cache import ResponseCache

        test_set_obj: Optional[TestSet] = (
            dataset if isinstance(dataset, TestSet) else None
        )
        if test_set_obj is not None and test_set_obj.is_frozen:
            # Fail fast before spending LLM calls if a frozen TestSet was tampered in memory.
            test_set_obj.verify_integrity()
        run_items: Union[pd.DataFrame, Sequence[Dict[str, Any]]] = (
            test_set_obj.to_dataset(split=split)
            if test_set_obj is not None
            else (
                dataset
                if isinstance(dataset, (list, tuple, pd.DataFrame))
                else list(dataset)
            )
        )

        judge_calls = self.estimate_judge_calls(run_items)
        if judge_calls > 0:
            n_samples = (
                len(run_items)
                if isinstance(run_items, (list, tuple, pd.DataFrame))
                else len(list(run_items))
            )
            warnings.warn(
                f"[PromptPrism] Notice: This experiment includes LLM Judge metric(s). "
                f"Estimated judge calls: {judge_calls} ({len(self.design.runs)} runs × {n_samples} samples). "
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
            dataset=run_items,
            experiment_id=experiment_id,
            progress_callback=progress_callback,
        )

        eff_model_id = (
            model_id if model_id is not None else getattr(client, "model_id", None)
        )
        if eff_model_id:
            self.last_results.metadata["model_id"] = str(eff_model_id)

        if test_set_obj is not None:
            self._last_test_set = test_set_obj
            self._last_split = split
            self.last_results.metadata["test_set_sha256"] = (
                test_set_obj.manifest.cases_sha256
            )
            self.last_results.metadata["test_set_split"] = split
            self.last_results.metadata["coverage_source"] = test_set_obj.provenance
            self.last_results.metadata["judge_scored_fraction"] = (
                test_set_obj.judge_scored_fraction(split=split)
            )
            gen_model = test_set_obj.manifest.generator.get("model_id")
            if gen_model:
                self.last_results.metadata["generator_model_id"] = str(gen_model)
        else:
            self._last_test_set = None
            self._last_split = None

        return self.last_results

    def analyze(
        self,
        results: Optional[ExperimentResults] = None,
        target_metric: Optional[str] = None,
        block_by: Optional[str] = "sample_id",
        include_interactions: bool = False,
        alpha: float = 0.05,
        maximize: Optional[bool] = None,
        strategy: str = "additive",
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
            strategy=strategy,
        )
        return self.last_report

    def diagnose(
        self,
        results: Optional[ExperimentResults] = None,
        target_metric: Optional[str] = None,
        *,
        sample_col: str = "sample_id",
        alpha: float = 0.05,
        maximize: Optional[bool] = None,
        equivalence_margin: float = 0.01,
        token_costs: Optional[Dict[str, int]] = None,
        n_sign_flip: int = 2000,
        seed: int = 42,
        test_set: Optional[TestSet] = None,
        split: Optional[SplitSelector] = None,
        allow_unfrozen: bool = False,
        coverage: Optional[Mapping[str, Collection[str]]] = None,
        baseline_sample_ids: Optional[Collection[str]] = None,
        min_targeted_items: int = 20,
        targeted_margin: Optional[float] = None,
        coverage_source: Optional[str] = None,
        test_set_sha256: Optional[str] = None,
        judge_scored_fraction: Optional[float] = None,
        generator_model_id: Optional[str] = None,
        model_id: Optional[str] = None,
        env_col: Optional[str] = None,
        ref_env: str = "ref",
    ):
        """Run PromptPrism 2.0 calibrated item-clustered attribution, noise-floor, TOST token-pruning, and coverage-gate audit.

        When `token_costs` is omitted, estimates incremental token count per binary factor
        (`max(0, ceil((len(lvl1) - len(lvl0)) / 4))`) from the configured Factor definitions.
        When `test_set` is passed (or automatically retained from `exp.run(test_set, ...)`),
        verifies frozen integrity, supplies coverage/baseline splits, checks for split leakage,
        and attaches manifest provenance. Factor names in `coverage`, `targets_factors`, or
        `token_costs` are automatically resolved against `self.factors` to match `design.factor_ids`.
        """
        from .analysis.diagnostics import (
            _normalize_id_collection,
            _parse_targets_factors_cell,
            run_calibrated_attribution_audit,
        )

        exp_res = results or self.last_results
        if exp_res is None:
            raise ValueError(
                "No experiment results provided or available. Run the experiment first."
            )

        eff_test_set: Optional[TestSet] = (
            test_set
            if test_set is not None
            else (self._last_test_set if results is None else None)
        )
        eff_split: SplitSelector = (
            split
            if split is not None
            else exp_res.metadata.get("test_set_split")
            or (self._last_split if results is None else None)
            or "discovery"
        )

        eff_coverage = coverage
        eff_baseline_ids = baseline_sample_ids
        eff_coverage_source = coverage_source
        eff_test_set_sha = (
            test_set_sha256
            if test_set_sha256 is not None
            else exp_res.metadata.get("test_set_sha256")
        )
        eff_judge_frac = (
            judge_scored_fraction
            if judge_scored_fraction is not None
            else exp_res.metadata.get("judge_scored_fraction")
        )
        eff_gen_model = (
            generator_model_id
            if generator_model_id is not None
            else exp_res.metadata.get("generator_model_id")
        )

        if eff_test_set is not None:
            if not eff_test_set.is_frozen and not allow_unfrozen:
                raise ValueError(
                    "TestSet must be frozen before running diagnose(). "
                    "Call test_set.freeze() or pass allow_unfrozen=True."
                )
            if eff_test_set.is_frozen:
                eff_test_set.verify_integrity()
            if eff_coverage is None:
                eff_coverage = eff_test_set.coverage(split=eff_split)
            if eff_baseline_ids is None:
                eff_baseline_ids = eff_test_set.baseline_sample_ids(split=eff_split)
            if eff_coverage_source is None:
                eff_coverage_source = eff_test_set.provenance
            if eff_test_set_sha is None:
                eff_test_set_sha = eff_test_set.manifest.cases_sha256
            if eff_judge_frac is None:
                eff_judge_frac = eff_test_set.judge_scored_fraction(split=eff_split)
            if eff_gen_model is None:
                gen_m = eff_test_set.manifest.generator.get("model_id")
                if gen_m:
                    eff_gen_model = str(gen_m)
        elif eff_coverage_source is None:
            eff_coverage_source = exp_res.metadata.get("coverage_source")

        metric_name = target_metric or self.target_metric
        if maximize is None:
            maximize = True
            for m in self.evaluator.metrics:
                if m.name == metric_name:
                    maximize = getattr(m, "higher_is_better", True)
                    break

        factor_ids = [str(fid) for fid in exp_res.design.factor_ids]
        factor_id_set = set(factor_ids)
        alias_to_col: Dict[str, str] = {fid: fid for fid in factor_ids}
        for factor in self.factors:
            if factor.id in factor_id_set and factor.name not in factor_id_set:
                alias_to_col[factor.name] = factor.id
            elif factor.name in factor_id_set and factor.id not in factor_id_set:
                alias_to_col[factor.id] = factor.name

        eff_token_costs: Dict[str, int] = {}
        if token_costs is not None:
            for k, v in token_costs.items():
                eff_token_costs[alias_to_col.get(str(k), str(k))] = int(v)
        else:
            for factor in self.factors:
                if len(factor.levels) >= 2:
                    c0 = len(str(factor.get_level(0).content))
                    c1 = len(str(factor.get_level(1).content))
                    key = alias_to_col.get(factor.id, factor.id)
                    eff_token_costs[key] = max(0, (c1 - c0 + 3) // 4)

        df = exp_res.to_dataframe()
        resolved_coverage: Optional[Dict[str, set]] = None
        if eff_coverage is not None:
            resolved_coverage = {}
            for k, sids in eff_coverage.items():
                target_key = alias_to_col.get(str(k), str(k))
                norm_ids = _normalize_id_collection(sids, param_name=f"coverage[{k!r}]")
                resolved_coverage.setdefault(target_key, set()).update(norm_ids)
        elif "targets_factors" in df.columns:
            df = df.copy()
            df["targets_factors"] = [
                [alias_to_col.get(tag, tag) for tag in _parse_targets_factors_cell(val)]
                for val in df["targets_factors"]
            ]

        eff_model_id = (
            model_id if model_id is not None else exp_res.metadata.get("model_id")
        )

        audit = run_calibrated_attribution_audit(
            data=df,
            factor_cols=factor_ids,
            target_col=metric_name,
            sample_col=sample_col,
            alpha=alpha,
            maximize=maximize,
            equivalence_margin=equivalence_margin,
            token_costs=eff_token_costs or None,
            n_sign_flip=n_sign_flip,
            seed=seed,
            coverage=resolved_coverage,
            baseline_sample_ids=eff_baseline_ids,
            min_targeted_items=min_targeted_items,
            targeted_margin=targeted_margin,
            coverage_source=eff_coverage_source,
            test_set_sha256=eff_test_set_sha,
            judge_scored_fraction=eff_judge_frac,
            generator_model_id=eff_gen_model,
            model_id=eff_model_id,
            env_col=env_col,
            ref_env=ref_env,
        )

        if (
            eff_test_set is not None
            and sample_col in df.columns
            and eff_test_set.detect_split_leakage(df[sample_col].astype(str).unique())
        ):
            audit.warnings.append(
                "Split leakage detected: evaluated results mix both 'discovery' and "
                "'confirmation' items from the TestSet."
            )

        return audit

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

    def selection_report(
        self,
        report: Optional[AnalysisReport] = None,
        *,
        strategy: str = "additive",
        confirmation_scores: Optional[Sequence[float]] = None,
        confirmation_envelope: Optional[OperatingEnvelopeSpec] = None,
        stability_certificate: Optional[PromptStabilityCertificate] = None,
        request_envelope: Optional[OperatingEnvelopeSpec] = None,
    ) -> PromptSelectionReport:
        """Return a `PromptSelectionReport` explicitly distinguishing `predicted`,
        `confirmation-validated`, and `stability-certified` evidence states.

        Existing `additive`, `screen_best`, and `interaction_2fi` callers default strictly to
        `evidence_state="predicted"` (`can_display_passing_stability_badge=False`) unless
        untouched confirmation outcomes and matching envelope/certificate metadata are provided.
        """
        from .analysis.selection import build_selection_report_from_recommendation

        rep = report
        if rep is None:
            if self.last_report is not None and (
                self.last_report.optimal_recommendation is not None
                and self.last_report.optimal_recommendation.strategy == strategy
            ):
                rep = self.last_report
            else:
                rep = self.analyze(
                    strategy=strategy,
                    include_interactions=(strategy == "interaction_2fi"),
                )

        if rep.optimal_recommendation is None:
            raise ValueError(
                "No optimal recommendation available to build selection report."
            )

        return build_selection_report_from_recommendation(
            rep.optimal_recommendation,
            confirmation_scores=confirmation_scores,
            confirmation_envelope=confirmation_envelope,
            stability_certificate=stability_certificate,
            request_envelope=request_envelope,
        )

    def experimental_select_two_stage(
        self,
        *,
        validation_scores_by_env: Dict[str, Dict[int, Sequence[float]]],
        experimental_opt_in: bool = False,
        screen_results: Optional[ExperimentResults] = None,
        target_metric: Optional[str] = None,
        group_ids: Optional[Sequence[str]] = None,
        confirmation_scores: Optional[Sequence[float]] = None,
        confirmation_envelope: Optional[OperatingEnvelopeSpec] = None,
        stability_certificate: Optional[PromptStabilityCertificate] = None,
        request_envelope: Optional[OperatingEnvelopeSpec] = None,
        max_candidates: int = 4,
    ) -> PromptSelectionReport:
        """Experimental opt-in two-stage screen -> multi-environment validation selector.

        Requires `experimental_opt_in=True` because the prospective Real-v4 promotion gate
        (`PASS_BOTH`) was not met (`FAIL_BOTH`); does not claim a default stability assurance.
        """
        from .analysis.selection import ExperimentalTwoStageSelector

        selector = ExperimentalTwoStageSelector(
            experimental_opt_in=experimental_opt_in,
            max_candidates=max_candidates,
        )
        exp_res = screen_results or self.last_results
        if exp_res is None:
            raise ValueError(
                "No screening ExperimentResults available. Run `experiment.run(...)` first."
            )

        metric_name = target_metric or self.target_metric
        maximize = True
        for m in self.evaluator.metrics:
            if m.name == metric_name:
                maximize = getattr(m, "higher_is_better", True)
                break

        df = exp_res.to_dataframe()
        run_levels = {int(r.run_id): dict(r.factor_levels) for r in exp_res.design.runs}
        screen_scores = {
            int(rid): grp.sort_values("sample_id")[metric_name].astype(float).tolist()
            for rid, grp in df.groupby("run_id")
        }
        cands = selector.generate_candidates_from_screen(
            factor_ids=self.factors.ids,
            run_factor_levels=run_levels,
            screen_scores_by_run=screen_scores,
            maximize=maximize,
        )
        return selector.select_on_validation(
            candidates=cands,
            validation_scores_by_env=validation_scores_by_env,
            group_ids=group_ids,
            maximize=maximize,
            confirmation_scores=confirmation_scores,
            confirmation_envelope=confirmation_envelope,
            stability_certificate=stability_certificate,
            request_envelope=request_envelope,
        )
