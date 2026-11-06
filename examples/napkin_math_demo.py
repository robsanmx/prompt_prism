"""
Napkin math demo — reproduces every number in the Medium article
"The Math Behind 16-Run Prompt Experiments, Derived on a Napkin".

A synthetic ground truth is baked into fake scores over the real 2^(5-1)_V
design, then recovered with prompt_prism's ANOVA engine, with and without
RCBD blocking on sample_id.

Run it:

    python examples/napkin_math_demo.py
"""

import numpy as np
import pandas as pd

from prompt_prism.analysis.anova import ANOVAEngine
from prompt_prism.design.aliasing import AliasStructure
from prompt_prism.design.generators import FractionalFactorialGenerator

SEED = 13          # the seed used in the article
N_ITEMS = 10        # benchmark items -> RCBD blocks
INTERCEPT = 0.55
ITEM_DIFFICULTY_SD = 0.12
NOISE_SD = 0.06

# Ground-truth main effects (factor letter -> true beta)
BETA = {"A": 0.12, "B": 0.20, "C": 0.00, "D": 0.15, "E": -0.10}
# Ground-truth B x D interaction
BD_INTERACTION = 0.08

FACTOR_NAMES = {
    "A": "persona",
    "B": "few_shot",
    "C": "cot",
    "D": "strict_json",
    "E": "guardrail",
}


def build_dataset() -> pd.DataFrame:
    """Bake the ground truth into synthetic scores: 16 runs x 10 items = 160 obs."""
    rng = np.random.default_rng(SEED)
    design = FractionalFactorialGenerator.from_plan_id("2(5-1)V")
    item_difficulty = rng.normal(0.0, ITEM_DIFFICULTY_SD, N_ITEMS)

    rows = []
    for item_id in range(N_ITEMS):
        for run in design.runs:
            levels = run.factor_levels
            y = INTERCEPT + item_difficulty[item_id] + NOISE_SD * rng.normal()
            for factor, beta in BETA.items():
                y += beta * levels[factor]
            y += BD_INTERACTION * levels["B"] * levels["D"]
            rows.append({"sample_id": item_id, **levels, "f1": round(y, 4)})
    return pd.DataFrame(rows)


def main() -> None:
    df = build_dataset()
    alias = AliasStructure(["E=ABCD"])
    print(f"observations: {len(df)}   design resolution: {alias.resolution}")

    for label, block_col in [
        ("WITH RCBD blocking on sample_id (the library default)", "sample_id"),
        ("WITHOUT blocking (you would have to opt in to this)", None),
    ]:
        result = ANOVAEngine.run_anova(
            data=df,
            factor_cols=list("ABCDE"),
            target_col="f1",
            block_col=block_col,
            factor_name_map=FACTOR_NAMES,
            alias_structure=alias,
            include_interactions=True,
            ss_type=2,
        )
        print("\n" + "=" * 76)
        print(label)
        print("=" * 76)
        print(result.to_dataframe().to_string(index=False))
        print(f"residual std error: {result.residual_std_error:.4f}")
        print(f"residual df: {result.residual_df:.0f}")
        print(f"R^2: {result.r_squared:.4f}  adj R^2: {result.r_squared_adj:.4f}")
        print("\nMain effects (from the fitted model):")
        for e in result.main_effects:
            print(
                f"  {e.factor_id} ({e.factor_name:12s})  delta={e.effect_delta:+.4f}"
                f"  se={e.std_error:.4f}  t={e.t_statistic:+.2f}  p={e.p_value:.4f}"
                f"  cohens_d={e.cohens_d:+.2f}  -> {e.action_recommendation}"
            )

    # Napkin checks quoted in the article:
    print("\n" + "=" * 76)
    print("Napkin arithmetic: marginal contrasts and hand-computed SS")
    print("=" * 76)
    n = len(df)
    for factor in "ABCDE":
        delta = df[df[factor] == 1]["f1"].mean() - df[df[factor] == 0]["f1"].mean()
        ss = n * delta * delta / 4
        print(f"  {factor}: marginal delta = {delta:+.4f}   SS = {n}*d^2/4 = {ss:.4f}")


if __name__ == "__main__":
    main()
