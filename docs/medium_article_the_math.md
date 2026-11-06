# The Math Behind 16-Run Prompt Experiments, Derived on a Napkin

### XOR, orthogonality, aliasing, F-tests, and the free lunch hiding inside your benchmark dataset.

---

![PromptPrism banner](https://raw.githubusercontent.com/robsanmx/prompt_prism/main/main_effects_plot.png)

A few weeks ago I published an article claiming you can test ten prompt factors in 16 LLM runs instead of 1,024, and know which lines of your prompt actually earn their tokens. The polite version of the most common comment was "how?"

So this is the how. All of it. I'm going to write down every assumption, build the design by hand, run the analysis on data where I already know the answers, and check the arithmetic as we go. No steps skipped, no "it can be shown that." If you have a pencil, you can verify every number in this article.

I'll use five factors throughout because ten factors makes every table twice as wide and your patience half as long. The math doesn't care.

## First, write down what you believe

Every experiment starts as a claim about the world. Here is ours, in one line. When you run prompt configuration $c$ on benchmark item $i$, the score you get is:

```
y(c, i) = μ + b_i + β_A·A + β_B·B + β_C·C + β_D·D + β_E·E
          + β_AB·(A·B) + β_BD·(B·D) + ... + ε
```

In words: the score is a baseline (`μ`), plus a per-item offset (`b_i` — some test questions are just hard), plus one contribution per prompt factor you switched on, plus some pairwise interaction terms, plus noise.

Two things are quietly hiding in that equation, and the entire method stands on both of them.

**The factors combine additively.** A persona's effect doesn't depend on which few-shot examples are present. Mostly. The `β_AB` terms exist precisely to let the data tell us when that's false, but we expect most of them to be zero.

**High-order interactions are a fantasy.** There is no meaningful four-way interaction between "persona," "chain-of-thought," "temperature," and "output format." Nobody has ever had a hypothesis about one. The model above doesn't even include terms like `β_ABCDE`; for prompt engineering, believing they're zero isn't optimism, it's just paying attention to what a prompt is.

If you accept those two assumptions, the full factorial is a bad deal, and I can prove it with one paragraph of counting.

## The full factorial mostly measures things nobody believes in

A full factorial over $k$ binary factors runs $2^k$ combinations. Ten factors, 1,024 runs. But what does that *buy* you statistically?

A $2^{10}$ design lets you estimate $2^{10} - 1 = 1{,}023$ effects: 10 main effects, 45 two-way interactions, 120 three-way, 210 four-way, and so on up to one heroic ten-way interaction. Now count what you actually care about: 10 main effects plus 45 two-way interactions is 55 numbers, plus an intercept, 56 parameters. A full factorial spends roughly 95% of its estimation power on third-order and higher interactions — terms that, for prompts, exist only to be zero.

We don't need 1,024 runs. We need enough runs to estimate the 56 numbers we believe in, arranged so that each column of the design tells us about one factor and not a blender of factors. That's the whole trick, and it has a precise construction.

## The design: 16 runs, five factors, one XOR

Take four factors — A through D — and write out their full factorial: all $2^4 = 16$ combinations of on/off. That part is trivial; here it is as 0/1 columns, where 0 means "factor absent" and 1 means "factor present":

```
run | A B C D
  1 | 0 0 0 0
  2 | 0 0 0 1
  3 | 0 0 1 0
  4 | 0 0 1 1
  5 | 0 1 0 0
  6 | 0 1 0 1
  7 | 0 1 1 0
  8 | 0 1 1 1
  9 | 1 0 0 0
 10 | 1 0 0 1
 11 | 1 0 1 0
 12 | 1 0 1 1
 13 | 1 1 0 0
 14 | 1 1 0 1
 15 | 1 1 1 0
 16 | 1 1 1 1
```

Now the fifth factor, the generator trick. Instead of giving E its own independent on/off schedule, we *define* E as the XOR of the other four columns:

```
E = A ⊕ B ⊕ C ⊕ D
```

E is on exactly when an odd number of A, B, C, D are on. Compute it down the table and you get the full 16-run, five-factor design:

```
run | A B C D E
  1 | 0 0 0 0 0
  2 | 0 0 0 1 1
  3 | 0 0 1 0 1
  4 | 0 0 1 1 0
  5 | 0 1 0 0 1
  6 | 0 1 0 1 0
  7 | 0 1 1 0 0
  8 | 0 1 1 1 1
  9 | 1 0 0 0 1
 10 | 1 0 0 1 0
 11 | 1 0 1 0 0
 12 | 1 0 1 1 1
 13 | 1 1 0 0 0
 14 | 1 1 0 1 1
 15 | 1 1 1 0 1
 16 | 1 1 1 1 0
```

Sixteen runs. Five factors. Every factor on in exactly 8 of them. This is the $2^{5-1}$ fractional factorial — a half-fraction of the 32-run full design — and it is the exact design my library's `2(5-1)V` plan generates, row for row.

It looks like cheating. E was *derived* from A, B, C, D — how can we possibly measure E's effect? Doesn't E's column just re-broadcast whatever the other columns say?

## The load-bearing fact: orthogonality

Here is the thing that makes fractional factorials work, and you can check it by eyeball on the table above.

Question: does knowing A tell you anything about E? Look at the 8 runs where A = 1 (runs 9–16). E is on in 4 of them. Now the 8 runs where A = 0: E is on in 4 of those too. Same for B, same for C, same for D. E is on half the time *within every slice* of any other factor. Knowing one column gives you zero information about another.

Statisticians call this orthogonality. Code the columns as −1/+1 instead of 0/1 and take dot products: every pair of distinct columns multiplies to zero across the 16 runs. I verified all 10 pairs while writing this; the invitation is for you to do it again. In matrix form, if X is the design matrix, then $X^T X = N \cdot I$: the columns are uncorrelated, each of length N.

Why does that matter? Because when the columns are orthogonal, fitting the regression model decomposes cleanly. The estimate of $\beta_A$ is just the A contrast — the difference between your average score with A on and with A off — and *that contrast is contaminated by nobody*. If E is secretly a wonderful instruction, it raises the A-on and A-off groups equally, so the subtraction cancels it out. Every other factor's contribution cancels the same way. One subtraction per factor, five subtractions total, and each one is a clean measurement of one $\beta$.

That is the entire miracle. No correlations to untangle, no multicollinearity, no "we adjusted for..." hand-waving. The design guarantees it, structurally, before you've spent a single API call.

And it goes one level deeper. Build a product column by multiplying two ±1 columns entrywise — say $A \cdot B$ — and check the ten possible product columns against the five main columns: still orthogonal, all fifty checks pass. So the pairwise interaction terms can also be estimated cleanly from the same 16 runs. This particular design is Resolution V, which is the name for that property, and I'll explain the name in a moment.

## Nothing is free: the alias structure

We did not get all this for nothing. The price is recorded in one short equation called the defining relation. Since $E = A \oplus B \oplus C \oplus D$, in the algebra of these columns (where multiplying a column by itself gives the identity, $A \cdot A = I$), we can write:

```
I = ABCDE
```

This is not decoration; it's a billing contract. Multiply both sides by A, and remember $A \cdot A = I$:

```
A = BCDE
```

Read that as: the A column and the four-way interaction column BCDE are *the same column* in this design (up to an overall sign, which doesn't change what's confounded with what). The estimate I read out as "$\beta_A$" is actually $\beta_A + \beta_{BCDE}$, always, forever, in this design. When I compute "persona is worth +0.09," the honest statement is "persona is worth +0.09, *or* the four-way interaction of few-shot × CoT × JSON × guardrail is worth +0.09, or some blend."

That's why the additive-beliefs section at the top wasn't optional. The design's trade is: *you give up the ability to see the terms you declared fiction, and in exchange you keep only the fraction of runs you need* — half, in the design above; a sixty-fourth, if you're screening ten factors in 16 runs. If you were wrong about the fiction, the data can lie to you. The design is a machine for converting assumptions into budget.

**Resolution** is just the length of the shortest word in the defining relation, and it tells you exactly which lies are possible:

- **Resolution III** — shortest word has 3 letters. Some main effect is aliased with a two-way interaction. Cheap screening; treat results as leads, not verdicts.
- **Resolution IV** — main effects are clear of two-way interactions, but two-ways are aliased with each other.
- **Resolution V and up** — main effects and two-way interactions are all unaliased. What I built above. Safe for production decisions.

Our design's defining relation has exactly one word, `ABCDE`, of length five. Resolution V. The only thing $\beta_A$ could be confused with is a four-way interaction we already declared fictional.

Now the cautionary tale that makes resolution real. My library also ships a 16-run design for *ten* factors, $2^{10-6}_{III}$, built with six generators. Its defining relation has $2^6 - 1 = 63$ words, the shortest being three letters, like `ABK`. That K deserves a look, because the letter placement is not free: in that design, the effect measured for the factor in slot K is really

```
β_K + β_AB + β_CE + β_DH + β_FG + (higher-order junk)
```

If you assigned "temperature" to slot K, and your persona (A) and few-shot (B) factors have a strong interaction, its effect arrives on temperature's bill. You would read the table and conclude "temperature matters a lot" and ship a hyperparameter change that was actually two other factors holding hands. Resolution III designs are for ranking candidates cheaply; anything they flag needs a confirmation run in a cleaner design before it goes to production. This is the single most important discipline in the whole method, and it's why the tooling computes and prints the alias map for every design before you run it.

## From columns to numbers: effects, sums of squares, and the F-test

So far this is all structure. Now the measurement machinery — three formulas, and then we'll check them against real output.

**Effect.** For factor B, the contrast is the difference of means:

```
Δ_B = mean(score | B on) − mean(score | B off)
```

**Sum of squares.** For a balanced two-level factor over N total observations, the variance attributable to B collapses to a beautiful closed form:

```
SS_B = N · Δ_B² / 4
```

(The 4 is not mysterious. In ±1 coding the regression coefficient is half the contrast, $\beta_B = \Delta_B / 2$, and for a balanced factor $SS = N \beta^2$ — so the 2 is that halving, squared.)

**F-test.** Divide each factor's sum of squares by its degrees of freedom (1, for a two-level factor) to get a mean square, then compare it to the residual mean square — the variance the model couldn't explain:

```
F_B = MS_B / MS_residual = (SS_B / 1) / (SS_residual / df_residual)
```

The F-statistic asks one question: *how many error-sized chunks of variance does this factor explain?* If the answer is "about 1," the factor is noise. If the answer is "120," the factor is the show. The p-value is just the tail area of the F distribution past your statistic — the probability of seeing an F that big if the factor truly did nothing.

Every ANOVA table ever printed is those three formulas repeated down a page. There's nothing else in it.

## The experiment: I baked the answers in

To show the machinery works, I ran the one experiment where the truth is known: a fake one. I kept the real $2^{5-1}_V$ design matrix from above, ten fake benchmark items, and generated each score directly from a ground-truth equation, so I know exactly what the math is supposed to recover:

```
score = 0.55
      + item difficulty          ~ Normal(0, 0.12)   per item, same across runs
      + 0.12 × persona (A)       + 0.20 × few_shot (B)
      + 0.00 × cot (C)           + 0.15 × strict_json (D)
      − 0.10 × guardrail (E)
      + 0.08 × few_shot × strict_json (B·D interaction)
      + noise                    ~ Normal(0, 0.06)   per observation
```

Persona helps a little, few-shot helps a lot, CoT does nothing, strict JSON helps a lot, the guardrail *hurts*, and few-shot and strict JSON amplify each other. Sixteen runs times ten items is 160 observations, and then I handed the raw scores to the library's ANOVA engine with no hints. Here's the trimmed output, with the noise-free napkin check alongside:

```
Source        SS       df   MS        F        p
persona      0.5033    1   0.5033   126.77   3.8e-21
few_shot     2.0700    1   2.0700   521.39   3.3e-48
cot          0.0001    1   0.0001     0.02     0.88
strict_json  1.4703    1   1.4703   370.33   1.6e-40
guardrail    0.4764    1   0.4764   119.98   2.3e-20
few_shot:strict_json  0.0889  1  0.0889  22.40  5.5e-06
Residual     0.5360  135   0.003970
```

First, the napkin check. The marginal contrast for few_shot across the 160 observations came out to Δ_B = +0.2275. So:

```
SS_B = 160 × 0.2275² / 4 = 2.0700   ✓ (table says 2.0700)
F_B  = 2.0700 / 0.003970 = 521.4    ✓ (table says 521.39)
```

The F for persona, by hand: $0.5033 / 0.003970 = 126.8$. Table says 126.77. The whole table is three formulas and a design matrix; you could rebuild it in a spreadsheet.

Second, the recovery. Persona, few_shot, and strict_json: flagged as significant positive effects. Guardrail: significant *negative*. CoT: p = 0.88, correctly acquitted. And the B×D interaction — the one term I buried in the middle of the equation — came out with F = 22.4, p = 5.5×10⁻⁶. The model-level readout, with standard errors and effect sizes, says the same thing:

```
Factor        Δ (coef)   SE      t       p        Cohen's d   Verdict
few_shot      +0.1840   0.0223  +8.26   <0.001     +0.75      ENABLE
strict_json   +0.1477   0.0223  +6.63   <0.001     +0.60      ENABLE
guardrail     −0.1059   0.0223  −4.75   <0.001     −0.43      DISABLE
persona       +0.0855   0.0223  +3.84   0.0002     +0.35      ENABLE
cot           +0.0186   0.0223  +0.83    0.406     +0.08      OMIT (neutral)
```

Every estimate landed within about a standard error and a half of the value I baked in (persona at +0.086 vs. a true +0.12 is about 1.5 SE — that's what standard errors mean; one dataset, one draw). One subtlety worth flagging honestly: because this model carries the B×D interaction term, the *coefficient* for few_shot (+0.184) is few_shot's effect with strict_json held at its baseline, while the marginal contrast (+0.228) includes the interaction's average contribution. Two slices of the same truth; with no interactions present they'd coincide. The ANOVA F-tests the marginal slice, the coefficient table tests the conditional slice. If that sentence takes a second read, that's normal — it took me a week of staring at statsmodels output to stop being confused by it, and it's the kind of thing the library's report renders explicitly so you don't have to.

Also notice: every factor has the same standard error, 0.0223. Not luck. In a balanced orthogonal design, every factor is measured with identical precision. No factor is privileged; none gets a bigger slice of the noise.

## The free lunch: your benchmark items are blocks

Now the part I care about most, because it's where most people quietly lose half their statistical power for no reason.

Look back at the ground truth: item difficulty had a standard deviation of 0.12, and in my draw the easiest and hardest items differed by 0.59 — *ten times* the per-observation noise (0.06). Some questions are just hard. If you don't tell the analysis about that, all of that item-to-item variance lands in the residual, and your residual mean square — the denominator of every F-test — balloons.

Here is the fix, and it costs zero extra API calls. Every one of the 16 runs was executed on every item, so each item is a *complete block*: a miniaturized copy of the whole experiment run on one piece of terrain. Add one term to the regression:

```
score ~ C(block) + A + B + C + D + E + all pairwise interactions
```

`C(block)` is statsmodels for "treat block as a categorical variable" — one offset column per benchmark item. The term absorbs the per-item difficulty, and that variance transfers out of the residual. (Implementation aside you'll appreciate if you ever build this: a prompt factor literally named "C" collides with R's categorical syntax, so the library silently renames factor columns to safe identifiers before fitting. I lost an afternoon to that one.) This is the Randomized Complete Block Design, Fisher's workhorse from actual agriculture, where the blocks were fields and the "items" were plots of soil. The df bookkeeping: ten items spend 9 degrees of freedom on block offsets (160 observations − 25 parameters = 135 residual df), and in exchange the residual variance collapses.

How much did it matter in our experiment? Run the *identical 160 scores* through the same engine, once with the block term and once without:

```
With blocks:    residual SE = 0.063    R² = 0.94
Without blocks: residual SE = 0.186    R² = 0.48
```

The unblocked analysis is *three times blinder*. Same data, same effects, same design. And watch what that does to the verdicts — same 160 observations, model coefficients:

```
Factor         With blocks              Without blocks
guardrail      t = −4.75, p < 0.001     t = −1.61, p = 0.110
               DISABLE ✓               "neutral" ✗ (harmful, kept!)
persona        t = +3.84, p = 0.0002    t = +1.30, p = 0.196
               ENABLE ✓                "neutral" ✗ (dropped!)
few_shot × strict_json interaction:
               p = 5.5e-06 ✓            p = 0.111 ✗ (invisible!)
```

The unblocked analysis keeps the two loudest signals and loses everything else. It would ship the guardrail that actively costs 10 points of F1, discard the persona that earns 12, and never notice the few-shot × JSON amplification. And it would report all of this with a straight face, because every number it printed was "correct" given the model it was allowed to use. The failure isn't an arithmetic error; it's a specification error — variance that belonged to items got billed to noise.

This is why the library blocks on `sample_id` by default — running the blinder analysis is a decision you'd have to make on purpose. It's also why "just use more test items" is the wrong instinct: more unblocked items make the residual *estimate* more precise while the residual itself stays huge — you get confidently wrong conclusions faster.

## How big is big? Effect sizes and p-values

A p-value tells you whether an effect is distinguishable from zero. It does not tell you whether you should care. Give a tiny +0.02 effect enough observations and it becomes "significant"; starve a real +0.20 effect of observations and it reads as noise. Neither result is a shipping decision.

So the report carries effect sizes next to every p-value. Three of them, each answering a different question:

```
Partial η²  = SS_factor / (SS_factor + SS_residual)
ω² (omega)  = (SS_factor − df·MS_residual) / (SS_total + MS_residual)
Cohen's d   = Δ / overall SD of the score
```

Partial η² is "what fraction of the leftover variance does this factor explain" — few_shot clocked in at 0.79, meaning it accounts for the overwhelming majority of explainable variation. Omega-squared is the same idea with a bias correction that shrinks toward zero when effects are small (it's the honest one; η² is optimistic in small samples, and the block term's ω² of 0.46 in our run quietly quantifies the "blocks matter" section: item difficulty is 46% of the total story). Cohen's d puts the effect in units of the score's own spread — guardrail's −0.43 says "removing it moves the average score by nearly half a standard deviation," which is a sentence a product manager can act on.

The decision rule the library applies is deliberately boring: significant and positive → enable; significant and negative → remove; not significant → drop anyway, because a factor that can't prove it pays for itself is still charging you tokens and latency. Note the asymmetry — the burden of proof is on the factor, not on your skepticism. That's the prompt-bloat prevention, as a statistical rule.

## The correction you didn't know you needed

One more piece of honesty in the machinery. That ANOVA table ran 15 hypothesis tests (5 factors + 10 interactions). At α = 0.05, running 15 independent tests on pure noise produces *at least one* "significant" finding with probability $1 - 0.95^{15} \approx 54\%$. If you test enough factors, you will always find winners. That's the multiple comparisons problem, and it's how cargo-cult prompt tuning stays alive.

The report therefore carries corrected p-values in two flavors. Bonferroni is the blunt one: multiply each p by the number of tests ($p_{Bonf} = \min(1, m \cdot p)$), controlling the chance of even one false positive. It's conservative — in an earlier run of this same simulation with a different random seed, one of the ten truly-zero interactions popped out with a raw p of 0.008; Bonferroni corrected it to 0.12, correctly calling it noise. Benjamini–Hochberg FDR is the surgical one: it controls not the chance of any false positive but the *expected proportion* of false positives among your flagged results, by comparing the sorted p-values to their rank-adjusted thresholds. In our main run, the few_shot × strict_json interaction went from p = 5.5×10⁻⁶ to 8.3×10⁻⁵ under Bonferroni and stayed decisive — the finding survives the corrections, which is exactly what you want a shippable finding to do.

A detail that matters if you ever implement this: the block term is excluded from the correction. It's not a finding; it's a control. Correcting for it would waste your error budget on a nuisance variable you never intended to interpret.

## Plackett–Burman, for when even 16 runs is too many

One last design, briefly, because it's the extreme end of the budget spectrum. Plackett–Burman designs screen up to N−1 factors in N runs for N ∈ {8, 12, 16, 20, 24}: eleven factors in twelve runs. The construction is almost comically simple: take one row of +1/−1 values chosen so that all its cyclic rotations are mutually orthogonal, write out the N−1 rotations, and add one all-−1 row. That's it — the resulting matrix has orthogonal columns (a Hadamard matrix), so every main effect is still a clean contrast.

The catch is the same as before, but sharper: at 12 runs you cannot even estimate the interactions, and the alias structure is tangled (each main effect is partially aliased with many interactions rather than cleanly swapped with one). It's a ranking instrument. It tells you which factors deserve a real experiment. It is never the last experiment.

## Where the math stops being your friend

I've been selling this method hard, so let me itemize its failure modes honestly.

**The sparsity assumption can be wrong.** If your prompt's factors interact promiscuously, a Resolution III screen will happily deliver confident nonsense. The defense is procedural: screen at low resolution, confirm at Resolution V, ship only what survives both.

**Effects are population averages.** Every β here is an average over your benchmark dataset. Average over a different distribution of real traffic — or a different model version, since providers silently update weights — and the betas move. DoE output is a snapshot with an expiry date, not a law of nature.

**Binary factors only.** The $2^{k-p}$ machinery is for on/off factors. "Three examples vs five" is a two-level factor; "how many examples, 1 to 10" is a different design family (and if the response is nonlinear in that dial, main effects will average over curvature). The 0.1-vs-0.7 temperature debate fits; a fine-grained sweep doesn't.

**Noise from the judge.** If your metric is an LLM-as-a-judge score rather than an exact match, the ε in the model has two sources, and the judge's inconsistency lands in the residual too, inflating every p-value. The fix is boring and effective: more items, and replicate the judge or cache it so replication is free.

And the deepest one: **the statistics decide what's real, not what matters.** A factor can be truly, significantly positive and still not worth shipping because +1.2% doesn't justify 400 extra tokens per request. The ANOVA table is an input to a decision, not a decision.

## The napkin, summarized

The whole method in one breath: assume your prompt is mostly additive; build a matrix whose columns are orthogonal by defining some factors as XORs of others; accept that the defining relation bills each effect to its aliases, and choose a resolution that keeps the billing away from anything you'd act on; run each design row on every benchmark item so items become blocks; read each factor off as a difference of means; convert contrasts to sums of squares; divide by residual mean squares to get F; correct the p-values for the number of questions you asked; and let effect sizes, not p-values, argue for shipping.

None of this is new. XOR generators are Box and Hunter, 1961. Blocks are Fisher, sugar beet fields, the 1920s. F-tests are Fisher again. The alias algebra is in every chapter 8 of every design-of-experiments textbook. The only new thing is the observation that an LLM prompt is a multi-factor process with an API bill attached — which makes it, rather delightfully, a classic industrial process optimization problem that just happens to be made of words.

The library that does all of this, with the alias maps printed before you spend money and blocking on by default, is [prompt_prism on GitHub](https://github.com/robsanmx/prompt_prism) (`pip install prompt-prism`). The friendlier, less equation-heavy tour of the workflow is the first piece in this series, *Stop Guessing Your Prompts*. And if you made it this far with a pencil and found an arithmetic slip anywhere, please tell me in the comments — that would be the most statistically rigorous thing that happened to me all week.
