"""
Live Stability-Upgrade Pilot for PromptPrism (No DSPy):
Tests whether PromptPrism-native upgrades:
  1. Multi-turn chat demo turns + explicit XML field framing (`<allowed_intents>`, `<query>`)
  2. Error-mined in-domain candidate demos from `screen` (zero confirmation leakage)
  3. 16-run 2(6-2)IV orthogonal screening + TOST dead-demo pruning + stability-aware selection

lift PromptPrism on `banking77_intent` (`gemini-2.5-flash`, temperature=0.0, thinkingBudget=0)
from Real-v4's `95.42% / U95_sim=0.0542 (FAIL)` to `PASS` stability (`U95_sim < 0.02`)
without DSPy, under BOTH inner-field perturbations (matching DSPy eval) and harsh
outer-shell perturbations (matching Real-v4).
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from prompt_prism.analysis.diagnostics import run_calibrated_attribution_audit
from prompt_prism.analysis.selection import (
    evaluate_simultaneous_perturbation_certificate,
)
from prompt_prism.core.models import OperatingEnvelopeSpec
from prompt_prism.design.generators import FractionalFactorialGenerator
from research.experiments.dspy_benchmark import (
    PAIRED_ENV_IDS,
    PERTURBATION_ONLY_ENV_IDS,
    GeminiMultiTurnClient,
    load_real_v4_splits,
    paired_item_comparison,
)
from research.real_v4.perturbations import (
    parser_hashes_by_task,
    score_task_response,
)

ROOT = Path(__file__).resolve().parents[2]


def _format_user_input(
    item: Mapping[str, Any],
    *,
    xml_framing: bool,
    env_id: str = "ref",
    outer_shell_perturb: bool = False,
) -> str:
    opts = ", ".join(str(o) for o in item["options"])
    query = str(item["input_text"])

    if not outer_shell_perturb:
        # Match DSPy benchmark inner-field perturbation
        if env_id == "fmt_whitespace":
            query_fmt = f"\n\n{query}\n\n"
        elif env_id == "fmt_delimiter":
            query_fmt = f"=== QUERY ===\n{query}\n=== END QUERY ==="
        elif env_id == "word_paraphrase_1":
            query_fmt = f"Carefully review the provided input below and supply your response.\n{query}"
        elif env_id == "word_paraphrase_2":
            query_fmt = f"Examine the input material below and provide the requested response.\n{query}"
        else:
            query_fmt = query

        if xml_framing:
            return (
                f"<allowed_intents>{opts}</allowed_intents>\n"
                f"<query>{query_fmt}</query>"
            )
        return f"Allowed Intents: {opts}\nQuery: {query_fmt}"

    # Harsh outer-shell perturbation (alters outer structural tags / preamble / whitespace)
    preamble = {
        "ref": "Classify the customer banking query into one allowed intent.",
        "repeat_r2": "Classify the customer banking query into one allowed intent.",
        "fmt_whitespace": "Classify the customer banking query into one allowed intent.",
        "fmt_delimiter": "Classify the customer banking query into one allowed intent.",
        "word_paraphrase_1": "Carefully review the provided input below and supply your response.",
        "word_paraphrase_2": "Examine the input material below and provide the requested response.",
    }[env_id]
    sep = "\n\n\n" if env_id == "fmt_whitespace" else "\n"
    if env_id == "fmt_delimiter":
        if xml_framing:
            return (
                f"=== TASK ===\n{preamble}\n"
                f"<allowed_intents>{opts}</allowed_intents>\n"
                f"<query>{query}</query>"
            )
        return f"=== TASK ===\n{preamble}\n=== ALLOWED INTENTS ===\n{opts}\n=== QUERY ===\n{query}"

    if xml_framing:
        return (
            f"{preamble}{sep}"
            f"<allowed_intents>{opts}</allowed_intents>{sep}"
            f"<query>{query}</query>"
        )
    return f"{preamble}{sep}Allowed Intents: {opts}{sep}Query: {query}"


def build_messages(
    *,
    item: Mapping[str, Any],
    levels: Mapping[str, int],
    demos: Sequence[Mapping[str, Any]],
    env_id: str = "ref",
    outer_shell_perturb: bool = False,
) -> List[Dict[str, str]]:
    """Render PromptPrism structured multi-turn or flat chat prompt."""
    xml_multi = int(levels["xml_multiturn"]) == 1
    strict_contract = int(levels["strict_contract"]) == 1

    sys_parts = [
        "You are a precision banking intent classification specialist. Follow all task rules accurately.",
        (
            "STRICT CONTRACT: Respond ONLY with `FINAL_ANSWER: <exact_intent_label>` using exactly one label from the allowed intents. No commentary."
            if strict_contract
            else "Return the matching intent label from the allowed list without extra commentary."
        ),
    ]
    system_text = "\n\n".join(sys_parts)

    active_demos = [
        demos[i] for i in range(len(demos)) if int(levels[f"demo_{i + 1}"]) == 1
    ]

    if xml_multi:
        msgs: List[Dict[str, str]] = [{"role": "system", "content": system_text}]
        for d in active_demos:
            d_opts = ", ".join(str(o) for o in d["options"])
            msgs.append(
                {
                    "role": "user",
                    "content": (
                        f"<allowed_intents>{d_opts}</allowed_intents>\n"
                        f"<query>{d['input_text']}</query>"
                    ),
                }
            )
            msgs.append(
                {
                    "role": "assistant",
                    "content": f"FINAL_ANSWER: {d['target']}",
                }
            )
        msgs.append(
            {
                "role": "user",
                "content": _format_user_input(
                    item,
                    xml_framing=True,
                    env_id=env_id,
                    outer_shell_perturb=outer_shell_perturb,
                ),
            }
        )
        return msgs

    # Flat single-user prompt (Real-v4 style)
    blocks: List[str] = []
    if active_demos:
        demo_lines = [
            f"Example {i + 1}:\nQuery: {d['input_text']}\nOutput: FINAL_ANSWER: {d['target']}"
            for i, d in enumerate(active_demos)
        ]
        hdr = (
            "=== EXAMPLES ==="
            if (outer_shell_perturb and env_id == "fmt_delimiter")
            else "[EXAMPLES]"
        )
        blocks.append(f"{hdr}\n" + "\n".join(demo_lines))
    blocks.append(
        _format_user_input(
            item,
            xml_framing=False,
            env_id=env_id,
            outer_shell_perturb=outer_shell_perturb,
        )
    )
    sep = "\n\n\n" if (outer_shell_perturb and env_id == "fmt_whitespace") else "\n\n"
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": sep.join(blocks)},
    ]


def mine_candidate_demos_from_screen(
    screen_items: Sequence[Mapping[str, Any]],
    client: GeminiMultiTurnClient,
    n_demos: int = 4,
    max_workers: int = 10,
) -> List[Dict[str, Any]]:
    """Zero-shot error-mine `n_demos` distinct-intent candidates strictly from `screen` (no leakage)."""
    base_levels = {
        "xml_multiturn": 1,
        "strict_contract": 1,
        "demo_1": 0,
        "demo_2": 0,
        "demo_3": 0,
        "demo_4": 0,
    }
    scores = [0.0] * len(screen_items)

    def _probe(idx: int) -> Tuple[int, float]:
        it = screen_items[idx]
        msgs = build_messages(item=it, levels=base_levels, demos=[], env_id="ref")
        r = client.generate(msgs)
        sc = (
            0.0
            if r.error
            else float(score_task_response("banking77_intent", r.content, it))
        )
        return idx, sc

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for fut in as_completed(
            [pool.submit(_probe, i) for i in range(len(screen_items))]
        ):
            idx, sc = fut.result()
            scores[idx] = sc

    # Pick distinct-intent errors first, then distinct-intent items if fewer than n_demos errors
    picked: List[Dict[str, Any]] = []
    seen_targets = set()
    for idx, sc in enumerate(scores):
        if sc < 1.0:
            tgt = str(screen_items[idx]["target"])
            if tgt not in seen_targets:
                picked.append(dict(screen_items[idx]))
                seen_targets.add(tgt)
                if len(picked) >= n_demos:
                    break
    if len(picked) < n_demos:
        for it in screen_items:
            tgt = str(it["target"])
            if tgt not in seen_targets:
                picked.append(dict(it))
                seen_targets.add(tgt)
                if len(picked) >= n_demos:
                    break
    print(
        f"[banking77_intent] Zero-shot screen acc={np.mean(scores):.4f}; "
        f"mined {len(picked)} distinct-intent candidate demos: "
        f"{[d['target'] for d in picked]}",
        flush=True,
    )
    return picked


def eval_config_on_envs(
    *,
    levels: Mapping[str, int],
    demos: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
    env_ids: Sequence[str],
    outer_shell_perturb: bool,
    client: GeminiMultiTurnClient,
    max_workers: int = 10,
) -> Dict[str, Any]:
    scores_by_env: Dict[str, List[float]] = {e: [0.0] * len(items) for e in env_ids}
    p_toks: List[int] = []
    c_toks: List[int] = []
    lock = threading.Lock()
    jobs = [(e, i, it) for e in env_ids for i, it in enumerate(items)]

    def _run(job):
        e, i, it = job
        msgs = build_messages(
            item=it,
            levels=levels,
            demos=demos,
            env_id=e,
            outer_shell_perturb=outer_shell_perturb,
        )
        r = client.generate(msgs)
        sc = (
            0.0
            if r.error
            else float(score_task_response("banking77_intent", r.content, it))
        )
        u = r.token_usage or {}
        return (
            e,
            i,
            sc,
            int(u.get("prompt_tokens") or 0),
            int(u.get("completion_tokens") or 0),
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for fut in as_completed([pool.submit(_run, j) for j in jobs]):
            e, i, sc, pt, ct = fut.result()
            with lock:
                scores_by_env[e][i] = sc
                if e == "ref":
                    p_toks.append(pt)
                    c_toks.append(ct)

    p_hashes = parser_hashes_by_task()
    env_spec = OperatingEnvelopeSpec(
        task_id=f"banking77_intent:prism_stabilized:{'outer' if outer_shell_perturb else 'inner'}",
        model_id="gemini-2.5-flash",
        model_version="gemini-2.5-flash",
        dataset_partition_hash="banking77_intent_clean_confirmation_240",
        parser_hash=p_hashes["banking77_intent"],
        decoding_config={
            "temperature": 0.0,
            "top_p": 1.0,
            "max_output_tokens": 512,
            "thinking_budget": 0,
        },
        perturbation_env_ids=list(PERTURBATION_ONLY_ENV_IDS),
    )
    cert = evaluate_simultaneous_perturbation_certificate(
        envelope=env_spec,
        reference_scores=scores_by_env["ref"],
        perturbation_scores_by_env={
            e: scores_by_env[e] for e in PERTURBATION_ONLY_ENV_IDS
        },
        group_ids=[str(it["group_id"]) for it in items],
        epsilon_stability=0.02,
        n_bootstrap=2000,
        seed=42,
    )
    worst_env = min(
        PERTURBATION_ONLY_ENV_IDS, key=lambda e: float(np.mean(scores_by_env[e]))
    )
    return {
        "scores_by_env": scores_by_env,
        "env_means": {e: float(np.mean(v)) for e, v in scores_by_env.items()},
        "mean_prompt_tokens_ref": float(np.mean(p_toks)),
        "mean_completion_tokens_ref": float(np.mean(c_toks)),
        "cert": {
            "status": cert.status,
            "reference_mean": float(cert.reference_mean),
            "max_drop_est": float(cert.d_stability_estimate),
            "u95_sim": float(cert.d_stability_upper_95_sim),
            "worst_env_id": worst_env,
        },
    }


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument(
        "--out",
        type=str,
        default=str(
            ROOT
            / "research"
            / "results"
            / "dspy_benchmark"
            / "banking77_prism_stabilized"
        ),
    )
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = load_real_v4_splits("banking77_intent")
    screen_items = splits["screen"]
    val_items = splits["validation"]
    conf_items = splits["clean_confirmation"]

    client = GeminiMultiTurnClient(
        "gemini-2.5-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.0,
        max_output_tokens=512,
        thinking_budget=0,
    )

    # 1. Error-mine 4 distinct-intent candidate demos strictly from screen (120 calls)
    demos = mine_candidate_demos_from_screen(
        screen_items, client, n_demos=4, max_workers=args.max_workers
    )

    # 2. 16-run 2(6-2)IV screen across 6 factors on screen (ref) + fmt_delimiter on val (stability!)
    factor_names = [
        "xml_multiturn",
        "strict_contract",
        "demo_1",
        "demo_2",
        "demo_3",
        "demo_4",
    ]
    design = FractionalFactorialGenerator.from_plan_id("2(6-2)IV")
    fid_to_name = dict(zip(design.factor_ids, factor_names))

    print(
        "[Screen] Running 2(6-2)IV (16 runs x 120 screen ref + 60 val outer fmt_delimiter)...",
        flush=True,
    )
    screen_rows: List[Dict[str, Any]] = []
    val_stab_rows: List[Dict[str, Any]] = []
    lock = threading.Lock()

    jobs_screen = []
    jobs_val = []
    for run in design.runs:
        nl = {
            fid_to_name[fid]: int(run.factor_levels[fid]) for fid in design.factor_ids
        }
        for it in screen_items:
            jobs_screen.append((int(run.run_id), nl, it))
        for it in val_items:
            jobs_val.append((int(run.run_id), nl, it, "ref"))
            jobs_val.append((int(run.run_id), nl, it, "fmt_delimiter"))

    def _run_s(job):
        rid, nl, it = job
        msgs = build_messages(
            item=it, levels=nl, demos=demos, env_id="ref", outer_shell_perturb=True
        )
        r = client.generate(msgs)
        sc = (
            0.0
            if r.error
            else float(score_task_response("banking77_intent", r.content, it))
        )
        u = r.token_usage or {}
        rec = {
            "run_id": rid,
            "sample_id": str(it["sample_id"]),
            "score": sc,
            "prompt_tokens": int(u.get("prompt_tokens") or 0),
        }
        rec.update(nl)
        return rec

    def _run_v(job):
        rid, nl, it, env = job
        msgs = build_messages(
            item=it, levels=nl, demos=demos, env_id=env, outer_shell_perturb=True
        )
        r = client.generate(msgs)
        sc = (
            0.0
            if r.error
            else float(score_task_response("banking77_intent", r.content, it))
        )
        rec = {
            "run_id": rid,
            "sample_id": str(it["sample_id"]),
            "env_id": env,
            "score": sc,
        }
        rec.update(nl)
        return rec

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        for fut in as_completed([pool.submit(_run_s, j) for j in jobs_screen]):
            with lock:
                screen_rows.append(fut.result())
        for fut in as_completed([pool.submit(_run_v, j) for j in jobs_val]):
            with lock:
                val_stab_rows.append(fut.result())

    df_s = pd.DataFrame(screen_rows)
    token_costs = {
        fn: max(
            0,
            int(
                round(
                    df_s.loc[df_s[fn] == 1, "prompt_tokens"].mean()
                    - df_s.loc[df_s[fn] == 0, "prompt_tokens"].mean()
                )
            ),
        )
        for fn in factor_names
    }
    audit_clean = run_calibrated_attribution_audit(
        data=df_s,
        factor_cols=factor_names,
        target_col="score",
        sample_col="sample_id",
        alpha=0.05,
        maximize=True,
        equivalence_margin=0.02,
        token_costs=token_costs,
        n_sign_flip=2000,
        seed=42,
    )
    (out_dir / "screen_audit_clean.md").write_text(
        audit_clean.to_markdown() + "\n", encoding="utf-8"
    )

    # Compute stability drop contrast on val (drop = ref - fmt_delimiter, minimize drop!)
    df_v = pd.DataFrame(val_stab_rows)
    piv_v = df_v.pivot_table(
        index=["run_id", "sample_id"] + factor_names, columns="env_id", values="score"
    ).reset_index()
    piv_v["drop"] = piv_v["ref"] - piv_v["fmt_delimiter"]
    audit_drop = run_calibrated_attribution_audit(
        data=piv_v,
        factor_cols=factor_names,
        target_col="drop",
        sample_col="sample_id",
        alpha=0.05,
        maximize=False,  # lower drop = better stability!
        equivalence_margin=0.02,
        n_sign_flip=2000,
        seed=42,
    )
    (out_dir / "val_audit_stability_drop.md").write_text(
        audit_drop.to_markdown() + "\n", encoding="utf-8"
    )

    # Select recipe: start from all-ON, prune REMOVE_HARMFUL or SAFE_TO_PRUNE on clean (unless KEEP on stability!), and disable FRAGILE factors
    clean_v = {f.factor_id: f.pruning_verdict for f in audit_clean.factors}
    stab_v = {f.factor_id: f.pruning_verdict for f in audit_drop.factors}
    selected_levels = {}
    for fn in factor_names:
        if stab_v[fn] == "KEEP":
            selected_levels[fn] = 1
        elif clean_v[fn] == "REMOVE_HARMFUL" or stab_v[fn] == "REMOVE_HARMFUL":
            selected_levels[fn] = 0
        elif clean_v[fn] == "SAFE_TO_PRUNE":
            selected_levels[fn] = 0
        else:
            selected_levels[fn] = 1
    # Always keep xml_multiturn=1 if not harmful/fragile to test structured chat hypothesis cleanly
    print(
        f"[Selection] clean_verdicts={clean_v} stab_verdicts={stab_v} -> selected={selected_levels}",
        flush=True,
    )

    # 3. Validate selected PromptPrism-Native Stabilized prompt on untouched 240 confirmation x 6 envs
    #    under BOTH inner-field perturbation (same as DSPy) AND harsh outer-shell perturbation (same as Real-v4)
    print(
        "[Confirm] Evaluating selected prompt on 240 confirmation x 6 envs (inner + outer)...",
        flush=True,
    )
    eval_inner = eval_config_on_envs(
        levels=selected_levels,
        demos=demos,
        items=conf_items,
        env_ids=PAIRED_ENV_IDS,
        outer_shell_perturb=False,
        client=client,
        max_workers=args.max_workers,
    )
    eval_outer = eval_config_on_envs(
        levels=selected_levels,
        demos=demos,
        items=conf_items,
        env_ids=PAIRED_ENV_IDS,
        outer_shell_perturb=True,
        client=client,
        max_workers=args.max_workers,
    )

    # Compare directly against saved DSPy Full & Real-v4 PromptPrism confirmation scores
    prev = json.loads(
        (
            ROOT
            / "research"
            / "results"
            / "dspy_benchmark"
            / "banking77_intent"
            / "confirmation_scores.json"
        ).read_text()
    )
    gids = [str(it["group_id"]) for it in conf_items]
    vs_dspy = paired_item_comparison(
        eval_inner["scores_by_env"]["ref"],
        prev["dspy_scores_by_env"]["ref"],
        group_ids=gids,
    )
    vs_v4 = paired_item_comparison(
        eval_inner["scores_by_env"]["ref"],
        prev["promptprism_scores_by_env"]["ref"],
        group_ids=gids,
    )

    summary = {
        "task_id": "banking77_intent",
        "mined_demo_targets": [d["target"] for d in demos],
        "selected_levels": selected_levels,
        "clean_verdicts": clean_v,
        "stability_verdicts": stab_v,
        "token_costs": token_costs,
        "eval_inner_perturb": {
            k: v for k, v in eval_inner.items() if k != "scores_by_env"
        },
        "eval_outer_perturb": {
            k: v for k, v in eval_outer.items() if k != "scores_by_env"
        },
        "vs_dspy_full_clean": vs_dspy,
        "vs_real_v4_prism_clean": vs_v4,
        "total_flash_calls": client.total_calls,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "\n=== PROMPTPRISM NATIVE STABILITY UPGRADE SUMMARY ===\n"
        + json.dumps(summary, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
