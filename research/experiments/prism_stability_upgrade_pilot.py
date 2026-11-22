"""
Live Stability-Upgrade Pilot for PromptPrism (No DSPy):
Tests whether PromptPrism-native upgrades:
  1. Multi-turn chat demo turns + explicit XML field framing
  2. Screen-mined in-domain candidate demos (zero confirmation leakage, zero Pro proposer calls)
  3. 16-run 2(6-2)IV orthogonal screening + TOST dead-demo pruning + stability-aware selection

lift PromptPrism (`gemini-2.5-flash`, temperature=0.0, thinkingBudget=0) on `banking77_intent`,
`squad2_grounded_qa`, and `gsm8k_math` without DSPy, evaluated on untouched `clean_confirmation`
(`N = 240`) under BOTH inner-field perturbations (matching DSPy eval) and harsh outer-shell
perturbations (matching Real-v4).
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

TASK_OUT_SUBDIRS: Dict[str, str] = {
    "banking77_intent": "banking77_prism_stabilized",
    "squad2_grounded_qa": "squad2_prism_stabilized",
    "gsm8k_math": "gsm8k_prism_stabilized",
}


def _perturb_field(text: str, *, field_label: str, env_id: str) -> str:
    """Apply inner-field perturbation matching `dspy_benchmark.py`."""
    if env_id == "fmt_whitespace":
        return f"\n\n{text}\n\n"
    if env_id == "fmt_delimiter":
        return f"=== {field_label} ===\n{text}\n=== END {field_label} ==="
    if env_id == "word_paraphrase_1":
        return f"Carefully review the provided input below and supply your response.\n{text}"
    if env_id == "word_paraphrase_2":
        return f"Examine the input material below and provide the requested response.\n{text}"
    return text


def _task_system_text(task_id: str, *, strict_contract: bool) -> str:
    if task_id == "banking77_intent":
        return "\n\n".join(
            [
                "You are a precision banking intent classification specialist. Follow all task rules accurately.",
                (
                    "STRICT CONTRACT: Respond ONLY with `FINAL_ANSWER: <exact_intent_label>` using exactly one label from the allowed intents. No commentary."
                    if strict_contract
                    else "Return the matching intent label from the allowed list without extra commentary."
                ),
            ]
        )
    if task_id == "squad2_grounded_qa":
        return "\n\n".join(
            [
                "You are a precision grounded reading-comprehension specialist. Answer strictly from the provided context without extrapolating.",
                (
                    "STRICT CONTRACT: Respond ONLY with `FINAL_ANSWER: <shortest exact span from context>` or `FINAL_ANSWER: UNANSWERABLE` if the context does not contain the answer. No commentary."
                    if strict_contract
                    else "Return only the shortest exact answer span from context, or UNANSWERABLE if the context does not state the answer."
                ),
            ]
        )
    if task_id == "gsm8k_math":
        return "\n\n".join(
            [
                "You are a precision grade-school arithmetic specialist. Solve the word problem carefully step by step.",
                (
                    "STRICT CONTRACT: Show concise step-by-step arithmetic, then finish on the last line with `FINAL_ANSWER: <number>` and nothing else after it."
                    if strict_contract
                    else "Work through the arithmetic step by step and state the final numeric answer clearly at the end."
                ),
            ]
        )
    raise ValueError(f"Unsupported task_id={task_id!r}")


def _task_preamble(task_id: str, env_id: str) -> str:
    if env_id == "word_paraphrase_1":
        return "Carefully review the provided input below and supply your response."
    if env_id == "word_paraphrase_2":
        return "Examine the input material below and provide the requested response."
    return {
        "banking77_intent": "Classify the customer banking query into one allowed intent.",
        "squad2_grounded_qa": "Answer the question using only the shortest exact span from context, or reply UNANSWERABLE.",
        "gsm8k_math": "Solve the grade-school arithmetic word problem step by step and give the final numeric answer.",
    }[task_id]


def _format_user_input(
    task_id: str,
    item: Mapping[str, Any],
    *,
    xml_framing: bool,
    env_id: str = "ref",
    outer_shell_perturb: bool = False,
) -> str:
    if task_id == "banking77_intent":
        opts = ", ".join(str(o) for o in item["options"])
        query = str(item["input_text"])
        if not outer_shell_perturb:
            q_fmt = _perturb_field(query, field_label="QUERY", env_id=env_id)
            if xml_framing:
                return (
                    f"<allowed_intents>{opts}</allowed_intents>\n<query>{q_fmt}</query>"
                )
            return f"Allowed Intents: {opts}\nQuery: {q_fmt}"
        preamble = _task_preamble(task_id, env_id)
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
            return f"{preamble}{sep}<allowed_intents>{opts}</allowed_intents>{sep}<query>{query}</query>"
        return f"{preamble}{sep}Allowed Intents: {opts}{sep}Query: {query}"

    if task_id == "squad2_grounded_qa":
        ctx = str(item["context"])
        q = str(item["question"])
        if not outer_shell_perturb:
            q_fmt = _perturb_field(q, field_label="QUESTION", env_id=env_id)
            if xml_framing:
                return f"<context>{ctx}</context>\n<question>{q_fmt}</question>"
            return f"Context: {ctx}\nQuestion: {q_fmt}"
        preamble = _task_preamble(task_id, env_id)
        sep = "\n\n\n" if env_id == "fmt_whitespace" else "\n"
        if env_id == "fmt_delimiter":
            if xml_framing:
                return (
                    f"=== TASK ===\n{preamble}\n"
                    f"<context>{ctx}</context>\n"
                    f"<question>{q}</question>"
                )
            return f"=== TASK ===\n{preamble}\n=== CONTEXT ===\n{ctx}\n=== QUESTION ===\n{q}"
        if xml_framing:
            return (
                f"{preamble}{sep}<context>{ctx}</context>{sep}<question>{q}</question>"
            )
        return f"{preamble}{sep}Context: {ctx}{sep}Question: {q}"

    if task_id == "gsm8k_math":
        q = str(item["question"])
        if not outer_shell_perturb:
            q_fmt = _perturb_field(q, field_label="QUESTION", env_id=env_id)
            if xml_framing:
                return f"<question>{q_fmt}</question>"
            return f"Question: {q_fmt}"
        preamble = _task_preamble(task_id, env_id)
        sep = "\n\n\n" if env_id == "fmt_whitespace" else "\n"
        if env_id == "fmt_delimiter":
            if xml_framing:
                return f"=== TASK ===\n{preamble}\n<question>{q}</question>"
            return f"=== TASK ===\n{preamble}\n=== QUESTION ===\n{q}"
        if xml_framing:
            return f"{preamble}{sep}<question>{q}</question>"
        return f"{preamble}{sep}Question: {q}"

    raise ValueError(f"Unsupported task_id={task_id!r}")


def _demo_user_and_assistant(
    task_id: str, demo: Mapping[str, Any], *, xml_framing: bool
) -> Tuple[str, str]:
    if task_id == "banking77_intent":
        d_opts = ", ".join(str(o) for o in demo["options"])
        if xml_framing:
            u = f"<allowed_intents>{d_opts}</allowed_intents>\n<query>{demo['input_text']}</query>"
        else:
            u = f"Query: {demo['input_text']}"
        a = f"FINAL_ANSWER: {demo['target']}"
        return u, a

    if task_id == "squad2_grounded_qa":
        gold_list = demo["target"]
        gold_ans = str(gold_list[0]) if gold_list else "UNANSWERABLE"
        if xml_framing:
            u = f"<context>{demo['context']}</context>\n<question>{demo['question']}</question>"
        else:
            u = f"Context: {demo['context']}\nQuestion: {demo['question']}"
        a = f"FINAL_ANSWER: {gold_ans}"
        return u, a

    if task_id == "gsm8k_math":
        if xml_framing:
            u = f"<question>{demo['question']}</question>"
        else:
            u = f"Question: {demo['question']}"
        a = str(demo.get("demo_assistant") or f"FINAL_ANSWER: {demo['target']}")
        return u, a

    raise ValueError(f"Unsupported task_id={task_id!r}")


def build_messages(
    *,
    task_id: str,
    item: Mapping[str, Any],
    levels: Mapping[str, int],
    demos: Sequence[Mapping[str, Any]],
    env_id: str = "ref",
    outer_shell_perturb: bool = False,
) -> List[Dict[str, str]]:
    """Render PromptPrism structured multi-turn or flat chat prompt."""
    xml_multi = int(levels["xml_multiturn"]) == 1
    strict_contract = int(levels["strict_contract"]) == 1
    system_text = _task_system_text(task_id, strict_contract=strict_contract)

    active_demos = [
        demos[i] for i in range(len(demos)) if int(levels[f"demo_{i + 1}"]) == 1
    ]

    if xml_multi:
        msgs: List[Dict[str, str]] = [{"role": "system", "content": system_text}]
        for d in active_demos:
            du, da = _demo_user_and_assistant(task_id, d, xml_framing=True)
            msgs.append({"role": "user", "content": du})
            msgs.append({"role": "assistant", "content": da})
        msgs.append(
            {
                "role": "user",
                "content": _format_user_input(
                    task_id,
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
        demo_lines = []
        for i, d in enumerate(active_demos):
            du, da = _demo_user_and_assistant(task_id, d, xml_framing=False)
            demo_lines.append(f"Example {i + 1}:\n{du}\nOutput: {da}")
        hdr = (
            "=== EXAMPLES ==="
            if (outer_shell_perturb and env_id == "fmt_delimiter")
            else "[EXAMPLES]"
        )
        blocks.append(f"{hdr}\n" + "\n".join(demo_lines))
    blocks.append(
        _format_user_input(
            task_id,
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
    task_id: str,
    screen_items: Sequence[Mapping[str, Any]],
    client: GeminiMultiTurnClient,
    n_demos: int = 4,
    max_workers: int = 10,
) -> List[Dict[str, Any]]:
    """Zero-shot mine `n_demos` diverse in-domain candidates strictly from `screen` (no leakage)."""
    base_levels = {
        "xml_multiturn": 1,
        "strict_contract": 1,
        "demo_1": 0,
        "demo_2": 0,
        "demo_3": 0,
        "demo_4": 0,
    }
    scores = [0.0] * len(screen_items)
    raw_outputs: List[str] = [""] * len(screen_items)

    def _probe(idx: int) -> Tuple[int, float, str]:
        it = screen_items[idx]
        msgs = build_messages(
            task_id=task_id, item=it, levels=base_levels, demos=[], env_id="ref"
        )
        r = client.generate(msgs)
        sc = 0.0 if r.error else float(score_task_response(task_id, r.content, it))
        return idx, sc, ("" if r.error else str(r.content or "").strip())

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for fut in as_completed(
            [pool.submit(_probe, i) for i in range(len(screen_items))]
        ):
            idx, sc, txt = fut.result()
            scores[idx] = sc
            raw_outputs[idx] = txt

    picked: List[Dict[str, Any]] = []
    if task_id == "banking77_intent":
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
        desc = [d["target"] for d in picked]

    elif task_id == "squad2_grounded_qa":
        # Balance lowest-scoring answerable (2) and unanswerable (2) items across distinct groups
        ans_indices = sorted(
            [i for i, it in enumerate(screen_items) if bool(it["target"])],
            key=lambda i: (scores[i], len(str(screen_items[i]["context"]))),
        )
        unans_indices = sorted(
            [i for i, it in enumerate(screen_items) if not bool(it["target"])],
            key=lambda i: (scores[i], len(str(screen_items[i]["context"]))),
        )
        seen_groups = set()
        half = n_demos // 2
        for bucket, quota in (
            (ans_indices, half),
            (unans_indices, n_demos - half),
            (ans_indices + unans_indices, n_demos),
        ):
            added = 0
            for i in bucket:
                if len(picked) >= n_demos or added >= quota:
                    break
                gid = str(screen_items[i]["group_id"])
                if gid in seen_groups:
                    continue
                picked.append(dict(screen_items[i]))
                seen_groups.add(gid)
                added += 1
        desc = [f"{d['sample_id']}:{'ANS' if d['target'] else 'UNANS'}" for d in picked]

    elif task_id == "gsm8k_math":
        # Pick diverse, challenging verified-correct (score==1.0) Flash CoT traces from screen
        # (zero Pro calls, zero external gold rationales, concise <= 800 chars with FINAL_ANSWER)
        verified = [
            i
            for i, sc in enumerate(scores)
            if sc >= 1.0
            and "FINAL_ANSWER:" in raw_outputs[i]
            and 40 <= len(raw_outputs[i]) <= 800
        ]
        # Prefer richer multi-step questions (longer question text) without bloat
        verified.sort(key=lambda i: -len(str(screen_items[i]["question"])))
        seen_groups = set()
        for i in verified:
            gid = str(screen_items[i]["group_id"])
            if gid in seen_groups:
                continue
            d = dict(screen_items[i])
            d["demo_assistant"] = raw_outputs[i]
            picked.append(d)
            seen_groups.add(gid)
            if len(picked) >= n_demos:
                break
        if len(picked) < n_demos:
            for i, sc in enumerate(scores):
                if len(picked) >= n_demos:
                    break
                gid = str(screen_items[i]["group_id"])
                if sc >= 1.0 and gid not in seen_groups:
                    d = dict(screen_items[i])
                    d["demo_assistant"] = (
                        raw_outputs[i] or f"FINAL_ANSWER: {d['target']}"
                    )
                    picked.append(d)
                    seen_groups.add(gid)
        desc = [f"{d['sample_id']}={d['target']}" for d in picked]

    else:
        raise ValueError(f"Unsupported task_id={task_id!r}")

    print(
        f"[{task_id}] Zero-shot screen acc={np.mean(scores):.4f}; "
        f"mined {len(picked)} candidate demos: {desc}",
        flush=True,
    )
    return picked


def eval_config_on_envs(
    *,
    task_id: str,
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
            task_id=task_id,
            item=it,
            levels=levels,
            demos=demos,
            env_id=e,
            outer_shell_perturb=outer_shell_perturb,
        )
        r = client.generate(msgs)
        sc = 0.0 if r.error else float(score_task_response(task_id, r.content, it))
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
        task_id=f"{task_id}:prism_stabilized:{'outer' if outer_shell_perturb else 'inner'}",
        model_id="gemini-2.5-flash",
        model_version="gemini-2.5-flash",
        dataset_partition_hash=f"{task_id}_clean_confirmation_240",
        parser_hash=p_hashes[task_id],
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
            "worst_env_id": cert.worst_env_id,
            "repeat_noise_se": cert.repeat_noise_se,
            "recommended_min_n": cert.recommended_min_n,
            "diagnostic_note": cert.diagnostic_note,
        },
    }


def run_task_stability_pilot(
    task_id: str,
    *,
    out_root: Path,
    max_workers: int = 10,
) -> Dict[str, Any]:
    out_dir = out_root / TASK_OUT_SUBDIRS[task_id]
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = load_real_v4_splits(task_id)
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

    # 1. Screen-mine 4 in-domain candidate demos (120 calls, zero confirmation leakage)
    demos = mine_candidate_demos_from_screen(
        task_id, screen_items, client, n_demos=4, max_workers=max_workers
    )

    # 2. 16-run 2(6-2)IV screen across 6 factors on screen (ref) + val (ref & fmt_delimiter)
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
        f"[{task_id}][Screen] Running 2(6-2)IV (16 runs x 120 screen ref + 60 val ref/outer fmt_delimiter)...",
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
            task_id=task_id,
            item=it,
            levels=nl,
            demos=demos,
            env_id="ref",
            outer_shell_perturb=True,
        )
        r = client.generate(msgs)
        sc = 0.0 if r.error else float(score_task_response(task_id, r.content, it))
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
            task_id=task_id,
            item=it,
            levels=nl,
            demos=demos,
            env_id=env,
            outer_shell_perturb=True,
        )
        r = client.generate(msgs)
        sc = 0.0 if r.error else float(score_task_response(task_id, r.content, it))
        rec = {
            "run_id": rid,
            "sample_id": str(it["sample_id"]),
            "env_id": env,
            "score": sc,
        }
        rec.update(nl)
        return rec

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
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

    # Native stability-aware attribution via `env_col="env_id"` on val (ref vs fmt_delimiter)
    df_v = pd.DataFrame(val_stab_rows)
    audit_val_stab = run_calibrated_attribution_audit(
        data=df_v,
        factor_cols=factor_names,
        target_col="score",
        sample_col="sample_id",
        alpha=0.05,
        maximize=True,
        equivalence_margin=0.02,
        token_costs=token_costs,
        n_sign_flip=2000,
        seed=42,
        env_col="env_id",
        ref_env="ref",
    )
    (out_dir / "val_audit_stability_drop.md").write_text(
        audit_val_stab.to_markdown() + "\n", encoding="utf-8"
    )

    clean_v = {f.factor_id: f.pruning_verdict for f in audit_clean.factors}
    stab_v = {
        f.factor_id: (f.stability_verdict or "NEUTRAL") for f in audit_val_stab.factors
    }
    selected_levels: Dict[str, int] = {}
    for fn in factor_names:
        if stab_v[fn] == "STABILIZING":
            selected_levels[fn] = 1
        elif clean_v[fn] == "REMOVE_HARMFUL" or stab_v[fn] == "FRAGILE":
            selected_levels[fn] = 0
        elif clean_v[fn] == "SAFE_TO_PRUNE":
            selected_levels[fn] = 0
        else:
            selected_levels[fn] = 1
    print(
        f"[{task_id}][Selection] clean={clean_v} stab={stab_v} -> selected={selected_levels}",
        flush=True,
    )

    # 3. Validate selected prompt on untouched 240 confirmation x 6 envs (inner + outer)
    print(
        f"[{task_id}][Confirm] Evaluating selected prompt on 240 confirmation x 6 envs (inner + outer)...",
        flush=True,
    )
    eval_inner = eval_config_on_envs(
        task_id=task_id,
        levels=selected_levels,
        demos=demos,
        items=conf_items,
        env_ids=PAIRED_ENV_IDS,
        outer_shell_perturb=False,
        client=client,
        max_workers=max_workers,
    )
    eval_outer = eval_config_on_envs(
        task_id=task_id,
        levels=selected_levels,
        demos=demos,
        items=conf_items,
        env_ids=PAIRED_ENV_IDS,
        outer_shell_perturb=True,
        client=client,
        max_workers=max_workers,
    )

    prev = json.loads(
        (out_root / task_id / "confirmation_scores.json").read_text(encoding="utf-8")
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
        "task_id": task_id,
        "mined_demo_ids": [str(d["sample_id"]) for d in demos],
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
    (out_dir / "confirmation_scores.json").write_text(
        json.dumps(
            {
                "inner_scores_by_env": eval_inner["scores_by_env"],
                "outer_scores_by_env": eval_outer["scores_by_env"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"\n=== PROMPTPRISM NATIVE STABILITY UPGRADE SUMMARY [{task_id}] ===\n"
        + json.dumps(summary, indent=2),
        flush=True,
    )
    return summary


def run_scaffold_held_confirmation(
    task_id: str,
    *,
    out_root: Path,
    max_workers: int = 14,
) -> Dict[str, Any]:
    """Re-confirm holding structural framing (`xml_multiturn=1`, `strict_contract=1`) fixed while pruning only demos."""
    out_dir = out_root / TASK_OUT_SUBDIRS[task_id]
    prev_summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    splits = load_real_v4_splits(task_id)
    screen_by_id = {str(it["sample_id"]): it for it in splits["screen"]}
    conf_items = splits["clean_confirmation"]

    # Reuse exact previously mined screen demo items (only demo_1..4 kept=1 are used)
    demos = [dict(screen_by_id[sid]) for sid in prev_summary["mined_demo_ids"]]
    selected_levels = dict(prev_summary["selected_levels"])
    selected_levels["xml_multiturn"] = 1
    selected_levels["strict_contract"] = 1

    client = GeminiMultiTurnClient(
        "gemini-2.5-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.0,
        max_output_tokens=512,
        thinking_budget=0,
    )
    print(
        f"[{task_id}][Scaffold-Held Confirm] levels={selected_levels} on 240 confirmation x 6 envs (inner + outer)...",
        flush=True,
    )
    eval_inner = eval_config_on_envs(
        task_id=task_id,
        levels=selected_levels,
        demos=demos,
        items=conf_items,
        env_ids=PAIRED_ENV_IDS,
        outer_shell_perturb=False,
        client=client,
        max_workers=max_workers,
    )
    eval_outer = eval_config_on_envs(
        task_id=task_id,
        levels=selected_levels,
        demos=demos,
        items=conf_items,
        env_ids=PAIRED_ENV_IDS,
        outer_shell_perturb=True,
        client=client,
        max_workers=max_workers,
    )

    prev_bench = json.loads(
        (out_root / task_id / "confirmation_scores.json").read_text(encoding="utf-8")
    )
    prev_native = json.loads(
        (out_dir / "confirmation_scores.json").read_text(encoding="utf-8")
    )
    gids = [str(it["group_id"]) for it in conf_items]
    vs_dspy = paired_item_comparison(
        eval_inner["scores_by_env"]["ref"],
        prev_bench["dspy_scores_by_env"]["ref"],
        group_ids=gids,
    )
    vs_v4 = paired_item_comparison(
        eval_inner["scores_by_env"]["ref"],
        prev_bench["promptprism_scores_by_env"]["ref"],
        group_ids=gids,
    )
    vs_unconstrained = paired_item_comparison(
        eval_inner["scores_by_env"]["ref"],
        prev_native["inner_scores_by_env"]["ref"],
        group_ids=gids,
    )

    res = {
        "task_id": task_id,
        "selected_levels": selected_levels,
        "eval_inner_perturb": {
            k: v for k, v in eval_inner.items() if k != "scores_by_env"
        },
        "eval_outer_perturb": {
            k: v for k, v in eval_outer.items() if k != "scores_by_env"
        },
        "vs_dspy_full_clean": vs_dspy,
        "vs_real_v4_prism_clean": vs_v4,
        "vs_unconstrained_pruned_clean": vs_unconstrained,
        "total_flash_calls": client.total_calls,
    }
    (out_dir / "scaffold_held_confirm.json").write_text(
        json.dumps(res, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"\n=== SCAFFOLD-HELD CONFIRMATION [{task_id}] ===\n"
        + json.dumps(res, indent=2),
        flush=True,
    )
    return res


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["banking77_intent"],
        choices=list(TASK_OUT_SUBDIRS.keys()),
    )
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--confirm-scaffold-held", action="store_true")
    parser.add_argument(
        "--out-root",
        type=str,
        default=str(ROOT / "research" / "results" / "dspy_benchmark"),
    )
    args = parser.parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    for task_id in args.tasks:
        if args.confirm_scaffold_held:
            run_scaffold_held_confirmation(
                task_id, out_root=out_root, max_workers=args.max_workers
            )
        else:
            run_task_stability_pilot(
                task_id, out_root=out_root, max_workers=args.max_workers
            )


if __name__ == "__main__":
    main()
