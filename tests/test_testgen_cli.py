"""
Tests for PR 4: CLI `prism tests {generate,review,freeze,stats}`, YAML/JSON loaders,
`--client module:callable` + `cloud`-gated LiteLLM `--model`, CSV & Markdown review
workflows, and `prism diagnose --tests` with split-leakage & integrity guards.

All tests run 100% offline with zero network calls.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from prompt_prism.cli.main import main as cli_main
from prompt_prism.core.factors import Factor, FactorSet, FactorType
from prompt_prism.runner.client import CallableLLM, LLMClient
from prompt_prism.template.composer import PromptTemplate
from prompt_prism.testgen.loaders import (
    LiteLLMClient,
    load_client_from_spec,
    load_factors_file,
    load_seed_examples_file,
    load_template_file,
    resolve_cli_client,
)
from prompt_prism.testgen.models import CheckSpec, TestCase, TestSet
from prompt_prism.testgen.review import (
    REVIEW_CSV_COLUMNS,
    export_review,
    format_testset_stats,
    import_review,
)


def _write_mock_client_module(
    tmp_path: Path, module_name: str = "mock_cli_clients"
) -> Path:
    """Write a deterministic offline generator + tagger module importable via module:callable."""
    code = """
import json
import re
from prompt_prism.runner.client import LLMClient, LLMResponse

_COUNTER = {"base": 0, "A": 0, "B": 0}
_TOPICS = [
    "wire", "debit", "credit", "iban", "pin", "refund", "dispute", "overdraft",
    "atm", "fx", "applepay", "sepa", "swift", "directdebit", "chargeback",
    "limit", "statement", "mortgage", "savings", "ledger", "invoice", "payroll",
    "escrow", "ach", "treasury", "cashback", "rewards", "autopay", "kyc", "vault",
]


def _word(idx: int) -> str:
    return f"{_TOPICS[(idx - 1) % len(_TOPICS)]}_{idx}"


def generator_fn(prompt, **kwargs):
    text = str(prompt)
    if "[PROMPTPRISM_TASK:CLASSIFY_ROLES]" in text:
        return json.dumps({"roles": {"A": "global_scaffold", "B": "conditional_rule"}})

    m_n = re.search(r"Generate (\\d+)", text)
    n = int(m_n.group(1)) if m_n else 4

    if "scaffold `A`" in text or "clause `A`" in text:
        cases = []
        for _ in range(n):
            _COUNTER["A"] += 1
            idx = _COUNTER["A"]
            w = _word(idx)
            cases.append({
                "inputs": {"query": f"Ambiguous scaffold {w} mixing fee and transfer rules"},
                "target": "billing",
                "checks": [{"kind": "exact", "value": "billing", "weight": 1.0, "description": "task label"}],
                "reference_pass": "billing",
                "reference_fail": "shipping",
            })
        return json.dumps({"cases": cases})

    if "clause `B`" in text or "scaffold `B`" in text:
        cases = []
        for _ in range(n):
            _COUNTER["B"] += 1
            idx = _COUNTER["B"]
            w = _word(idx + 7)
            # Item #1 lacks trigger keyword so blind tagger disagrees with intended_factor B
            if idx == 1:
                q = f"Standard routine inquiry {w} checking normal account status"
            else:
                q = f"Out of scope weather forecast {w} for tomorrow afternoon"
            cases.append({
                "inputs": {"query": q},
                "target": "OUT_OF_SCOPE",
                "checks": [{"kind": "contains", "value": "OUT_OF_SCOPE", "weight": 1.0, "description": "oos rule"}],
                "reference_pass": "OUT_OF_SCOPE",
                "reference_fail": "billing",
            })
        return json.dumps({"cases": cases})

    # Baseline cases
    cases = []
    for _ in range(n):
        _COUNTER["base"] += 1
        idx = _COUNTER["base"]
        w = _word(idx + 13)
        cases.append({
            "inputs": {"query": f"Routine banking question {w} about card settlement"},
            "target": "billing",
            "checks": [{"kind": "exact", "value": "billing", "weight": 1.0, "description": "baseline label"}],
            "reference_pass": "billing",
            "reference_fail": "OUT_OF_SCOPE",
        })
    return json.dumps({"cases": cases})


generator_fn.model_id = "mock-gen-v1"


class TaggerClient(LLMClient):
    model_id = "mock-tagger-v1"

    def generate(self, prompt, **kwargs):
        text = str(prompt)
        m = re.search(r"Cases:\\n(\\[.*\\])\\s*$", text, flags=re.DOTALL)
        items = json.loads(m.group(1)) if m else []
        tags = {}
        for item in items:
            q = str(item.get("inputs", {}).get("query", "")).lower()
            matched = []
            if "ambiguous" in q or "scaffold" in q:
                matched.append("A")
            if "out of scope" in q or "weather" in q:
                matched.append("B")
            tags[item["id"]] = matched
        return LLMResponse(content=json.dumps({"tags": tags}))


tagger_instance = TaggerClient()
"""
    mod_path = tmp_path / f"{module_name}.py"
    mod_path.write_text(code.strip() + "\n", encoding="utf-8")
    return mod_path


def _write_sample_yaml_configs(tmp_path: Path) -> Dict[str, Path]:
    factors_p = tmp_path / "factors.yaml"
    factors_p.write_text(
        """
factors:
  - id: A
    name: persona
    clause_role: global_scaffold
    description: Specialist billing persona
    level_0: ""
    level_1: "You are a senior banking support specialist."
  - id: B
    name: oos_rule
    clause_role: conditional_rule
    description: Out of scope refusal rule
    level_0_content: ""
    level_1_content: "If the request is out of scope, output OUT_OF_SCOPE."
""".lstrip(),
        encoding="utf-8",
    )

    template_p = tmp_path / "template.yaml"
    template_p.write_text(
        """
system_prompt: "Classify each customer query."
data_template: "Query: {{ query }}"
""".lstrip(),
        encoding="utf-8",
    )

    seeds_p = tmp_path / "seeds.jsonl"
    seeds_p.write_text(
        json.dumps({"query": "Why was my card declined?", "target": "billing"})
        + "\n"
        + json.dumps(
            {"query": "What is the weather in Tokyo?", "target": "OUT_OF_SCOPE"}
        )
        + "\n",
        encoding="utf-8",
    )
    return {"factors": factors_p, "template": template_p, "seeds": seeds_p}


def test_loaders_factors_template_and_seeds(tmp_path: Path):
    """YAML/JSON loaders map 1:1 to Factor.binary/Factor, PromptTemplate, and seed files."""
    cfg = _write_sample_yaml_configs(tmp_path)

    fset = load_factors_file(cfg["factors"])
    assert len(fset) == 2
    assert fset["A"].name == "persona"
    assert fset["A"].metadata["clause_role"] == "global_scaffold"
    assert (
        fset["A"].get_level(1).content == "You are a senior banking support specialist."
    )
    assert fset["B"].metadata["clause_role"] == "conditional_rule"
    assert "OUT_OF_SCOPE" in fset["B"].get_level(1).content

    # JSON top-level list with explicit levels
    factors_json = tmp_path / "factors.json"
    factors_json.write_text(
        json.dumps(
            [
                {
                    "id": "X",
                    "name": "strict_json",
                    "factor_type": "section",
                    "clause_role": "conditional_rule",
                    "levels": [
                        {"code": 0, "name": "Off", "content": ""},
                        {"code": 1, "name": "On", "content": "Output valid JSON."},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    fset_j = load_factors_file(factors_json)
    assert fset_j["X"].factor_type == FactorType.SECTION
    assert fset_j["X"].metadata["clause_role"] == "conditional_rule"

    # Error validation on typos, duplicates, invalid clause_role, mixed keys
    bad_typo = tmp_path / "bad_typo.yaml"
    bad_typo.write_text("factors:\n  - name: foo\n    level1: bar\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown key"):
        load_factors_file(bad_typo)

    bad_role = tmp_path / "bad_role.yaml"
    bad_role.write_text(
        "factors:\n  - name: foo\n    clause_role: invalid_role\n    level_1: hi\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid clause_role"):
        load_factors_file(bad_role)

    bad_dup = tmp_path / "bad_dup.yaml"
    bad_dup.write_text(
        "factors:\n  - id: A\n    name: f1\n  - id: A\n    name: f2\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Duplicate factor id"):
        load_factors_file(bad_dup)

    # Template formats: shorthand, canonical sections dict, master_template
    tmpl_short = load_template_file(cfg["template"], factors=fset)
    assert tmpl_short.required_data_variables(fset) == {"query"}
    assert any(s.factor_id == "A" for s in tmpl_short.sections)

    tmpl_dict_p = tmp_path / "template_dict.json"
    tmpl_dict_p.write_text(json.dumps(tmpl_short.to_dict()), encoding="utf-8")
    tmpl_round = load_template_file(tmpl_dict_p)
    assert tmpl_round.template_sha256() == tmpl_short.template_sha256()

    tmpl_master_p = tmp_path / "template_master.yaml"
    tmpl_master_p.write_text(
        'master_template: "{{ persona }}\\n{{ oos_rule }}\\nInput: {{ query }}"\n',
        encoding="utf-8",
    )
    tmpl_master = load_template_file(tmpl_master_p)
    assert tmpl_master.required_data_variables(fset) == {"query"}

    # Shorthand without factors fails clearly
    with pytest.raises(ValueError, match="requires factors"):
        load_template_file(cfg["template"])

    # Seed examples: jsonl, json, yaml, csv
    seeds_jsonl = load_seed_examples_file(cfg["seeds"])
    assert len(seeds_jsonl) == 2

    seeds_csv_p = tmp_path / "seeds.csv"
    pd.DataFrame([{"query": "hello", "note": np.nan}]).to_csv(seeds_csv_p, index=False)
    seeds_csv = load_seed_examples_file(seeds_csv_p)
    assert seeds_csv == [{"query": "hello", "note": None}]

    seeds_yaml_p = tmp_path / "seeds.yaml"
    seeds_yaml_p.write_text("examples:\n  - query: hi\n", encoding="utf-8")
    assert load_seed_examples_file(seeds_yaml_p) == [{"query": "hi"}]


def test_client_resolution_and_cloud_gated_litellm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Verify `--client module:callable` resolution and `cloud`-gated LiteLLM `--model`."""
    _write_mock_client_module(tmp_path, module_name="tmp_client_res_mod")
    monkeypatch.chdir(tmp_path)
    # Remove tmp_path from sys.path if present to prove CWD fallback works cleanly
    monkeypatch.setattr(
        sys, "path", [p for p in sys.path if p not in ("", str(tmp_path))]
    )

    c_fn = load_client_from_spec("tmp_client_res_mod:generator_fn")
    assert isinstance(c_fn, CallableLLM)
    assert c_fn.model_id == "mock-gen-v1"

    c_inst = load_client_from_spec("tmp_client_res_mod:tagger_instance")
    assert isinstance(c_inst, LLMClient)
    assert c_inst.model_id == "mock-tagger-v1"

    c_cls = load_client_from_spec("tmp_client_res_mod:TaggerClient")
    assert isinstance(c_cls, LLMClient)
    assert c_cls.model_id == "mock-tagger-v1"

    with pytest.raises(ValueError, match="module.path:callable"):
        load_client_from_spec("no_colon_here")

    with pytest.raises(ValueError, match="not both"):
        resolve_cli_client(
            client_spec="tmp_client_res_mod:generator_fn", model="gpt-4o"
        )

    with pytest.raises(ValueError, match="Must specify"):
        resolve_cli_client()

    # Cloud-gated LiteLLM: missing litellm gives clean prompt-prism[cloud] ImportError
    monkeypatch.setitem(sys.modules, "litellm", None)
    with pytest.raises(ImportError, match="prompt-prism\\[cloud\\]"):
        resolve_cli_client(model="gemini/gemini-2.5-flash")

    # Stub litellm installed: LiteLLMClient formats messages, captures content/usage/model_id
    calls: List[Dict[str, Any]] = []

    def _fake_completion(**kwargs):
        calls.append(kwargs)
        if kwargs.get("fail_now"):
            raise RuntimeError("boom")
        return {
            "choices": [{"message": {"content": "stubbed answer"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    fake_litellm = types.SimpleNamespace(completion=_fake_completion)
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    llm = resolve_cli_client(model="gemini/gemini-2.5-flash")
    assert isinstance(llm, LiteLLMClient)
    assert llm.model_id == "gemini/gemini-2.5-flash"

    resp = llm.generate("Hello world")
    assert resp.error is None
    assert resp.content == "stubbed answer"
    assert resp.token_usage["total_tokens"] == 15
    assert calls[-1]["model"] == "gemini/gemini-2.5-flash"
    assert calls[-1]["messages"] == [{"role": "user", "content": "Hello world"}]

    err_resp = llm.generate([{"role": "user", "content": "Hi"}], fail_now=True)
    assert err_resp.error == "boom"
    assert err_resp.content == ""


def test_cli_tests_generate_review_freeze_stats_and_refresh_e2e(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """End-to-end CLI workflow: generate -> stats -> CSV & MD review -> freeze -> refresh-confirmation."""
    _write_mock_client_module(tmp_path, module_name="e2e_mock_clients")
    cfg = _write_sample_yaml_configs(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))

    tests_dir = tmp_path / "tests_out"

    # 1. Generate draft TestSet (--no-freeze)
    cli_main(
        [
            "tests",
            "generate",
            "--factors",
            str(cfg["factors"]),
            "--template",
            str(cfg["template"]),
            "--task",
            "Banking support intent classification",
            "--seed-examples",
            str(cfg["seeds"]),
            "--n-baseline-discovery",
            "6",
            "--n-baseline-confirmation",
            "4",
            "--n-per-factor-discovery",
            "4",
            "--n-per-factor-confirmation",
            "2",
            "--client",
            "e2e_mock_clients:generator_fn",
            "--tagger-client",
            "e2e_mock_clients:tagger_instance",
            "--no-freeze",
            "--out",
            str(tests_dir),
        ]
    )
    out = capsys.readouterr().out
    assert "Generated TestSet" in out
    assert "Status:              draft" in out

    ts_draft = TestSet.load(tests_dir)
    assert not ts_draft.is_frozen
    assert ts_draft.manifest.generator["model_id"] == "mock-gen-v1"
    assert ts_draft.manifest.generator["tagger_model_id"] == "mock-tagger-v1"
    assert len(ts_draft.cases) == (6 + 4) + 2 * (4 + 2)
    disagreed_cases = [c for c in ts_draft.cases if c.tagger_agrees is False]
    assert len(disagreed_cases) == 1
    disagreed_id = disagreed_cases[0].id

    # 2. Stats (--json) reports draft warning, thin-factor warnings, and tagger disagreement
    cli_main(["tests", "stats", str(tests_dir), "--min-targeted-items", "10", "--json"])
    stats_data = json.loads(capsys.readouterr().out)
    assert stats_data["status"] == "draft"
    assert any("unfrozen" in w.lower() for w in stats_data["warnings"])
    assert any("tagger disagreement" in w.lower() for w in stats_data["warnings"])
    assert any("min_targeted_items=10" in w for w in stats_data["warnings"])

    # 3. Review export CSV: disagreed item must be sorted FIRST
    csv_review = tmp_path / "review.csv"
    cli_main(
        [
            "tests",
            "review",
            "export",
            str(tests_dir),
            "--format",
            "csv",
            "--out",
            str(csv_review),
        ]
    )
    assert "tagger-disagreed first" in capsys.readouterr().out

    with csv_review.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == REVIEW_CSV_COLUMNS
    assert rows[0]["id"] == disagreed_id
    assert rows[0]["tagger_agrees"] == "False"

    # Edit CSV: accept the disagreed targeted case, reject one baseline case
    baseline_row = next(r for r in rows if r["kind"] == "baseline")
    rejected_base_id = baseline_row["id"]
    rows[0]["accept"] = "yes"
    baseline_row["accept"] = "no"
    with csv_review.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    # 4. Review import CSV: updates review, retains rejected row, restores coverage, and freezes
    cli_main(["tests", "review", "import", str(tests_dir), str(csv_review)])
    imp_out = capsys.readouterr().out
    assert "re-frozen" in imp_out

    ts_after_csv = TestSet.load(tests_dir)
    assert ts_after_csv.is_frozen
    assert len(ts_after_csv.cases) == len(ts_draft.cases)  # rejected row retained!
    by_id = {c.id: c for c in ts_after_csv.cases}
    assert by_id[rejected_base_id].review == "rejected"
    assert by_id[disagreed_id].review == "accepted"
    assert "B" in by_id[disagreed_id].targets_factors
    assert (
        ts_after_csv.manifest.cases_sha256 != ts_after_csv.manifest.active_cases_sha256
    )
    # Rejected baseline row excluded from dataset export
    all_dataset_ids = {r["id"] for r in ts_after_csv.to_dataset(split="all")}
    assert rejected_base_id not in all_dataset_ids
    assert disagreed_id in all_dataset_ids

    # 5. Review export Markdown + edit + import (auto-detects .md)
    md_review = tmp_path / "review.md"
    cli_main(
        [
            "tests",
            "review",
            "export",
            str(tests_dir),
            "--format",
            "md",
            "--out",
            str(md_review),
        ]
    )
    capsys.readouterr()
    md_text = md_review.read_text(encoding="utf-8")
    assert f"## Case `{disagreed_id}`" in md_text

    # Pick an active pending targeted case to reject via [!] and a pending baseline case to accept via GFM checkbox
    pending_targeted = next(
        c for c in ts_after_csv.cases if c.kind == "targeted" and c.review == "pending"
    )
    pending_baseline = next(
        c for c in ts_after_csv.cases if c.kind == "baseline" and c.review == "pending"
    )

    def _replace_case_accept(doc: str, cid: str, new_line: str) -> str:
        marker = f"## Case `{cid}`"
        head, tail = doc.split(marker, 1)
        old_line = "- **accept:** `[ ]` <!-- set to [x] / yes (accept) or [!] / no (reject) -->"
        assert old_line in tail
        return head + marker + tail.replace(old_line, new_line, 1)

    md_text = _replace_case_accept(md_text, pending_targeted.id, "- **accept:** `[!]`")
    md_text = _replace_case_accept(md_text, pending_baseline.id, "- [x] **accept:**")
    md_review.write_text(md_text, encoding="utf-8")

    cli_main(["tests", "review", "import", str(tests_dir), str(md_review)])
    assert "[md]" in capsys.readouterr().out

    ts_after_md = TestSet.load(tests_dir)
    assert ts_after_md.is_frozen
    by_id_md = {c.id: c for c in ts_after_md.cases}
    assert by_id_md[pending_targeted.id].review == "rejected"
    assert by_id_md[pending_baseline.id].review == "accepted"

    # 6. Freeze command refuses already-frozen without --force, succeeds with --force
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["tests", "freeze", str(tests_dir)])
    assert exc_info.value.code == 1
    assert "already frozen" in capsys.readouterr().err

    cli_main(["tests", "freeze", str(tests_dir), "--force"])
    assert "Frozen TestSet" in capsys.readouterr().out

    # 7. --refresh-confirmation in-place preserves discovery verbatim and replaces confirmation
    old_disc_ids = {c.id for c in ts_after_md.cases if c.split == "discovery"}
    old_conf_ids = {c.id for c in ts_after_md.cases if c.split == "confirmation"}
    old_sha = ts_after_md.manifest.cases_sha256

    cli_main(
        [
            "tests",
            "generate",
            "--factors",
            str(cfg["factors"]),
            "--template",
            str(cfg["template"]),
            "--task",
            "Banking support intent classification",
            "--n-baseline-confirmation",
            "4",
            "--n-per-factor-confirmation",
            "2",
            "--client",
            "e2e_mock_clients:generator_fn",
            "--tagger-client",
            "e2e_mock_clients:tagger_instance",
            "--refresh-confirmation",
            str(tests_dir),
        ]
    )
    assert "Refreshed confirmation split" in capsys.readouterr().out

    ts_refreshed = TestSet.load(tests_dir)
    assert ts_refreshed.is_frozen
    new_disc_ids = {c.id for c in ts_refreshed.cases if c.split == "discovery"}
    new_conf_ids = {c.id for c in ts_refreshed.cases if c.split == "confirmation"}
    assert new_disc_ids == old_disc_ids
    assert new_conf_ids.isdisjoint(old_disc_ids)
    assert new_conf_ids.isdisjoint(old_conf_ids)
    assert ts_refreshed.manifest.cases_sha256 != old_sha


def test_review_import_rejects_unknown_duplicate_and_invalid_tokens(tmp_path: Path):
    """Review import fails fast on unknown IDs, duplicate IDs, and malformed accept values."""
    c1 = TestCase(
        inputs={"query": "hello"},
        target="ok",
        checks=[CheckSpec(kind="exact", value="ok")],
        kind="baseline",
        split="discovery",
    )
    ts = TestSet.create([c1], freeze=True)

    out_csv = tmp_path / "r.csv"
    export_review(ts, out_csv, fmt="csv")

    bad_id_csv = tmp_path / "bad_id.csv"
    bad_id_csv.write_text(
        "id,accept,review\ntc_unknown,yes,pending\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Unknown case ID"):
        import_review(ts, bad_id_csv)

    dup_csv = tmp_path / "dup.csv"
    dup_csv.write_text(
        f"id,accept,review\n{c1.id},yes,pending\n{c1.id},no,pending\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Duplicate case ID"):
        import_review(ts, dup_csv)

    bad_tok_csv = tmp_path / "bad_tok.csv"
    bad_tok_csv.write_text(
        f"id,accept,review\n{c1.id},maybe,pending\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Invalid accept value"):
        import_review(ts, bad_tok_csv)


def test_cli_diagnose_with_tests_coverage_gate_and_guards(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """`prism diagnose --tests` resolves factor names/IDs, enforces freeze/integrity, and guards split leakage."""
    f_a = Factor.binary(
        id="A", name="persona", level_0_content="", level_1_content="Be an expert."
    )
    f_b = Factor.binary(
        id="B",
        name="rare_rule",
        level_0_content="",
        level_1_content="Return NULL if date missing.",
    )
    fset = FactorSet([f_a, f_b])
    tmpl = PromptTemplate.from_factors(fset, data_template="Input: {{ text }}")

    cases: List[TestCase] = []
    # 30 discovery baseline + 10 confirmation baseline
    for i in range(40):
        cases.append(
            TestCase(
                inputs={"text": f"baseline_{i}"},
                target="ok",
                checks=[CheckSpec(kind="exact", value="ok")],
                kind="baseline",
                split="discovery" if i < 30 else "confirmation",
            )
        )
    # 25 discovery targeted + 10 confirmation targeted for B (rare_rule); 0 targeted for A (persona)
    for i in range(35):
        cases.append(
            TestCase(
                inputs={"text": f"missing_date_{i}"},
                target="NULL",
                checks=[CheckSpec(kind="exact", value="NULL")],
                kind="targeted",
                intended_factor="B",
                tagger_factors=["B"],
                tagger_agrees=True,
                targets_factors=["B"],
                split="discovery" if i < 25 else "confirmation",
            )
        )

    ts = TestSet.create(
        cases,
        factors=fset,
        template=tmpl,
        factor_roles={"A": "global_scaffold", "B": "conditional_rule"},
        generator={"model_id": "gen-model-x", "tagger_model_id": "tag-model-y"},
        freeze=True,
    )
    tests_dir = tmp_path / "frozen_tests"
    ts.save(tests_dir)

    # Build results CSV using FACTOR NAMES ('persona', 'rare_rule') to test ID<->name resolution!
    rng = np.random.default_rng(123)
    disc_cases = ts.cases_for_split("discovery")
    conf_cases = ts.cases_for_split("confirmation")

    def _build_rows(case_list: List[TestCase]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        combos = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for run_id, (a_val, b_val) in enumerate(combos):
            for c in case_list:
                if c.kind == "targeted" and c.intended_factor == "B":
                    score = 1.0 if b_val == 1 else 0.0
                else:
                    # Baseline: neither factor has effect
                    score = float(np.clip(0.80 + rng.normal(0.0, 0.002), 0.0, 1.0))
                rows.append(
                    {
                        "run_id": run_id,
                        "sample_id": c.id,
                        "persona": a_val,
                        "rare_rule": b_val,
                        "check_suite": score,
                    }
                )
        return rows

    disc_csv = tmp_path / "disc_results.csv"
    pd.DataFrame(_build_rows(disc_cases)).to_csv(disc_csv, index=False)

    out_md = tmp_path / "audit.md"
    out_json = tmp_path / "audit.json"

    # 1. Clean discovery run with --tests and --model-id matching generator -> KEEP_TARGETED + UNTESTED + gen==model warning
    cli_main(
        [
            "diagnose",
            "--data",
            str(disc_csv),
            "--target",
            "check_suite",
            "--tests",
            str(tests_dir),
            "--model-id",
            "gen-model-x",
            "--token-costs",
            "A=40,B=55",
            "--sign-flip",
            "200",
            "--output-report",
            str(out_md),
            "--json-out",
            str(out_json),
        ]
    )
    capsys.readouterr()

    audit_obj = json.loads(out_json.read_text(encoding="utf-8"))
    md_text = out_md.read_text(encoding="utf-8")
    assert audit_obj["schema_version"] == "2.1.0"
    assert audit_obj["evidence_tier"] == "predicted (synthetic tests)"
    assert audit_obj["baseline_only_full_set"] is True
    assert audit_obj["test_set_sha256"] == ts.manifest.cases_sha256
    assert audit_obj["generator_model_id"] == "gen-model-x"
    assert audit_obj["model_id"] == "gen-model-x"
    assert any("equals model under test" in w for w in audit_obj["warnings"])
    assert not any("Split leakage" in w for w in audit_obj["warnings"])

    by_f = {d["factor_id"]: d for d in audit_obj["factors"]}
    assert by_f["rare_rule"]["pruning_verdict"] == "KEEP_TARGETED"
    assert by_f["rare_rule"]["n_targeted"] == 25
    assert (
        by_f["rare_rule"]["token_cost"] == 55
    )  # resolved from A/B -> persona/rare_rule
    assert by_f["persona"]["pruning_verdict"] == "UNTESTED"
    assert by_f["persona"]["n_targeted"] == 0
    assert "Covered N" in md_text
    assert "NI p (T)" in md_text

    # 2. Split-leakage guard: mixed discovery + confirmation CSV surfaces prominent warning
    mixed_csv = tmp_path / "mixed_results.csv"
    pd.DataFrame(_build_rows(disc_cases + conf_cases)).to_csv(mixed_csv, index=False)
    mixed_json = tmp_path / "mixed_audit.json"
    cli_main(
        [
            "diagnose",
            "--data",
            str(mixed_csv),
            "--target",
            "check_suite",
            "--tests",
            str(tests_dir),
            "--sign-flip",
            "0",
            "--json-out",
            str(mixed_json),
        ]
    )
    mixed_stdout = capsys.readouterr().out
    mixed_obj = json.loads(mixed_json.read_text(encoding="utf-8"))
    assert any("Split leakage detected" in w for w in mixed_obj["warnings"])
    assert "Split leakage detected" in mixed_stdout

    # 3. Confirmation-only data with default --split discovery -> helpful split hint; --split confirmation works
    conf_csv = tmp_path / "conf_results.csv"
    pd.DataFrame(_build_rows(conf_cases)).to_csv(conf_csv, index=False)
    with pytest.raises(SystemExit) as exc_split:
        cli_main(
            [
                "diagnose",
                "--data",
                str(conf_csv),
                "--target",
                "check_suite",
                "--tests",
                str(tests_dir),
                "--sign-flip",
                "0",
            ]
        )
    assert exc_split.value.code == 1
    assert "--split confirmation" in capsys.readouterr().err

    cli_main(
        [
            "diagnose",
            "--data",
            str(conf_csv),
            "--target",
            "check_suite",
            "--tests",
            str(tests_dir),
            "--split",
            "confirmation",
            "--min-targeted-items",
            "5",
            "--sign-flip",
            "0",
        ]
    )
    assert "KEEP_TARGETED" in capsys.readouterr().out

    # 4. Unfrozen draft refusal (and --allow-unfrozen override)
    draft_dir = tmp_path / "draft_tests"
    ts_draft = TestSet.create(cases, factors=fset, template=tmpl, freeze=False)
    ts_draft.save(draft_dir)

    with pytest.raises(SystemExit) as exc_draft:
        cli_main(
            [
                "diagnose",
                "--data",
                str(disc_csv),
                "--target",
                "check_suite",
                "--tests",
                str(draft_dir),
                "--sign-flip",
                "0",
            ]
        )
    assert exc_draft.value.code == 1
    assert "not frozen" in capsys.readouterr().err

    cli_main(
        [
            "diagnose",
            "--data",
            str(disc_csv),
            "--target",
            "check_suite",
            "--tests",
            str(draft_dir),
            "--allow-unfrozen",
            "--sign-flip",
            "0",
        ]
    )
    assert "KEEP_TARGETED" in capsys.readouterr().out

    # 5. Tampered frozen TestSet refusal
    tampered_cases = tests_dir / "cases.jsonl"
    tampered_cases.write_text(
        tampered_cases.read_text(encoding="utf-8").replace("baseline_0", "HACKED_0"),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc_tamper:
        cli_main(
            [
                "diagnose",
                "--data",
                str(disc_csv),
                "--target",
                "check_suite",
                "--tests",
                str(tests_dir),
                "--sign-flip",
                "0",
            ]
        )
    assert exc_tamper.value.code == 1
    assert "integrity check failed" in capsys.readouterr().err


def test_cli_tests_subprocess_smoke(tmp_path: Path):
    """Real subprocess execution of `python -m prompt_prism.cli.main tests ...`."""
    _write_mock_client_module(tmp_path, module_name="subproc_clients")
    cfg = _write_sample_yaml_configs(tmp_path)
    out_dir = tmp_path / "sub_tests"

    env = dict(os.environ)
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{tmp_path}{os.pathsep}{existing_pp}" if existing_pp else str(tmp_path)
    )

    gen_res = subprocess.run(
        [
            sys.executable,
            "-m",
            "prompt_prism.cli.main",
            "tests",
            "generate",
            "--factors",
            str(cfg["factors"]),
            "--template",
            str(cfg["template"]),
            "--n-baseline-discovery",
            "4",
            "--n-baseline-confirmation",
            "2",
            "--n-per-factor-discovery",
            "2",
            "--n-per-factor-confirmation",
            "2",
            "--client",
            "subproc_clients:generator_fn",
            "--tagger-client",
            "subproc_clients:tagger_instance",
            "--out",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert gen_res.returncode == 0, gen_res.stderr
    assert "Generated TestSet" in gen_res.stdout
    assert (out_dir / "cases.jsonl").exists()
    assert (out_dir / "manifest.json").exists()

    stats_res = subprocess.run(
        [
            sys.executable,
            "-m",
            "prompt_prism.cli.main",
            "tests",
            "stats",
            str(out_dir),
            "--min-targeted-items",
            "2",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert stats_res.returncode == 0, stats_res.stderr
    assert "Per-Factor Coverage" in stats_res.stdout
    assert "conditional_rule" in stats_res.stdout
    assert "global_scaffold" in stats_res.stdout


def test_pr4_review_hardening_null_yaml_bom_review_conflicts_and_split_hint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """Regression coverage for PR 4 review findings across loaders, review, and CLI."""
    # 1. YAML nulls don't stringify to 'None' or crash int(None)
    null_yaml = tmp_path / "null_factors.yaml"
    null_yaml.write_text(
        """
factors:
  - id:
    name: persona
    level_0:
    level_1: "Be concise."
    level_0_name:
    position:
    default_level_code:
""".lstrip(),
        encoding="utf-8",
    )
    fset_null = load_factors_file(null_yaml)
    assert fset_null["A"].id == "A"
    assert fset_null["A"].get_level(0).content == ""
    assert fset_null["A"].get_level(0).name == "Off"
    assert fset_null["A"].position == 0

    # Explicit vs auto-assigned ID collision raises ValueError
    collide_yaml = tmp_path / "collide_factors.yaml"
    collide_yaml.write_text(
        """
factors:
  - name: first_auto
    level_1: "one"
  - id: A
    name: second_explicit
    level_1: "two"
""".lstrip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Factor ID collision"):
        load_factors_file(collide_yaml)

    # 2. Malformed YAML wraps YAMLError into ValueError (and CLI exits cleanly)
    bad_yaml = tmp_path / "broken.yaml"
    bad_yaml.write_text("factors:\n  - [unclosed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Failed to parse"):
        load_factors_file(bad_yaml)

    bad_sec_tmpl = tmp_path / "bad_sec.yaml"
    bad_sec_tmpl.write_text("sections:\n  - not_a_dict\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_template_file(bad_sec_tmpl)

    with pytest.raises(SystemExit) as exc_yaml:
        cli_main(
            [
                "tests",
                "generate",
                "--factors",
                str(bad_yaml),
                "--template",
                str(bad_sec_tmpl),
                "--client",
                "unused:fn",
                "--out",
                str(tmp_path / "never"),
            ]
        )
    assert exc_yaml.value.code == 1
    assert "Failed to parse" in capsys.readouterr().err

    # 3. Tagger role error names --tagger-client / --tagger-model; bool usage ignored
    with pytest.raises(ValueError, match="--tagger-client"):
        resolve_cli_client(client_spec="a:b", model="m", role="tagger")
    assert LiteLLMClient._extract_usage(
        {"usage": {"prompt_tokens": True, "total_tokens": 12}}
    ) == {"total_tokens": 12}

    # 4. CSV uses clean LF newlines & Excel UTF-8 BOM imports cleanly
    c_acc = TestCase(
        inputs={"query": "hello"},
        target="ok",
        checks=[CheckSpec(kind="exact", value="ok")],
        kind="targeted",
        intended_factor="A",
        tagger_factors=["A"],
        tagger_agrees=True,
        targets_factors=["A"],
        review="accepted",
        split=None,
    )
    ts = TestSet.create(
        [c_acc],
        factors=fset_null,
        factor_roles={"A": "conditional_rule"},
        freeze=False,
    )

    csv_p = tmp_path / "exported.csv"
    export_review(ts, csv_p, fmt="csv")
    raw_bytes = csv_p.read_bytes()
    assert b"\r" not in raw_bytes

    # 5a. Editing `review` on already-accepted row (leaving prefilled accept=yes) applies edit; BOM handled
    bom_csv = tmp_path / "excel_bom.csv"
    bom_csv.write_bytes(
        b"\xef\xbb\xbf"
        + f"id, review, accept\n{c_acc.id},rejected,yes\n".encode("utf-8")
    )
    res_bom = import_review(ts, bom_csv, freeze=False)
    assert res_bom["rejected"] == 1
    assert ts.cases[0].review == "rejected"

    # 5b. Conflicting simultaneous edits raise ValueError (CSV & Markdown)
    conflict_csv = tmp_path / "conflict.csv"
    conflict_csv.write_text(
        f"id,review,accept\n{c_acc.id},pending,yes\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Conflicting review edits"):
        import_review(ts, conflict_csv, freeze=False)

    conflict_md = tmp_path / "conflict.md"
    conflict_md.write_text(
        f"## Case `{c_acc.id}`\n\n- [x] accept: no\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Conflicting checkbox"):
        import_review(ts, conflict_md, freeze=False)

    # Unbolded GFM checkbox `- [x] accept` works cleanly
    unbold_md = tmp_path / "unbold.md"
    unbold_md.write_text(f"## Case `{c_acc.id}`\n\n- [x] accept\n", encoding="utf-8")
    res_unbold = import_review(ts, unbold_md, freeze=False)
    assert res_unbold["accepted"] == 1

    # 5c. Unsplit TestSet stats includes Unassigned column and unsplit warning
    stats_text = format_testset_stats(ts, min_targeted_items=1)
    assert "Unassigned" in stats_text
    assert "no split assignment" in stats_text

    # 6. Targeted-only TestSet (no baseline items) still detects confirmation-vs-discovery split mismatch
    c_disc = TestCase(
        inputs={"query": "disc_only"},
        target="ok",
        checks=[CheckSpec(kind="exact", value="ok")],
        kind="targeted",
        intended_factor="A",
        tagger_factors=["A"],
        tagger_agrees=True,
        targets_factors=["A"],
        split="discovery",
    )
    c_conf = TestCase(
        inputs={"query": "conf_only"},
        target="ok",
        checks=[CheckSpec(kind="exact", value="ok")],
        kind="targeted",
        intended_factor="A",
        tagger_factors=["A"],
        tagger_agrees=True,
        targets_factors=["A"],
        split="confirmation",
    )
    ts_targ_only = TestSet.create([c_disc, c_conf], factors=fset_null, freeze=True)
    targ_dir = tmp_path / "targ_only_ts"
    ts_targ_only.save(targ_dir)

    conf_only_csv = tmp_path / "targ_conf.csv"
    pd.DataFrame(
        [
            {"run_id": 0, "sample_id": c_conf.id, "A": 0, "check_suite": 0.0},
            {"run_id": 1, "sample_id": c_conf.id, "A": 1, "check_suite": 1.0},
        ]
    ).to_csv(conf_only_csv, index=False)

    with pytest.raises(SystemExit) as exc_targ_split:
        cli_main(
            [
                "diagnose",
                "--data",
                str(conf_only_csv),
                "--target",
                "check_suite",
                "--tests",
                str(targ_dir),
                "--sign-flip",
                "0",
            ]
        )
    assert exc_targ_split.value.code == 1
    assert "--split confirmation" in capsys.readouterr().err
