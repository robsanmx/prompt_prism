"""
Review export/import (CSV & Markdown) and coverage statistics for PromptPrism TestSets.

Supports:
- Exporting cases sorted with tagger-disagreed and untagged items first (`csv` or `md`)
- Importing human/agent review decisions (`accepted` / `rejected` / `pending`), retaining
  rejected rows in `cases.jsonl` for audit, syncing effective coverage tags, and re-freezing
- Computing and formatting per-split, per-factor, clause-role, and judge-share stats
"""

from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

from tabulate import tabulate

from .models import SplitSelector, TestCase, TestSet

ReviewFormat = Literal["csv", "md"]

REVIEW_CSV_COLUMNS: List[str] = [
    "id",
    "split",
    "kind",
    "intended_factor",
    "intended_factor_name",
    "tagger_factors",
    "tagger_agrees",
    "inputs",
    "checks",
    "reference_pass",
    "reference_fail",
    "review",
    "accept",
]

_ACCEPT_TRUE_TOKENS = {"yes", "y", "true", "1", "accept", "accepted", "[x]", "x"}
_ACCEPT_FALSE_TOKENS = {
    "no",
    "n",
    "false",
    "0",
    "reject",
    "rejected",
    "[!]",
    "[-]",
    "-",
}
_ACCEPT_PENDING_TOKENS = {"", "pending", "[ ]"}
_VALID_REVIEW_STATUSES = {"pending", "accepted", "rejected"}

_MD_CASE_HEADER_RE = re.compile(r"^##\s+Case\s+`?([^\s`]+)`?\s*$")
_MD_GFM_ACCEPT_RE = re.compile(
    r"^\s*-\s*\[([ xX!\-])\]\s*(?:\*\*)?accept:?(?:\*\*)?:?\s*(.*)$", re.IGNORECASE
)
_MD_ACCEPT_RE = re.compile(r"^\s*-\s*\*\*accept:?\*\*:?\s*(.*)$", re.IGNORECASE)
_MD_REVIEW_RE = re.compile(r"^\s*-\s*\*\*review:?\*\*:?\s*(.*)$", re.IGNORECASE)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->")


def _clean_md_value(raw: str) -> str:
    cleaned = _HTML_COMMENT_RE.sub("", raw).strip()
    if len(cleaned) >= 2 and cleaned.startswith("`") and cleaned.endswith("`"):
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _detect_format(path: Path, text: Optional[str] = None) -> ReviewFormat:
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        return "md"
    if suffix == ".csv":
        return "csv"
    if text is not None:
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("# ") or stripped.startswith("## Case "):
                return "md"
            break
    return "csv"


def sort_cases_for_review(cases: Sequence[TestCase]) -> List[TestCase]:
    """Sort cases so tagger-disagreed and untagged targeted items appear first."""

    def _key(c: TestCase) -> Tuple[int, str, str, str]:
        if c.tagger_agrees is False:
            prio = 0
        elif c.kind == "targeted" and c.tagger_agrees is None:
            prio = 1
        elif c.kind == "targeted":
            prio = 2
        else:
            prio = 3
        return (prio, c.split or "", c.intended_factor or "", c.id)

    return sorted(cases, key=_key)


def _default_accept_token(review_status: str, *, fmt: ReviewFormat) -> str:
    if fmt == "md":
        if review_status == "accepted":
            return "[x]"
        if review_status == "rejected":
            return "[!]"
        return "[ ]"
    if review_status == "accepted":
        return "yes"
    if review_status == "rejected":
        return "no"
    return ""


def _render_markdown_review(
    cases: Sequence[TestCase], factor_id_to_name: Dict[str, str]
) -> str:
    lines: List[str] = [
        "# PromptPrism Test Set Review",
        "",
        "> Edit each case's **accept** line to `[x]` / `yes` (accept), `[!]` / `no` (reject), or leave `[ ]` (unchanged).",
        "> Tagger-disagreed and untagged targeted cases are listed first.",
        "",
    ]
    for c in cases:
        intended_label = ""
        if c.intended_factor:
            fname = factor_id_to_name.get(c.intended_factor)
            intended_label = (
                f"`{c.intended_factor}` ({fname})"
                if fname
                else f"`{c.intended_factor}`"
            )
        else:
            intended_label = "`-`"

        inputs_json = json.dumps(c.inputs, sort_keys=True, ensure_ascii=False)
        checks_json = json.dumps(
            [chk.model_dump(mode="json") for chk in c.checks],
            sort_keys=True,
            ensure_ascii=False,
        )
        tagger_json = json.dumps(c.tagger_factors, ensure_ascii=False)
        ref_pass = (
            json.dumps(c.reference_pass, ensure_ascii=False)
            if c.reference_pass is not None
            else "null"
        )
        ref_fail = (
            json.dumps(c.reference_fail, ensure_ascii=False)
            if c.reference_fail is not None
            else "null"
        )
        accept_tok = _default_accept_token(c.review, fmt="md")
        agrees_str = "null" if c.tagger_agrees is None else str(c.tagger_agrees)

        lines.extend(
            [
                f"## Case `{c.id}`",
                "",
                f"- **accept:** `{accept_tok}` <!-- set to [x] / yes (accept) or [!] / no (reject) -->",
                f"- **review:** `{c.review}`",
                f"- **split:** `{c.split or 'unassigned'}`",
                f"- **kind:** `{c.kind}`",
                f"- **intended_factor:** {intended_label}",
                f"- **tagger_agrees:** `{agrees_str}`",
                f"- **tagger_factors:** `{tagger_json}`",
                f"- **inputs:** `{inputs_json}`",
                f"- **checks:** `{checks_json}`",
                f"- **reference_pass:** `{ref_pass}`",
                f"- **reference_fail:** `{ref_fail}`",
                "",
            ]
        )
    return "\n".join(lines)


def export_review(
    test_set: TestSet,
    out_path: Union[str, Path],
    *,
    fmt: Optional[ReviewFormat] = None,
    split: SplitSelector = "all",
    only_disagreed: bool = False,
) -> Dict[str, Any]:
    """Export test cases for human or agent review in CSV or Markdown format."""
    target_path = Path(out_path)
    resolved_fmt: ReviewFormat = fmt or _detect_format(target_path)
    if resolved_fmt not in ("csv", "md"):
        raise ValueError(
            f"Unsupported review format {resolved_fmt!r}; expected 'csv' or 'md'."
        )

    selected = test_set.cases_for_split(split=split, include_rejected=True)
    if only_disagreed:
        selected = [
            c
            for c in selected
            if c.tagger_agrees is False
            or (c.kind == "targeted" and c.tagger_agrees is None)
        ]

    ordered = sort_cases_for_review(selected)
    id_to_name = dict(test_set.manifest.factor_id_to_name)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if resolved_fmt == "csv":
        with target_path.open("w", encoding="utf-8", newline="") as out_f:
            writer = csv.DictWriter(
                out_f, fieldnames=REVIEW_CSV_COLUMNS, lineterminator="\n"
            )
            writer.writeheader()
            for c in ordered:
                writer.writerow(
                    {
                        "id": c.id,
                        "split": c.split or "",
                        "kind": c.kind,
                        "intended_factor": c.intended_factor or "",
                        "intended_factor_name": (
                            id_to_name.get(c.intended_factor, "")
                            if c.intended_factor
                            else ""
                        ),
                        "tagger_factors": json.dumps(
                            c.tagger_factors, ensure_ascii=False
                        ),
                        "tagger_agrees": (
                            "" if c.tagger_agrees is None else str(c.tagger_agrees)
                        ),
                        "inputs": json.dumps(
                            c.inputs, sort_keys=True, ensure_ascii=False
                        ),
                        "checks": json.dumps(
                            [chk.model_dump(mode="json") for chk in c.checks],
                            sort_keys=True,
                            ensure_ascii=False,
                        ),
                        "reference_pass": (
                            "" if c.reference_pass is None else c.reference_pass
                        ),
                        "reference_fail": (
                            "" if c.reference_fail is None else c.reference_fail
                        ),
                        "review": c.review,
                        "accept": _default_accept_token(c.review, fmt="csv"),
                    }
                )
    else:
        md_text = _render_markdown_review(ordered, id_to_name)
        target_path.write_text(md_text, encoding="utf-8")

    n_disagreed = sum(1 for c in ordered if c.tagger_agrees is False)
    return {
        "path": str(target_path),
        "format": resolved_fmt,
        "exported": len(ordered),
        "disagreed": n_disagreed,
    }


def parse_review_decision(
    accept_val: Optional[str],
    review_val: Optional[str],
    current_review: str,
    *,
    case_id: str,
) -> Literal["pending", "accepted", "rejected"]:
    """Resolve a reviewer's `accept` / `review` fields into a valid review status."""
    acc = _clean_md_value(str(accept_val or "")).lower()
    rev = _clean_md_value(str(review_val or "")).lower()

    acc_target: Optional[Literal["pending", "accepted", "rejected"]]
    if acc in _ACCEPT_TRUE_TOKENS:
        acc_target = "accepted"
    elif acc in _ACCEPT_FALSE_TOKENS:
        acc_target = "rejected"
    elif acc == "pending":
        acc_target = "pending"
    elif acc in _ACCEPT_PENDING_TOKENS:
        acc_target = None
    else:
        raise ValueError(
            f"Invalid accept value {accept_val!r} for case {case_id!r}. "
            "Use yes/[x]/accept, no/[!]/reject, or blank/[ ]/pending."
        )

    rev_target: Optional[Literal["pending", "accepted", "rejected"]] = None
    if rev:
        if rev not in _VALID_REVIEW_STATUSES:
            raise ValueError(
                f"Invalid review status {review_val!r} for case {case_id!r}; "
                f"expected one of {sorted(_VALID_REVIEW_STATUSES)}."
            )
        rev_target = rev  # type: ignore[assignment]

    acc_changed = acc_target is not None and acc_target != current_review
    rev_changed = rev_target is not None and rev_target != current_review

    if acc_changed and rev_changed and acc_target != rev_target:
        raise ValueError(
            f"Conflicting review edits for case {case_id!r}: "
            f"accept={accept_val!r} (-> {acc_target}) vs review={review_val!r} (-> {rev_target})."
        )
    if acc_changed and acc_target is not None:
        return acc_target
    if rev_changed and rev_target is not None:
        return rev_target
    return current_review  # type: ignore[return-value]


def _parse_csv_review(text: str) -> List[Tuple[str, str, str]]:
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("CSV review file must contain an 'id' column.")
    cleaned_fields = [f.strip().lstrip("\ufeff") for f in reader.fieldnames]
    reader.fieldnames = cleaned_fields
    if "id" not in cleaned_fields:
        raise ValueError("CSV review file must contain an 'id' column.")
    if "accept" not in cleaned_fields and "review" not in cleaned_fields:
        raise ValueError(
            "CSV review file must contain an 'accept' and/or 'review' column."
        )
    rows: List[Tuple[str, str, str]] = []
    for row in reader:
        cid = str(row.get("id") or "").strip()
        if not cid:
            continue
        rows.append((cid, str(row.get("accept") or ""), str(row.get("review") or "")))
    return rows


def _parse_markdown_review(text: str) -> List[Tuple[str, str, str]]:
    rows: List[Tuple[str, str, str]] = []
    current_id: Optional[str] = None
    current_accept = ""
    current_review = ""

    def _flush() -> None:
        if current_id is not None:
            rows.append((current_id, current_accept, current_review))

    for line in text.splitlines():
        hdr = _MD_CASE_HEADER_RE.match(line.strip())
        if hdr:
            _flush()
            current_id = hdr.group(1).strip()
            current_accept = ""
            current_review = ""
            continue
        if current_id is None:
            continue

        gfm_m = _MD_GFM_ACCEPT_RE.match(line)
        if gfm_m:
            box = gfm_m.group(1).lower()
            rest = _clean_md_value(gfm_m.group(2))
            rest_lower = rest.lower()
            if box == "x":
                if rest_lower in _ACCEPT_FALSE_TOKENS:
                    raise ValueError(
                        f"Conflicting checkbox '[x]' and value {rest!r} for case {current_id!r}."
                    )
                current_accept = "[x]"
            elif box in ("!", "-"):
                if rest_lower in _ACCEPT_TRUE_TOKENS:
                    raise ValueError(
                        f"Conflicting checkbox '[{box}]' and value {rest!r} for case {current_id!r}."
                    )
                current_accept = "[!]"
            elif rest:
                current_accept = rest
            else:
                current_accept = "[ ]"
            continue

        acc_m = _MD_ACCEPT_RE.match(line)
        if acc_m:
            current_accept = _clean_md_value(acc_m.group(1))
            continue

        rev_m = _MD_REVIEW_RE.match(line)
        if rev_m:
            current_review = _clean_md_value(rev_m.group(1))
            continue

    _flush()
    if not rows:
        raise ValueError("No '## Case <id>' sections found in Markdown review file.")
    return rows


def import_review(
    test_set: TestSet,
    review_path: Union[str, Path],
    *,
    fmt: Optional[ReviewFormat] = None,
    freeze: bool = True,
) -> Dict[str, Any]:
    """Import review decisions from CSV or Markdown, keep rejected rows, and re-freeze."""
    src_path = Path(review_path)
    if not src_path.exists():
        raise FileNotFoundError(f"Review file not found: {src_path}")

    text = src_path.read_text(encoding="utf-8-sig")
    resolved_fmt: ReviewFormat = fmt or _detect_format(src_path, text=text)
    if resolved_fmt not in ("csv", "md"):
        raise ValueError(
            f"Unsupported review format {resolved_fmt!r}; expected 'csv' or 'md'."
        )

    parsed_rows = (
        _parse_csv_review(text)
        if resolved_fmt == "csv"
        else _parse_markdown_review(text)
    )

    by_id: Dict[str, TestCase] = {c.id: c for c in test_set.cases}
    seen_ids: set[str] = set()
    updates: List[Tuple[TestCase, Literal["pending", "accepted", "rejected"]]] = []

    for cid, acc_raw, rev_raw in parsed_rows:
        if cid in seen_ids:
            raise ValueError(f"Duplicate case ID {cid!r} in review file '{src_path}'.")
        seen_ids.add(cid)
        case = by_id.get(cid)
        if case is None:
            raise ValueError(
                f"Unknown case ID {cid!r} in review file '{src_path}' (not in TestSet)."
            )
        new_status = parse_review_decision(acc_raw, rev_raw, case.review, case_id=cid)
        updates.append((case, new_status))

    changed = 0
    for case, new_status in updates:
        if case.review != new_status:
            changed += 1
        case.review = new_status
        # Sync effective targets_factors so human acceptance of tagger-disagreed cases
        # immediately restores intended_factor coverage (and reversion removes it).
        case.targets_factors = case.effective_targets_factors(
            require_tag_agreement=True
        )

    if freeze:
        test_set.freeze(force=True)
    else:
        test_set.manifest.frozen_at = None
        test_set.recompute_metadata()

    n_accepted = sum(1 for c in test_set.cases if c.review == "accepted")
    n_rejected = sum(1 for c in test_set.cases if c.review == "rejected")
    n_pending = sum(1 for c in test_set.cases if c.review == "pending")

    return {
        "format": resolved_fmt,
        "rows_read": len(parsed_rows),
        "updated": changed,
        "accepted": n_accepted,
        "rejected": n_rejected,
        "pending": n_pending,
        "frozen": test_set.is_frozen,
        "cases_sha256": test_set.manifest.cases_sha256,
        "active_cases_sha256": test_set.manifest.active_cases_sha256,
    }


def compute_testset_stats(
    test_set: TestSet, *, min_targeted_items: int = 20
) -> Dict[str, Any]:
    """Compute per-split, per-factor, clause-role, and judge-share statistics + warnings."""
    active = [c for c in test_set.cases if c.review != "rejected"]
    rejected = [c for c in test_set.cases if c.review == "rejected"]
    accepted = [c for c in test_set.cases if c.review == "accepted"]
    pending = [c for c in test_set.cases if c.review == "pending"]

    split_buckets: Dict[str, List[TestCase]] = {
        "discovery": [],
        "confirmation": [],
        "unassigned": [],
    }
    for c in active:
        split_buckets[c.split or "unassigned"].append(c)

    splits_summary: Dict[str, Dict[str, Any]] = {}
    for s_name in ("discovery", "confirmation", "unassigned"):
        bucket = split_buckets[s_name]
        if s_name == "unassigned" and not bucket:
            continue
        n_tot = len(bucket)
        n_base = sum(1 for c in bucket if c.kind == "baseline")
        n_targ = sum(1 for c in bucket if c.kind == "targeted")
        n_judge = sum(
            1 for c in bucket if any(chk.kind == "llm_judge" for chk in c.checks)
        )
        splits_summary[s_name] = {
            "total": n_tot,
            "baseline": n_base,
            "targeted": n_targ,
            "judge_items": n_judge,
            "judge_fraction": float(n_judge / n_tot) if n_tot > 0 else 0.0,
        }

    all_judge = sum(
        1 for c in active if any(chk.kind == "llm_judge" for chk in c.checks)
    )
    judge_fractions = {
        "all": float(all_judge / len(active)) if active else 0.0,
        "discovery": splits_summary["discovery"]["judge_fraction"],
        "confirmation": splits_summary["confirmation"]["judge_fraction"],
    }

    # Collect all canonical factor IDs (manifest + any appearing on active cases)
    factor_ids: List[str] = list(test_set.manifest.factor_id_to_name.keys())
    known_fids = set(factor_ids)
    for c in active:
        candidates = list(c.effective_targets_factors(require_tag_agreement=True))
        if c.intended_factor:
            candidates.append(c.intended_factor)
        for raw_fid in candidates:
            canon = test_set._canonical_factor_key(raw_fid, key_by="id")
            if canon not in known_fids:
                known_fids.add(canon)
                factor_ids.append(canon)

    per_factor: List[Dict[str, Any]] = []
    has_discovery = len(split_buckets["discovery"]) > 0
    for fid in factor_ids:
        fname = test_set.manifest.factor_id_to_name.get(fid, fid)
        role = test_set.manifest.factor_roles.get(fid, "unknown")
        disc_cov = 0
        conf_cov = 0
        unass_cov = 0
        disc_int = 0
        conf_int = 0
        disagreed_pending = 0
        untagged = 0

        for c in active:
            eff_tags = {
                test_set._canonical_factor_key(t, key_by="id")
                for t in c.effective_targets_factors(require_tag_agreement=True)
            }
            if fid in eff_tags:
                if c.split == "discovery":
                    disc_cov += 1
                elif c.split == "confirmation":
                    conf_cov += 1
                else:
                    unass_cov += 1

            if c.kind == "targeted" and c.intended_factor:
                int_canon = test_set._canonical_factor_key(
                    c.intended_factor, key_by="id"
                )
                if int_canon == fid:
                    if c.split == "discovery":
                        disc_int += 1
                    elif c.split == "confirmation":
                        conf_int += 1
                    if c.tagger_agrees is False and c.review != "accepted":
                        disagreed_pending += 1
                    elif c.tagger_agrees is None:
                        untagged += 1

        effective_check_n = disc_cov if has_discovery else (disc_cov + unass_cov)
        per_factor.append(
            {
                "factor_id": fid,
                "factor_name": fname,
                "clause_role": role,
                "discovery_covered": disc_cov,
                "confirmation_covered": conf_cov,
                "unassigned_covered": unass_cov,
                "discovery_intended": disc_int,
                "confirmation_intended": conf_int,
                "tagger_disagreed": disagreed_pending,
                "tagger_untagged": untagged,
                "thin_discovery": effective_check_n < min_targeted_items,
            }
        )

    warnings_list: List[str] = []
    if not test_set.is_frozen:
        warnings_list.append(
            "TestSet is unfrozen (draft). Run `prism tests freeze <dir>` before running diagnose."
        )
    if split_buckets["unassigned"]:
        warnings_list.append(
            f"{len(split_buckets['unassigned'])} active item(s) have no split assignment (split=None); "
            "call split_cases() before running discovery/confirmation experiments."
        )

    for row in per_factor:
        cov_n = (
            row["discovery_covered"]
            if has_discovery
            else (row["discovery_covered"] + row["unassigned_covered"])
        )
        if cov_n == 0:
            warnings_list.append(
                f"Factor '{row['factor_id']}' ({row['factor_name']}) has 0 effective covered items "
                f"in discovery -> diagnose will mark it UNTESTED (or CONFLICT_REVIEW if baseline-harmful)."
            )
        elif cov_n < min_targeted_items:
            warnings_list.append(
                f"Factor '{row['factor_id']}' ({row['factor_name']}) has only {cov_n} effective "
                f"covered items in discovery (< min_targeted_items={min_targeted_items}) -> "
                "cannot reach SAFE_TO_PRUNE or unguarded REMOVE_HARMFUL."
            )

    total_disagreed = sum(r["tagger_disagreed"] for r in per_factor)
    if total_disagreed > 0:
        warnings_list.append(
            f"{total_disagreed} active targeted item(s) have pending tagger disagreements "
            "(excluded from strict coverage until accepted). Run `prism tests review export` to inspect."
        )

    total_untagged = sum(r["tagger_untagged"] for r in per_factor)
    if total_untagged > 0:
        warnings_list.append(
            f"{total_untagged} active targeted item(s) were not tagged by the blind tagger (tagger_agrees=None)."
        )

    if judge_fractions["discovery"] > 0.5:
        warnings_list.append(
            f"High LLM-judge share in discovery ({judge_fractions['discovery']:.0%}): "
            "prefer deterministic checks where possible to minimize judge cost and variance."
        )

    gen_meta = test_set.manifest.generator or {}
    if gen_meta.get("shortfalls"):
        warnings_list.append(
            f"Generator recorded post-filter shortfalls during creation: {gen_meta['shortfalls']}"
        )
    if gen_meta.get("confirmation_shortfalls"):
        warnings_list.append(
            f"Generator recorded post-filter shortfalls during confirmation refresh: "
            f"{gen_meta['confirmation_shortfalls']}"
        )

    return {
        "status": "frozen" if test_set.is_frozen else "draft",
        "frozen_at": test_set.manifest.frozen_at,
        "cases_sha256": test_set.manifest.cases_sha256,
        "active_cases_sha256": test_set.manifest.active_cases_sha256,
        "provenance": test_set.provenance,
        "generator_model_id": gen_meta.get("model_id"),
        "tagger_model_id": gen_meta.get("tagger_model_id"),
        "min_targeted_items": min_targeted_items,
        "counts": {
            "total": len(test_set.cases),
            "active": len(active),
            "accepted": len(accepted),
            "pending": len(pending),
            "rejected": len(rejected),
        },
        "judge_scored_fraction": judge_fractions,
        "splits": splits_summary,
        "factors": per_factor,
        "warnings": warnings_list,
    }


def format_testset_stats(test_set: TestSet, *, min_targeted_items: int = 20) -> str:
    """Render a concise human-readable statistics report for `prism tests stats`."""
    stats = compute_testset_stats(test_set, min_targeted_items=min_targeted_items)
    cnt = stats["counts"]
    lines: List[str] = [
        "=== PromptPrism TestSet Stats ===",
        f"Status:              {stats['status']}"
        + (f" ({stats['frozen_at']})" if stats["frozen_at"] else ""),
        f"Provenance:          {stats['provenance']}",
        f"Generator / Tagger:  {stats['generator_model_id'] or 'unknown'} / "
        f"{stats['tagger_model_id'] or 'unknown'}",
        f"Cases:               {cnt['active']} active ({cnt['accepted']} accepted, "
        f"{cnt['pending']} pending) / {cnt['rejected']} rejected / {cnt['total']} total",
        f"Judge-Scored Share:  all={stats['judge_scored_fraction']['all']:.1%}, "
        f"discovery={stats['judge_scored_fraction']['discovery']:.1%}, "
        f"confirmation={stats['judge_scored_fraction']['confirmation']:.1%}",
        f"cases_sha256:        {stats['cases_sha256']}",
        f"active_cases_sha256: {stats['active_cases_sha256']}",
        "",
        "Splits:",
    ]

    split_rows = [
        [
            s_name,
            s_info["total"],
            s_info["baseline"],
            s_info["targeted"],
            f"{s_info['judge_items']} ({s_info['judge_fraction']:.1%})",
        ]
        for s_name, s_info in stats["splits"].items()
    ]
    lines.append(
        tabulate(
            split_rows,
            headers=["Split", "Active", "Baseline", "Targeted", "Judge Items"],
            tablefmt="github",
        )
    )

    if stats["factors"]:
        lines.extend(
            ["", f"Per-Factor Coverage (min_targeted_items={min_targeted_items}):"]
        )
        has_unassigned = any(f["unassigned_covered"] > 0 for f in stats["factors"])
        factor_rows = []
        for f in stats["factors"]:
            status_str = "THIN" if f["thin_discovery"] else "OK"
            row = [
                f["factor_id"],
                f["factor_name"],
                f["clause_role"],
                f["discovery_covered"],
                f["confirmation_covered"],
            ]
            if has_unassigned:
                row.append(f["unassigned_covered"])
            row.extend([f["tagger_disagreed"], status_str])
            factor_rows.append(row)
        headers = [
            "ID",
            "Name",
            "Clause Role",
            "Disc. Covered",
            "Conf. Covered",
        ]
        if has_unassigned:
            headers.append("Unassigned")
        headers.extend(["Disagreed", "Gate Ready"])
        lines.append(
            tabulate(
                factor_rows,
                headers=headers,
                tablefmt="github",
            )
        )

    if stats["warnings"]:
        lines.extend(["", "Warnings:"])
        for w in stats["warnings"]:
            lines.append(f"  - WARNING: {w}")

    return "\n".join(lines)
