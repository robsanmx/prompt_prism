"""Build the post-compilation release manifest for the paper PDF."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict


ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper_hf"
RESULTS_DIR = ROOT / "research" / "hf_multi_prompt" / "results"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_release_manifest() -> Dict[str, object]:
    pdf_path = PAPER_DIR / "main.pdf"
    analysis_manifest_path = RESULTS_DIR / "summary_manifest.json"
    if not pdf_path.exists() or not analysis_manifest_path.exists():
        raise FileNotFoundError(
            "Compile paper_hf/main.pdf and generate results/summary_manifest.json first"
        )

    package_files = sorted((ROOT / "prompt_prism").rglob("*.py"))
    package_hashes = {
        path.relative_to(ROOT).as_posix(): sha256_file(path) for path in package_files
    }
    package_payload = "".join(
        f"{relative_path}:{digest}\n"
        for relative_path, digest in package_hashes.items()
    )

    return {
        "release_contract": "paper_hf_v4",
        "final_pdf": {
            "path": "paper_hf/main.pdf",
            "sha256": sha256_file(pdf_path),
            "bytes": pdf_path.stat().st_size,
        },
        "reader_friendly_pdf": (
            {
                "path": "paper_hf/reader_friendly.pdf",
                "sha256": sha256_file(PAPER_DIR / "reader_friendly.pdf"),
                "bytes": (PAPER_DIR / "reader_friendly.pdf").stat().st_size,
            }
            if (PAPER_DIR / "reader_friendly.pdf").exists()
            else None
        ),
        "analysis_manifest": {
            "path": "research/hf_multi_prompt/results/summary_manifest.json",
            "sha256": sha256_file(analysis_manifest_path),
        },
        "real_v4_prospective_manifest": (
            {
                "release_manifest_sha256": sha256_file(
                    ROOT / "research" / "results" / "real_v4" / "release_manifest.json"
                ),
                "summary_sha256": sha256_file(
                    ROOT / "research" / "results" / "real_v4" / "summary.json"
                ),
                "stability_certificate_sha256": sha256_file(
                    ROOT / "research" / "results" / "real_v4" / "stability_certificate.json"
                ),
                "claim_evidence_map_sha256": sha256_file(
                    ROOT / "research" / "results" / "real_v4" / "claim_evidence_map.csv"
                ),
            }
            if (ROOT / "research" / "results" / "real_v4" / "release_manifest.json").exists()
            else None
        ),
        "prompt_prism_source_manifest": {
            "file_count": len(package_hashes),
            "sha256": hashlib.sha256(package_payload.encode("utf-8")).hexdigest(),
            "files": package_hashes,
        },
    }


def main() -> None:
    output_path = PAPER_DIR / "release_manifest.json"
    output_path.write_text(
        json.dumps(build_release_manifest(), indent=2) + "\n", encoding="utf-8"
    )
    print(f"Release manifest written to {output_path}")


if __name__ == "__main__":
    main()
