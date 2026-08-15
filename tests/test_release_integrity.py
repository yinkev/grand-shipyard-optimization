from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINAL_VECTOR = [11280, 31368, 103065, 6464170, 16443998, 42333422]
FINAL_ZIP_SHA256 = "0b3442ebc2513cd0bedece780f33331f719293513667f075bb3cface08c4cf4e"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_official_results_are_complete_and_arithmetically_consistent() -> None:
    path = ROOT / "results/official-evaluation-history.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10
    assert [int(rows[-1][f"p{i}"]) for i in range(1, 7)] == FINAL_VECTOR
    assert all(int(row["feasible_outputs"]) == 6 for row in rows)
    assert sum(int(row["feasible_outputs"]) for row in rows) == 60
    assert int(rows[0]["descriptive_raw_sum"]) == 91_216_059
    assert int(rows[-1]["descriptive_raw_sum"]) == 65_387_303
    reduction = 1 - int(rows[-1]["descriptive_raw_sum"]) / int(rows[0]["descriptive_raw_sum"])
    assert round(100 * reduction, 3) == 28.316


def test_public_source_matches_frozen_private_source_hashes() -> None:
    manifest = json.loads((ROOT / "reproducibility/source-manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["release"] == "1.1.0"
    assert manifest["public_name"] == "final solver"
    assert manifest["provenance"]["internal_development_identifier"] == "Candidate 23"
    assert manifest["provenance"]["chronological_evaluation_label"] == 10
    assert manifest["evaluation_history"]["private_evidence_id"] == "EVID-044"
    assert manifest["evaluation_history"]["sha256"] == sha256(
        ROOT / "results/official-evaluation-history.csv"
    )
    for relative, expected in manifest["released_source_sha256"].items():
        assert sha256(ROOT / relative) == expected
    assert manifest["private_submission_zip_sha256"] == FINAL_ZIP_SHA256
    assert manifest["submitted_binary_released"] is False


def test_public_tree_excludes_private_and_organizer_material() -> None:
    forbidden_suffixes = {".zip", ".xpr"}
    forbidden_names = {
        "utils.py",
        "problem-statement-v1.2.pdf",
        "CANDIDATE23_SUBMISSION.zip",
    }
    violations = []
    for path in ROOT.rglob("*"):
        if (
            not path.is_file()
            or ".git" in path.parts
            or ".venv" in path.parts
            or path == Path(__file__)
        ):
            continue
        if path.suffix in forbidden_suffixes or path.name in forbidden_names:
            violations.append(str(path.relative_to(ROOT)))
        if path.suffix in {".md", ".py", ".json", ".yaml", ".yml", ".csv", ".txt", ".cff"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "/Users/kyin/" in text or "sk-proj-" in text or "xpauth" in text:
                violations.append(str(path.relative_to(ROOT)))
    assert not violations, violations


def test_report_contains_required_claim_boundaries_and_semantic_names() -> None:
    report = (ROOT / "report/technical-retrospective.md").read_text(encoding="utf-8")
    required = (
        "60/60",
        "28.316%",
        "not the competition score",
        "exact final preliminary-round rank was not preserved",
        "AI systems were used throughout",
        "chronological labels used by this retrospective",
        "organizer-assigned submission identifiers",
        "final solver",
    )
    lower = " ".join(report.lower().split())
    for phrase in required:
        assert phrase.lower() in lower
    main_narrative = report.split("# Provenance appendix", 1)[0]
    assert "Candidate 23" not in main_narrative
    assert "Submission 10" not in main_narrative
    prohibited = ("competition-best", "finalist", "Top 40")
    for phrase in prohibited:
        assert phrase not in report


def test_readme_is_a_standalone_public_entry_point() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "report/technical-retrospective.pdf" in readme
    assert "60/60" in readme
    assert "final solver" in readme.lower()
    assert "AI-assisted" in readme
    assert "Candidate 23" not in readme[:1200]
    assert "Submission 10" not in readme[:1200]


def test_required_public_artifacts_exist() -> None:
    required = (
        "README.md",
        "report/technical-retrospective.md",
        "report/technical-retrospective.pdf",
        "results/official-evaluation-history.csv",
        "results/results-notes.md",
        "methodology/research-and-validation-process.md",
        "reproducibility/source-manifest.json",
        "reproducibility/release-boundary.md",
        "CITATION.cff",
        "LICENSE",
        "NOTICE.md",
        "figures/submission-progression.svg",
        "figures/architecture.svg",
        "figures/research-funnel.svg",
        "figures/constraint-coupling.svg",
        "solver/final-release/myalgorithm.py",
    )
    assert not [relative for relative in required if not (ROOT / relative).is_file()]
