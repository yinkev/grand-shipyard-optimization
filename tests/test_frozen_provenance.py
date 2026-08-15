from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("build_public_assets", ROOT / "scripts/build_public_assets.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
FROZEN_SOURCE_SHA256 = MODULE.FROZEN_SOURCE_SHA256
verify_frozen_source_hashes = MODULE.verify_frozen_source_hashes


def test_frozen_source_hashes_are_the_private_release_identities() -> None:
    assert FROZEN_SOURCE_SHA256 == {
        "solver/submission-10/c19_protocol.py": "eb10a979c47f73c02c84426c19d13f3361885987199eca6ed8fc5a500a1d9fc8",
        "solver/submission-10/c19_workers.py": "c99d487a490fb2673bbfeb8c86a892982128b8dc36f9cec7db7ccd948e0d5023",
        "solver/submission-10/candidate11_core.py": "d1902d6d2ef9d450eda73d6493a389fbaa031e60876bad33a1e83536a5912ed0",
        "solver/submission-10/candidate19_core.py": "104859099032abbdb1062b9e5061c032bc389af4e816b1aca9367929bfbdd3d6",
        "solver/submission-10/frontier_offense.py": "56d2f830d2f2bd708f0b42b1c3eef7dc35fd8b1b1e2fb4a63ebf7d2371b0d619",
        "solver/submission-10/myalgorithm.py": "238e8138a97e750c19c2bff176e3f59cf40eaa79733c32fc6d80d098e3ea3467",
        "solver/submission-10/submission8_floor.py": "993a3242eaa828260b7ed52cfe5b8cd425240c310c9af4cdb6493d0d628bf309",
        "solver/native-kernel/feasible_map_core.cpp": "a15c6c43c47deb81412126e4434ff61b2a266543fcc5cfb3acf4efe9f5fb1e1c",
        "solver/native-kernel/feasible_map_core.hpp": "ed5822a9de7415662d7e766aaad308e0a243c7c99c9a82c96b8b20b11f4263c8",
        "solver/native-kernel/module.cpp": "d1f2139480c016e039b9aba94df4c556adaade93c84a63d59406ae5b2d79b130",
        "solver/native-kernel/setup.py": "fb9cf91f62bdd420812ff45582f1132b8d1433e7e4a45fa65698f7a938ce374c",
    }
    verify_frozen_source_hashes(ROOT)


def test_manifest_builder_rejects_source_drift(tmp_path: Path) -> None:
    for relative in FROZEN_SOURCE_SHA256:
        source = ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    drifted = tmp_path / "solver/submission-10/myalgorithm.py"
    drifted.write_bytes(drifted.read_bytes() + b"\n# drift\n")
    with pytest.raises(RuntimeError, match="frozen source drift"):
        verify_frozen_source_hashes(tmp_path)


def test_public_asset_generation_is_deterministic() -> None:
    generated = [
        ROOT / "results/official-submissions.csv",
        ROOT / "reproducibility/source-manifest.json",
        ROOT / "figures/architecture.svg",
        ROOT / "figures/constraint-coupling.svg",
        ROOT / "figures/research-funnel.svg",
        ROOT / "figures/submission-progression.svg",
    ]
    MODULE.main()
    first = {path: path.read_bytes() for path in generated}
    MODULE.main()
    second = {path: path.read_bytes() for path in generated}
    assert first == second


def test_generated_svgs_have_no_trailing_whitespace() -> None:
    MODULE.main()
    violations = []
    for path in sorted((ROOT / "figures").glob("*.svg")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line != line.rstrip():
                violations.append(f"{path.name}:{line_number}")
    assert not violations, violations
