from __future__ import annotations

import importlib.util
from pathlib import Path

import pikepdf

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("normalize_pdf", ROOT / "scripts/normalize_pdf.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
normalize_pdf = MODULE.normalize_pdf


def make_pdf(path: Path, created: str) -> None:
    pdf = pikepdf.new()
    pdf.add_blank_page(page_size=(100, 100))
    pdf.docinfo["/Title"] = "Determinism fixture"
    pdf.docinfo["/CreationDate"] = created
    pdf.docinfo["/ModDate"] = created
    pdf.save(path)


def test_normalizer_removes_timestamp_and_identifier_variance(tmp_path: Path) -> None:
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    make_pdf(first, "D:20260101000000Z")
    make_pdf(second, "D:20261231235959Z")

    normalize_pdf(first)
    normalize_pdf(second)

    assert first.read_bytes() == second.read_bytes()
    with pikepdf.open(first) as pdf:
        assert str(pdf.docinfo["/CreationDate"]) == "D:20260815000000Z"
        assert str(pdf.docinfo["/ModDate"]) == "D:20260815000000Z"
