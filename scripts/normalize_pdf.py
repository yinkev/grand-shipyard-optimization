from __future__ import annotations

import os
from pathlib import Path

import pikepdf

FIXED_PDF_DATE = "D:20260815000000Z"


def normalize_pdf(path: str | Path) -> None:
    target = Path(path)
    temporary = target.with_name(f".{target.name}.normalized")
    temporary.unlink(missing_ok=True)

    with pikepdf.open(target) as pdf:
        pdf.docinfo["/CreationDate"] = FIXED_PDF_DATE
        pdf.docinfo["/ModDate"] = FIXED_PDF_DATE
        pdf.docinfo["/Creator"] = "Pandoc and wkhtmltopdf"
        pdf.docinfo["/Producer"] = "Grand Shipyard public release pipeline"
        if "/Metadata" in pdf.Root:
            del pdf.Root["/Metadata"]
        fixed_id = pikepdf.String(bytes.fromhex("6772616e642d73686970796172642d7631"))
        pdf.trailer["/ID"] = pikepdf.Array([fixed_id, fixed_id])
        pdf.save(
            temporary,
            deterministic_id=True,
            object_stream_mode=pikepdf.ObjectStreamMode.generate,
        )

    os.replace(temporary, target)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Normalize PDF metadata and identifiers.")
    parser.add_argument("pdf", type=Path)
    args = parser.parse_args()
    normalize_pdf(args.pdf)
