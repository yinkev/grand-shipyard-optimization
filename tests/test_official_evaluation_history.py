from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = [
    (1, "2026-06-18 05:58:44", [11280, 31368, 130705, 10153470, 28945493, 51943743]),
    (2, "2026-07-05 16:32:16", [11280, 31368, 130705, 9593960, 28945493, 51943743]),
    (3, "2026-07-06 08:15:58", [11280, 31368, 130705, 9593960, 28945493, 51943743]),
    (4, "2026-07-13 02:42:13", [11280, 31368, 123880, 7992878, 26711863, 50955045]),
    (5, "2026-07-14 22:08:30", [11280, 31368, 105855, 7215499, 22333093, 47073282]),
    (6, "2026-07-16 08:48:35", [11280, 31368, 105855, 7215499, 22243379, 47073282]),
    (7, "2026-07-21 18:47:52", [11280, 31368, 103065, 7136114, 17492603, 41698677]),
    (8, "2026-07-24 20:13:57", [11280, 31368, 103065, 6187873, 17447412, 41698677]),
    (9, "2026-07-26 16:40:58", [11280, 31368, 110230, 6368452, 17447412, 41939247]),
    (10, "2026-07-28 04:55:41", [11280, 31368, 103065, 6464170, 16443998, 42333422]),
]


def test_public_history_matches_direct_organizer_email_matrix() -> None:
    path = ROOT / "results/official-evaluation-history.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10
    for row, (sequence, submitted_at, expected) in zip(rows, EXPECTED, strict=True):
        assert int(row["evaluation_sequence"]) == sequence
        assert row["submitted_at_utc"] == submitted_at
        actual = [int(row[f"p{i}"]) for i in range(1, 7)]
        assert actual == expected
        assert int(row["descriptive_raw_sum"]) == sum(expected)
        assert int(row["feasible_outputs"]) == 6


def test_chronology_is_labeled_as_retrospective_not_organizer_identity() -> None:
    report = (ROOT / "report/technical-retrospective.md").read_text(encoding="utf-8")
    notes = (ROOT / "results/results-notes.md").read_text(encoding="utf-8")
    phrase = "chronological labels used by this retrospective"
    report_text = " ".join(report.lower().split())
    notes_text = " ".join(notes.lower().split())
    assert phrase in report_text
    assert phrase in notes_text
    assert "organizer-assigned submission identifiers" in report_text
