from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results/official-evaluation-history.csv"

FROZEN_SOURCE_SHA256 = {
    "solver/final-release/c19_protocol.py": "eb10a979c47f73c02c84426c19d13f3361885987199eca6ed8fc5a500a1d9fc8",
    "solver/final-release/c19_workers.py": "c99d487a490fb2673bbfeb8c86a892982128b8dc36f9cec7db7ccd948e0d5023",
    "solver/final-release/candidate11_core.py": "d1902d6d2ef9d450eda73d6493a389fbaa031e60876bad33a1e83536a5912ed0",
    "solver/final-release/candidate19_core.py": "104859099032abbdb1062b9e5061c032bc389af4e816b1aca9367929bfbdd3d6",
    "solver/final-release/frontier_offense.py": "56d2f830d2f2bd708f0b42b1c3eef7dc35fd8b1b1e2fb4a63ebf7d2371b0d619",
    "solver/final-release/myalgorithm.py": "238e8138a97e750c19c2bff176e3f59cf40eaa79733c32fc6d80d098e3ea3467",
    "solver/final-release/submission8_floor.py": "993a3242eaa828260b7ed52cfe5b8cd425240c310c9af4cdb6493d0d628bf309",
    "solver/native-kernel/feasible_map_core.cpp": "a15c6c43c47deb81412126e4434ff61b2a266543fcc5cfb3acf4efe9f5fb1e1c",
    "solver/native-kernel/feasible_map_core.hpp": "ed5822a9de7415662d7e766aaad308e0a243c7c99c9a82c96b8b20b11f4263c8",
    "solver/native-kernel/module.cpp": "d1f2139480c016e039b9aba94df4c556adaade93c84a63d59406ae5b2d79b130",
    "solver/native-kernel/setup.py": "fb9cf91f62bdd420812ff45582f1132b8d1433e7e4a45fa65698f7a938ce374c",
}

# Direct transcriptions of the ten completed-evaluation emails retained in the
# registered Team Smoop mailbox. The sequence number is retrospective bookkeeping,
# not an organizer-assigned submission identifier.
EVALUATIONS = [
    (1, "2026-06-18 05:58:44", [11280, 31368, 130705, 10153470, 28945493, 51943743], "Initial feasible baseline"),
    (2, "2026-07-05 16:32:16", [11280, 31368, 130705, 9593960, 28945493, 51943743], "Instance-conditional pricing"),
    (3, "2026-07-06 08:15:58", [11280, 31368, 130705, 9593960, 28945493, 51943743], "Safety hardening; exact result repeat"),
    (4, "2026-07-13 02:42:13", [11280, 31368, 123880, 7992878, 26711863, 50955045], "Compiled geometry kernel"),
    (5, "2026-07-14 22:08:30", [11280, 31368, 105855, 7215499, 22333093, 47073282], "Versioned forbidden-map cache"),
    (6, "2026-07-16 08:48:35", [11280, 31368, 105855, 7215499, 22243379, 47073282], "Objective-aware constructor"),
    (7, "2026-07-21 18:47:52", [11280, 31368, 103065, 7136114, 17492603, 41698677], "Structure-gated constructor"),
    (8, "2026-07-24 20:13:57", [11280, 31368, 103065, 6187873, 17447412, 41698677], "Protected-floor refinement"),
    (9, "2026-07-26 16:40:58", [11280, 31368, 110230, 6368452, 17447412, 41939247], "Topology-based repair experiment"),
    (10, "2026-07-28 04:55:41", [11280, 31368, 103065, 6464170, 16443998, 42333422], "Final current-run portfolio"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen_source_hashes(root: Path = ROOT) -> None:
    drift = []
    for relative, expected in FROZEN_SOURCE_SHA256.items():
        path = root / relative
        if not path.is_file():
            drift.append(f"missing:{relative}")
            continue
        actual = sha256(path)
        if actual != expected:
            drift.append(f"hash:{relative}:{actual}")
    if drift:
        raise RuntimeError("frozen source drift: " + "; ".join(drift))


def write_results() -> None:
    with RESULTS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([
            "evaluation_sequence",
            "submitted_at_utc",
            "p1",
            "p2",
            "p3",
            "p4",
            "p5",
            "p6",
            "descriptive_raw_sum",
            "feasible_outputs",
            "stage",
        ])
        for sequence, timestamp, vector, stage in EVALUATIONS:
            writer.writerow([sequence, timestamp, *vector, sum(vector), 6, stage])


def write_manifest() -> None:
    verify_frozen_source_hashes(ROOT)
    payload = {
        "schema_version": 2,
        "release": "1.1.0",
        "public_name": "final solver",
        "provenance": {
            "internal_development_identifier": "Candidate 23",
            "chronological_evaluation_label": 10,
            "private_release_commit": "731b3ca9a59c898f4b097e95b75f43336ccb55ee",
            "private_release_tag": "candidate23-release-20260727",
            "private_evidence_correction_commit": "7841de97e22ce753667abfa236f76836c57511d3",
            "private_evidence_correction_tag": "ogc2026-evaluation-history-erratum-v1.0",
        },
        "evaluation_history": {
            "authority": "direct organizer completed-evaluation emails",
            "private_evidence_id": "EVID-044",
            "sequence_labels": "retrospective chronological labels",
            "sha256": sha256(RESULTS),
        },
        "private_submission_zip_sha256": "0b3442ebc2513cd0bedece780f33331f719293513667f075bb3cface08c4cf4e",
        "submitted_binary_sha256": "a1f0309bc7d30a6482528bf9a6b52107e02e622365a33c1517d65fbaa1636737",
        "submitted_binary_released": False,
        "released_source_sha256": dict(sorted(FROZEN_SOURCE_SHA256.items())),
        "excluded_material": [
            "organizer problem statement",
            "organizer checker and evaluation instances",
            "participant correspondence and certificate",
            "exact submission ZIP",
            "compiled Linux CPython 3.12 extension",
            "private experiment corpus and ResearchLab runs",
        ],
    }
    (ROOT / "reproducibility/source-manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _diagram_canvas(title: str, figsize=(10.5, 5.6)):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=16, pad=12)
    return fig, ax


def _box(ax, x, y, w, h, text, face="#F8FAFC", edge="#334155", fontsize=9.2, linewidth=1.2):
    from matplotlib.patches import FancyBboxPatch

    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=linewidth,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def _cluster(ax, x, y, w, h, label):
    from matplotlib.patches import FancyBboxPatch

    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=1.15,
        edgecolor="#94A3B8",
        facecolor="none",
        linestyle="--",
    )
    ax.add_patch(patch)
    ax.text(x + 0.012, y + h - 0.022, label, ha="left", va="top", fontsize=9.2, weight="bold")
    return patch


def _arrow(ax, a, b, label=None, dashed=False):
    style = "--" if dashed else "-"
    ax.annotate(
        "",
        xy=b,
        xytext=a,
        arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#64748B", linestyle=style),
    )
    if label:
        ax.text(
            (a[0] + b[0]) / 2,
            (a[1] + b[1]) / 2 + 0.014,
            label,
            ha="center",
            va="bottom",
            fontsize=7.4,
            color="#475569",
        )


def render_architecture() -> None:
    import matplotlib.pyplot as plt

    fig, ax = _diagram_canvas("Final solver architecture: two mutually exclusive CPU topologies", (12.8, 7.2))
    _box(ax, 0.015, 0.43, 0.12, 0.13, "Problem instance\n+ wall deadline", "#E0F2FE")
    _box(ax, 0.17, 0.43, 0.13, 0.13, "Structural regime gate\nblock count + pressure", "#E0F2FE")
    ax.text(0.235, 0.585, "one topology per instance", ha="center", va="bottom", fontsize=7.5, color="#475569")

    _cluster(ax, 0.34, 0.56, 0.33, 0.38, "Lower-pressure topology - four CPUs")
    lower = [
        (0.37, 0.75, "Protected feasible path\n2 CPUs"),
        (0.37, 0.63, "Calendar solver\n1 CPU"),
        (0.51, 0.63, "Calendar-seeded\njoint repair\n1 CPU"),
    ]
    for x, y, label in lower:
        _box(ax, x, y, 0.13, 0.09, label)

    _cluster(ax, 0.34, 0.06, 0.33, 0.43, "High-pressure topology - four one-CPU workers")
    high = [
        (0.37, 0.31, "Scored constructor"),
        (0.51, 0.31, "Calendar solver"),
        (0.37, 0.17, "Calendar-first\njoint repair"),
        (0.51, 0.17, "Barrier-crossing\nspace-time repair"),
    ]
    for x, y, label in high:
        _box(ax, x, y, 0.13, 0.09, label)

    _box(ax, 0.71, 0.39, 0.13, 0.20, "Atomic incumbent stream\ncomplete current-run\nsolutions only", "#FEF3C7")
    _box(ax, 0.865, 0.63, 0.12, 0.11, "Organizer checker\nfeasibility + objective", "#DCFCE7")
    _box(ax, 0.865, 0.43, 0.12, 0.11, "Best valid\ncurrent-run incumbent", "#DCFCE7")
    _box(ax, 0.865, 0.23, 0.12, 0.11, "Returned schedule")
    _box(ax, 0.71, 0.10, 0.13, 0.11, "Emergency feasible\nfallback", "#FEE2E2")

    _arrow(ax, (0.135, 0.495), (0.17, 0.495))
    _arrow(ax, (0.30, 0.52), (0.34, 0.76))
    _arrow(ax, (0.30, 0.47), (0.34, 0.28))
    for x, y, _ in lower + high:
        _arrow(ax, (x + 0.13, y + 0.045), (0.71, 0.49))
    _arrow(ax, (0.84, 0.52), (0.865, 0.685))
    _arrow(ax, (0.925, 0.63), (0.925, 0.54), "valid only")
    _arrow(ax, (0.925, 0.43), (0.925, 0.34))
    _arrow(ax, (0.135, 0.45), (0.71, 0.155), dashed=True)
    _arrow(ax, (0.84, 0.155), (0.865, 0.285), dashed=True)

    fig.savefig(FIGURES / "architecture.svg", metadata={"Date": None})
    fig.savefig(FIGURES / "architecture.png", dpi=220, metadata={"Date": None})
    plt.close(fig)


def render_constraint_coupling() -> None:
    import matplotlib.pyplot as plt

    fig, ax = _diagram_canvas("Why a good calendar can still fail in the yard", (11.2, 4.9))
    _box(ax, 0.03, 0.62, 0.18, 0.16, "Temporal schedule\nrelease, duration, due date", "#E0F2FE")
    _box(ax, 0.03, 0.22, 0.18, 0.16, "Spatial placement\nbay, orientation, anchor", "#E0F2FE")
    _box(ax, 0.30, 0.42, 0.17, 0.16, "Space-time residence\nENTRY -> EXIT occupancy", "#FEF3C7")
    _box(ax, 0.55, 0.62, 0.17, 0.16, "Directional crane\nreachability + precedence", "#FEF3C7")
    _box(ax, 0.55, 0.22, 0.17, 0.16, "Objective\ntardiness + imbalance + preference", "#F3E8FF")
    _box(ax, 0.79, 0.42, 0.18, 0.16, "Checker-valid\nrealizable schedule", "#DCFCE7")
    _box(ax, 0.30, 0.02, 0.17, 0.14, "Hard runtime\ncrash / timeout / infeasible = -1", "#FEE2E2")
    _arrow(ax, (0.21, 0.70), (0.30, 0.52))
    _arrow(ax, (0.21, 0.30), (0.30, 0.48))
    _arrow(ax, (0.47, 0.52), (0.55, 0.70))
    _arrow(ax, (0.21, 0.70), (0.55, 0.70))
    _arrow(ax, (0.21, 0.30), (0.55, 0.70))
    _arrow(ax, (0.47, 0.48), (0.55, 0.30))
    _arrow(ax, (0.72, 0.70), (0.79, 0.52))
    _arrow(ax, (0.72, 0.30), (0.79, 0.48))
    _arrow(ax, (0.47, 0.09), (0.79, 0.44))
    fig.savefig(FIGURES / "constraint-coupling.svg", metadata={"Date": None})
    fig.savefig(FIGURES / "constraint-coupling.png", dpi=220, metadata={"Date": None})
    plt.close(fig)


def render_research_funnel() -> None:
    import matplotlib.pyplot as plt

    fig, ax = _diagram_canvas("Research gate: prove opportunity before efficacy", (12.0, 4.9))
    nodes = [
        (0.02, "Question or\nmechanism claim", "#E0F2FE"),
        (0.18, "Opportunity /\nengagement gate", "#FEF3C7"),
        (0.34, "Bounded mechanism\nbuild", "#F8FAFC"),
        (0.50, "Paired efficacy +\npersisted replay", "#F8FAFC"),
        (0.66, "Retain, defer,\nor kill", "#FEE2E2"),
        (0.82, "Architecture\ndecision", "#DCFCE7"),
    ]
    for x, label, face in nodes:
        _box(ax, x, 0.42, 0.13, 0.18, label, face)
    for (x0, _, _), (x1, _, _) in zip(nodes, nodes[1:]):
        _arrow(ax, (x0 + 0.13, 0.51), (x1, 0.51))

    ax.text(0.50, 0.82, "49 separately reviewed investigations", ha="center", fontsize=11, weight="bold")
    ax.text(0.50, 0.74, "27 bounded tests  ·  13 closures  ·  9 deferrals", ha="center", fontsize=9.5)
    ax.text(0.50, 0.18, "Six selected opportunity-first gates reached valid terminal kills", ha="center", fontsize=10, weight="bold")
    ax.text(0.50, 0.10, "The terminal sprint used a lower-latency workflow when formal coordination cost became dominant.", ha="center", fontsize=8.8, color="#475569")

    fig.savefig(FIGURES / "research-funnel.svg", metadata={"Date": None})
    fig.savefig(FIGURES / "research-funnel.png", dpi=220, metadata={"Date": None})
    plt.close(fig)


def render_evaluation_figure() -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    numbers = [row[0] for row in EVALUATIONS]
    vectors = [row[2] for row in EVALUATIONS]
    sums = [sum(vector) for vector in vectors]
    baseline = sums[0]
    reduction = [(1 - value / baseline) * 100 for value in sums]

    fig, ax = plt.subplots(figsize=(9.4, 4.9))
    fig.subplots_adjust(left=0.10, right=0.88, top=0.88, bottom=0.22)
    ax.plot(numbers, sums, marker="o", linewidth=2.3)
    ax.set_title("Official hidden-evaluation history")
    ax.set_xlabel("Chronological official evaluation (retrospective labels)", labelpad=10)
    ax.set_ylabel("Descriptive raw objective sum")
    ax.set_xticks(numbers)
    ax.grid(axis="y", alpha=0.25)
    ax.ticklabel_format(axis="y", style="plain")
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
    for sequence in (1, 5, 8, 9, 10):
        value = sums[sequence - 1]
        label = "Final" if sequence == 10 else f"E{sequence}"
        ax.annotate(
            f"{label}\n{value:,.0f}",
            (sequence, value),
            xytext=(0, 7 if sequence == 1 else 9),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax2 = ax.twinx()
    ax2.plot(numbers, reduction, linestyle="--", linewidth=1.6, alpha=0.65)
    ax2.set_ylabel("Reduction vs first evaluation")
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    fig.text(
        0.49,
        0.055,
        "Sequence labels are retrospective. The raw sum is descriptive, not the competition score.",
        ha="center",
        fontsize=8,
    )
    fig.savefig(FIGURES / "submission-progression.svg", metadata={"Date": None})
    fig.savefig(FIGURES / "submission-progression.png", dpi=220, metadata={"Date": None})
    plt.close(fig)


def normalize_svg(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    path.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n", encoding="utf-8")


def main() -> None:
    import matplotlib as mpl

    mpl.rcParams["svg.hashsalt"] = "grand-shipyard-v1.1.0"
    FIGURES.mkdir(exist_ok=True)
    write_results()
    write_manifest()
    render_architecture()
    render_constraint_coupling()
    render_research_funnel()
    render_evaluation_figure()
    for path in sorted(FIGURES.glob("*.svg")):
        normalize_svg(path)


if __name__ == "__main__":
    main()
