from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"

FROZEN_SOURCE_SHA256 = {
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

SUBMISSIONS = [
    (1, "2026-06-21 18:05:18", [11280, 31368, 130705, 10153470, 28945493, 51943743], "Initial feasible baseline"),
    (2, "2026-06-25 16:25:35", [11280, 31368, 130705, 10153470, 28945493, 51943743], "Zero-loss retest"),
    (3, "2026-07-03 20:54:03", [11280, 31368, 106650, 8691553, 24764083, 49673443], "Early structural improvement"),
    (4, "2026-07-05 21:04:48", [11280, 31368, 107570, 8231627, 24608769, 50085398], "Tradeoff, not aggregate record"),
    (5, "2026-07-12 19:07:07", [11280, 31368, 105855, 7369833, 23166627, 47853733], "Native-kernel/cache release"),
    (6, "2026-07-16 16:15:31", [11280, 31368, 105855, 7215499, 22243379, 47073282], "Objective-aware constructor"),
    (7, "2026-07-23 03:03:39", [11280, 31368, 103065, 7136168, 17492599, 41698677], "Portfolio expansion"),
    (8, "2026-07-24 03:21:20", [11280, 31368, 103065, 6187873, 17447412, 41698677], "Strong protected floor"),
    (9, "2026-07-26 22:33:38", [11280, 31368, 103065, 7160951, 17449131, 43769078], "Candidate 14 hidden transfer failure"),
    (10, "2026-07-28 04:55:41", [11280, 31368, 103065, 6464170, 16443998, 42333422], "Candidate 23 terminal submission"),
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
    path = ROOT / "results/official-submissions.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "submission", "evaluated_at_utc", "p1", "p2", "p3", "p4", "p5", "p6",
            "raw_sum", "feasible_outputs", "note",
        ])
        for number, timestamp, vector, note in SUBMISSIONS:
            writer.writerow([number, timestamp, *vector, sum(vector), 6, note])


def write_manifest() -> None:
    verify_frozen_source_hashes(ROOT)
    payload = {
        "schema_version": 1,
        "release": "1.0.0",
        "candidate": "Candidate 23 / Submission 10",
        "private_release_commit": "731b3ca9a59c898f4b097e95b75f43336ccb55ee",
        "private_release_tag": "candidate23-release-20260727",
        "private_closeout_commit": "c38042b48e6d7cc2ad53cd56a4f5aa7d3e6c6059",
        "private_closeout_tag": "ogc2026-closeout-v1.0",
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
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

def _diagram_canvas(title: str, figsize=(10.5, 5.6)):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=16, pad=12)
    return fig, ax


def _box(ax, x, y, w, h, text, face="#F8FAFC", edge="#334155", fontsize=9.5):
    from matplotlib.patches import FancyBboxPatch
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.2, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def _arrow(ax, a, b, label=None, dashed=False):
    style = "--" if dashed else "-"
    ax.annotate(
        "", xy=b, xytext=a,
        arrowprops=dict(arrowstyle="-|>", lw=1.25, color="#64748B", linestyle=style),
    )
    if label:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.018, label,
                ha="center", va="bottom", fontsize=7.6, color="#475569")


def render_architecture() -> None:
    import matplotlib.pyplot as plt
    fig, ax = _diagram_canvas("Candidate 23: current-run, checker-gated portfolio", (12.2, 6.2))
    _box(ax, .02, .42, .12, .14, "Problem instance\n+ wall deadline", "#E0F2FE")
    _box(ax, .18, .42, .12, .14, "Deterministic\nregime gate", "#E0F2FE")
    roles = [
        (.36, .80, "Protected floor\n2 CPUs"),
        (.36, .62, "Scored constructor\n1 CPU"),
        (.36, .44, "Calendar solver\n1 CPU"),
        (.36, .26, "Calendar-first\njoint repair\n1 CPU"),
        (.36, .08, "C2PM calendar +\ncausal repair\n1 CPU"),
    ]
    for x, y, label in roles:
        _box(ax, x, y, .17, .13, label)
    _box(ax, .61, .42, .15, .18, "Atomic incumbent stream\nsource + objective + solution", "#FEF3C7")
    _box(ax, .81, .56, .15, .13, "Organizer checker\nfeasibility + objective", "#DCFCE7")
    _box(ax, .81, .31, .15, .13, "Best valid\ncurrent-run incumbent", "#DCFCE7")
    _box(ax, .61, .08, .15, .13, "Emergency feasible\nfallback", "#FEE2E2")
    _box(ax, .81, .08, .15, .13, "Returned schedule")
    _arrow(ax, (.14, .49), (.18, .49))
    for _, y, _ in roles:
        _arrow(ax, (.30, .49), (.36, y + .065))
        _arrow(ax, (.53, y + .065), (.61, .51))
    _arrow(ax, (.76, .51), (.81, .625))
    _arrow(ax, (.885, .56), (.885, .44), "valid only")
    _arrow(ax, (.14, .44), (.61, .145), dashed=True)
    _arrow(ax, (.76, .145), (.81, .145), dashed=True)
    _arrow(ax, (.885, .31), (.885, .21))
    fig.savefig(FIGURES / "architecture.svg", metadata={"Date": None})
    fig.savefig(FIGURES / "architecture.png", dpi=220, metadata={"Date": None})
    plt.close(fig)


def render_constraint_coupling() -> None:
    import matplotlib.pyplot as plt
    fig, ax = _diagram_canvas("Why independent scheduling and packing are insufficient", (11.2, 4.9))
    _box(ax, .03, .62, .18, .16, "Temporal schedule\nrelease, duration, due date", "#E0F2FE")
    _box(ax, .03, .22, .18, .16, "Spatial placement\nbay, orientation, anchor", "#E0F2FE")
    _box(ax, .30, .42, .17, .16, "Space-time residence\nENTRY → EXIT occupancy", "#FEF3C7")
    _box(ax, .55, .62, .17, .16, "Directional crane\nreachability + precedence", "#FEF3C7")
    _box(ax, .55, .22, .17, .16, "Objective\ntardiness + imbalance + preference", "#F3E8FF")
    _box(ax, .79, .42, .18, .16, "Checker-valid\nrealizable schedule", "#DCFCE7")
    _box(ax, .30, .02, .17, .14, "Hard runtime\ncrash / timeout / infeasible = -1", "#FEE2E2")
    _arrow(ax, (.21, .70), (.30, .52))
    _arrow(ax, (.21, .30), (.30, .48))
    _arrow(ax, (.47, .52), (.55, .70))
    _arrow(ax, (.21, .70), (.55, .70))
    _arrow(ax, (.21, .30), (.55, .70))
    _arrow(ax, (.47, .48), (.55, .30))
    _arrow(ax, (.72, .70), (.79, .52))
    _arrow(ax, (.72, .30), (.79, .48))
    _arrow(ax, (.47, .09), (.79, .44))
    fig.savefig(FIGURES / "constraint-coupling.svg", metadata={"Date": None})
    fig.savefig(FIGURES / "constraint-coupling.png", dpi=220, metadata={"Date": None})
    plt.close(fig)


def render_research_funnel() -> None:
    import matplotlib.pyplot as plt
    fig, ax = _diagram_canvas("Research governance and terminal sprint", (10.0, 6.2))
    _box(ax, .34, .82, .32, .11, "80 planned specialist roles", "#E0F2FE")
    _box(ax, .34, .65, .32, .11, "49 completed, reviewed runs\nB01–B09", "#E0F2FE")
    _box(ax, .10, .46, .32, .12, "27 bounded Test recommendations\n13 closures · 9 deferrals", "#FEF3C7")
    _box(ax, .10, .27, .32, .12, "6 selected opportunity-first\nempirical gates", "#FEF3C7")
    _box(ax, .10, .08, .32, .12, "6 valid terminal kills\nno formal-campaign promotion", "#FEE2E2")
    _box(ax, .58, .46, .32, .12, "Direct terminal sprint\nCandidates 9–23", "#F3E8FF")
    _box(ax, .58, .27, .32, .12, "Candidate 23\nSubmission 10", "#DCFCE7")
    _box(ax, .58, .08, .32, .12, "31 roles cancelled at closeout\nB10–B17", "#F1F5F9")
    _arrow(ax, (.50, .82), (.50, .76))
    _arrow(ax, (.42, .65), (.26, .58))
    _arrow(ax, (.26, .46), (.26, .39))
    _arrow(ax, (.26, .27), (.26, .20))
    _arrow(ax, (.58, .70), (.74, .58))
    ax.text(.77, .70, "workflow bypass under deadline", ha="center", va="bottom", fontsize=7.6, color="#475569")
    _arrow(ax, (.74, .46), (.74, .39))
    ax.plot([.66, .95, .95, .90], [.70, .70, .14, .14], linestyle="--", linewidth=1.25, color="#64748B")
    ax.annotate("", xy=(.90, .14), xytext=(.94, .14), arrowprops=dict(arrowstyle="-|>", lw=1.25, color="#64748B", linestyle="--"))
    fig.savefig(FIGURES / "research-funnel.svg", metadata={"Date": None})
    fig.savefig(FIGURES / "research-funnel.png", dpi=220, metadata={"Date": None})
    plt.close(fig)


def render_graphviz() -> None:
    render_architecture()
    render_constraint_coupling()
    render_research_funnel()

def render_submission_figure() -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    numbers = [row[0] for row in SUBMISSIONS]
    vectors = [row[2] for row in SUBMISSIONS]
    sums = [sum(v) for v in vectors]
    baseline = sums[0]
    reduction = [(1 - value / baseline) * 100 for value in sums]

    fig, ax = plt.subplots(figsize=(9.4, 4.7), constrained_layout=True)
    ax.plot(numbers, sums, marker="o", linewidth=2.3)
    ax.set_title("Official hidden-evaluation progression")
    ax.set_xlabel("Official submission")
    ax.set_ylabel("Descriptive raw objective sum")
    ax.set_xticks(numbers)
    ax.grid(axis="y", alpha=0.25)
    ax.ticklabel_format(axis="y", style="plain")
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
    for n, value in ((1, sums[0]), (8, sums[7]), (9, sums[8]), (10, sums[9])):
        ax.annotate(f"S{n}\n{value:,.0f}", (n, value), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(numbers, reduction, linestyle="--", linewidth=1.6, alpha=0.65)
    ax2.set_ylabel("Reduction vs Submission 1")
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    fig.savefig(FIGURES / "submission-progression.svg", metadata={"Date": None})
    fig.savefig(FIGURES / "submission-progression.png", dpi=220, metadata={"Date": None})
    plt.close(fig)


def normalize_svg(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    normalized = "\n".join(line.rstrip() for line in text.splitlines()) + "\n"
    path.write_text(normalized, encoding="utf-8")


def main() -> None:
    import matplotlib as mpl

    mpl.rcParams["svg.hashsalt"] = "grand-shipyard-v1.0.0"
    FIGURES.mkdir(exist_ok=True)
    write_results()
    write_manifest()
    render_graphviz()
    render_submission_figure()
    for path in sorted(FIGURES.glob("*.svg")):
        normalize_svg(path)


if __name__ == "__main__":
    main()
