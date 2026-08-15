# Engineering a Reliable Anytime Solver for the OGC 2026 Grand Shipyard Challenge

**Kevin Yin · Team Smoop · Technical retrospective and source release**

The Grand Shipyard challenge coupled irregular multilayer packing, temporal
residence, bay assignment, weighted tardiness, workload balance, preferences, and
directional crane precedence under a four-core hard deadline. The final system,
**Candidate 23 / Submission 10**, used a deterministic regime gate, current-run
worker portfolio, compiled geometry primitive, joint space-time repair,
checker-gated incumbent protocol, and protected feasible fallback.

## Result

- **10** official submissions
- **60/60** hidden outputs feasible
- final vector: `11,280 · 31,368 · 103,065 · 6,464,170 · 16,443,998 · 42,333,422`
- descriptive raw six-instance sum reduced **28.316%** from first to final
  submission

The raw sum is **not the competition score**; official scoring was ordinal and
instance-specific. The exact final preliminary-round rank was not preserved, so
this repository makes no rank or award claim.

![Candidate 23 architecture](figures/architecture.svg)

## Read at three depths

**Overview:** this README and the architecture figure.

**Technical case study:** [technical retrospective](report/technical-retrospective.pdf)
([Markdown source](report/technical-retrospective.md)).

**Audit trail:** [official result table](results/official-submissions.csv),
[methodology](methodology/research-and-validation-process.md),
[source manifest](reproducibility/source-manifest.json), and
[release boundary](reproducibility/release-boundary.md).

## Repository map

```text
report/                retrospective, PDF, and bibliography
figures/               architecture, coupling, research, and result figures
results/               official submission vectors and interpretation boundary
methodology/           research and validation controls
solver/submission-10/  final submitted Python source
solver/native-kernel/  source release of the compiled geometry primitive
reproducibility/       hashes and explicit public/private boundary
tests/                 integrity, worker, protocol, topology, and kernel tests
```

## Source release boundary

The seven Python files under `solver/submission-10/` are byte-identical to the
frozen private submission source. The native-kernel source builds and is tested
against a direct NumPy reference. The repository does not redistribute the
organizer checker, problem data, evaluation instances, exact submission ZIP,
compiled submitted extension, emails, or participation certificate.

The full solver expects the organizer-provided `utils` module inside the challenge
environment. See [reproducibility/release-boundary.md](reproducibility/release-boundary.md).

## Validation

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.txt
(cd solver/native-kernel && python setup.py build_ext --inplace)
pytest
```

CI runs the source-integrity, protocol, worker-topology, release-boundary, and
native-kernel reference-equivalence checks on Python 3.11 and 3.12.

## Contribution and AI assistance

Kevin Yin retained responsibility for problem decomposition, architecture,
research questions, experiment and promotion gates, evidence interpretation,
submission decisions, and the final public record. **AI-assisted** systems
materially supported literature synthesis, implementation, testing, adversarial
review, and documentation. Organizer evaluation and frozen artifacts remained the
empirical authority.

## License

- solver source: Apache License 2.0
- report, figures, methodology, and result tables: CC BY 4.0
- third-party dependencies and organizer materials retain their respective rights
