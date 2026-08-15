# Reproducibility and release boundary

This repository is a source-only public artifact. It preserves the final Python
portfolio and C++/pybind11 geometry-kernel source, but deliberately omits:

- the organizer problem statement, checker, and evaluation instances;
- the exact submitted ZIP and its compiled CPython 3.12 Linux extension;
- organizer emails, leaderboard captures, and the participation certificate;
- private experimental traces, candidate worktrees, and ResearchLab run packets.

The private submission ZIP is bound by SHA-256 in `source-manifest.json`. The seven
released Python files under `solver/final-release/` are byte-identical to the
submitted source members. The released native-kernel source builds and passes a
randomized reference-equivalence test, but a byte-for-byte reproduction of the
submitted Linux binary is not claimed because the full original build-container
identity was not preserved.

The corrected result history under `results/official-evaluation-history.csv` was
rebuilt from direct organizer completed-evaluation emails. The public repository
contains the numerical transcription and its hash, not the private messages. The
sequence numbers are retrospective chronological labels, not organizer-assigned
submission identifiers.

The full solver expects the organizer-provided `utils` module and challenge data.
Without those materials, this repository can validate source integrity, protocol
behavior, worker topology, deterministic asset generation, and the native geometry
primitive, but it cannot reproduce private official scores.
