# Public source and validation boundary

This repository publishes participant-authored source, documentation, and the
transcribed organizer evaluation history. It contains the final Python portfolio
and C++/pybind11 geometry-kernel source. Organizer-owned challenge materials are not
redistributed, and private participant records remain outside the public release.

The private submission package is bound by SHA-256 in `source-manifest.json`. The
seven released Python files under `solver/final-release/` are byte-identical to the
submitted source members. The released native-kernel source builds and passes a
randomized reference-equivalence test. The public release intentionally ships
source rather than the private competition package.

The corrected result history under `results/official-evaluation-history.csv` was
rebuilt from direct organizer completed-evaluation emails. The public repository
contains the numerical transcription and its hash; the private messages remain in
the evidence archive. The sequence numbers are retrospective chronological labels,
not organizer-assigned submission identifiers.

The full solver imports the organizer-provided `utils` module when executed in the
challenge environment. Public tests therefore validate the released components:
source integrity, protocol behavior, worker topology, deterministic asset
generation, and the native geometry primitive. Official hidden-evaluation results
are represented by the organizer-issued result records transcribed under `results/`.
