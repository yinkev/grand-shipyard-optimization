# Research and validation process

The campaign separated four record types that are often conflated in heuristic
optimization work:

1. **Hypothesis:** a mechanism that might change a decision.
2. **Opportunity evidence:** proof that the relevant decision surface actually
   occurs under the accepted solver.
3. **Causal experiment:** a frozen treatment compared with matched controls.
4. **Promotion decision:** adoption only after checker validity, objective value,
   wall-time safety, and artifact reproducibility all pass.

## Controls used

- Source identities and candidate packages were frozen by SHA-256.
- Fixed-wall comparisons used paired or order-balanced runs where practical.
- Engagement telemetry was not treated as objective improvement.
- Stored artifacts were replayed from disk; in-memory success was insufficient.
- Every returned candidate was passed through the organizer checker before
  selection.
- Negative results closed the exact tested lane instead of silently changing the
  panel, threshold, trigger, or mechanism.
- The protected feasible floor and emergency fallback remained available while
  experimental workers searched.

## Formal ResearchLab campaign

CAM-0001 planned 80 specialist roles and completed 49 reviewed runs across B01–B09.
The accepted reports produced 27 bounded Test recommendations, 13 closures, and 9
deferrals. Six selected opportunity-first empirical gates—SEMS-4, CEEC-NG,
CPRI-1, CVCGD-1, FACS-4 A0, and TXV-2 Stage O—reached valid terminal kills.

The formal campaign was cancelled at 49/80 roles when the competition and solver
campaign ended. Candidates 9–23 were developed through a faster direct research
and implementation sprint. That transition is part of the methodological result:
the high-control workflow was useful for pruning, but too coordination-heavy for
the terminal competition window.

## Evidence hierarchy

Public claims in this repository follow this order:

1. organizer evaluation and certificate records;
2. exact private artifacts and checker-backed local results;
3. derived arithmetic from those records;
4. interpretation, explicitly labeled as such.

No exact rank, award, or optimality claim is made.
