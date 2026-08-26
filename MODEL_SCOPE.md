# Model and algorithm scope

The active development contract is V7
(`hierarchical_group_area_bay_pattern_v7`). Its specification is in
[docs/V7_MODEL_CONTRACT.md](docs/V7_MODEL_CONTRACT.md).

V6.1 (`row_aware_bay_zone_v6_1`) is frozen for historical reproduction. V7
does not modify the V6 schema, objective, solvers, tests, or recorded reports.

## V7 formal model

- Export group identity remains exactly `(voyage, size, height, discharge port)`.
- A zone is no longer a formal decision, feasibility object, objective term, or
  pricing column.
- The compact model uses integer `q[group,bay]`, binary group-bay and
  group-area use, binary legal row atoms, and physical group-bay use states.
- Size, 20/40/45 footprint, 45-ft edge eligibility, physical capacity,
  height-no-mix, exact row-group compatibility, and V6.1 historical-state
  non-worsening semantics are retained.
- Each physical bay may be used by at most three new export groups. The count
  is based on every physical footprint member, not only the anchor bay.
- Anonymous imports remain flow-size capacity reservations, with one import
  size per physical bay and export/import physical-bay exclusivity.
- Peak utilization remains a data-derived hard epsilon constraint. V7 has no
  area-balance or peak-minimization secondary objective.

## V7 objective

The normalized weighted sum has three categories:

1. spatial consolidation, composed of extra group areas, extra group bays, and
   normalized within-area bay span;
2. proximity to an existing exact same group;
3. berth transport distance.

The current weights are explicitly a provisional development baseline. Scale
derivation uses the complete legal V7 atom domain and is independent of the
Stage-1 active set. Zone dispersion, unused zone capacity, and voyage-area
dispersion are absent.

## V7 algorithm

1. Build the full legal V7 row-atom domain.
2. Derive an analytic peak cap and certify it with a zone-free compact
   feasibility witness.
3. Solve a coarse group-to-area Stage-1 MIP and construct a sparse algorithmic
   active set. Its quantities are guidance only and are never fixed in Stage 2.
4. Solve a global bay-pattern RMP. Each pattern represents one anchor bay's
   complete legal row-capacity structure and contains at most three groups;
   actual `q[group,bay]` remains in the master.
5. Run exact active-domain pricing. Before root closure, run exact full-domain
   pricing; any improving excluded group-area pair is activated dynamically.
6. Solve a restricted integer master over root, witness, and Stage-1-guided
   patterns and validate the incumbent independently.

Complete pattern enumeration is available only as a tiny correctness oracle.
There is no hidden Complete-MIP or exhaustive-pattern fallback in the V7
pipeline.

## Current evidence boundary

V7 code and micro correctness gates are implemented. No 24/48/96 benchmark,
multi-seed experiment, scalability test, performance comparison, sensitivity,
ablation, weight calibration, formal UB/LB/gap evaluation, or runtime tuning is
part of this implementation round. All such evidence is deferred.
