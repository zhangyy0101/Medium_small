# Model and algorithm scope

The active business-model contract is V7
(`hierarchical_group_area_bay_pattern_v7`), and the active algorithm is V7.1
(`hierarchical_v7_1_restricted_area_bay_cg`). Its specification is in
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
Stage-1 restricted set. Zone dispersion, unused zone capacity, and voyage-area
dispersion are absent.

## V7 algorithm

1. Build the full legal V7 row-atom domain.
2. Derive an analytic peak cap and obtain a time-limited feasibility-only
   witness from the zone-free compact model. Its zero objective makes it a
   feasibility certificate and source of fixed proof columns, not a
   business-quality optimizer or a candidate-domain oracle.
3. Solve a strengthened coarse group-to-area Stage-1 MIP. Preserve Stage-1
   best support, distinguish solution-pool members by their
   `Y[group,area]` support, and add a capped mixture of coarse-pool and
   bay-local placement alternatives. The bay-local ranking greedily opens
   actual compatible anchors and evaluates the formal flow, extra-bay, and
   span coefficients for one group at a time. By default, at most one of five
   optional slots comes from the coarse pool. The cap applies only to
   additional candidates; best support is never removed. Stage-1 quantities
   remain guidance and are never fixed in Stage 2.
4. Solve a restricted bay-pattern RMP. Each pattern represents one anchor bay's
   complete legal row-capacity structure and contains at most three groups;
   actual `q[group,bay]` remains in the master.
5. Run exact pricing only inside the frozen restricted area domain and close
   that restricted LP. Feasibility-witness patterns remain fixed, explicitly
   registered proof columns even when outside the restricted domain; they do
   not activate an excluded area or permit new pricing there.
6. Solve a Restricted Integer Master over root, witness, Stage-1-guided, and
   deterministic one-group patterns, then validate the incumbent independently.
   Witness, Stage 1, root CG, and RIM share one production wall-clock budget;
   the first three stages have safety soft caps and RIM receives all remaining
   time. The optional global audit and Complete-MIP baseline are outside that
   production budget.

Full-domain exact pricing is available only as a read-only diagnostic audit.
It never changes the production RMP or restricted mapping. Without such a
certificate, the restricted LP value is not reported as a global lower bound,
and global gap fields remain unset.

Complete pattern enumeration is available only as a tiny correctness oracle.
There is no hidden Complete-MIP or exhaustive-pattern fallback in the V7
pipeline.

## Current evidence boundary

V7 code and 141 unit/micro tests (plus two subtests) are implemented. The first
24-group oracle audit found that the cap-2/3 coarse policy covered only 4 of the
Complete-MIP incumbent's 28 group-area pairs (85 of 640 boxes). The bay-local
repair with five optional candidates, at most one from the coarse pool, covers
22/28 pairs and 534/640 boxes. Under the same 120-second production budget it
returned UB `0.045080`, down from the prior V7 UB `0.060389` and 1.47% below the
saved 120-second Complete-MIP incumbent `0.045754`. The MIP baseline was not
rerun for this comparison. At 48 groups the repaired UB `0.079309` improves
5.20% over the prior V7 result but remains 1.73% above the saved MIP incumbent.
At 96 groups the restricted root still had negative reduced-cost columns after
34 rounds and 61.67 seconds, so no formal UB was produced. These are
single-seed directional diagnostics, not a formal performance claim. No
multi-seed scalability test, sensitivity,
ablation, weight calibration, or paper-level statistical evaluation has been
completed.
