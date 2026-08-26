# V7.1 Implementation Report

## 1. Git

- base commit: `92f5c7e` (`feat: establish V7 two-stage yard planning baseline`)
- final commit: `NOT CREATED`（用户要求先提交 V7 基线；V7.1 修改保留在当前工作树中供审阅）
- branch: `refactor/v7-model-redesign`
- new branch created: NO

## 2. Algorithmic change

- Stage 1 changed from a temporary active set to the formal restricted-domain
  decision layer.
- Stage 2 changed from active/full-domain alternating CG to exact CG only inside
  one frozen restricted area mapping.
- Production global expansion was removed. The production solve never activates
  a group-area pair discovered outside Stage 1.
- Full-domain exact pricing was moved to an optional, read-only diagnostic audit.
  An audit finding a negative reduced-cost pattern does not add the pattern or
  continue CG.
- Algorithm version: `hierarchical_v7_1_restricted_area_bay_cg`.

The V7 business model, objective, objective weights, peak hard constraint, row
atoms, Bay Pattern definition, size/height/footprint/legacy/import constraints,
and at-most-three-new-groups rule were not changed.

## 3. Stage1 strengthening

- Witness mandatory support: group-area support recovered from the peak
  feasibility witness is always included in `A_restricted[g]` and cannot be
  deleted by the preferred cap.
- Best-solution mandatory support: all areas used by the best available Stage-1
  pool solution are mandatory.
- Solution-pool alternatives: optional areas are ranked first by pool frequency,
  then total assigned quantity, then first objective rank; capacity slack,
  compatible-bay count, existing proximity, berth distance, and deterministic
  area order are tie breakers.
- Preferred area cap: `preferred_candidate_area_cap=4` controls only optional
  pool alternatives. Mandatory support may exceed four areas.
- Packing-awareness proxy: Stage 1 now precomputes reachable capacity,
  compatible anchor count, maximum single-anchor capacity, usable physical-bay
  count, and footprint width. Integer `N[g,a]` satisfies
  `Q[g,a] <= max_single_anchor_capacity[g,a] * N[g,a]` and the area-level
  necessary condition `sum_g footprint_width[g] * N[g,a] <= 3 * usable_bays[a]`.
- Stage-1 `Q[g,a]` remains guidance only and is not fixed in Stage 2.

## 4. Restricted Stage2

- `restricted_areas_by_group` is normalized to an immutable mapping of
  `frozenset` values.
- Peak-witness patterns are injected before CG and fail fast if their support is
  outside the frozen mapping.
- Initial columns combine witness patterns, Stage-1 best-solution guidance, and
  deterministic one-group patterns, all inside the restricted domain.
- Production pricing remains exact for one-, two-, and three-group Bay Patterns,
  row allocations, size/height states, footprint, and legacy compatibility, but
  only permits a group when the anchor's area belongs to `A_restricted[g]`.
- A complete restricted-domain pricing sweep with no negative reduced-cost
  pattern sets `restricted_root_closed=True`.
- The Restricted Integer Master rejects any pool containing a pattern outside
  the frozen domain and independently validates its incumbent with
  `V7ModelEvaluator`.

## 5. Bound semantics

- restricted LP value: `restricted_lp_bound`
- restricted integer incumbent: `restricted_integer_ub`
- restricted master gap: `restricted_mip_gap`
- globally feasible incumbent: `global_feasible_ub`
- without a separate full-domain certificate:
  - `global_lower_bound = None`
  - `global_gap = None`
  - `global_root_certified = False`

The restricted LP value is not reported as a lower bound for the complete V7
MIP. The independently validated integer solution remains a feasible UB for the
complete business model.

## 6. Global audit API

- API: `audit_full_domain_pricing(...)`
- implemented: YES
- default production execution: NO
- mutates production state: NO
- reports: minimum reduced cost, negative-pattern count, affected groups,
  affected areas, affected group-area pairs, and global-root certification.
- adding audit columns, expanding the frozen mapping, or continuing production
  CG after an audit is prohibited and not implemented.

## 7. Fast tests

Only unit, tiny synthetic, and micro exhaustive tests were run.

- witness support survives cap: PASS
- Stage-1 best support survives cap: PASS
- cap trims only optional pool alternatives: PASS
- packing proxy rejects an impossible coarse allocation: PASS
- production Stage 2 cannot generate a pattern in an excluded area: PASS
- closed restricted CG equals exhaustive restricted-pattern LP: PASS
- global audit finds an excluded negative column without mutation: PASS
- global audit can certify a globally complete restricted domain: PASS
- uncertified global lower-bound/gap fields are `None`: PASS
- witness-backed tiny Restricted Integer Master is feasible and passes the
  independent evaluator: PASS

Commands:

```text
.venv/bin/python -m unittest \
  tests.test_v7_stage1_area \
  tests.test_v7_column_generation \
  tests.test_v7_integer \
  tests.test_v7_pipeline -v

.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

Final fast-suite result: **133 tests passed, 0 failed**.

## 8. Deferred

- 24/48/96 group experiments
- scalability experiments
- multi-seed evaluation
- performance comparison
- preferred-cap sensitivity
- objective-weight sensitivity
- constraint ablation
- runtime or top-K tuning
- dual stabilization
- F&O / adaptive enrichment / local branching
- valid inequalities and branch-and-price

No deferred experiment or performance mechanism was run or implemented in this
round.

## 9. Subsequent user-authorized experiment

After the implementation round was closed, the user separately authorized one
24-group run with a read-only full-domain pricing audit. That later result is
recorded in `V7_1_24_GROUP_GLOBAL_AUDIT_REPORT.md`; it does not alter the fast
test evidence or retroactively add an experiment to the original implementation
round.
