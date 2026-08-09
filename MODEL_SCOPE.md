# Model and algorithm scope

## 1. Planning boundary

Detailed decisions cover only declared export containers that have not entered the yard. Every declared export box must be allocated; export forecasts and unallocated-demand variables are excluded.

Import voyages are represented by anonymous capacity reservations. Only large-plan `new_qty` is incremental import demand. `planned_qty` already includes in-yard boxes and is never read as new demand. Import reservation obeys area-function compatibility, bay-size compatibility, and shared physical capacity, but it does not inherit export voyage, discharge-port, height, or other detailed no-mix rules.

Large-plan size must be `20` or `40`; `40` denotes the physical 40/45-ft category. `ALL`, blank, `45`, and unknown values are rejected. Detailed 45-ft export demand may use only available edge large bays, subject to all ordinary footprint, size, row, and no-mix constraints. This restriction does not block feasible non-45-ft demand from edge positions.

## 2. Contiguous-zone formulation

For each export group, area, and physical row, compatible atomic row footprints are ordered by bay number. A zone is a contiguous interval of those footprints. Selecting a zone dedicates every resource in the interval to that group and reserves the interval's full size-compatible capacity. Integer group-to-bay flow uses the reserved capacity and sums exactly to declared group demand.

The explicit capacity-protection rule is

`zone capacity <= group demand + largest atomic-row capacity on the strip`.

It excludes excessive unused reservation and bounds the implicit interval family. It is a model constraint, not a dominance claim.

The master coordinates:

- exact export demand flow;
- full physical-footprint reservation and overlap;
- bay physical and size capacity;
- stack resources;
- area-function eligibility;
- hard bay/row no-mix states;
- voyage, discharge-port, and height separation for exports;
- 40/45-ft paired footprints and 45-ft edge eligibility;
- anonymous import capacity and export/import large-plan guidance.

An active group-area pair must have a selected zone, and actual area flow is bounded by the sum of selected-zone capacities clipped at group demand. A proof-only group-area zone-count inequality strengthens the root certificate; it is removed before primal integer search because it is valid for the root proof purpose but empirically delays incumbent discovery in the primal phase.

## 3. Unified objective

All optimization, pricing, certification, upper bounds, and lower bounds use one normalized objective:

| Component | Weight | Natural scale |
|---|---:|---|
| extra group areas | 0.25 | reachable group-area activations |
| extra disconnected zones | 0.22 | natural interval expansion by group |
| existing-group proximity | 0.08 | reachable anchored quantity and normalized bay distance |
| export/import large-plan L1 deviation | 0.22 | guided export and reserved import quantity |
| unused reserved zone capacity | 0.13 | total declared export demand |
| quantity-weighted berth distance | 0.10 | assigned quantity and voyage-specific distance range |

The unavoidable first area and first zone of each positive-demand group are removed from the two dispersion terms. All required berth-area distances must exist; missing data is an input error.

Exact row filling has no primary-objective term. It is a feasibility recourse with a separately reported secondary row-quality diagnostic, so it cannot change the zone-model bound or gap.

## 4. Exact root pricing

The restricted LP starts from a feasible zone subset. Dual prices decompose by group and ordered strip. Prefix sums convert each interval's reduced cost into a range-minimum query; a deterministic RMQ heap returns the exact top-k intervals without explicitly materializing every legal zone.

Pricing uses adaptive per-group batches but terminates only after an exhaustive exact pass proves that no excluded legal interval has reduced cost below tolerance. At termination, the restricted-master objective equals the complete zone-LP optimum and is a valid global lower bound for the integer zone model. The restricted integer-master bound is never reported as global.

Automated micro-instance tests compare the generated root directly with full zone enumeration and independently compare RMQ pricing with exhaustive interval reduced costs.

## 5. Primal search and recourse

The root support is enriched by a dimension-aware integer pool and solved as a restricted integer master. The incumbent is then improved through one objective-guided Fix-and-Optimize neighborhood. Groups are sorted by their attributable incumbent objective; selection stops after reaching the target objective mass or the candidate-zone budget. Defaults are 60% objective mass and 35% of all candidate zones. All decisions outside the chosen groups are fixed, while every legal zone for the selected groups is made available.

The resulting group-bay export flows and anonymous import reservation are fixed in an exact row-level recourse MIP. A constructive capacity certificate and the solver solution must agree on all fixed flows. The final output is accepted only after independent validation of demand, footprint, capacity, size, area-function, 45-ft edge, stack, and no-mix constraints. Any recourse or validation failure is a hard error; there is no greedy or M0 fallback chain.

## 6. Bounds and comparison policy

The reported certificate is

- `LB`: the closed exact-pricing root objective;
- `UB`: the independently reconstructed objective of the validated integer zone solution;
- `gap = (UB - LB) / |UB|`.

`analyze_complete_zone_mip()` solves the fully materialized version of the same zone model and is the valid small-scale direct comparison. `DirectMilpPlanner` allocates directly at row locations and is retained because the zone algorithm inherits its shared row model for recourse. As a standalone M0 it has a different concentration representation and feasible set; cross-model objective subtraction is therefore deliberately disabled.

## 7. Reproducible stage-gate results

With one Gurobi thread and a common solver budget:

- Base instance, 60 s: `LB=0.1450871517`, adaptive-neighborhood `UB=0.1518251117`, gap 4.44%. The neighborhood selects 3 of 9 groups and exposes 27.46% of candidate zones while covering 64.79% of attributable objective.
- 72-group instance, 120 s: `LB=0.1792403938`, initial restricted-MIP `UB=0.1993817657`, final `UB=0.1918854490`, gap 6.59%. The neighborhood selects 23 groups and improves the incumbent by 3.76%. Exact recourse realizes all 2,241 export boxes and 624 import-reserved boxes.
- Same-model complete zone MIP on the 72-group instance, 120 s: `UB=0.2073750105`, `LB=0.1797169064`, gap 13.34%.

These measurements establish correctness and a scale-stage algorithmic advantage over the same-model complete MIP. They are development evidence, not a substitute for the final paper experiment design, multi-seed robustness analysis, or formal complexity discussion.
