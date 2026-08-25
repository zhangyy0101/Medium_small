# Model and algorithm scope

## 1. Planning boundary

The integrated paper model uses deterministic known-box demand only: in-yard containers plus declared documents, deduplicated by container ID. It does not read an upstream large-plan file. Forecast fields may remain in a legacy input payload but are ignored, and forecast-only voyages are excluded. Detailed decisions cover declared export containers that have not entered the yard, and every such box must be allocated; unallocated-demand variables are excluded from the final integer model.

Import voyages are represented by anonymous capacity reservations. Incremental import demand is counted directly from declared import documents after excluding container IDs already in the yard. It is aggregated by operational flow and physical size. Import reservation obeys area-function compatibility, bay-size compatibility, shared physical capacity, and the peak-utilization epsilon constraint, but it does not inherit export voyage, discharge-port, height, detailed no-mix rules, or any export spatial-quality objective.

Detailed 45-ft export demand may use only available edge large bays, subject to all ordinary footprint, size, row, and no-mix constraints. This restriction does not block feasible non-45-ft demand from edge positions. In anonymous import capacity, 45-ft documents are conservatively aggregated into the physical 40-ft category.

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
- voyage-area and group-area activation;
- anonymous import capacity;
- a data-derived peak residual-capacity-consumption epsilon constraint.

An active group-area pair must have a selected zone, and actual area flow is bounded by the sum of selected-zone capacities clipped at group demand. A proof-only group-area zone-count inequality strengthens the root certificate; it is removed before primal integer search because it is valid for the root proof purpose but empirically delays incumbent discovery in the primal phase.

## 3. Unified objective

All optimization, pricing, certification, upper bounds, and lower bounds use one normalized objective:

| Component | Weight | Natural scale |
|---|---:|---|
| extra voyage areas | 0.1500 | reachable voyage-area activations |
| extra disconnected zones | 0.3500 | natural interval expansion by group |
| existing-group proximity | 0.1250 | reachable anchored quantity and normalized bay distance |
| unused reserved export-zone capacity | 0.2125 | total declared export demand |
| quantity-weighted berth distance | 0.1625 | assigned export quantity and voyage-specific distance range |

Extra group areas remain a reported diagnostic but have zero objective coefficient because every cross-area group split already requires extra contiguous zones. The unavoidable first area of each positive-demand voyage and the first zone of each positive-demand group are removed from the corresponding dispersion terms. All required berth-area distances must exist; missing data is an input error. Anonymous imports have zero coefficient in every objective component.

The declared unused-capacity ablation sets its weight to zero and proportionally renormalizes the other four weights. Raw objectives from the baseline and ablation are not subtracted because their objective definitions differ; comparisons use physical KPI values under paired inputs, seeds, threads, and time budgets.

Peak use is handled as an epsilon constraint rather than a seventh objective. Let `rho_LB` be the strongest planned-slot-load/residual-capacity lower bound computed for the whole instance, export/import subsets, voyages, groups, and import flow-size classes. The calculation includes integer area-capacity breakpoints, so fractional capacity that cannot hold another slot unit is not counted. With headroom parameter `h`, the cap is `rho_cap = rho_LB + h(1-rho_LB)`; the default is `h=0.50`. Both export footprints and anonymous import footprints count toward each area's load. This is a reproducible experimental proxy, not a terminal-approved safety threshold, and `h` must be covered by sensitivity analysis.

Exact row filling has no primary-objective term. It is a feasibility recourse with a separately reported secondary row-quality diagnostic, so it cannot change the zone-model bound or gap.

## 4. Exact root pricing

The restricted LP starts from a feasible zone subset. Dual prices decompose by group and ordered strip. Prefix sums convert each interval's reduced cost into a range-minimum query; a deterministic RMQ heap returns the exact top-k intervals without explicitly materializing every legal zone.

Pricing uses adaptive per-group batches but terminates only after an exhaustive exact pass proves that no excluded legal interval has reduced cost below tolerance. At termination, the restricted-master objective equals the complete zone-LP optimum and is a valid global lower bound for the integer zone model. The restricted integer-master bound is never reported as global.

Automated micro-instance tests compare the generated root directly with full zone enumeration and independently compare RMQ pricing with exhaustive interval reduced costs.

## 5. Primal search and recourse

The exact-root snapshot is compressed into the Phase 2 group-specific, five-channel diversified primal pool and solved as a restricted integer master. The incumbent is then improved by at most three conflict-aware Fix-and-Optimize rounds. Each round uses attributable incumbent objective as the seed signal, then expands the neighborhood with an equal-weight conflict graph built from same-voyage activation, shared candidate areas, atomic physical-resource competition, and incumbent peak-utilization exchange pressure. Candidate-zone budget fractions increase deterministically through 12%, 22%, and 35%. If a fully searched seed does not improve the incumbent, the next round rotates to the next untried objective-ranked seed; an improvement recomputes attribution and conflicts from the new incumbent. All legal zones are materialized only for the selected groups, while decisions outside the neighborhood remain fixed. Used local columns are injected into the existing primal master before a time-bounded reoptimization.

The resulting group-bay export flows and anonymous import reservation are fixed in an exact row-level recourse MIP. A constructive capacity certificate and the solver solution must agree on all fixed flows. The final output is accepted only after independent validation of demand, footprint, capacity, size, area-function, 45-ft edge, stack, and no-mix constraints. Any recourse or validation failure is a hard error; there is no greedy or M0 fallback chain.

## 6. Bounds and comparison policy

The reported certificate is

- `LB`: the closed exact-pricing root objective;
- `UB`: the independently reconstructed objective of the validated integer zone solution;
- `gap = (UB - LB) / |UB|`.

`analyze_complete_zone_mip()` solves the fully materialized version of the same zone model and is the valid small-scale direct comparison. `DirectMilpPlanner` allocates directly at row locations and is retained because the zone algorithm inherits its shared row model for recourse. As a standalone M0 it has a different concentration representation and feasible set; cross-model objective subtraction is therefore deliberately disabled.

## 7. Experiment-version boundary

Results produced with earlier objective definitions are not comparable with this integrated model's UB, LB, or gap. All stage-gate, ablation, and complete-zone-MIP experiments must be regenerated under model schema `integrated_zone_v4`. Correctness checks still require the generated root to match the fully enumerated zone LP on micro instances, followed by multi-seed comparisons under equal solver budgets.
