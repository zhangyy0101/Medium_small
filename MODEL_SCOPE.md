# Model and algorithm scope

## 1. Planning boundary

The detailed row-allocation demand consists only of declared export containers that have not yet entered the yard. Every such container must be allocated. Export forecasts are excluded from both demand and capacity reservation.

Import voyages are represented by anonymous capacity reservations. For an import large-plan record, only `new_qty` is treated as future demand; `planned_qty` already contains in-yard boxes and is not an incremental quantity. Import reservations obey:

1. yard-area function compatibility;
2. bay-size compatibility;
3. shared physical capacity.

They do not inherit export voyage, discharge-port, height, or other no-mix constraints and are not assigned to detailed rows.

The large-plan size is restricted to `20` and `40`; `40` covers physical 40/45-ft containers. `ALL`, blank, `45`, and unknown values are rejected.

## 2. Decisions

The underlying compact model uses integer export quantities at row locations, binary group-area activation, binary group-row activation, anonymous integer import reservations at bays, and auxiliary binary variables for configured no-mix states.

The paper algorithm applies Dantzig-Wolfe decomposition by yard area. One column is a complete integer configuration of one area and contains:

- export group quantities at row locations;
- anonymous import reservations at compatible bays;
- the corresponding local capacity, footprint, stack, no-mix, area-use, and row-use states;
- its normalized business cost.

The master selects exactly one configuration for each modeled area. It enforces every export group demand equality, every import flow-size reservation equality, and the export/import large-plan deviation balances.

## 3. Hard constraints

Both the paper algorithm and M0 enforce the same constraints:

- complete allocation of declared export demand;
- shared physical bay and row capacity;
- size-specific bay and row capacity;
- large-bay paired footprints for 40/45-ft containers;
- stack availability and stack-height conversion;
- area-function eligibility;
- configured bay-level and row-level no-mix attributes;
- mandatory row separation of export voyage and discharge port;
- mandatory height compatibility within a row;
- 45-ft containers restricted to available edge large bays.

The 45-ft rule does not prohibit non-45-ft containers from other feasible edge positions.

Import reservations share physical and size capacity with exports but receive no detailed no-mix constraints.

## 4. Objective

The objective is a directly normalized weighted sum. The five components and empirical weights are:

| Component | Weight | Natural normalization |
|---|---:|---|
| group-area dispersion | 0.290 | maximum reachable group-area activations |
| group-row dispersion | 0.240 | maximum reachable group-row activations |
| distance from existing same-group boxes | 0.070 | reachable anchored demand and normalized bay-order distance |
| export/import large-plan deviation | 0.270 | guided export and reserved import quantities |
| berth-area travel distance | 0.130 | assigned export quantity and voyage-specific distance range |

All required berth-area distances must exist. A missing distance is an input error rather than a zero-cost fallback.

## 5. Adaptive area pricing

Area difficulty is determined from the instance rather than area names. Candidate row locations and large-bay footprints form a graph inside each area.

- A small or single-component area is priced by one complete exact MIP.
- A large multi-component area uses nested exact pricing.

For nested pricing, each physical component has a persistent exact block MIP. A block configuration satisfies every local packing restriction. A persistent coordination LP chooses one configuration per block and links their group quantities to one group-area activation variable.

Blocks with identical physical data, candidate semantics, and constraint structure are detected without using area names. One representative exact MIP supplies a small pool that is mapped bijectively to every block in the equivalence class. Because the block-convexity dual is a constant in each subproblem, this reuse preserves both the optimal configuration and its exact lower bound. Sharing is disabled whenever a row- or bay-specific branch decision breaks the equivalence.

For a fixed outer-master dual vector, block reduced-cost lower bounds are obtained from the global MIP bounds. The coordination-LP objective plus the sum of negative block lower-bound corrections is a valid lower bound for the complete integer area-pricing problem. If that value, after the outer area-convexity dual, is nonnegative, the area is exactly certified. The restricted coordination MIP is used only to construct feasible negative columns. If the nested bound is inconclusive and no new column is found, the solver falls back to the complete area MIP.

This fallback is part of the exactness mechanism and is not a secondary heuristic solver chain.

## 6. Restricted master and convergence

The initial pool contains a zero configuration for each area. Temporary Phase-I artificial variables establish restricted-master feasibility; they are fixed to zero before the business objective is optimized and never enter the final model.

Business pricing alternates between productive-area sweeps and periodic complete sweeps. A selective sweep may add columns but cannot update a valid global lower bound or declare closure. Exact node closure requires every area to have a valid nonnegative reduced-cost certificate under the same raw master dual vector.

The root and every branch node use the same column-generation engine. If the time limit interrupts pricing, the node retains its last valid bound and remains open.

## 7. Branching and primal upper bound

The branching order is:

1. group-area allocated quantity;
2. group-row allocated quantity;
3. group-row use;
4. import reservation quantity;
5. remaining original auxiliary integer states when required.

Each decision is imposed in both the restricted master and subsequent area pricing, so generated configurations obey the node partition.

A screened compact row MILP is solved once at the root to obtain an incumbent. Its support is built from LP-active area configurations, the two most recent configurations in each area, and the two most recent configurations of each nested physical block. This procedure supplies only a feasible upper bound and has no role in lower-bound certification.

## 8. M0 baseline

M0 is the complete compact row-location MILP. It creates all feasible row locations, uses the same hard constraints, import-reservation representation, normalization, and empirical weights, and is solved directly by Gurobi. It has no decomposition, pricing, Phase I, or alternative solution chain.

Small-instance regression tests require M0 and the paper algorithm to return the same objective and independently valid row output.
