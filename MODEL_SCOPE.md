# Model and algorithm scope

## 1. Planning boundary

Detailed row-allocation demand consists only of declared export containers that have not yet entered the yard. Every such container must be allocated. Export forecasts are excluded from demand and capacity reservation.

Import voyages are represented by anonymous capacity reservations. Only `new_qty` is future demand; `planned_qty` already includes in-yard boxes and is not incremental. Import reservations obey area-function compatibility, bay-size compatibility, and shared physical capacity. They do not inherit export voyage, discharge-port, height, or other no-mix constraints and do not receive detailed row assignments.

Large-plan size is restricted to `20` and `40`; `40` covers physical 40/45-ft containers. `ALL`, blank, `45`, and unknown values are rejected.

## 2. Common mathematical model

The compact M0 formulation uses integer export quantities at row locations, binary operational-group area and row activations, anonymous integer import reservations at bays, and binary auxiliaries for configured no-mix states.

Both M0 and the paper algorithm enforce:

- complete allocation of declared export demand;
- shared physical bay and row capacity;
- size-specific bay and row capacity;
- paired large-bay footprints for 40/45-ft containers;
- stack availability and stack-height conversion;
- area-function eligibility;
- configured bay-level and row-level no-mix attributes;
- mandatory row separation by export voyage and discharge port;
- mandatory height compatibility within a row;
- 45-ft containers restricted to available edge large bays.

The 45-ft rule does not prohibit non-45-ft containers from other feasible edge positions. Import reservations share physical and size capacity with exports but receive no detailed no-mix constraints.

## 3. Objective

The objective is a directly normalized weighted sum:

| Component | Weight | Natural normalization |
|---|---:|---|
| operational-group area dispersion | 0.290 | maximum reachable group-area activations |
| operational-group row dispersion | 0.240 | maximum reachable group-row activations |
| distance from existing same-group boxes | 0.070 | reachable anchored demand and normalized bay-order distance |
| export/import large-plan deviation | 0.270 | guided export and reserved import quantities |
| berth-area travel distance | 0.130 | assigned export quantity and voyage-specific distance range |

All required berth-area distances must exist. Missing distance data is an input error.

## 4. Outer complete-voyage decomposition

The paper algorithm applies Dantzig-Wolfe decomposition by export voyage. One outer column is a complete integer row allocation for all groups of one voyage across all feasible yard areas. It contains the corresponding shared-resource and no-mix coefficients and its exact normalized business cost.

The outer restricted master selects exactly one complete plan per voyage. Export demand is already satisfied inside every column. The master coordinates shared bay/row resources, cross-voyage no-mix states, export large-plan deviation, and anonymous import reservation. Thus the number of convexity blocks grows with the number of voyages, not with voyage-area pairs.

Initial complete plans are obtained from single-voyage feasibility MIPs. Phase-I artificials relax only global coupling constraints and import balance. They are fixed to zero before activating the business objective and can never enter the final solution.

## 5. Nested voyage pricing

For a fixed outer dual vector, each voyage is priced independently. Its inner master selects one integer local pattern for every feasible area and enforces all group demands of that voyage. A local pattern may be zero or may assign partial group quantities to row locations in one area.

Each area-pattern pricer is an integer MIP containing exact local physical, size, footprint, stack, bay/row no-mix, group-area activation, and group-row activation constraints. Full sweeps use raw inner duals, and time is allocated by the square root of candidate-row count. Up to three local patterns are returned per solve.

Local MIP bounds yield a valid lower-bound correction for the inner voyage relaxation. That bound, less the outer voyage-convexity dual, is a valid bound on complete-voyage reduced cost. A negative integer inner-master solution supplies a new complete outer column.

Because the inner local-pattern relaxation may have an integrality gap, it is not by itself an exact pricing certificate. When it cannot certify nonnegative reduced cost, or when its integer improvement is weak relative to its bound, the persistent complete-voyage row MIP performs strict certification. A timed certification may still contribute its finite Gurobi bound, but exact outer closure is declared only when every voyage is certified under the same outer dual vector.

Every complete plan found by strict certification is split back into its area patterns, so information flows in both directions between the two pricing levels.

## 6. Bounds and convergence

For each complete outer sweep, the restricted-master objective plus the sum of negative valid voyage-pricing bounds is a valid lower bound for the complete-voyage Dantzig-Wolfe relaxation. If a sweep is interrupted, the algorithm retains the best bound from an earlier complete finite sweep. A skipped voyage or nonfinite pricing bound cannot form a certificate.

The implementation uses no stabilization, branch-and-price, isomorphic reuse, case-specific area selection, runtime solver switch, greedy backup, or alternative incumbent chain.

## 7. Integer recovery

After root column generation, one restricted compact row MILP is solved. Its support contains row locations exposed by LP-active complete-voyage plans, the eight most recent plans per voyage, and the minimum additional locations required to give every export group sufficient raw support capacity.

This MILP enforces the complete original model and is solved once with zero requested MIP gap within its allowance. It supplies only a feasible primal upper bound. Its restricted-model bound is never used as a complete-model lower bound; the reported global gap always uses the valid column-generation lower bound.

## 8. M0 baseline and regression checks

M0 is the complete compact row-location MILP. It creates all feasible row locations and uses the same hard constraints, import representation, normalization, and weights. It has no decomposition, pricing, Phase I, or alternate solution chain.

Small-instance tests require M0 and the paper algorithm to return the same objective and independently valid row output. Additional tests verify strict root closure on a complete-column case and cross-voyage row separation.
