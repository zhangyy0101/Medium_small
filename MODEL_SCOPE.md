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

## 9. Alternative strengthened logic-based Benders solver

The optional `lbbd` solver is isolated from both complete-voyage column generation and M0. Its master contains integer group-to-bay quantities, anonymous integer import reservations, binary voyage-row-footprint ownership states indexed by the declared row no-mix class, and continuous routing flow from group-bay quantities to compatible row footprints. A footprint may activate at most one row class, and overlapping footprints across voyages and sizes share the same physical-row packing constraints.

The master also carries operational-group row counts and area activations, so all five normalized business objectives are represented before decomposition. Exact bay physical/size capacity, stack capacity, configured bay no-mix states, large-plan guidance, import guidance, row-capacity envelopes, and maximal conflict-clique/Hall inequalities strengthen the relaxation. Routing remains continuous; consequently the master is not a duplicate of M0 and its bound remains a valid lower bound for the common model.

The extended formulation removes unused-resource symmetry with four canonical links: every active row-class owner carries at least one routed box; an operational row count cannot exceed either its assigned boxes or its compatible active owners; and the number of owners in one voyage-bay-row-class cannot exceed the compatible operational row counts. These links do not add a business rule or alter the projected feasible set.

For each export voyage, one persistent integer subproblem fixes the master group-bay quantities and active row-class footprints, then performs the exact row allocation under physical/size footprint constraints and row no-mix rules. The current master routing flow is supplied as a MIP start. The union of all voyage solutions is globally feasible because physical footprint ownership is already coordinated in the master; no global compact repair is used.

An infeasible voyage subproblem first attempts a globally valid Hall-capacity cut on its IIS demand core. If aggregate capacity is not the cause, a monotone conditional logic cut requires either a reduced core allocation or a newly activated compatible row-class footprint. A solved voyage subproblem can analogously add a conditional lower-bound cut when its exact row-dispersion cost exceeds the master theta value. Repeated exact voyage assignments are cached.

Initialization uses the same resource master in two phases: a short zero-objective solve obtains a feasible skeleton, after which the original normalized objective is restored and the remaining initialization budget polishes that incumbent. This is not an alternative model or fallback chain. The first formal master solve then keeps one continuous search tree for all time not reserved for exact voyage validation. A 20-second reoptimization slice is used only after a newly generated cut changes the master.

This implementation is an experimental alternative and does not modify the `cg` or `direct` execution paths. It shares preprocessing, objective coefficients, and independent output validation with the other solvers, but it never invokes either solver internally.

## 10. Row-profile resource aggregation variant

The separate `lbbd_profile` solver replaces concrete voyage-row-class ownership binaries with integer counts of exchangeable row profiles. A safe profile preserves its anchor bay, every bay in the 20/40/45-ft physical footprint, per-slot physical and size capacity, candidate group compatibility, height, and row no-mix class. Only the concrete row labels are removed. The same aggregation is applied to continuous group-profile routing edges.

Profiles remain voyage-specific for routing and objective accounting. Profiles whose complete sets of concrete physical footprints are identical share one static physical-pool capacity. In addition, connected sets of partially overlapping footprint pools receive Hall-style packing bounds: the total number of activated profile states cannot exceed the maximum number of mutually disjoint physical footprints in that pool. Aggregate bay-row counts provide further necessary packing constraints. All these constraints are relaxations of concrete physical placement and therefore preserve a valid master lower bound.

For each profile/no-mix state, the sum of its routing flows is bounded by the physical row capacity times its integer profile count. This prevents several compatible group flows from independently reusing the same aggregate row capacity. The weaker profile-wide capacity envelope is not used.

For a fixed profile-count incumbent, a global binary footprint-matching problem selects concrete templates across all voyages and row classes. It enforces one state per concrete template and at most one selected footprint on every physical row slot. A proven-infeasible matching extracts an IIS profile-state core; a separate maximum set-packing solve first attempts a stronger valid upper bound for that core. If that short packing solve does not yield a violated bound, an exact conditional matching-feasibility cut still excludes the certified infeasible aggregate state. A feasible matching supplies concrete row owners to the same exact voyage subproblems used by `lbbd`.

Failure of the voyage solves under one feasible footprint matching is not treated as proof that the aggregate point is infeasible. In that case, an exact global recourse MIP jointly chooses all concrete footprints and row allocations while fixing only the master group-bay quantities and profile-state counts. The same oracle is called when a feasible fast disaggregation has row cost above the master theta estimate. Exact infeasibility yields an IIS-core conditional aggregate feasibility cut. A finite recourse lower bound above theta yields a conditional optimality cut on the sum of voyage theta variables. These cuts are active only at the certified aggregate integer assignment (or its IIS core) and become inactive after a relevant quantity/profile-state change, so they cannot remove a different feasible aggregate plan. IIS-core quantity changes are encoded in both directions; full-assignment cuts exploit fixed group demand to omit redundant zero-valued quantity terms.

The master is reoptimized after every generated packing, feasibility, or optimality cut. Without a wall-clock or iteration limit, the finite aggregate integer state space and exact conditional cuts give the standard finite LBBD closure argument. Under a time limit, the reported master bound remains a valid lower bound and convergence is reported only when the master is optimal, every aggregate incumbent has exact feasible recourse, and no violated cut remains. The global oracle is built lazily; when the fast disaggregation attains the valid theta lower estimate, that equality itself certifies exact recourse and the larger oracle is skipped.

Initialization solves only a zero-objective aggregate feasibility skeleton. It immediately performs global footprint matching and all exact voyage row subproblems. If this disaggregation succeeds, the resulting complete row allocation is mapped back to every principal master variable and supplied as the formal master MIP start. There is no separate aggregate-objective polishing phase and no M0 repair.

The profile solver is deliberately isolated from `lbbd`, M0, and column generation. It has its own benchmark entry point and can be removed or revised without changing any established solver path. It is currently an experimental compression variant, not the default paper algorithm.
