# TRE-oriented model scope

## Planning hierarchy

The upstream big plan is an exogenous area-capacity allocation. For the
long-format big-plan file, this project reads `new_qty` only:

- `new_qty`: allocation for containers that have not entered the yard;
- `planned_qty`: snapshot occupancy plus `new_qty`, and is never used as
  downstream demand or reservation.

The upstream size field is mandatory and must be either `20` or `40`. The
upstream `40` class combines physical 40-ft and 45-ft containers. Blank,
unknown, `45`, and aggregate-size values are rejected instead of silently
converted.

The detailed model allocates declared, not-yet-arrived export containers to
yard rows. Imports receive anonymous bay-size capacity reservations, not
container-group, individual-container, or row assignments.

## Demand and reservations

- Export detailed demand: declared containers (`doc_cntrs`) only.
- Export big-plan guidance: export `new_qty` is normalized by voyage and size
  and rescaled to the declared export demand. It supplies a soft area target
  only; forecast-only export quantity is neither allocated nor reserved.
- Import capacity total: import big-plan `new_qty`, conserved by flow and size.
  Its upstream area allocation is a soft reference. Anonymous reservations may
  move between areas that support the import flow when the original area lacks
  usable size capacity.
- Incumbent import and export containers: already reflected in available bay
  and row capacity derived from the yard snapshot.
- Incumbent import containers are conservatively assumed not to leave during
  the planning horizon because release-time data are unavailable.

The model is static and conservative: incumbent containers remain occupied
throughout the horizon, while every declared export container in the demand
set is assumed to enter before the end of the horizon and still occupy yard
capacity at that point.

Import reservations consume bay-size and joint physical capacity. A 20-ft
reservation uses a 20-ft-enabled bay; an upstream `40` reservation uses the
paired physical footprint of a 40-ft-enabled large bay. The upstream data do
not distinguish physical 40-ft and 45-ft imports, so no import-specific 45-ft
edge rule is inferred.

## Detailed decision level

Every placement column contains a row allocation. The mathematical decisions
therefore remain row-level; an area-bay summary is derived only for reporting.

There is one operational group definition throughout the model:

`(voyage, flow, destination port, size, height)`.

The former coarse/fine grouping distinction is removed from the model input.
All active grouping rules resolve to the single operational group above.

## Core hard constraints

- declared export demand balance with an explicit unplaced slack variable;
- physical, size-specific, stack, bay-row, and row-size capacity;
- paired-slot footprint for 40 ft and 45 ft containers;
- area-function compatibility;
- no size mixing within a bay;
- no height mixing within a bay;
- different voyages cannot share a row, including conflicts with incumbent
  containers;
- destination-port row compatibility;
- import flow-size total conservation and anonymous bay-size capacity;
- 45-ft containers may use only the first or last feasible large-bay position
  of an area. They remain subject to the same paired-bay, capacity, row, size,
  height, voyage, and destination-port constraints as every other container.
  This rule does not reserve every edge position for 45-ft containers and does
  not impose area-wide mutual exclusion between 45-ft and non-45-ft demand.

Area-function compatibility is always hard. An area appearing in the upstream
big plan is not exempt from its flow-function requirement; an incompatible
upstream allocation contributes a soft guidance deviation but never creates a
detailed-placement candidate.

## Retained objectives

Each column is a feasible operational-group/bay/row unit flow. Its master
variable is an integer container quantity, so arbitrary feasible integer row
allocations are represented by combining unit-flow columns; there is no capped
list of multi-row packing templates. The initial restricted master contains no
export placement column. In every lexicographic phase, the restricted-master
LP is solved first and its row duals define an independent finite pricing
problem for each export group. Feasible group/bay/row unit flows are generated
transiently; only flows with reduced cost below the numerical tolerance enter
the master. Each phase keeps one persistent Gurobi master and inserts priced
variables with their master-row coefficients, preserving the simplex basis.
Dual-simplex pricing is deterministic for a fixed solver configuration. The
number of entering columns is adaptive by group: it is a bounded fraction of
the currently negative candidates rather than a fixed batch. The complete
compatible column universe is neither stored nor solved as one auxiliary model
during pricing. Pricing stops only when every group pricing problem has no
negative-reduced-cost flow. A time limit cannot silently terminate this
certificate phase.

The final integer restricted master over the generated columns is solved
in two lexicographic stages. Stage 1
minimizes the number of unplaced declared containers. Stage 2 fixes that
minimum exactly and minimizes the following operational criteria:

- transferred boxes relative to upstream area-size guidance: normalized export
  guidance plus the import reservation's original area reference, measured by
  their combined L1 deviation;
- quantity-weighted berth-to-yard distance;
- operational-group area dispersion;
- operational-group row dispersion;
- proximity to incumbent containers of the exact same operational group;

Every secondary criterion is first converted to a dimensionless natural
instance scale. Area and row dispersion count only activations beyond the first
one used by each placed operational group, divided respectively by the maximum
number of additional feasible areas and rows. Incumbent proximity is a
quantity-weighted bay distance in `[0,1]`, divided by the demand of groups that
have a reachable incumbent anchor area. An anchor is neutral and excluded from
the normalization scale when no function-compatible bay in its area has a
feasible residual position; this removes decision-independent constant costs.
Big-plan deviation is divided by twice the sum of
guided export demand and import reservation demand, because moving one box
creates one shortage and one excess in the L1 vector. For each voyage, berth
distance is mapped from its closest and farthest compatible areas to `[0,1]`
and then averaged by declared quantity.

The baseline empirical weights are 0.240 for area dispersion, 0.205 for row
dispersion, 0.157 for incumbent-group proximity, 0.265 for big-plan guidance,
and 0.133 for berth distance. They sum to one. The first three terms jointly
receive 0.602, expressing the policy order concentration and layout continuity
> big-plan inheritance > travel efficiency. The weights therefore express
policy preference only, rather than compensate for incompatible raw units.

The stage-2 integer solution is the final reported allocation. There is no
post-solve intra-area row relayout or separate heuristic objective; row-level
row assignments must be combinations of the declared unit flows and are
selected by the same final master.

The stage-1 optimum for total unplaced demand is imposed as an equality in
every stage-2 restricted master. Consequently, stage-2 unplaced variables have
zero objective coefficients; no artificial million-scale penalty is mixed
with the normalized operational objective. Activation links use
constraint-specific bounds rather than one global Big-M. In particular, an
operational-group/area bound is the smaller of its relevant demand and its
feasible bay-row capacity in that area, with each bay additionally capped by
its size, footprint, and stack capacity. It does not use unrelated total area
capacity.

The no-negative-reduced-cost certificate applies to the LP relaxation over the
complete compatible unit-flow universe. The code separately reports the
restricted-master MIP gap and the valid gap between the integer incumbent and
the complete column-generation LP lower bound. For small cases below a
configured potential-column threshold, a verification layer appends every
missing column, warm-starts a complete two-stage integer master, and reports
its complete-universe MIP bound and gap. Larger cases retain standard column
generation followed by integer recovery. This is not branch-and-price;
without complete-column verification, no full integer optimality claim is made.

The import variables are anonymous capacity reservations indexed only by flow,
size, and bay. Their totals equal the corresponding import `new_qty`. Their
area totals may deviate from the big plan and the same big-plan L1 criterion
selects the minimum adjustment. Import variables enter only area-function,
bay-size, paired-footprint, and joint physical-capacity constraints. They do
not enter size/height no-mix, voyage no-mix, destination-port compatibility,
stack, or row constraints. If all compatible bays together cannot reserve the
full import total, the model reports infeasibility instead of truncating demand.

Berth-to-area distance is weighted by assigned quantity. Every export voyage
in the detailed model must have a berth mapping, and every candidate area must
have a positive finite distance to that berth. Missing or invalid values are
input errors; the model neither imputes them nor silently excludes the travel
criterion.

The independent output validator reconstructs every export footprint from the
input group and bay data. It checks group identity and dynamic attributes,
area-function eligibility, 40/45-ft paired-row signatures, configured bay and
row no-mix attributes (including incumbent conflicts), demand and unplaced
balances, joint export/import capacities, anonymous import metadata, and the
45-ft edge rule without consulting the solver incumbent.

## Removed from the paper model

- detailed placement of forecast export containers;
- aggregate reservation of forecast-only export containers;
- detailed group, container, or row placement of import containers;
- weight classes;
- reefer, dangerous, over-limit, and other special-container rules;
- manual required/allowed/blocked area or bay overrides;
- E-area naming rules;
- fixed multi-bay planning blocks;
- coarse/fine group-specific dispersion and balancing terms;
- bay-count dispersion (replaced by row-count dispersion);
- fixed maximum run length for consecutive 20 ft bays;
- tiered alternative-area penalties;
- misplaced-bay exclusion ratio;
- post-window loading rewards;
- document-floor and forecast-substitution demand construction.
- concurrent-operation conflict penalties.

When an entire voyage-flow-size demand has no matching upstream allocation, it
is excluded from the area-guidance transfer objective. Once that combination
has valid guidance, however, every candidate area is evaluated: areas absent
from its upstream allocation have target zero. This makes one half of the L1
deviation exactly equal to the number of boxes transferred away from the
normalized upstream area pattern.
