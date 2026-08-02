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
yard rows. Import containers do not receive bay or row decisions.

## Demand and reservations

- Export detailed demand: declared containers (`doc_cntrs`) only.
- Export big-plan guidance: export `new_qty` is normalized by voyage and size
  and rescaled to the declared export demand. It supplies a soft area target
  only; forecast-only export quantity is neither allocated nor reserved.
- Import aggregate reservation: import big-plan `new_qty`, by area and size.
- Incumbent import and export containers: already reflected in available bay
  and row capacity derived from the yard snapshot.
- Incumbent import containers are conservatively assumed not to leave during
  the planning horizon because release-time data are unavailable.

The model is static and conservative: incumbent containers remain occupied
throughout the horizon, while every declared export container in the demand
set is assumed to enter before the end of the horizon and still occupy yard
capacity at that point.

Import reservations consume aggregate area capacity but are not assigned to a
specific bay or row. Large-container import quantities additionally require
sufficient usable 40/45-ft pair capacity.

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
- aggregate import capacity reservation and large-pair preservation.

## Retained objectives

The integer restricted master is solved in two lexicographic stages. Stage 1
minimizes the number of unplaced declared containers. Stage 2 fixes that
minimum exactly and minimizes the following operational criteria:

- transferred boxes relative to the normalized upstream area-size guidance
  target, measured as one half of the L1 deviation;
- quantity-weighted berth-to-yard distance;
- operational-group area dispersion;
- operational-group row dispersion;
- proximity to incumbent containers of the exact same operational group;
- exact loss of usable 40/45-ft pair capacity caused by assigning 20-ft
  containers to a pair member.

Secondary criteria are normalized by a natural instance scale before their
weights are applied: guided demand for area deviation, operational-group count
for area activation, declared demand for row activation and incumbent
proximity, total usable pair capacity for large-pair loss, and the product of
maximum berth distance and declared demand for travel. Thus the reported
weights express policy trade-offs rather than compensate for incompatible raw
units.

The stage-2 integer solution is the final reported allocation. There is no
post-solve intra-area row relayout or separate heuristic objective; row-level
placement patterns must enter the generated column pool and are selected by
the same restricted master.

One pair-state variable is used for both large-container preservation and
import reservation. Assigning a 20-ft container to either member makes the
pair unavailable; the same state enters the pair-loss objective and the hard
lower bound on pair capacity reserved for incoming 40/45-ft imports. No
separate isolated-bay reward or auxiliary pair-loss score is used.

Berth-to-area distance is weighted by assigned quantity. Every export voyage
in the detailed model must have a berth mapping, and every candidate area must
have a positive finite distance to that berth. Missing or invalid values are
input errors; the model neither imputes them nor silently excludes the travel
criterion.

## Removed from the paper model

- detailed placement of forecast export containers;
- aggregate reservation of forecast-only export containers;
- detailed placement of import containers;
- weight classes;
- reefer, dangerous, over-limit, and other special-container rules;
- manual required/allowed/blocked area or bay overrides;
- E-area naming rules;
- six-bay-block objective terms;
- coarse/fine group-specific dispersion and balancing terms;
- bay-count dispersion (replaced by row-count dispersion);
- fixed maximum run length for consecutive 20 ft bays;
- tiered fallback-area penalties;
- misplaced-bay exclusion ratio;
- post-window loading rewards;
- document-floor and forecast-fallback demand construction.
- concurrent-operation conflict penalties.

When an entire voyage-flow-size demand has no matching upstream allocation, it
is excluded from the area-guidance transfer objective. Once that combination
has valid guidance, however, every candidate area is evaluated: areas absent
from its upstream allocation have target zero. This makes one half of the L1
deviation exactly equal to the number of boxes transferred away from the
normalized upstream area pattern.
