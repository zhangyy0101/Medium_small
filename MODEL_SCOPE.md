# TRE-oriented model scope

## Planning hierarchy

The upstream big plan is an exogenous area-capacity allocation. For the
long-format big-plan file, this project reads `new_qty` only:

- `new_qty`: allocation for containers that have not entered the yard;
- `planned_qty`: snapshot occupancy plus `new_qty`, and is never used as
  downstream demand or reservation.

The detailed model allocates declared, not-yet-arrived export containers to
yard rows. Import containers do not receive bay or row decisions.

## Demand and reservations

- Export detailed demand: declared containers (`doc_cntrs`) only.
- Export aggregate reservation: the positive residual between export big-plan
  `new_qty` and declared export demand, distributed according to the big-plan
  area-size pattern.
- Import aggregate reservation: import big-plan `new_qty`, by area and size.
- Incumbent import and export containers: already reflected in available bay
  and row capacity derived from the yard snapshot.
- Incumbent import containers are conservatively assumed not to leave during
  the planning horizon because release-time data are unavailable.

Aggregate reservations consume area capacity but are not assigned to a
specific bay or row.

## Detailed decision level

Every placement column contains a row allocation. The mathematical decisions
therefore remain row-level even though medium output can also be aggregated for
reporting.

There is one operational group definition throughout the model:

`(voyage, flow, destination port, size, height)`.

The former coarse/fine grouping distinction has no mathematical effect in this
branch. Legacy internal field names are retained only to avoid coupling model
changes to the column-generation implementation.

## Core hard constraints

- declared export demand balance, with penalized unplaced quantity;
- physical, size-specific, stack, bay-row, and row-size capacity;
- paired-slot footprint for 40 ft and 45 ft containers;
- area-function compatibility;
- no size mixing within a bay;
- no height mixing within a bay;
- different voyages cannot share a row, including conflicts with incumbent
  containers;
- destination-port row compatibility;
- aggregate import and export-residual capacity reservations.

## Retained objectives

- unplaced declared containers;
- deviation from the upstream big-plan area-size pattern;
- berth-to-yard distance and concurrent-operation conflict;
- operational-group area dispersion;
- operational-group row dispersion;
- proximity to incumbent containers of the exact same operational group;
- opportunity loss caused by assigning 20 ft containers to space that can
  support future 40/45 ft placements.

## Removed from the paper model

- detailed placement of forecast export containers;
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
