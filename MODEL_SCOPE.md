# Model and algorithm scope

The active model contract is V6.1 (`row_aware_bay_zone_v6_1`). Its full Chinese specification is in [docs/V6_MODEL_CONTRACT.md](docs/V6_MODEL_CONTRACT.md).

## Current status

The V6 mathematical model, complete small-instance zone oracle, V6-native analytic peak policy with a compact feasibility certificate, optional exact min-max oracle, complete business MIP, objective normalization, independent evaluator, projected restricted master, exact root pricing, restricted integer master, and compact primal coverage V1 are implemented. The exact root proof pool remains separate from a row-atom compact MIP that jointly creates integer-feasible coordination columns without enumerating zones.

`yard_planning/contiguous_zone_generation.py`, `yard_planning/direct_milp.py`, `benchmark_contiguous_zones.py`, and the positive-time `preexperiment` paper runner remain frozen V5 implementations. They are retained only for historical reproduction and do not produce V6-comparable UB, LB, gap, or objective values.

## V6 decisions

- Export group: exactly `(voyage, size, height, discharge port)`.
- Zone: one group, one yard area, one contiguous anchor-bay interval, and a nonempty selectable row set at every bay. Row numbers may change across bays.
- Export flow: integer boxes by selected zone and anchor bay, positive at every bay in a selected zone.
- Import flow: anonymous integer capacity by flow, physical size, and anchor bay.

Different groups cannot share a physical row. Groups with the same size and height may share a physical bay through different rows. Anonymous imports cannot share any physical bay with new exports.

Existing violations are frozen under a non-worsening policy. Size compatibility remains strict. A new export height may reuse one value from an existing bay-height set, and a new export group may reuse one exact group from an existing row-group set; newly assigned boxes still use one height per bay and one group per row.

## V6 objective

The normalized weighted sum has three declared categories:

1. spatial concentration (`0.6250`), internally composed of zone dispersion (`0.56`), voyage-area dispersion (`0.24`), and existing-group proximity (`0.20`);
2. berth transport (`0.1625`);
3. reserved-capacity efficiency (`0.2125`).

Peak utilization is a hard epsilon constraint. The production cap is derived from V6-native reachable-workload/capacity lower bounds and must have a compact full-model feasibility witness. Exact `rho*` is an offline oracle and sensitivity diagnostic, not a required production phase. No upstream large plan, TOPS plan, forecast demand, row-dispersion target, group-area target, shortage, or demand-plus-one-atom zone cap belongs to V6.

## Next gate

The next gate is scalability evaluation of the analytic-cap/root-proof/compact-primal split: measure feasibility-certificate time, time to first incumbent, compact-MIP trajectory, candidate-column diversity, final UB, and root gap on several seeds and genuinely larger instances. The old V5 solver components must not be bulk-migrated, and full enumeration remains a test oracle rather than a hidden production fallback.
