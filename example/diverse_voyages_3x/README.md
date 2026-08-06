# Diversified 3x-voyage case

This case enlarges the two detailed export voyages in the base input to six
voyages while leaving the physical yard snapshot and all import commitments
unchanged. It is intended to replace exact voyage replication in algorithmic
tests where artificial symmetry can distort column-generation behaviour.

Generate it deterministically with:

```bash
python -X utf8 -B example/generate_diverse_voyages_case.py --copies 3 --overwrite
```

The generator enforces the following invariants:

- total declared export demand is exactly `3 x 747 = 2241` containers;
- only size-height-destination combinations observed on the corresponding
  source voyage are used;
- the two synthetic volumes in each source family sum to exactly twice the
  source volume, but individual volumes and group shares differ;
- synthetic large-plan `new_qty` equals declared demand by size, with a
  diversified distribution over source-supported areas;
- synthetic `snapshot_qty` is zero and `planned_qty = new_qty`;
- estimated berths are selected only from columns with complete values in the
  supplied berth-area distance matrix;
- synthetic receiving, berthing, and departure times form consistent future
  schedules relative to the planning time;
- container IDs and numbers are unique.

Exact input profiles, berth assignments, and large-plan distributions are
recorded in `manifest.json`. The previous `more_voyages_3x` instance remains
available as a replicated-symmetry stress case; it should not be the sole
large-instance result in a paper.

Run the proposed algorithm on this case with:

```bash
python -X utf8 -B benchmark_voyage_plans.py \
  --input example/diverse_voyages_3x/input_data.json \
  --large-plan example/diverse_voyages_3x/large_plan.csv \
  --total-time-limit 120 \
  --solver-threads 1
```
