# Natural peak-conflict benchmark

This deterministic stress case strengthens operational conflicts without
changing the real yard snapshot, closing areas, prescribing placements, or
copying an identical voyage profile.  It starts from the diversified
six-voyage, twelve-group case and models a simultaneous booking peak:

- total declared export demand rises from 2,241 to 3,137 boxes (1.4x);
- voyage totals and group shares receive small deterministic, non-identical
  perturbations;
- 9 of 1,887 large boxes are changed from 40-ft high-cube to 45-ft high-cube,
  close to the small 45-ft share observed in the yard snapshot;
- the yard snapshot and all 624 anonymous import commitments remain unchanged;
- the upstream large-plan distribution is rescaled from `new_qty`, while every
  row continues to satisfy `planned_qty = snapshot_qty + new_qty`.

The resulting instance has six voyages, 81 positive
size-height-destination groups, and 5,024 export physical-slot units.  Its
conflicts arise from the simultaneous action of voyage and destination-port
row ownership, bay-level height separation, paired 40/45-ft footprints, the
45-ft edge-bay rule, and import capacity reservation.  These are constraints
of the common paper model rather than generator-only restrictions.

## Recorded feasibility check

Using one Gurobi thread, the selective-state LBBD found and exactly validated a
complete row allocation within a 90-second total limit:

| Method | Limit | Complete incumbent | Objective | Valid lower bound | Gap |
|---|---:|---:|---:|---:|---:|
| Selective-state LBBD | 90 s | yes | 0.32267634 | 0.27102025 | 16.01% |
| M0 direct MILP | 120 s | no | - | - | - |

The current cut-active LBBD spent 13.96 seconds preparing row locations, 8.43
seconds building the master, and 83.56 seconds inside its total solve
accounting.  The coarse quantity probe was exactly infeasible and generated
one monotone IIS feasibility cut.  Conflict-directed repair reduced the
initial upper bound from 0.36122102 to 0.33273818, and the next restricted
row-level neighbourhood reduced it again to 0.32267634.  The first formal
master bound was 0.27102025.  Thus this seed now demonstrates a genuinely
active feasibility cut and a separate exact primal-repair path, although its
remaining gap also shows that further upper-bound work is still justified.
A family of independently generated instances remains necessary for a
statistical paper experiment.

## Reproduction

```bash
python -X utf8 -B example/generate_natural_conflict_case.py --overwrite

python -X utf8 -B benchmark_selective_benders.py \
  --input example/natural_conflict_peak/input_data.json \
  --large-plan example/natural_conflict_peak/large_plan.csv \
  --total-time-limit 90 \
  --solver-threads 1 \
  --output outputs/selective_cut_repair_natural_final_90s.json

python -X utf8 -B run_yard_plan.py \
  --solver direct \
  --input example/natural_conflict_peak/input_data.json \
  --large-plan example/natural_conflict_peak/large_plan.csv \
  --total-time-limit 120 \
  --solver-threads 1 \
  --run-name direct_natural_conflict_peak_120s \
  --quiet
```

The generator exposes `--volume-scale`, `--forty-five-share`, and
`--voyage-copies` for later instance-family construction.  Values should be
declared before experiments rather than tuned separately for each algorithm.
