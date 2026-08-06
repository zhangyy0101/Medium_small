# 3x-voyage benchmark

The pressure instance is generated from the current input format by copying voyage-level export demand and large-plan records while keeping the real yard geometry fixed:

```bash
python -X utf8 -B example/generate_more_voyages_case.py --copies 3 --overwrite
```

It enlarges the number of voyages, export groups, and containers rather than the number of bays or rows. This matches the expected growth mode of the application.

## Current 120-second comparison

Both methods use one Gurobi thread and the same mathematical model.

| Method | Upper bound | Valid lower bound | Relative gap | Proven optimal |
|---|---:|---:|---:|---:|
| Nested area-configuration Branch-and-Price | 0.25222222 | 0.24365804 | 3.40% | no |
| M0 direct MILP | **0.25138088** | **0.24446865** | **2.75%** | no |

A separate root-only run of the nested algorithm obtained a valid lower bound of `0.24401836`. The base instance root relaxation is solved exactly at `0.1901725424` in about 14 seconds.

The current algorithm reuses exact pricing results across data-identical physical blocks and screens the root row-recombination MIP with LP-active and recent configurations. Relative to the preceding implementation, its development-run gap fell from 4.93% to 3.40%; M0 remains slightly ahead on this instance. These are single-run development measurements, not a publication experiment. A formal study still needs multiple generated sizes, fixed seeds and threads, repeated runs where appropriate, and ablation of nested pricing, equivalent-block sharing, and the root primal heuristic.

Reproduce the paper algorithm with:

```bash
python -X utf8 -B benchmark_area_configuration.py \
  --input example/more_voyages_3x/input_data.json \
  --large-plan example/more_voyages_3x/large_plan.csv \
  --algorithm branch-price \
  --total-time-limit 120 \
  --solver-threads 1
```
