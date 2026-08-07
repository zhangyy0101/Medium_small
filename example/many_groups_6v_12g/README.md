# Six-voyage, many-group benchmark

This controlled stress case keeps the diversified six-voyage yard snapshot,
large plan, 2,241 declared export containers, and 624 anonymous import
reservations unchanged.  It changes only the declared export group structure:
each detailed voyage has twelve positive size-height-destination combinations,
for 72 export groups in total instead of 27.

The four destination ports and the size-specific height values all occur in the
source data.  Their deterministic recombination is synthetic and is intended
to isolate group-count growth rather than claim a new empirical demand sample.
Voyage totals and size totals are conserved exactly.

## Formal 120-second result

All methods use one Gurobi thread, the same mathematical model, a zero target
MIP gap, and a 120-second total limit that includes each method's own model
preparation and construction.

| Method | Upper bound | Valid lower bound | Relative gap |
|---|---:|---:|---:|
| Strict row-profile LBBD | **0.25060736** | 0.23280542 | **7.10%** |
| Voyage-row-resource LBBD | 0.25403877 | 0.23265116 | 8.42% |
| M0 direct MILP | 0.25703998 | **0.23296371** | 9.37% |

The voyage-row-resource master has 148,693 variables: 11,568 integer
group-bay quantities, 11,408 voyage-specific row-footprint templates, 45,626
binary footprint/class ownership states, and 55,418 continuous routing-flow
variables.  Exact voyage subproblems produced every incumbent; no global M0
repair or fallback solver was used.

Relative to M0 under the same budget, the concrete-row LBBD improves the incumbent by about
1.17% and reduces the reported gap by 0.95 percentage points, while M0 retains
a slightly stronger lower bound.  The six voyage-subproblem solves were all
optimal when completed and took roughly 0.04--0.06 seconds apiece after using
the master routing flow as a MIP start.  No dynamic feasibility or optimality
cut was needed on this instance; the initial conflict cliques, Hall inequalities,
and row-class ownership states already closed the tested feasibility conflicts.

This is a controlled algorithm-development result, not final computational
evidence for a paper.  The crossover is small and must later be tested on
multiple independently generated instances and time budgets.  Objective
magnitudes across differently grouped cases should not be compared directly
because natural instance normalization changes with group count.

The separate row-profile variant compresses the master from 148,693 to 95,845
variables.  Concrete templates fall from 11,408 to 5,504 resource profiles,
owner variables from 45,626 to 22,010, and routing flows from 55,418 to 26,186.
Its strengthened master adds 174 Hall-style bounds over partially overlapping
physical-footprint pools and a capacity constraint for every profile/no-mix
state.  Initialization stops after the first aggregate skeleton, exactly
disaggregates it, and supplies the verified row allocation as the formal
master MIP start.  In the recorded strict run, preparation took 12.62 seconds,
master construction 8.09 seconds, aggregate initialization 10.64 seconds, and
the formal master search 72.91 seconds.

The strict implementation also contains a lazily built global exact
disaggregation oracle.  A proven-infeasible footprint matching first produces
a packing or conditional matching cut.  The larger oracle is invoked only if
a feasible matching cannot be disaggregated by the voyage subproblems or its
exact row cost exceeds the master theta estimate; exact infeasibility and
recourse lower bounds then generate conditional aggregate feasibility and
optimality cuts, respectively.  In this run, the fast disaggregation attained
theta, so neither the large oracle nor a dynamic cut was needed.  The strict
closure therefore did not change either bound relative to the immediately
preceding strengthened run.

Against the earlier unstrengthened profile result (0.25451526 upper bound,
0.23189254 lower bound, 8.89% gap), the strict version improves both
bounds and reduces the gap by 1.79 percentage points.  Its incumbent is about
1.35% better than the concrete-row LBBD and 2.50% better than M0 under the same
budget.  The M0 lower bound remains higher by about 0.00016, so this is still a
controlled development result rather than final computational evidence.

## Reproduction

```bash
python -X utf8 -B example/generate_many_groups_case.py --overwrite

python -X utf8 -B benchmark_logic_benders.py \
  --input example/many_groups_6v_12g/input_data.json \
  --large-plan example/many_groups_6v_12g/large_plan.csv \
  --total-time-limit 120 \
  --solver-threads 1 \
  --master-feasibility-time-limit 30 \
  --master-time-limit 20 \
  --voyage-time-limit 8 \
  --compare-direct \
  --output outputs/voyage_row_resource_many_groups_120s.json

python -X utf8 -B benchmark_profile_benders.py \
  --input example/many_groups_6v_12g/input_data.json \
  --large-plan example/many_groups_6v_12g/large_plan.csv \
  --total-time-limit 120 \
  --solver-threads 1 \
  --output outputs/profile_resource_strict_lbbd_many_groups_120s.json
```
