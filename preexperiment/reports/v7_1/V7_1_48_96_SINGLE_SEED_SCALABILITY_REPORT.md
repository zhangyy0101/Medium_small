# V7.1 48/96-Group Single-Seed Scalability Report

## Scope

- cases: `scale_g048_s601`, `scale_g096_s801`;
- production algorithm total budget: 120 seconds per case;
- quality-witness / Stage-1 / root soft caps: 10 / 10 / 60 seconds;
- RIM: all remaining production time;
- Complete MIP baseline: independent 120-second solver budget;
- threads / solver seed: 1 / 0;
- full-domain audit requested, but executable only after a completed root.

The benchmark runner was corrected so that an independent Complete-MIP
baseline is still executed and archived when the production algorithm fails.
This changes reporting only; it does not add a fallback to the V7 algorithm.

## Problem sizes

| Case | Export groups | Export boxes | Import boxes | Row atoms | Group-bay edges |
|---|---:|---:|---:|---:|---:|
| 48 groups | 48 | 1,280 | 499 | 38,482 | 8,944 |
| 96 groups | 96 | 2,560 | 497 | 76,962 | 17,888 |

## Current V7.1 algorithm

| Measure | 48 groups | 96 groups |
|---|---:|---:|
| Quality-witness solver limit | 10.00 s | 10.00 s |
| Witness stage wall time | 13.36 s | 17.14 s |
| Witness solution count | 0 | 0 |
| Witness status | time limit | time limit |
| Unused production budget at failure | 106.64 s | 102.86 s |
| Stage 1 entered | no | no |
| Restricted root entered | no | no |
| RIM entered | no | no |
| V7 feasible UB | unavailable | unavailable |

Both failures occurred in the business-objective compact quality witness. They
do not establish model infeasibility. A historical 48-group run using the same
model with `feasibility_only=True` found a validated witness in about one solver
second, confirming that the business-objective witness responsibility is the
immediate failure mechanism for that case.

Because there is no V7 integer solution, no UB improvement percentage or V7
optimality gap can be reported. The full-domain audit is also inapplicable
because no restricted root exists.

## Same-model Complete MIP baseline

| Measure | 48 groups | 96 groups |
|---|---:|---:|
| Time to first incumbent | 57.81 s | 34.42 s |
| Final UB | 0.077957971 | 0.106229104 |
| Final LB | 0.071590940 | 0.094745558 |
| Final gap | 8.17% | 10.81% |
| Solver status | time limit | time limit |
| Stage wall time | 122.94 s | 126.18 s |

The baseline found and independently validated incumbents at both scales. Its
wall time exceeds the Gurobi limit because model construction and solution
validation are outside Gurobi's internal 120-second timer.

## Interpretation

The requested comparison cannot yet answer whether V7 produces a better UB
than Complete MIP at 48/96 groups. The current V7 flow produces no UB before its
quality-witness handoff fails, whereas Complete MIP does produce incumbents.
Treating the missing V7 values as poor numeric UBs would be incorrect.

This is a workflow-completeness failure at scale, not a root-CG or RIM
bottleneck. The next isolated redesign should make feasibility and quality two
separate witness responsibilities:

1. first obtain a validated feasibility witness quickly;
2. optionally improve its business quality within a soft budget or from a MIP
   start;
3. if quality improvement times out, retain the feasible witness and continue
   instead of discarding the remaining production budget.

That redesign is not implemented in this report.

Raw results:

- `preexperiment_outputs/v7_1_dynamic_budget_48_96_baseline_20260827/scale_g048_s601/result.json`
- `preexperiment_outputs/v7_1_dynamic_budget_48_96_baseline_20260827/scale_g096_s801/result.json`
