# V7.1 Shared-Budget 24-Group Report

## Question

Does replacing the fixed `10 + 10 + 60 + 40` phase limits with one 120-second
production budget improve the incumbent when the candidate domain and root
column pool are unchanged?

The production budget includes the quality witness, Stage 1, restricted root
CG, and Restricted Integer Master. The read-only global-pricing audit and the
same-model Complete MIP use separate time and do not reduce the production
budget.

## Implementation

- total production limit: 120 seconds;
- quality-witness / Stage-1 / root soft caps: 10 / 10 / 60 seconds;
- minimum protected RIM launch time: 5 seconds;
- RIM default ceiling: none; it receives all remaining time;
- after RIM model construction, the Gurobi time limit is recalibrated against
  the absolute production deadline;
- full-domain audit moved after RIM and remains read-only;
- an optional `integer_time_limit` is retained only for fixed-ceiling ablations.

No model, objective, candidate-domain, initial-column, pricing, or solver-seed
change was made.

## Budget execution

| Stage | Soft/allocated limit | Wall time |
|---|---:|---:|
| Quality witness | 10.00 s | 3.64 s |
| Stage 1 | 10.00 s | 10.35 s |
| Restricted root CG | 60.00 s | 7.83 s |
| Restricted Integer Master | 98.16 s | 98.19 s |
| Production total | 120.00 s | 120.04 s |
| Read-only global audit, excluded | -- | 0.87 s |
| Complete MIP, independent baseline | 120.00 s | 121.55 s wall |

The RIM model took about 1.96 seconds to build, so its solver received an
effective 96.20-second Gurobi limit. The production deadline overrun was only
0.04 seconds.

## Fixed versus shared budget

| Measure | Fixed 40-second RIM | Shared 120-second budget |
|---|---:|---:|
| Restricted group-area / group-bay edges | 114 / 1,082 | 114 / 1,082 |
| Restricted LP value | 0.038760095 | 0.038760095 |
| Root patterns / rounds | 1,601 / 17 | 1,601 / 17 |
| RIM solver time | 40.00 s | 96.21 s |
| RIM UB | 0.047316576 | 0.047316576 |
| RIM bound | 0.046277134 | 0.046471057 |
| RIM gap | 2.20% | 1.79% |
| Complete-MIP UB | 0.045754036 | 0.045754036 |
| V7 UB disadvantage | 3.42% | 3.42% |

An additional 56.20 solver seconds did not produce a better incumbent. It only
raised the restricted-master bound and reduced its internal gap by 0.41
percentage points (18.66% relative reduction).

## Residual bottleneck

The new restricted-master bound `0.046471057` is already 1.57% above the
Complete-MIP incumbent `0.045754036`. Therefore the unchanged restricted column
pool cannot reproduce that Complete-MIP incumbent, even if its integer search
is completed.

Of the observed objective difference `0.001562539`:

- at least `0.000717021` (45.89%) is now certified as restricted-pool loss;
- at most `0.000845519` (54.11%) remains unresolved integer search.

The read-only audit is identical to the fixed-budget run, as expected from the
identical root:

- global minimum reduced cost: `-0.001007880`;
- returned top-3 negative patterns: 600;
- affected excluded group-area pairs: 235;
- global root certified: no.

## Conclusion

The shared-budget policy is the more coherent default and enforces a fair total
algorithm budget, but it does not improve UB on this case. The experiment rules
out “the fixed 40-second RIM is the main UB bottleneck” as the next development
direction. Further RIM-only time increases are not justified before candidate
coverage is improved.

The next isolated experiment should keep this shared-budget controller and add
a small, controlled audit-guided expansion of excluded group-area candidates.
That mechanism is not implemented in this report.

Raw result:

`preexperiment_outputs/v7_1_dynamic_budget_24_audit_20260827/scale_g024_s401/result.json`
