# V7.1 Bay-local Stage-1: 48/96-group Check

## Scope

- cases: `scale_g048_s601` and `scale_g096_s801`;
- production budget: 120 seconds per case;
- witness / Stage 1 / restricted-root soft limits: 10 / 10 / 60 seconds;
- Stage-1 additional-area cap: 5;
- maximum coarse-pool alternatives: 1;
- threads / solver seed: 1 / 0;
- Complete MIP was not rerun. The comparison uses the previously saved
  same-model 120-second incumbents.

## Results

| Measure | 48 groups | 96 groups |
|---|---:|---:|
| Restricted group-area edges | 297 | 601 |
| Average active areas/group | 6.19 | 6.26 |
| Root rounds completed | 44 | 34 |
| Root time | 39.22 s | 61.67 s |
| Final root pattern count | 3,823 | 7,105 |
| Restricted root closed | yes | no |
| Restricted LP / last RMP value | 0.070954 | 0.102484 |
| V7 feasible UB | 0.079309 | unavailable |
| V7 restricted MIP gap | 5.75% | unavailable |
| Saved Complete-MIP UB | 0.077958 | 0.106229 |
| V7 difference from saved MIP | +1.73% | not comparable |

The repaired 48-group UB improves by 5.20% over the earlier V7 result
`0.083663`, but remains 1.73% above the saved Complete-MIP incumbent. Its root
closed within the soft limit, leaving 64.79 seconds for the integer master.

The 96-group root did not close before its declared limit. At round 34 the last
RMP value was `0.102484`, the exact restricted pricing sweep still found
minimum reduced cost `-0.0001452`, and the round added 53 columns. Of the root
time, 41.47 seconds were spent in pricing and 14.38 seconds in master solves.
Because negative reduced-cost columns remained, this value is not a completed
restricted-root result and no RIM UB or gap is reported.

## Interpretation

The bay-local Stage-1 repair continues to improve primal quality at 48 groups,
but it does not yet yield a complete 96-group pipeline under the current phase
budget. The bottleneck has moved from candidate-area coverage to restricted
pricing/root closure as the number of groups doubles. The 96-group failure is
therefore a scalability result, not evidence that the model is infeasible or
that its UB is worse than Complete MIP.

No parameter was relaxed and no incomplete root was converted into a formal
result. These remain two single-seed diagnostics rather than paper-level
performance evidence.

Raw results:

- `preexperiment_outputs/v7_1_bay_local_stage1_48_96_20260827/scale_g048_s601/result.json`
- `preexperiment_outputs/v7_1_bay_local_stage1_48_96_20260827/scale_g096_s801/result.json`
