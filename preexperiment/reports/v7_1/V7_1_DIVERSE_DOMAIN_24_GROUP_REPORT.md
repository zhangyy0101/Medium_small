# V7.1 Diverse Restricted-Domain 24-Group Report

## Scope

- case: `scale_g024_s401`
- solver threads / seed: `1 / 0`
- quality-witness / Stage 1 / restricted root / integer limits: `10 / 10 / 60 / 40` seconds
- quality-witness accepted MIP gap: `20%`
- Stage-1 pool: eight distinct `Y[group,area]` supports within absolute normalized-objective envelope `0.10`
- additional candidate-area cap: `2` beyond mandatory support
- minimum metadata-safety candidates: `1`
- Complete MIP limit: `120` seconds
- global pricing audit: one read-only exact full-domain sweep

The comparison run before this candidate-domain repair is recorded in
`V7_1_24_GROUP_GLOBAL_AUDIT_REPORT.md`. Both runs use the same materialized
case, objective, peak cap, solver thread count and seed.

## Implemented repair

1. Solve the compact witness with the business objective instead of stopping at
   the first feasible solution.
2. Tell the Gurobi pool to distinguish solutions by `Y[group,area]`; differences
   only in Stage-1 quantity and incidence variables are ignored.
3. Preserve every witness and Stage-1 best area as mandatory.
4. Interpret the cap as two *additional* areas, not a total per-group cap.
5. Fill the additional channel with structurally diverse pool alternatives and
   metadata-safety candidates ranked by reachable-capacity slack, compatible-bay
   count, existing-group proximity and berth distance.

No production full-domain expansion, F&O, local branching, stabilization, cut,
pricing-model or business-objective change was introduced.

## Candidate-domain diagnostics

| Measure | Before repair | After repair |
|---|---:|---:|
| Raw / unique Stage-1 pool supports | not distinguished | 8 / 8 |
| Mandatory witness + best union edges | 105 | 66 |
| Selected pool-alternative edges | 0 | 12 |
| Selected metadata-safety edges | 0 | 36 |
| Restricted group-area edges | 105 | 114 |
| Restricted group-bay edges | 664 | 1,082 |
| Full group-area / group-bay edges | 976 / 4,472 | 976 / 4,472 |
| Mean restricted areas per group | 4.375 | 4.750 |

The quality witness used 32 group-area edges and had objective `0.048802023` at
the configured 20% stopping gap. The Stage-1 best support used 37 edges; their
union contained 66 mandatory edges. The final domain therefore contains 66
mandatory, 12 pool and 36 safety edges.

## Runtime and optimization results

| Measure | Before repair | After repair | Complete MIP |
|---|---:|---:|---:|
| Witness wall time | 2.25 s | 3.65 s | -- |
| Stage-1 wall time | 10.34 s | 10.34 s | -- |
| Restricted-root wall time | 4.40 s | 8.23 s | -- |
| Restricted pricing rounds | 9 | 17 | -- |
| Final root patterns | 1,168 | 1,601 | -- |
| Restricted LP value | 0.054511809 | 0.038760095 | -- |
| Integer wall time | 13.78 s | 42.00 s | 121.61 s |
| Feasible UB | 0.063443491 | 0.047316576 | 0.045754036 |
| Solver gap at stop | 0.00% | 2.20% | 7.70% |

The repaired restricted-domain UB is **25.42% lower** than the previous V7.1
UB. Its remaining disadvantage relative to the 120-second Complete MIP
incumbent is **3.42%**, down from **38.66%**. The restricted root still closes
well inside the 60-second budget, although the richer domain roughly doubles
root time and makes the Restricted Integer Master use its full 40-second
budget.

The LP-to-incumbent difference is 18.08%. This is not a globally valid gap
because the read-only audit did not certify the full pricing domain.

The time-limited Restricted Integer Master bound is `0.046277134`. Therefore,
even a complete search over the unchanged restricted column pool cannot be
expected to match the Complete-MIP incumbent `0.045754036` unless that RIM bound
later proves invalid numerically. At least `0.000523097` (1.14% of the
Complete-MIP incumbent) is a demonstrated restricted-pool handicap; the other
`0.001039442` between the current RIM UB and its bound is unresolved integer
search. In terms of the observed `0.001562539` UB difference, this is a lower
bound of roughly 33.5% coverage loss and up to 66.5% unfinished RIM search.

For historical context, the earlier production-expanding/full-domain V7 run on
the same case obtained UB `0.044820909`, but spent 40.06 seconds in the root and
43.07 seconds in its integer master. The repaired frozen-domain root is 4.87x
faster, while its current UB is 5.57% worse. This historical run is useful as a
coverage oracle, not as evidence that the older active/full-domain cycling
workflow should be restored.

## Read-only global pricing audit

| Audit measure | Before repair | After repair |
|---|---:|---:|
| Exact minimum reduced cost | -0.004017860 | -0.001007880 |
| Returned top-3 negative patterns | 1,097 | 600 |
| Affected groups | 18 | 22 |
| Affected areas | 46 | 36 |
| Affected excluded group-area pairs | 283 | 235 |
| Global root certified | no | no |

The minimum omitted reduced-cost magnitude fell by about 75%, and the number
of returned negative patterns and excluded group-area pairs also fell. More
groups have at least one small omitted negative because the repaired restricted
dual solution differs from the earlier one; that count does not mean the total
coverage deterioration increased.

## Conclusion

The repair confirms the diagnosis: duplicate/low-diversity Stage-1 pool support
and the old hard candidate construction were the main cause of the catastrophic
UB degradation. The modified domain recovers most of the lost incumbent
quality while retaining a seconds-scale restricted root.

It is not yet a final paper algorithm. The full-domain audit still finds
material omitted improving columns, the Restricted Integer Master stops with a
2.20% internal gap, and only one 24-group seed has been tested. The next safe
experiment should separate these two residual causes: first allow the existing
RIM more time on the unchanged column pool, then test a small audit-guided
candidate expansion as an ablation. Neither is implemented in this report.

Raw result:

`preexperiment_outputs/v7_1_diverse_domain_24_audit_20260826/scale_g024_s401/result.json`

The proposed unchanged-pool/longer-RIM follow-up was subsequently completed
with a shared 120-second production budget. It did not improve the incumbent;
see `V7_1_DYNAMIC_BUDGET_24_GROUP_REPORT.md`.
