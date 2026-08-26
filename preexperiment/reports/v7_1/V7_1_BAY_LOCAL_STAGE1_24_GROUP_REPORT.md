# V7.1 Bay-local Stage-1 Repair: 24-group Result

## Purpose

The prior oracle/coverage audit showed that the coarse Stage-1 pool omitted the
cross-group area combinations used by the 120-second Complete-MIP incumbent.
Cap 2 and cap 3 both covered only 4 of 28 oracle group-area pairs. This run
tests the corresponding Stage-1 repair only; the Complete MIP was not rerun.

## Repair

- Keep every area in the Stage-1 best support mandatory.
- Retain at most one optional area from the coarse Stage-1 solution pool.
- Fill the remaining optional slots from a bay-local ranking.
- For each group-area pair, greedily place that group on actual compatible
  anchor bays under three deterministic orderings and rank the best plan by
  shortage, then the formal flow, extra-bay, and span objective terms.
- Use five optional candidate areas per group. Stage-1 quantities remain
  guidance and are not fixed in Stage 2.

## Configuration

- case: `scale_g024_s401`
- V7 production budget: 120 seconds
- witness / Stage 1 / restricted root soft limits: 10 / 10 / 60 seconds
- Stage-1 pool solutions: 8
- additional candidate-area cap: 5
- maximum coarse-pool candidates: 1
- threads / seed: 1 / 0
- saved Complete-MIP reference: UB `0.045754036252225815`, 120.016 seconds

## Results

| Metric | Prior coarse Stage 1 | Bay-local repair |
|---|---:|---:|
| Oracle group-area pairs covered | 4 / 28 | 22 / 28 |
| Oracle quantity covered | 85 / 640 | 534 / 640 |
| Oracle groups fully covered | 3 / 24 | 18 / 24 |
| Restricted group-area edges | 85 | 157 |
| Restricted group-bay edges | 881 | 762 |
| Restricted-root rounds | 6 | 23 |
| Restricted-root closure time | — | 9.366 s |
| Restricted LP value | 0.052099 | 0.038496 |
| V7 feasible UB | 0.060389 | 0.045080 |
| V7 restricted MIP gap | — | 2.920% |

The repaired UB is 25.35% below the prior V7 UB and 1.47% below the saved
Complete-MIP incumbent. The integer master used 97.54 seconds and ended at its
time limit. The six historical oracle pairs still outside the repaired domain
carry 106 boxes in total.

## Interpretation boundary

This result confirms that Stage-1 candidate coverage, rather than the column
generation skeleton itself, was the direct cause of the earlier UB degradation.
The repair materially improves the incumbent on this seed while keeping 83.9%
of full group-area edges outside the restricted domain.

The reported 2.920% gap is a restricted-master gap, not a global optimality gap.
The restricted LP is not a valid lower bound for the full V7 model without a
full-domain certificate. This is one 24-group seed and therefore a directional
validation only; it is not a paper-level performance claim.
