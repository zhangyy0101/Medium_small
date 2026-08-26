# V7.1 24-Group Restricted-CG and Global-Audit Report

## Scope

- case: `scale_g024_s401`
- solver threads: 1
- solver seed: 0
- preferred candidate-area cap: 4
- peak witness / Stage 1 / restricted root / integer limits: 10 / 10 / 60 / 40 seconds
- same-model Complete MIP limit: 120 seconds
- global pricing audit: one read-only exact sweep

## Restricted-domain construction

- full legal group-area edges: 976
- restricted group-area edges: 105 (89.24% reduction)
- full legal group-bay edges: 4,472
- restricted group-bay edges: 664 (85.15% reduction)
- witness mandatory edges: 68
- Stage-1 best mandatory edges: 37
- overlap between witness and best edges: 0
- selected optional pool edges: 0
- groups whose mandatory support exceeded cap: 7/24

The eight Stage-1 pool solutions did not introduce an area outside the best
support. Consequently, the intended near-optimal pool-alternative channel was
inactive in this run.

## Runtime and solution

| Component | Result |
|---|---:|
| Stage 1 | 10.34 s (time limit) |
| Restricted root | 4.18 s internal / 4.40 s stage wall |
| Restricted pricing rounds | 9 |
| Initial / final patterns | 269 / 1,168 |
| Global audit | 0.82 s |
| Restricted Integer Master | 11.99 s solver / 13.78 s stage wall |
| Restricted LP bound | 0.054511809 |
| Restricted integer UB | 0.063443491 |
| LP-to-integer restricted gap | 14.08% |
| Restricted MIP solver gap | 0.00% (restricted optimum proved) |
| Complete MIP UB | 0.045754036 |
| Complete MIP LB | 0.042283480 |
| Complete MIP gap | 7.59% at 120 s |

The V7.1 incumbent is 38.66% worse than the Complete MIP incumbent. Compared
with the earlier full-domain-certified V7 run on the same materialized case,
the restricted root is 9.52x faster and uses 77.03% fewer patterns, but its UB
is 41.55% worse.

## Read-only global pricing audit

- exact global minimum reduced cost: `-0.004017860`
- returned negative patterns: 1,097
- affected groups: 18/24
- affected areas: 46
- affected excluded group-area pairs: 283
- globally root certified: NO
- production restricted mapping/pattern pool mutated: NO

The negative-pattern count is the number returned under the configured top-3
per-bay audit sweep; it is not an exhaustive count of every negative pattern.
The exact negative minimum reduced cost is sufficient to disprove global root
closure.

## Conclusion

V7.1 successfully removes the active/full-domain cycling bottleneck, but the
current Stage-1 restricted-domain construction sacrifices too much solution
quality. The present setting is not suitable as the default paper algorithm:
the optional solution-pool channel contributed no new areas, and the audit
shows widespread improving columns outside the frozen domain. No automatic
domain expansion was performed in this experiment.
